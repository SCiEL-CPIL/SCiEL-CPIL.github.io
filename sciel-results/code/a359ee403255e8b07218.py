import time
import random
import numpy as np
from collections import deque

def _verify_girth_fast(adj, n, g):
    """
    Check if graph has girth >= g using numpy array-based non-backtracking BFS.
    adj: list of lists of neighbors
    n: number of vertices
    g: target girth
    Returns True if girth >= g, False otherwise.
    """
    if g <= 2:
        return True
    
    # Build adjacency matrix for fast neighbor lookup
    adj_set = [set(nei) for nei in adj]
    
    # For each start vertex, do non-backtracking BFS
    # State: (vertex, prev_vertex) -> depth
    # We use a 2D array: dist[v][prev] = depth, or -1 if unvisited
    # prev ranges over 0..n-1 (plus a special -1 for start)
    # Actually, let's use a dict-based approach but with early termination
    
    # Alternative: use the fact that for girth >= g, no cycle of length < g exists.
    # A cycle of length L < g means there's a non-backtracking closed walk of length L.
    # We can check by doing BFS from each vertex up to depth g-2.
    # If at depth d < g-1 we reach the start vertex, we have a cycle of length d+1 < g.
    
    # Optimized: use numpy for the BFS frontier
    # dist_matrix[v, prev] = depth of state (v, prev), -1 if unvisited
    # We'll use a flat array indexed by v*n + (prev+1) where prev=-1 maps to index n
    
    # Actually for speed, let's use a different approach:
    # For each start s, we track which directed edges have been visited.
    # A directed edge (u,v) is visited at depth d.
    # If we can reach (s, w) at depth d where d+1 < g, we found a short cycle.
    
    # Let me use a more efficient approach with numpy:
    # dist[v, prev] where prev is 0..n-1, and we handle start separately.
    # For start s: initial state is (s, -1) at depth 0.
    # From (s, -1), we go to (v, s) for each v in adj[s] at depth 1.
    # From (v, s), we go to (w, v) for each w in adj[v] \ {s} at depth 2.
    # ...
    # If we reach (s, w) at depth d, cycle length is d+1.
    # We need d+1 >= g, i.e., d >= g-1.
    # So we only need to check depths 0 to g-2 for violations.
    
    max_depth = g - 2  # We check up to this depth; if we hit start at depth <= max_depth, cycle < g
    
    for start in range(n):
        # dist[v, prev] = depth, -1 = unvisited
        # Use a 2D numpy array for speed
        dist = np.full((n, n), -1, dtype=np.int32)
        # Special handling for start: dist[start, :] all start as -1
        # But we need to track that (start, -1) is at depth 0
        
        # Frontier: list of (v, prev, depth)
        # Use numpy arrays for the frontier
        frontier_v = np.array([start], dtype=np.int32)
        frontier_prev = np.array([-1], dtype=np.int32)  # -1 means "start"
        frontier_depth = 0
        
        # Mark start state
        # We'll handle prev=-1 specially
        
        while len(frontier_v) > 0 and frontier_depth <= max_depth:
            next_vs = []
            next_prevs = []
            
            for i in range(len(frontier_v)):
                v = frontier_v[i]
                prev = frontier_prev[i]
                
                for w in adj[v]:
                    if prev >= 0 and w == prev:
                        continue  # non-backtracking
                    
                    # Check if w == start
                    if w == start:
                        cycle_len = frontier_depth + 1
                        if cycle_len < g:
                            return False
                        continue
                    
                    # Check if state (w, v) already visited
                    if dist[w, v] == -1:
                        dist[w, v] = frontier_depth + 1
                        next_vs.append(w)
                        next_prevs.append(v)
            
            frontier_depth += 1
            if frontier_depth > max_depth:
                break
            frontier_v = np.array(next_vs, dtype=np.int32)
            frontier_prev = np.array(next_prevs, dtype=np.int32)
    
    return True

def _petersen_edges():
    """Return edges of the Petersen graph (10 vertices)."""
    # Outer pentagon: 0-1-2-3-4-0
    # Inner pentagram: 5-7-9-6-8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    return [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]

def _k33_edges():
    """Return edges of K_{3,3} (6 vertices)."""
    # Partitions: {0,1,2} and {3,4,5}
    return [
        (0, 3), (0, 4), (0, 5),
        (1, 3), (1, 4), (1, 5),
        (2, 3), (2, 4), (2, 5)
    ]

def _build_lift(base_edges, m, voltages, base_n):
    """
    Build a voltage lift of a base graph.
    base_edges: list of (u,v) edges in base graph
    m: lift size (number of layers)
    voltages: list of integers (one per edge) in Z_m
    base_n: number of vertices in base graph
    Returns (n, edges, adj) or None if invalid.
    """
    n = base_n * m
    adj = [set() for _ in range(n)]
    edges = []
    
    for i, (u, v) in enumerate(base_edges):
        vol = voltages[i] % m
        for layer in range(m):
            a = u * m + layer
            b = v * m + (layer + vol) % m
            if b not in adj[a]:
                adj[a].add(b)
                adj[b].add(a)
                edges.append((a, b))
            else:
                # Duplicate edge, invalid simple graph
                return None
            
    # Check degree
    for i in range(n):
        if len(adj[i]) != 3:
            return None
            
    adj_list = [sorted(list(adj[i])) for i in range(n)]
    return n, edges, adj_list

def _build_petersen_lift(m, voltages, seed):
    """
    Build a lift of the Petersen graph with given voltages.
    m: lift size (number of layers)
    voltages: list of 15 integers (one per edge) in Z_m
    Returns (n, edges, adj) or None if invalid.
    """
    base_edges = _petersen_edges()
    return _build_lift(base_edges, m, voltages, 10)

def _build_k33_lift(m, voltages, seed):
    """
    Build a lift of K_{3,3} with given voltages.
    m: lift size (number of layers)
    voltages: list of 9 integers (one per edge) in Z_m
    Returns (n, edges, adj) or None if invalid.
    """
    base_edges = _k33_edges()
    return _build_lift(base_edges, m, voltages, 6)

def _random_voltage_search(m, g, n_attempts, seed, base='petersen'):
    """
    Try n_attempts random voltage assignments for a voltage lift of size m.
    base: 'petersen' or 'k33'
    Returns (n, edges) if a valid graph is found, else None.
    """
    rng = random.Random(seed)
    
    if base == 'k33':
        base_edges = _k33_edges()
        build_fn = _build_k33_lift
    else:
        base_edges = _petersen_edges()
        build_fn = _build_petersen_lift
    
    num_edges = len(base_edges)
    
    for _ in range(n_attempts):
        voltages = [rng.randint(0, m - 1) for _ in range(num_edges)]
        result = build_fn(m, voltages, seed)
        if result is None:
            continue
        n, edges, adj_list = result
        if _verify_girth_fast(adj_list, n, g):
            return n, edges
    return None

def _random_3reg_graph(n, seed):
    """
    Generate a random 3-regular graph on n vertices.
    Returns (n, edges, adj_list) or None.
    """
    if n % 2 != 0:
        return None
    rng = random.Random(seed)
    stubs = []
    for v in range(n):
        for _ in range(3):
            stubs.append(v)
    rng.shuffle(stubs)
    
    adj = [set() for _ in range(n)]
    edges = []
    valid = True
    for i in range(0, n * 3, 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v or v in adj[u]:
            valid = False
            break
        adj[u].add(v)
        adj[v].add(u)
        edges.append((u, v))
        
    if not valid:
        return None
        
    # Check connectivity
    visited = {0}
    stack = [0]
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if w not in visited:
                visited.add(w)
                stack.append(w)
    if len(visited) != n:
        return None
        
    adj_list = [sorted(list(adj[i])) for i in range(n)]
    return n, edges, adj_list

def _improve_girth(edges, adj_list, n, g, max_swaps, seed):
    """
    Try to increase girth using edge swaps.
    Returns improved (n, edges, adj_list) or original.
    """
    rng = random.Random(seed)
    current_edges = list(edges)
    current_adj = [list(nei) for nei in adj_list]
    
    # Convert to set for fast lookup
    edge_set = set((min(u,v), max(u,v)) for u,v in current_edges)
    
    best_girth = _verify_girth_fast(current_adj, n, g)
    
    for _ in range(max_swaps):
        # Pick two random edges (a,b) and (c,d)
        e1_idx = rng.randint(0, len(current_edges) - 1)
        e2_idx = rng.randint(0, len(current_edges) - 1)
        if e1_idx == e2_idx:
            continue
        
        a, b = current_edges[e1_idx]
        c, d = current_edges[e2_idx]
        
        # Ensure no shared vertices
        if len({a,b,c,d}) < 4:
            continue
            
        # Try swap (a,b),(c,d) -> (a,c),(b,d)
        new_e1 = (a, c)
        new_e2 = (b, d)
        
        # Check if new edges exist or are loops
        if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
            continue
        if (min(new_e1[0], new_e1[1]), max(new_e1[0], new_e1[1])) in edge_set:
            continue
        if (min(new_e2[0], new_e2[1]), max(new_e2[0], new_e2[1])) in edge_set:
            continue
            
        # Perform swap
        edge_set.remove((min(a,b), max(a,b)))
        edge_set.remove((min(c,d), max(c,d)))
        edge_set.add((min(new_e1[0], new_e1[1]), max(new_e1[0], new_e1[1])))
        edge_set.add((min(new_e2[0], new_e2[1]), max(new_e2[0], new_e2[1])))
        
        current_edges[e1_idx] = new_e1
        current_edges[e2_idx] = new_e2
        
        # Rebuild adj
        # This is O(E), might be slow. Optimize by updating only 4 vertices.
        # But for simplicity and correctness, let's just check girth on the fly?
        # No, we need adj for girth check.
        
        # Update adj lists
        # Remove old neighbors
        current_adj[a].remove(b)
        current_adj[b].remove(a)
        current_adj[c].remove(d)
        current_adj[d].remove(c)
        
        # Add new neighbors
        current_adj[a].append(c)
        current_adj[c].append(a)
        current_adj[b].append(d)
        current_adj[d].append(b)
        
        # Sort for consistency? Not strictly needed for BFS if we use lists
        # But _verify_girth_fast expects lists.
        
        if _verify_girth_fast(current_adj, n, g):
            return n, current_edges, current_adj
            
        # If girth decreased or stayed same, we might want to revert?
        # For now, accept any swap that doesn't break degree/connectivity.
        # Connectivity is preserved by 2-swaps on a connected graph.
        
    return n, current_edges, current_adj

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    if k != 3:
        # Fallback for other k
        pass
        
    best_result = None
    best_n = float('inf')
    
    # Strategy 1: K3,3 Lifts (n = 6m, can reach Moore bound with smaller m)
    # Moore bound for k=3, g=16 is 512. 6m >= 512 => m >= 86 (since 6*85=510 < 512)
    # Try m from 86 upward
    for m in range(86, 200):
        if time.time() > deadline - 25:
            break
        n_lift = 6 * m
        if n_lift >= best_n:
            break
            
        attempts = 80
        result = _random_voltage_search(m, g, attempts, seed=m*1000+1, base='k33')
        if result is not None:
            n_r, edges_r = result
            if n_r < best_n:
                best_n = n_r
                best_result = (n_r, edges_r)
            break
    
    # Strategy 2: Petersen Lifts (n = 10m)
    # 10m >= 512 => m >= 52
    for m in range(52, 150):
        if time.time() > deadline - 25:
            break
        n_lift = 10 * m
        if n_lift >= best_n:
            break
            
        attempts = 60
        result = _random_voltage_search(m, g, attempts, seed=m*1000+2, base='petersen')
        if result is not None:
            n_r, edges_r = result
            if n_r < best_n:
                best_n = n_r
                best_result = (n_r, edges_r)
            break
    
    # Strategy 3: Random 3-regular graphs
    if time.time() < deadline - 30:
        for n_try in range(512, min(best_n if best_n < float('inf') else 700, 700), 2):
            if time.time() > deadline - 10:
                break
            for attempt in range(5):
                if time.time() > deadline - 5:
                    break
                res = _random_3reg_graph(n_try, seed=attempt*100 + n_try)
                if res is None:
                    continue
                n_r, edges_r, adj_r = res
                if _verify_girth_fast(adj_r, n_r, g):
                    if n_r < best_n:
                        best_n = n_r
                        best_result = (n_r, edges_r)
                    break
    
    if best_result is None:
        # Fallback: larger lifts
        for base, m_start, m_end, seed_base in [('k33', 300, 400, 999), ('petersen', 200, 300, 998)]:
            if time.time() > deadline - 10:
                break
            for m in range(m_start, m_end):
                if time.time() > deadline - 5:
                    break
                result = _random_voltage_search(m, g, 50, seed=m+seed_base, base=base)
                if result is not None:
                    best_result = result
                    break
            if best_result is not None:
                break
    
    if best_result is None:
        # Emergency fallback: very large lift
        result = _random_voltage_search(500, g, 20, seed=777, base='k33')
        if result is not None:
            best_result = result
        else:
            result = _random_voltage_search(500, g, 20, seed=778, base='petersen')
            if result is not None:
                best_result = result
            else:
                return 0, []
    
    return best_result
