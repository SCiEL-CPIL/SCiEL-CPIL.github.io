import time
import random
import numpy as np
from itertools import combinations

def _count_c6s_fast(colors, n):
    """Count total C6s across all 4 color classes using numpy."""
    total = 0
    adj_mats = [np.zeros((n, n), dtype=np.int32) for _ in range(4)]
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj_mats[c][i, j] = 1
            adj_mats[c][j, i] = 1
            idx += 1
    
    for c in range(4):
        A = adj_mats[c]
        # Count C6s: for each pair (u,v) that is an edge, count paths of length 4 from u to v
        # that don't use u or v internally. Each C6 is counted 6 times (once per edge).
        # So C6_count = sum over edges (u,v) of (paths of length 4 from u to v avoiding u,v) / 6
        # But we need to be careful: paths of length 4 from u to v that avoid u and v internally.
        # A path of length 4 from u to v: u-a-b-c-v where a,b,c are distinct and not u or v.
        
        # Use matrix powers: (A^4)[u,v] counts walks of length 4 from u to v.
        # But we need simple paths, not walks. For n <= 20, let's use a different approach.
        
        # Alternative: enumerate all 6-subsets and check if they form a C6.
        # C(n,6) for n=20 is 38760, which is manageable.
        
        # For each 6-subset, check if the induced subgraph is a C6 (each vertex has degree 2 in the induced subgraph).
        # Actually, we need to check if there exists a Hamiltonian cycle in the induced subgraph on 6 vertices.
        # For 6 vertices, we can check all (6-1)!/2 = 60 possible cycles.
        
        # But that's 38760 * 60 = 2.3M operations per color, times 4 = 9.2M. Might be slow.
        
        # Better: use the formula C6 = (1/12) * trace(A^6) - ... but that counts walks.
        
        # Let's use a direct enumeration with pruning for n <= 20.
        # For each starting vertex s (smallest in cycle), do DFS to find paths of length 5
        # where all vertices are distinct and > s (except s), and the last vertex is adjacent to s.
        
        count = 0
        for s in range(n):
            neighbors = [v for v in range(s+1, n) if A[s, v]]
            if not neighbors:
                continue
            
            # DFS from each neighbor of s, looking for paths of length 4 more
            # (total 5 edges, 6 vertices), ending at a vertex adjacent to s
            # Use iterative DFS for speed
            
            for start_n in neighbors:
                # Path: s -> start_n -> ... -> last, where last is adjacent to s
                # We need 4 more edges after start_n
                stack = [(start_n, 1, [start_n])]
                while stack:
                    cur, depth, path = stack.pop()
                    if depth == 5:
                        if A[s, cur]:
                            count += 1
                        continue
                    for nxt in range(s+1, n):
                        if nxt not in path and A[cur, nxt]:
                            stack.append((nxt, depth+1, path + [nxt]))
        
        total += count
    
    return total

def _has_c6_fast(colors, n):
    """Check if any color class contains a C6."""
    return _count_c6s_fast(colors, n) > 0

def _find_conflicting_edges(colors, n):
    """Find edges that are part of at least one C6. Returns set of edge indices."""
    conflicting = set()
    adj_mats = [np.zeros((n, n), dtype=np.int32) for _ in range(4)]
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj_mats[c][i, j] = 1
            adj_mats[c][j, i] = 1
            idx += 1
    
    for c in range(4):
        A = adj_mats[c]
        for s in range(n):
            neighbors = [v for v in range(s+1, n) if A[s, v]]
            if not neighbors:
                continue
            for start_n in neighbors:
                stack = [(start_n, 1, [start_n])]
                while stack:
                    cur, depth, path = stack.pop()
                    if depth == 5:
                        if A[s, cur]:
                            # Found a C6: s, start_n, ..., cur
                            # Mark all edges in this cycle
                            cycle = [s] + path
                            for k in range(6):
                                u, v = cycle[k], cycle[(k+1) % 6]
                                if u < v:
                                    # Find edge index
                                    edge_idx = u * (n - 1) - u * (u - 1) // 2 + (v - u - 1)
                                    conflicting.add(edge_idx)
                        continue
                    for nxt in range(s+1, n):
                        if nxt not in path and A[cur, nxt]:
                            stack.append((nxt, depth+1, path + [nxt]))
    
    return conflicting

def _simulated_annealing(colors, n, max_steps=20000, deadline=None, initial_temp=1.0):
    """Simulated annealing to minimize C6 count."""
    colors = list(colors)
    best_colors = list(colors)
    best_c6_count = _count_c6s_fast(colors, n)
    
    if best_c6_count == 0:
        return colors
    
    step = 0
    temp = initial_temp
    cooling_rate = 0.9995
    
    while step < max_steps:
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random edge to flip, with bias toward conflicting edges
        conflicting = _find_conflicting_edges(colors, n)
        if conflicting:
            # 70% chance to pick from conflicting edges
            if random.random() < 0.7:
                idx = random.choice(list(conflicting))
            else:
                idx = random.randint(0, len(colors) - 1)
        else:
            idx = random.randint(0, len(colors) - 1)
        
        old_color = colors[idx]
        new_color = random.choice([c for c in range(4) if c != old_color])
        colors[idx] = new_color
        
        new_c6_count = _count_c6s_fast(colors, n)
        delta = new_c6_count - best_c6_count
        
        if delta < 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
            if new_c6_count < best_c6_count:
                best_c6_count = new_c6_count
                best_colors = list(colors)
                if best_c6_count == 0:
                    return colors
        else:
            colors[idx] = old_color
        
        temp *= cooling_rate
        step += 1
    
    return best_colors

def _backtrack_search(n, deadline=None, max_nodes=50000):
    """Backtracking search for a valid 4-coloring of K_n with no C6 in any color class."""
    total_edges = n * (n - 1) // 2
    colors = [0] * total_edges
    
    # Precompute for each edge, which other edges it shares a vertex with
    # This helps with pruning
    
    def edge_to_vertices(idx):
        # Convert edge index to (i, j)
        # Edge indices: (0,1)=0, (0,2)=1, ..., (0,n-1)=n-2, (1,2)=n-1, ...
        # i * (n-1) - i*(i-1)//2 + (j-i-1) = idx
        # Solve for i: i*(n-1) - i*(i-1)//2 <= idx
        # Let's just iterate
        for i in range(n):
            base = i * (n - 1) - i * (i - 1) // 2
            if base <= idx < base + (n - 1 - i):
                j = i + 1 + (idx - base)
                return i, j
        return None
    
    # For each edge, precompute its vertices
    edge_vertices = [edge_to_vertices(i) for i in range(total_edges)]
    
    # For each vertex, precompute which edges are incident to it
    vertex_edges = {v: [] for v in range(n)}
    for idx, (i, j) in enumerate(edge_vertices):
        vertex_edges[i].append(idx)
        vertex_edges[j].append(idx)
    
    # For each color, maintain adjacency matrix
    adj = [np.zeros((n, n), dtype=np.int32) for _ in range(4)]
    
    def check_c6(c, u, v):
        """Check if adding edge (u,v) to color c creates a C6."""
        # A C6 containing (u,v) in color c: u-v-a-b-c-d-u where all edges are in color c
        # We need to check if there's a path of length 4 from u to v in color c
        # that doesn't use u or v internally.
        # After adding (u,v), we check if there's a path of length 4 from u to v
        # in the graph with (u,v) added.
        
        # Temporarily add (u,v)
        adj[c][u, v] = 1
        adj[c][v, u] = 1
        
        # Count paths of length 4 from u to v that don't use u or v internally
        # Path: u-a-b-c-v where a,b,c are distinct, not u or v
        count = 0
        for a in range(n):
            if a == u or a == v or not adj[c][u, a]:
                continue
            for b in range(n):
                if b == u or b == v or b == a or not adj[c][a, b]:
                    continue
                for c2 in range(n):
                    if c2 == u or c2 == v or c2 == a or c2 == b or not adj[c][b, c2]:
                        continue
                    if adj[c][c2, v]:
                        count += 1
        
        # Remove (u,v)
        adj[c][u, v] = 0
        adj[c][v, u] = 0
        
        return count > 0
    
    node_count = 0
    
    def backtrack(idx):
        nonlocal node_count
        if deadline is not None and time.time() > deadline:
            return False
        if idx == total_edges:
            return True
        
        node_count += 1
        if node_count > max_nodes:
            return False
        
        i, j = edge_vertices[idx]
        
        for c in range(4):
            # Check if assigning color c to edge (i,j) creates a C6
            if check_c6(c, i, j):
                continue
            
            # Assign
            colors[idx] = c
            adj[c][i, j] = 1
            adj[c][j, i] = 1
            
            if backtrack(idx + 1):
                return True
            
            # Undo
            adj[c][i, j] = 0
            adj[c][j, i] = 0
            colors[idx] = 0
        
        return False
    
    if backtrack(0):
        return colors
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Known valid coloring for n=9
    n0 = 9
    colors0 = [min(j - i, n0 - j + i) - 1 for i in range(n0) for j in range(i + 1, n0)]
    
    best_n = n0
    best_colors = list(colors0)
    
    # Try to extend from n=9 up to n=17 using simulated annealing
    for target_n in range(n0 + 1, 18):
        if time.time() > deadline - 15:
            break
        
        # Try multiple random restarts
        found = False
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            # Initialize: use colors0 for edges within first n0 vertices, random for new edges
            colors = []
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if j < n0:
                        idx0 = i * (n0 - 1) - i * (i - 1) // 2 + (j - i - 1)
                        colors.append(colors0[idx0])
                    else:
                        colors.append(random.randint(0, 3))
            
            # Run simulated annealing
            result = _simulated_annealing(colors, target_n, max_steps=15000, deadline=deadline - 5)
            
            if _count_c6s_fast(result, target_n) == 0:
                best_n = target_n
                best_colors = result
                found = True
                break
        
        if not found:
            # If we can't find a valid coloring for this n, try a few more restarts
            # with different initializations
            for attempt in range(3):
                if time.time() > deadline - 8:
                    break
                
                # Fully random initialization
                colors = [random.randint(0, 3) for _ in range(target_n * (target_n - 1) // 2)]
                result = _simulated_annealing(colors, target_n, max_steps=10000, deadline=deadline - 5)
                
                if _count_c6s_fast(result, target_n) == 0:
                    best_n = target_n
                    best_colors = result
                    found = True
                    break
            
            if not found:
                # Give up on this n and larger ones
                break
    
    # If we found n=17, try n=18 with remaining time
    if best_n == 17 and time.time() < deadline - 20:
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            
            # Initialize from best_colors for first 17 vertices
            colors = []
            for i in range(18):
                for j in range(i + 1, 18):
                    if j < 17:
                        idx0 = i * (17 - 1) - i * (i - 1) // 2 + (j - i - 1)
                        colors.append(best_colors[idx0])
                    else:
                        colors.append(random.randint(0, 3))
            
            result = _simulated_annealing(colors, 18, max_steps=20000, deadline=deadline - 5)
            
            if _count_c6s_fast(result, 18) == 0:
                best_n = 18
                best_colors = result
                break
    
    return {'n': best_n, 'colors': best_colors}
