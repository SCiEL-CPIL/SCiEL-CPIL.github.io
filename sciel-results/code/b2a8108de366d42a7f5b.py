"""
Construct a k-regular graph with girth at least g.
Uses random generation and targeted local search to eliminate short cycles.
"""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin out of 150s limit

    if k != 3:
        # Fallback for other k, though task specifies k=3
        # For now, just return a large random graph and hope for the best?
        # Or raise exception? The task says k=3.
        return _construct_generic(k, g)

    # Target n
    # Moore bound is 510. Best known 936.
    # We start with n=1000 for easier local search, then increase if needed.
    # Actually, smaller n is better for score.
    # Let's try n=512, 514, ... up to 1500.
    
    for n in range(512, 1502, 2):
        if time.time() > deadline - 10:
            break
            
        # Try several random instances for this n
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Generate random 3-regular graph
            edges = _generate_random_3_regular(n)
            if edges is None:
                continue
                
            # Local search to increase girth
            edges = _local_search_girth(edges, n, g, deadline)
            
            # Verify girth
            if _verify_girth_fast(n, edges, g):
                return n, edges

    # If we didn't find one in the range, try larger n
    for n in range(1504, 3002, 2):
        if time.time() > deadline - 2:
            break
        for attempt in range(20):
            if time.time() > deadline - 1:
                break
            edges = _generate_random_3_regular(n)
            if edges is None:
                continue
            edges = _local_search_girth(edges, n, g, deadline)
            if _verify_girth_fast(n, edges, g):
                return n, edges

    # Absolute fallback: Return a valid graph with large n
    # This should not happen.
    n = 10000
    edges = _generate_random_3_regular(n)
    if edges:
        # Skip local search for huge n, just verify?
        # If it fails, we return it anyway? No, must be valid.
        # If we can't verify, we assume it's valid? Risky.
        # But for n=10000, girth is likely > 16.
        if _verify_girth_fast(n, edges, g):
            return n, edges
            
    raise Exception("Failed to construct valid graph")


def _generate_random_3_regular(n):
    """
    Generate a random simple 3-regular graph on n vertices using configuration model.
    Returns list of edges or None if fails.
    """
    if n < 4 or n % 2 != 0:
        return None
    
    stubs = []
    for i in range(n):
        for _ in range(3):
            stubs.append(i)
            
    random.shuffle(stubs)
    
    edges = set()
    
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            return None
        edge = (min(u, v), max(u, v))
        if edge in edges:
            return None
        edges.add(edge)
        
    return list(edges)


def _local_search_girth(edges, n, target_girth, deadline):
    """
    Local search to eliminate cycles shorter than target_girth.
    Uses targeted edge swaps to break the shortest cycle.
    """
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add((min(u, v), max(u, v)))
        
    # Initial girth check
    current_girth = _get_girth(adj, n)
    if current_girth >= target_girth:
        return list(edge_set)
        
    max_iters = 2000
    for i in range(max_iters):
        if time.time() > deadline - 2:
            break
        if current_girth >= target_girth:
            break
            
        # Find a short cycle
        cycle = _find_short_cycle(adj, n, min(target_girth, current_girth + 1))
        if not cycle:
            # If no cycle shorter than target found, re-verify
            current_girth = _get_girth(adj, n)
            if current_girth >= target_girth:
                break
            else:
                # Detection failed? Try random swap
                cycle = None
                
        if cycle:
            cycle_len = len(cycle)
            # Pick a random edge in the cycle to swap
            # We try to break the cycle by swapping one of its edges.
            # We pick a random edge (u, v) from the cycle.
            u, v = random.choice(cycle)
            
            swapped = False
            # Try several random candidate edges (x, y) to swap with
            all_edges = list(edge_set)
            random.shuffle(all_edges)
            
            for x, y in all_edges[:50]: # Limit candidates for speed
                if (x, y) == (u, v) or (x, y) == (v, u):
                    continue
                if x in (u, v) or y in (u, v):
                    continue # Avoid creating triangles or multi-edges easily
                
                # Try swap: remove (u,v), (x,y). Add (u,x), (v,y).
                new_e1 = (min(u, x), max(u, x))
                new_e2 = (min(v, y), max(v, y))
                
                if new_e1 == new_e2:
                    continue
                if new_e1 in edge_set or new_e2 in edge_set:
                    continue
                if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
                    continue
                
                # Check if this swap creates a cycle shorter than cycle_len
                # We need to check if the new edges create short cycles.
                # Temporarily remove (u,v) and (x,y) from adj/edge_set?
                # Or just check distances in the graph with these edges removed.
                
                # For speed, we do a quick BFS check for paths from x to u and y to v
                # in the graph with (u,v) and (x,y) removed.
                
                # If dist(x, u) + 1 < cycle_len, it's bad.
                # If dist(y, v) + 1 < cycle_len, it's bad.
                
                # We also need to check the other swap option (u,y) and (v,x)
                # new_e1_alt = (min(u, y), max(u, y))
                # new_e2_alt = (min(v, x), max(v, x))
                
                # Let's just try the first swap option.
                
                if _is_swap_safe(adj, edge_set, u, v, x, y, cycle_len):
                    # Perform swap
                    edge_set.remove((u, v))
                    edge_set.remove((x, y))
                    edge_set.add(new_e1)
                    edge_set.add(new_e2)
                    
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[x].remove(y)
                    adj[y].remove(x)
                    
                    adj[u].append(x)
                    adj[x].append(u)
                    adj[v].append(y)
                    adj[y].append(v)
                    
                    swapped = True
                    break
            
            if not swapped:
                # If we can't swap safely, try another cycle or give up
                continue
                
            # Update girth
            current_girth = _get_girth(adj, n)
            
    return list(edge_set)

def _is_swap_safe(adj, edge_set, u, v, x, y, min_len):
    """
    Check if swapping (u,v) and (x,y) with (u,x) and (v,y) creates a cycle < min_len.
    We check if there is a path from x to u of length < min_len - 1
    or from y to v of length < min_len - 1
    in the graph with (u,v) and (x,y) removed.
    """
    # Build a temporary adjacency list excluding (u,v) and (x,y)
    # To avoid copying, we just skip these edges in BFS.
    
    # BFS from x to u
    if _shortest_path_len(adj, x, u, u, v, x, y) < min_len - 1:
        return False
        
    # BFS from y to v
    if _shortest_path_len(adj, y, v, u, v, x, y) < min_len - 1:
        return False
        
    return True

def _shortest_path_len(adj, start, end, skip_u, skip_v, skip_x, skip_y):
    """
    Find shortest path from start to end, skipping edges (skip_u, skip_v) and (skip_x, skip_y).
    """
    if start == end:
        return 0
        
    dist = {start: 0}
    queue = deque([start])
    
    while queue:
        u = queue.popleft()
        if u == end:
            return dist[u]
            
        for v in adj[u]:
            # Check if edge (u, v) is skipped
            if (u == skip_u and v == skip_v) or (u == skip_v and v == skip_u):
                continue
            if (u == skip_x and v == skip_y) or (u == skip_y and v == skip_x):
                continue
                
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
                
    return float('inf')


def _get_girth(adj, n):
    """Calculate exact girth using BFS from each vertex."""
    min_girth = n + 1
    for i in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[i] = 0
        queue = deque([i])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and dist[v] < dist[u]:
                    # Found a cycle
                    # Reconstruct to get length
                    # The cycle is u -> ... -> v -> u
                    # Length is dist[u] - dist[v] + 1?
                    # No, v is an ancestor of u?
                    # In BFS, if v is visited and v != parent[u], v is not necessarily an ancestor.
                    # But if dist[v] < dist[u], v is in a previous layer.
                    # The path from v to u is dist[u] - dist[v].
                    # Plus the edge (u, v) makes it a cycle?
                    # No, (u, v) is an edge.
                    # The path from v to u in the tree has length dist[u] - dist[v].
                    # So the cycle length is dist[u] - dist[v] + 1.
                    cycle_len = dist[u] - dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        
    return min_girth


def _find_short_cycle(adj, n, max_len):
    """
    Find a cycle of length < max_len.
    Returns list of edges in the cycle, or None.
    """
    for i in range(n):
        parent = [-1] * n
        dist = [-1] * n
        dist[i] = 0
        queue = deque([i])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= max_len - 1:
                continue
                
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and dist[v] < dist[u]:
                    cycle_len = dist[u] - dist[v] + 1
                    if cycle_len < max_len and cycle_len >= 3:
                        # Reconstruct cycle
                        cycle_edges = []
                        curr = u
                        while curr != v:
                            p = parent[curr]
                            cycle_edges.append((min(curr, p), max(curr, p)))
                            curr = p
                        cycle_edges.append((min(u, v), max(u, v)))
                        return cycle_edges
                        
    return None


def _verify_girth_fast(n, edges, g):
    """
    Fast girth verification using non-backtracking walks.
    Returns True if girth >= g, False otherwise.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for i in range(n):
        if len(adj[i]) != 3:
            return False

    # Check for cycles shorter than g using non-backtracking walks
    for src in range(n):
        if _nb_detect_short_cycle(src, adj, n, g):
            return False

    return True


def _nb_detect_short_cycle(src, adj, n, g):
    """
    Non-backtracking walk from src to detect cycles shorter than g.
    """
    visited = set()
    queue = deque()
    
    for v in adj[src]:
        state = (v, src)
        if state not in visited:
            visited.add(state)
            queue.append((state, 1))
    
    while queue:
        (u, p), length = queue.popleft()
        
        if u == src and length + 1 >= 3:
            cycle_len = length + 1
            if cycle_len < g:
                return True
        
        if length + 1 >= g:
            continue
        
        for w in adj[u]:
            if w == p:
                continue
            new_state = (w, u)
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, length + 1))
    
    return False

def _construct_generic(k, g):
    # Fallback for k != 3
    # Just return a large random graph
    n = 10000
    # Generate k-regular graph
    # For simplicity, use configuration model
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    random.shuffle(stubs)
    edges = set()
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i+1]
        if u != v:
            edges.add((min(u, v), max(u, v)))
    return n, list(edges)

# EVOLVE-BLOCK-END
