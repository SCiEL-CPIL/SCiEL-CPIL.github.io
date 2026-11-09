"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random

# EVOLVE-BLOCK-START

def _get_cycle_lengths(n, edges):
    """
    Enumerate all simple cycles and return the set of their lengths.
    Uses DFS from each vertex, only visiting vertices > start to avoid duplicates.
    Optimized for small n (12).
    """
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    lengths = set()
    
    def find_cycles_from(start):
        # Path is a list of vertices
        # Start with [start]
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            for nb in adj[node]:
                if nb == start:
                    # Found a cycle
                    if len(path) >= 4:
                        lengths.add(len(path))
                elif nb > start and nb not in path:
                    # Only visit vertices > start to avoid duplicate cycle counting
                    # and to ensure we start from the minimum vertex of the cycle
                    stack.append((nb, path + [nb]))
                    
    for s in range(n):
        find_cycles_from(s)
        
    return lengths

def _is_triangle_free(n, edges):
    """
    Check if graph is triangle-free.
    """
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for u in range(n):
        for v in adj[u]:
            if v > u:
                for w in adj[v]:
                    if w > v and w in adj[u]:
                        return False
    return True

def _min_degree(n, edges):
    """
    Calculate minimum degree.
    """
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return min(deg)

def _is_connected(n, edges):
    """
    Check if graph is connected.
    """
    if not edges:
        return n <= 1
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    visited = set()
    stack = [0]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for nb in adj[node]:
            if nb not in visited:
                stack.append(nb)
    return len(visited) == n

def _two_k33():
    """
    Construct two disjoint K_{3,3} components.
    Vertices 0-5 form first K_{3,3} (U={0,1,2}, V={3,4,5}).
    Vertices 6-11 form second K_{3,3} (U={6,7,8}, V={9,10,11}).
    Cycle lengths in K_{3,3}: 4 and 6.
    Value = 2.
    """
    edges = []
    for base in [0, 6]:
        U = [base, base+1, base+2]
        V = [base+3, base+4, base+5]
        for u in U:
            for v in V:
                edges.append([u, v])
    return edges

def _try_find_value1_fast(n=12, max_attempts=500):
    """
    Attempt to find a graph with value 1 (all cycles same length) using a limited random search.
    We restrict to 3-regular bipartite graphs to ensure triangle-free (girth >= 4).
    """
    random.seed(42)
    left = list(range(6))
    right = list(range(6, 12))
    
    best_edges = None
    best_value = float('inf')
    
    for _ in range(max_attempts):
        # Generate random 3-regular bipartite graph
        # Left stubs: 3 per vertex
        left_stubs = []
        for i in left:
            left_stubs.extend([i] * 3)
        
        right_stubs = []
        for i in right:
            right_stubs.extend([i] * 3)
        
        random.shuffle(right_stubs)
        
        edges_set = set()
        valid = True
        for i in range(18):
            l = left_stubs[i]
            r = right_stubs[i]
            e = (min(l, r), max(l, r))
            if e in edges_set:
                valid = False
                break
            edges_set.add(e)
        
        if not valid:
            continue
        
        edges = [list(e) for e in edges_set]
        
        # Quick checks
        if _min_degree(n, edges) < 3:
            continue
        # Bipartite by construction (left vs right), so triangle-free is guaranteed
        # if no multi-edges, which we checked.
        
        # Check for cycles
        # We need at least one cycle
        cycle_lengths = _get_cycle_lengths(n, edges)
        if not cycle_lengths:
            continue
            
        val = len(cycle_lengths)
        if val < best_value:
            best_value = val
            best_edges = edges
            if best_value == 1:
                return best_edges
                
    return best_edges

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    n = 12
    min_deg = 3
    
    # Fallback: Two K_{3,3} (Value = 2)
    # This is a valid solution with score 2.5 (assuming seed=5 or similar)
    best_edges = _two_k33()
    best_value = 2
    
    # Try to find a better solution (Value = 1)
    # We use a limited search to avoid timeout.
    # If found, it's a significant improvement.
    if time.time() < deadline:
        result = _try_find_value1_fast(n, max_attempts=1000)
        if result is not None:
            return result
        
        # If random search didn't find value 1, try a few structured candidates
        # Candidate: Circulant bipartite graph
        # Left i connects to Right {i, i+1, i+2} mod 6
        candidate_edges = []
        for i in range(6):
            for j in [i % 6, (i+1) % 6, (i+2) % 6]:
                u = i
                v = 6 + j
                candidate_edges.append([u, v])
        
        if _min_degree(n, candidate_edges) >= 3 and _is_triangle_free(n, candidate_edges):
            if _is_connected(n, candidate_edges):
                cl = _get_cycle_lengths(n, candidate_edges)
                if cl:
                    val = len(cl)
                    if val < best_value:
                        best_value = val
                        best_edges = candidate_edges
                        if best_value == 1:
                            return best_edges
                            
        # Candidate 2: Left i connects to Right {i, i+2, i+4} mod 6
        candidate_edges2 = []
        for i in range(6):
            for j in [i % 6, (i+2) % 6, (i+4) % 6]:
                u = i
                v = 6 + j
                candidate_edges2.append([u, v])
        
        if _min_degree(n, candidate_edges2) >= 3 and _is_triangle_free(n, candidate_edges2):
            if _is_connected(n, candidate_edges2):
                cl = _get_cycle_lengths(n, candidate_edges2)
                if cl:
                    val = len(cl)
                    if val < best_value:
                        best_value = val
                        best_edges = candidate_edges2
                        if best_value == 1:
                            return best_edges

    return best_edges

# EVOLVE-BLOCK-END
