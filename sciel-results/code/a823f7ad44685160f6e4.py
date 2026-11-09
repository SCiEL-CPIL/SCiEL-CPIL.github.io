"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random

# EVOLVE-BLOCK-START

def _get_cycle_lengths(n, edges):
    """
    Enumerate all simple cycles and return the set of their lengths.
    Uses DFS from each vertex, only visiting vertices > start to avoid duplicates.
    """
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    lengths = set()
    
    def find_cycles_from(start):
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            for nb in adj[node]:
                if nb == start and len(path) >= 4:
                    lengths.add(len(path))
                elif nb > start and nb not in path:
                    stack.append((nb, path + [nb]))
                    
    for s in range(n):
        find_cycles_from(s)
        
    return lengths

def _is_triangle_free(n, edges):
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
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return min(deg)

def _is_connected(n, edges):
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
    """Two disjoint K_{3,3} components. Cycle lengths: {4, 6}. Value = 2."""
    edges = []
    for base in [0, 6]:
        U = [base, base+1, base+2]
        V = [base+3, base+4, base+5]
        for u in U:
            for v in V:
                edges.append([u, v])
    return edges

def _try_construct_value1():
    """
    Try to find a 3-regular bipartite graph on 12 vertices (6 left, 6 right)
    where all simple cycles have the same length (ideally 4).
    
    Strategy: Use a structured approach. A 3-regular bipartite graph where
    every 4-cycle is "tight" and no longer cycles exist.
    
    One known construction: Take the 3-dimensional cube Q3 (8 vertices, 3-regular,
    bipartite, girth 4, all cycles are 4 or 6). That gives value 2.
    
    For 12 vertices: We need a different approach.
    
    Key insight: If we make the graph such that it's a "union of 4-cycles" sharing
    edges in a way that prevents 6-cycles, we might get value 1.
    
    Actually, let's try: a 3-regular bipartite graph that is a "chain" of 4-cycles
    where each new 4-cycle shares exactly one edge with the previous, creating
    a structure where no 6-cycle can form.
    
    Alternative: Use the incidence structure of a specific design.
    
    Let me try a systematic search with verification.
    """
    random.seed(42)
    
    n = 12
    left = list(range(6))    # 0-5
    right = list(range(6, 12))  # 6-11
    
    best_edges = None
    best_value = float('inf')
    
    # Try many random 3-regular bipartite graphs
    for attempt in range(20000):
        # Generate random 3-regular bipartite graph using configuration model
        # Each left vertex has 3 stubs, each right vertex has 3 stubs
        left_stubs = []
        for i in left:
            left_stubs.extend([i] * 3)
        right_stubs = []
        for i in right:
            right_stubs.extend([i] * 3)
        
        # Randomly pair stubs
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
        
        # Check constraints
        if _min_degree(n, edges) < 3:
            continue
        if not _is_triangle_free(n, edges):
            continue
        if not _is_connected(n, edges):
            continue
        
        # Get cycle lengths
        cycle_lengths = _get_cycle_lengths(n, edges)
        if not cycle_lengths:
            continue
        
        val = len(cycle_lengths)
        if val < best_value:
            best_value = val
            best_edges = edges
            if best_value == 1:
                break
    
    if best_value == 1:
        return best_edges
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill
    
    n = 12
    min_deg = 3
    
    # Fallback: two K_{3,3} (value = 2, score = 2.5)
    best_edges = _two_k33()
    best_value = 2
    
    # Try to find value = 1
    result = _try_construct_value1()
    if result is not None:
        return result
    
    # If we still have time, try more targeted constructions
    if time.time() < deadline:
        # Try specific structured graphs
        # Attempt: A graph built from 4-cycles in a way that prevents 6-cycles
        
        # Construction idea: Use a "ladder" of 4-cycles
        # Left: 0,1,2,3,4,5  Right: 6,7,8,9,10,11
        # Make 4-cycles: (0,6,1,7), (1,7,2,8), (2,8,3,9), (3,9,4,10), (4,10,5,11)
        # That's 5 4-cycles, 5*4=20 edge-uses but shared edges...
        # Actually let me think of this differently.
        
        # Try: 3-regular bipartite where the adjacency matrix has a specific structure
        # that only allows 4-cycles.
        
        # One approach: Use the complement of a matching in K_{6,6} won't work (degree 5).
        
        # Let's try: For left vertex i, connect to right vertices {i, i+1, i+2} mod 6
        # Check if this gives only 4-cycles.
        
        candidate_edges = []
        for i in range(6):
            for j in [i % 6, (i+1) % 6, (i+2) % 6]:
                u = i
                v = 6 + j
                candidate_edges.append([u, v])
        
        # Check validity
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
        
        # Try another structured pattern
        # Left i connects to right {i, (i+2)%6, (i+4)%6} mod 6
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
