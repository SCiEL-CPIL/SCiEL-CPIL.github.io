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

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    n = 12
    min_deg = 3
    
    # Fallback: two K_{3,3} (value = 2, score = 2.5)
    best_edges = _two_k33()
    best_value = 2
    
    # Try to find value = 1: all cycles same length
    # Strategy: try random 3-regular bipartite graphs
    # Bipartite partition: {0..5} and {6..11}
    
    random.seed(42)
    attempts = 0
    max_attempts = 5000
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # Generate random 3-regular bipartite graph
        # Left: 0-5, Right: 6-11
        # Each left vertex connects to 3 right vertices, each right to 3 left
        # Use random permutation approach
        
        # For each left vertex, randomly assign 3 right vertices
        # But we need each right vertex to have degree exactly 3
        # This is equivalent to a 3-regular bipartite multigraph -> simple
        
        # Use configuration model / random assignment
        # Right vertices 6-11 each need 3 connections
        # Total edges = 18, each left has degree 3, each right has degree 3
        
        # Create list of right slots: each right vertex appears 3 times
        right_slots = []
        for r in range(6, 12):
            right_slots.extend([r] * 3)
        
        # For each left vertex, shuffle and assign 3
        edges_set = set()
        valid = True
        for left in range(6):
            random.shuffle(right_slots)
            chosen = right_slots[:3]
            right_slots = right_slots[3:]
            for right in chosen:
                e = (min(left, right), max(left, right))
                if e in edges_set:
                    valid = False
                    break
                edges_set.add(e)
            if not valid:
                break
        
        if not valid:
            continue
        
        edges = [list(e) for e in edges_set]
        
        if len(edges) != 18:
            continue
        
        if _min_degree(n, edges) < min_deg:
            continue
        
        if not _is_triangle_free(n, edges):
            continue
        
        # Check if it contains a cycle
        # Since it's 3-regular bipartite on 12 vertices, it must have cycles
        # (bipartite 3-regular graphs on 12 vertices always have cycles)
        
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
    
    return best_edges

# EVOLVE-BLOCK-END
