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

def _random_3reg_bipartite():
    """
    Generate a random simple 3-regular bipartite graph on 12 vertices.
    Left: 0-5, Right: 6-11.
    Uses the configuration model with rejection.
    """
    left = list(range(6))
    right = list(range(6, 12))
    
    # Each side has 6 vertices, each needs degree 3
    # Total edges = 18
    
    # Build adjacency: for each left vertex, assign 3 right neighbors
    # Use random permutation of right vertices repeated 3 times, then partition
    for _ in range(100):  # max 100 retries
        # Create a random 3-regular bipartite multigraph
        # Assign each left vertex 3 right neighbors (with replacement initially)
        # Then check if simple (no duplicate edges)
        
        # Better approach: use a random 3-regular bipartite simple graph generator
        # via the "random perfect matching" composition
        
        # A 3-regular bipartite graph can be decomposed into 3 perfect matchings
        # Generate 3 random perfect matchings and take their union
        
        matchings = []
        valid = True
        for m in range(3):
            perm = right[:]
            random.shuffle(perm)
            matching = [(left[i], perm[i]) for i in range(6)]
            matchings.append(matching)
        
        # Check if the union is a simple graph (no duplicate edges)
        edge_set = set()
        for matching in matchings:
            for u, v in matching:
                e = (min(u, v), max(u, v))
                if e in edge_set:
                    valid = False
                    break
                edge_set.add(e)
            if not valid:
                break
        
        if valid and len(edge_set) == 18:
            return [list(e) for e in edge_set]
    
    return None

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
    max_attempts = 20000
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        edges = _random_3reg_bipartite()
        if edges is None:
            continue
        
        if len(edges) != 18:
            continue
        
        if _min_degree(n, edges) < min_deg:
            continue
        
        if not _is_triangle_free(n, edges):
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
    
    return best_edges

# EVOLVE-BLOCK-END
