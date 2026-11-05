import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for return overhead
    
    # Seed fallback
    seed_edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        seed_edges += [[a, b], [a, c], [b, c]]
    
    best_result = {'n': 30, 'edges': seed_edges}
    
    # Try to find a better graph
    # We iterate n from 43 down to 31
    for n in range(43, 30, -1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random constructions for this n
        for seed in range(50):
            if time.time() > deadline - 5:
                break
            
            edges, adj = greedy_c4_free(n, seed)
            
            # Check if alpha <= 10
            # i.e., no independent set of size 11
            if not has_independent_set(adj, n, 11):
                best_result = {'n': n, 'edges': edges}
                return best_result
                
    return best_result

def greedy_c4_free(n, seed):
    """
    Constructs a maximal C4-free graph on n vertices using greedy edge addition.
    Returns (edges_list, adjacency_bitmasks)
    """
    random.seed(seed)
    
    # Generate all possible edges
    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.append((u, v))
    
    random.shuffle(all_edges)
    
    adj = [0] * n  # Bitmasks for neighbors
    edges = []
    
    for u, v in all_edges:
        # Check if adding (u, v) creates a C4
        # C4 exists if there is a common neighbor w of u and v
        # i.e., adj[u] & adj[v] != 0
        if adj[u] & adj[v] == 0:
            # Safe to add
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append([u, v])
            
    return edges, adj

def has_independent_set(adj, n, k):
    """
    Returns True if there is an independent set of size k.
    Uses backtracking with bitsets.
    """
    # Pre-compute neighbor masks excluding self? 
    # adj[v] contains neighbors.
    
    def backtrack(candidates, count):
        if count == 0:
            return True
        if candidates == 0:
            return False
        
        # Pruning: if number of candidates < count, impossible
        # popcount
        if bin(candidates).count('1') < count:
            return False
            
        # Heuristic: Pick a vertex with high degree in the candidate subgraph?
        # Or just pick the lowest set bit.
        # To improve pruning, pick a vertex that is connected to many other candidates.
        # Let's pick the first one for simplicity, but we can optimize.
        
        # Pick v
        v = (candidates & -candidates).bit_length() - 1
        candidates_without_v = candidates & ~(1 << v)
        
        # Branch 1: Include v
        # New candidates must not be neighbors of v
        new_candidates = candidates_without_v & ~adj[v]
        if backtrack(new_candidates, count - 1):
            return True
        
        # Branch 2: Exclude v
        if backtrack(candidates_without_v, count):
            return True
        
        return False

    initial_candidates = (1 << n) - 1
    return backtrack(initial_candidates, k)
