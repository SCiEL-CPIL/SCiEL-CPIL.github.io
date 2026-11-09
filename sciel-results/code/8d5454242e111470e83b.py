import time
import numpy as np
import itertools

def paley_graph(q):
    """Construct the Paley graph on q vertices."""
    residues = {(x * x) % q for x in range(1, q)}
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                edges.append((i, j))
    return q, edges

def is_5_5_free(adj, n):
    """Check if graph with adjacency matrix adj on n vertices has no K5 and no independent 5-set."""
    # This is an O(n^5) check, but for n=38 it's 38 choose 5 = 501,942 sets.
    # 500k sets * 10 edges check is 5M ops. Very fast in Python/Numpy.
    for comb in itertools.combinations(range(n), 5):
        # Check if clique
        is_clique = True
        is_independent = True
        for i in range(5):
            for j in range(i+1, 5):
                u, v = comb[i], comb[j]
                if adj[u, v]:
                    is_independent = False
                else:
                    is_clique = False
                if not is_clique and not is_independent:
                    break
            if not is_clique and not is_independent:
                break
        if is_clique or is_independent:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # Fallback: Paley graph on 37 vertices
    n_seed, edges_seed = paley_graph(37)
    best_n = n_seed
    best_edges = edges_seed
    
    # Helper to build adjacency matrix from edges
    def build_adj(n, edges):
        adj = np.zeros((n, n), dtype=bool)
        for u, v in edges:
            adj[u, v] = True
            adj[v, u] = True
        return adj

    # Try deterministic constructions for n=38
    # Inspiration: 2-fold cover of Paley(19)
    
    n_base = 19
    _, base_edges = paley_graph(n_base)
    base_adj = build_adj(n_base, base_edges)
    
    # Candidate 1: Cross edges exist if base edge exists
    # V = 0..18 (Copy 0), 19..37 (Copy 1)
    # Internal edges: same as base
    # Cross edges: u in C0, v in C1. Let v' = v - 19.
    # Edge (u, v) iff base_adj[u, v']
    
    def build_candidate(cross_rule):
        n = 2 * n_base
        adj = np.zeros((n, n), dtype=bool)
        
        # Internal edges
        for i in range(n_base):
            for j in range(i + 1, n_base):
                if base_adj[i, j]:
                    adj[i, j] = True
                    adj[j, i] = True
                    adj[i + n_base, j + n_base] = True
                    adj[j + n_base, i + n_base] = True
        
        # Cross edges
        for i in range(n_base):
            for j in range(n_base):
                if cross_rule(i, j, base_adj[i, j]):
                    adj[i, j + n_base] = True
                    adj[j + n_base, i] = True
                    
        return n, adj

    # Rule 1: Connect if base edge exists
    def rule1(i, j, exists):
        return exists

    # Rule 2: Connect if base edge does NOT exist
    def rule2(i, j, exists):
        return not exists

    # Rule 3: Connect if i == j (perfect matching)
    def rule3(i, j, exists):
        return i == j

    # Rule 4: Connect if (i - j) % n_base is a quadratic residue (shifted Paley)
    residues = {(x * x) % n_base for x in range(1, n_base)}
    def rule4(i, j, exists):
        return (j - i) % n_base in residues

    candidates = [
        ("cover_match", rule1),
        ("cover_complement", rule2),
        ("cover_diagonal", rule3),
        ("cover_paley_shift", rule4)
    ]
    
    for name, rule in candidates:
        n, adj = build_candidate(rule)
        if is_5_5_free(adj, n):
            # Convert to edge list
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i, j]:
                        edges.append((i, j))
            return n, edges

    # If deterministic constructions fail, fallback to search (or just seed if time is short)
    # The search in the original code was very slow. 
    # We will implement a faster local search if time permits, but likely we just return seed.
    # Given the 300s limit and the difficulty of n=38, returning the seed is safer than a broken search.
    # However, we can try a very quick random search for n=38.
    
    if time.time() < deadline - 50:
        # Quick random search for n=38
        n_try = 38
        adj = np.zeros((n_try, n_try), dtype=bool)
        for i in range(n_try):
            for j in range(i+1, n_try):
                adj[i, j] = np.random.rand() < 0.5
                adj[j, i] = adj[i, j]
        
        # Simple hill climbing
        for _ in range(10000):
            if time.time() > deadline - 10:
                break
            u = np.random.randint(0, n_try)
            v = np.random.randint(0, n_try)
            if u == v: continue
            
            # Count violations for this edge? Too slow.
            # Just flip and check whole graph? 10k checks * 500k sets * 10 ops = 50 billion ops. Too slow.
            # Skip random search for n=38 as it's too slow in Python.
            break
            
    return best_n, best_edges
