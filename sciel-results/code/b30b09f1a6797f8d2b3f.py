"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin (150s limit)
    
    # Try deterministic algebraic construction: polarity graph over GF(23)
    # This gives a 21-vertex graph (projective plane PG(2,4) has 21 points)
    # Actually, let's use a simpler approach: construct a graph on 21 vertices
    # using a quadratic form over GF(7) or similar finite field structure
    
    # Strategy: Use a polarity graph from projective geometry
    # PG(2, q) has (q^2 + q + 1) points
    # For q=4: 17 points, q=5: 31 points (too many)
    # Let's try a different algebraic construction
    
    # Alternative: Use a Paley-type graph or conference matrix approach
    # For n=21, we can try to construct a graph where each vertex has degree ~10
    # and the adjacency structure avoids regular induced subgraphs
    
    # Let's try a specific construction: 
    # Label vertices 0..20, use quadratic residues mod 23 (prime)
    # Paley graph on 23 vertices minus 2 vertices? Or use a subset?
    
    # Actually, let's try: construct graph on 21 vertices using GF(23) quadratic residues
    # but remove 2 vertices to get 21
    
    p = 23
    n_target = 21
    
    # Quadratic residues mod p
    qr = set()
    for x in range(1, p):
        if pow(x, (p-1)//2, p) == 1:
            qr.add(x)
    
    # Create Paley graph on p vertices
    # Vertex i connected to vertex j if (j-i) mod p is in qr
    adj_p = np.zeros((p, p), dtype=np.int8)
    for i in range(p):
        for d in qr:
            j = (i + d) % p
            adj_p[i, j] = 1
            adj_p[j, i] = 1
    
    # Remove 2 vertices to get 21 vertices
    # Try removing vertices 0 and 1
    keep = list(range(2, p))  # vertices 2..22, that's 21 vertices
    n = len(keep)
    vertex_map = {old: new for new, old in enumerate(keep)}
    
    adj = np.zeros((n, n), dtype=np.int8)
    for old_i, new_i in vertex_map.items():
        for old_j, new_j in vertex_map.items():
            if old_i != old_j:
                adj[new_i, new_j] = adj_p[old_i, old_j]
    
    # Check if this graph is valid (no regular induced subgraph on 6+ vertices)
    # Use optimized vectorized check
    if _is_valid_graph(adj, n):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
    
    # If algebraic construction fails, try randomized search
    best_n = 0
    best_edges = []
    
    # Try several random graphs on 21 vertices
    rng = np.random.default_rng(42)
    attempts = 0
    while time.time() < deadline and attempts < 500:
        attempts += 1
        # Random graph with edge probability ~0.5
        prob = 0.5
        adj_rand = np.zeros((n_target, n_target), dtype=np.int8)
        for i in range(n_target):
            for j in range(i+1, n_target):
                if rng.random() < prob:
                    adj_rand[i, j] = 1
                    adj_rand[j, i] = 1
        
        if _is_valid_graph(adj_rand, n_target):
            edges = []
            for i in range(n_target):
                for j in range(i+1, n_target):
                    if adj_rand[i, j]:
                        edges.append([i, j])
            return {'n': n_target, 'edges': edges}
    
    # If 21 fails, try 20
    n_try = 20
    while time.time() < deadline and n_try >= 10:
        # Try to find valid graph on n_try vertices
        for attempt in range(200):
            if time.time() > deadline:
                break
            adj_try = np.zeros((n_try, n_try), dtype=np.int8)
            for i in range(n_try):
                for j in range(i+1, n_try):
                    if rng.random() < 0.5:
                        adj_try[i, j] = 1
                        adj_try[j, i] = 1
            
            if _is_valid_graph(adj_try, n_try):
                edges = []
                for i in range(n_try):
                    for j in range(i+1, n_try):
                        if adj_try[i, j]:
                            edges.append([i, j])
                return {'n': n_try, 'edges': edges}
        n_try -= 1
    
    # Fallback: return a small valid graph
    return {'n': 5, 'edges': []}

def _is_valid_graph(adj, n):
    """Check if graph has no regular induced subgraph on 6+ vertices.
    Uses vectorized NumPy operations for speed."""
    n = int(n)
    if n < 6:
        return True
    
    # Precompute degrees
    degrees = adj.sum(axis=1)
    
    # Check all subsets of size 6 to n
    # Use combinatorial generation but with early termination
    from itertools import combinations
    
    # Optimize: check sizes from 6 upward, stop if any regular found
    for size in range(6, n+1):
        if size > n:
            break
        # Generate combinations of size 'size'
        # For efficiency, we can use bit manipulation or numpy
        # But for n<=22, itertools should be fast enough with early exit
        
        for subset in combinations(range(n), size):
            # Check if induced subgraph is regular
            # Get subgraph
            sub_adj = adj[np.ix_(list(subset), list(subset))]
            sub_degrees = sub_adj.sum(axis=1)
            
            # Regular if all degrees are equal
            if len(set(sub_degrees.tolist())) == 1:
                return False
    
    return True
# EVOLVE-BLOCK-END
