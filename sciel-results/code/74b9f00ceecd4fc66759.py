import random
import itertools
import time

def _find_violation(n, adj_bits):
    """
    Finds an induced subgraph of size >= 6 that is regular.
    Returns the list of vertices in that subgraph, or None if none exists.
    """
    # Check sizes 6 to n
    for k in range(6, n + 1):
        for comb in itertools.combinations(range(n), k):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            
            # Check regularity
            first_d = -1
            is_reg = True
            for v in comb:
                d = (adj_bits[v] & mask).bit_count()
                if first_d == -1:
                    first_d = d
                elif d != first_d:
                    is_reg = False
                    break
            
            if is_reg:
                return list(comb)
    return None

def _random_graph(n, p=0.5):
    adj_bits = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
    return adj_bits

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds safety margin
    
    n = 20
    
    # Try local search
    best_adj = None
    best_violation_count = float('inf')
    
    while time.time() < deadline:
        adj_bits = _random_graph(n)
        
        # Check if valid
        violation = _find_violation(n, adj_bits)
        if violation is None:
            return {'n': n, 'edges': _adj_to_edges(n, adj_bits)}
            
        # Local search phase
        # Try to fix the violation
        for _ in range(1000):  # Max iterations per restart
            if time.time() > deadline:
                break
            
            # Find a violation
            violation = _find_violation(n, adj_bits)
            if violation is None:
                return {'n': n, 'edges': _adj_to_edges(n, adj_bits)}
                
            # Pick a random edge inside the violation to flip
            u, v = random.sample(violation, 2)
            
            # Flip edge (u, v)
            adj_bits[u] ^= (1 << v)
            adj_bits[v] ^= (1 << u)
            
            # Optionally, we could check if this flip reduced the number of violations
            # but that's expensive. Just re-checking for *any* violation is enough for termination.
            # However, we might oscillate.
            # A simple random walk on the edge space.
            
    # If we didn't find a valid graph on 20, try 19?
    # But we must return something.
    # Let's try one more time with n=19 just in case, or return the best effort.
    # Actually, if we didn't find n=20, we should probably return a valid n=19 if we can find one quickly.
    # But random search for 19 is easier.
    
    # Fallback: return a small valid graph if we failed completely?
    # The task says "Maximize n".
    # If we fail to find 20, we should try to find 19.
    
    n_fallback = 19
    while time.time() < deadline:
        adj_bits = _random_graph(n_fallback)
        violation = _find_violation(n_fallback, adj_bits)
        if violation is None:
            return {'n': n_fallback, 'edges': _adj_to_edges(n_fallback, adj_bits)}
            
    # Absolute fallback: return empty graph on 5 vertices (always valid)
    return {'n': 5, 'edges': []}

def _adj_to_edges(n, adj_bits):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj_bits[i] >> j) & 1:
                edges.append([i, j])
    return edges
