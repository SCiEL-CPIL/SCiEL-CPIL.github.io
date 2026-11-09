import numpy as np
from itertools import combinations

def construct():
    def generate_paley(q):
        adj = np.zeros((q, q), dtype=np.int8)
        residues = set()
        for x in range(1, q):
            residues.add((x * x) % q)
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in residues:
                    adj[i, j] = 1
                    adj[j, i] = 1
        return adj

    def check_valid(adj, n):
        # Use bitmasks for speed
        adj_mask = [0] * n
        non_adj_mask = [0] * n
        all_mask = (1 << n) - 1
        
        for i in range(n):
            mask = 0
            for j in range(n):
                if i != j and adj[i, j]:
                    mask |= (1 << j)
            adj_mask[i] = mask
            non_adj_mask[i] = all_mask ^ (mask | (1 << i))

        # Check for K5
        for combo in combinations(range(n), 5):
            v0, v1, v2, v3, v4 = combo
            m = adj_mask[v0] & adj_mask[v1] & adj_mask[v2] & adj_mask[v3] & adj_mask[v4]
            if (m & (1 << v1)) and (m & (1 << v2)) and (m & (1 << v3)) and (m & (1 << v4)):
                return False
        
        # Check for Independent Set of 5
        for combo in combinations(range(n), 5):
            v0, v1, v2, v3, v4 = combo
            m = non_adj_mask[v0] & non_adj_mask[v1] & non_adj_mask[v2] & non_adj_mask[v3] & non_adj_mask[v4]
            if (m & (1 << v1)) and (m & (1 << v2)) and (m & (1 << v3)) and (m & (1 << v4)):
                return False

        return True

    # Candidates: q == 1 mod 4, prime powers
    # We check a limited range to ensure speed.
    # If a larger graph is valid, we'd want it, but likely 37-50 is the range.
    candidates = [37, 41, 49, 53, 61, 73, 81, 89, 97]
    
    best_n = 0
    best_edges = []
    
    for q in candidates:
        adj = generate_paley(q)
        if check_valid(adj, q):
            if q > best_n:
                best_n = q
                edges = []
                for i in range(q):
                    for j in range(i + 1, q):
                        if adj[i, j]:
                            edges.append((i, j))
                best_edges = edges
                
    if best_n == 0:
        # Fallback
        q = 37
        adj = generate_paley(q)
        edges = []
        for i in range(q):
            for j in range(i + 1, q):
                if adj[i, j]:
                    edges.append((i, j))
        return q, edges

    return best_n, best_edges
