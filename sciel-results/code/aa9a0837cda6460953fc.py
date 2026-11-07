"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Bron-Kerbosch with pivoting to find maximum clique
    best_clique = []
    
    def bronk(R, P_set, X):
        nonlocal best_clique
        # If P and X are empty, R is a maximal clique
        if not P_set and not X:
            if len(R) > len(best_clique):
                best_clique = R[:]
            return
        
        # Pivot selection: choose u from P union X with max |P ∩ N(u)|
        U = P_set | X
        if not U:
            return
            
        # Find pivot
        pivot = -1
        max_common = -1
        temp_U = U
        while temp_U:
            lsb = temp_U & -temp_U
            u = lsb.bit_length() - 1
            common = P_set & adj[u]
            cnt = bin(common).count('1')
            if cnt > max_common:
                max_common = cnt
                pivot = u
            temp_U ^= lsb
        
        # If pivot is in P, we iterate over P \ N(pivot)
        # If pivot is in X (or both), we iterate over P \ N(pivot)
        # Standard Bron-Kerbosch: iterate over P \ N(u)
        candidates = P_set & ~adj[pivot]
        
        # Iterate over candidates
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            bronk(R + [v], P_set & adj[v], X & adj[v])
            P_set ^= lsb
            X |= lsb
            c ^= lsb

    bronk([], (1 << n) - 1, 0)
    
    # If no clique found (unlikely), fallback to edge
    if not best_clique:
        u, v = P['edges'][0]
        return {'branches': [u, v], 'paths': [[u, v]]}
    
    branches = [int(x) for x in best_clique]
    paths = []
    for i in range(len(branches)):
        for j in range(i + 1, len(branches)):
            paths.append([branches[i], branches[j]])
            
    return {'branches': branches, 'paths': paths}

# EVOLVE-BLOCK-END
