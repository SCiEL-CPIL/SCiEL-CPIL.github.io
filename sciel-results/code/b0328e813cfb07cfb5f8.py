import time
import random
import itertools
from fractions import Fraction

def construct():
    n, m, p = 2, 3, 7
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, n, m, p, tol=1e-9):
        rank = len(U[0])
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                        if abs(s - 1) > tol:
                            return False
                    else:
                        s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                        if abs(s) > tol:
                            return False
        return True

    def verify_exact(U, V, W, n, m, p):
        rank = len(U[0])
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        s = Fraction(0)
                        for r in range(rank):
                            s += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                        if s != 1:
                            return False
                    else:
                        s = Fraction(0)
                        for r in range(rank):
                            s += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                        if s != 0:
                            return False
        return True

    best = schoolbook()
    best_rank = n * m * p

    # Search for lower rank
    # Try ranks from 30 down to 24
    for target_rank in range(30, 23, -1):
        if time.time() > deadline - 10.0:
            break
            
        # Random search for this rank
        # We will try to find a valid assignment of non-zero entries
        # This is a heuristic search
        
        found = False
        for attempt in range(200):
            if time.time() > deadline - 5.0:
                break
                
            # Strategy: Build a structured decomposition where each rank column
            # corresponds to a "pattern" that covers multiple (i,j,k) triples.
            # The key insight: for rank < n*m*p, we need some rank columns to
            # handle multiple required entries with appropriate coefficients.
            
            # Approach: Group the 42 required (i,j,k) triples into target_rank groups.
            # For each group, we assign coefficients such that the tensor equation holds.
            
            all_triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
            
            # Create random groupings
            random.shuffle(all_triples)
            groups = [[] for _ in range(target_rank)]
            for idx, triple in enumerate(all_triples):
                groups[idx % target_rank].append(triple)
            
            # Now try to find coefficients for each group
            # For a group with triples, we need:
            # For each (i,j,k) in the group: U[i*m+j][col] * V[j*p+k][col] * W[k*n+i][col] = 1
            # For each (i,j,k) NOT in the group but sharing indices: the sum over all groups must be 0
            
            # Simple approach: set all coefficients to 1 for required entries in the group
            # This won't work directly, but let's try randomized signs and see if any work
            
            for sign_attempt in range(3):
                random.seed(attempt * 1000 + sign_attempt * 100 + target_rank)
                U = [[0] * target_rank for _ in range(n * m)]
                V = [[0] * target_rank for _ in range(m * p)]
                W = [[0] * target_rank for _ in range(p * n)]
                
                for col, triples in enumerate(groups):
                    if not triples:
                        continue
                    
                    # For each triple in this group, assign random signs
                    # The product of signs must be 1 for each triple
                    signs_u = {}
                    signs_v = {}
                    signs_w = {}
                    
                    for (i, j, k) in triples:
                        su = random.choice([1, -1])
                        sv = random.choice([1, -1])
                        sw = 1  # sw = su * sv to make product 1
                        # Actually, we need su * sv * sw = 1, so sw = 1/(su*sv) = su*sv
                        sw = su * sv
                        
                        signs_u[(i, j)] = su
                        signs_v[(j, k)] = sv
                        signs_w[(k, i)] = sw
                    
                    # Check consistency: if same (i,j) appears in multiple triples, signs must match
                    consistent = True
                    for (i, j), su in signs_u.items():
                        for (i2, j2), su2 in signs_u.items():
                            if (i, j) != (i2, j2) and i == i2 and j == j2:
                                if su != su2:
                                    consistent = False
                                    break
                        if not consistent:
                            break
                    if not consistent:
                        continue
                    
                    for (j, k), sv in signs_v.items():
                        for (j2, k2), sv2 in signs_v.items():
                            if (j, k) != (j2, k2) and j == j2 and k == k2:
                                if sv != sv2:
                                    consistent = False
                                    break
                        if not consistent:
                            break
                    if not consistent:
                        continue
                    
                    for (k, i), sw in signs_w.items():
                        for (k2, i2), sw2 in signs_w.items():
                            if (k, i) != (k2, i2) and k == k2 and i == i2:
                                if sw != sw2:
                                    consistent = False
                                    break
                        if not consistent:
                            break
                    if not consistent:
                        continue
                    
                    # Assign values
                    for (i, j), su in signs_u.items():
                        U[i * m + j][col] = su
                    for (j, k), sv in signs_v.items():
                        V[j * p + k][col] = sv
                    for (k, i), sw in signs_w.items():
                        W[k * n + i][col] = sw
                
                if verify(U, V, W, n, m, p):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    found = True
                    break
            
            if found:
                break
        
        if found:
            break

    return best
