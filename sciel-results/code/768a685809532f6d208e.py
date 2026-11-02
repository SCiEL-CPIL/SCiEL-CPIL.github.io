"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

import numpy as np
import time
import random

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    nm = n * m
    mp = m * p
    pn = p * n
    rank = len(U[0]) if U else 0
    
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # For each (i, j, k), check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # For all other (a, b, c) where the triple doesn't correspond to a valid (i,j,k), check == 0
    # But actually, the tensor is defined such that T[a,b,c] = 1 if there exists (i,j,k) with 
    # a = i*m+j, b = j*p+k, c = k*n+i, else 0.
    # Note: a, b, c must be consistent: from a = i*m+j we get j = a % m, i = a // m
    # from b = j*p+k we get k = b % p, and j must equal a % m
    # from c = k*n+i we get i = c % n, k = c // n, and i must equal a // m, k must equal b % p
    
    # Build the target tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check that all other entries are 0
    # For each a in [0, nm), b in [0, mp), c in [0, pn):
    # Determine if (a, b, c) corresponds to a valid (i, j, k)
    # a = i*m + j => i = a//m, j = a%m
    # b = j'*p + k => j' = b//p, k = b%p
    # c = k'*n + i' => i' = c%n, k' = c//n
    # Valid if j == j' and i == i' and k == k'
    for a in range(nm):
        i_a = a // m
        j_a = a % m
        for b in range(mp):
            j_b = b // p
            k_b = b % p
            if j_a != j_b:
                # This (a,b) pair doesn't correspond to any valid (i,j,k)
                # So for all c, the tensor entry should be 0
                for c in range(pn):
                    val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                    if abs(val) > 1e-9:
                        return False
            else:
                # j_a == j_b, so we need to check c
                for c in range(pn):
                    i_c = c % n
                    k_c = c // n
                    if i_c == i_a and k_c == k_b:
                        # This is a valid triple, already checked above
                        pass
                    else:
                        val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                        if abs(val) > 1e-9:
                            return False
    
    return True

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 12
    mp = m * p  # 16
    pn = p * n  # 12
    
    start_time = time.time()
    deadline = start_time + 120  # 120 second budget
    
    # Schoolbook decomposition
    rank = n * m * p  # 48
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best = {"U": U, "V": V, "W": W, "rank": rank}
    
    # Try to improve using random search
    # Strategy: try random small-rank decompositions and verify
    
    # Actually, let's try a more structured approach.
    # For now, return the schoolbook as baseline.
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
