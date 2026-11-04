"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 130  # Leave 20s safety margin before 150s kill
    
    def verify(U, V, W, rank):
        """Verify that U, V, W with given rank exactly decompose the tensor."""
        # Build the full tensor from decomposition
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # For each (i,j,k), check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # For all other (a,b,c), check sum_r U[a,r] * V[b,r] * W[c,r] == 0
        
        nm = n * m
        mp = m * p
        pn = p * n
        
        # Check all required entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :]) / 1  # elementwise then sum
                    # Actually: sum_r U[a,r] * V[b,r] * W[c,r]
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        
        # Check all forbidden entries (sample or full)
        # Full check is expensive; do it for small dimensions
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    # Skip if this is a valid (i,j,k) triple
                    i = a // m
                    j = a % m
                    # b = j*p + k => k = b % p, but need j match
                    j2 = b // p
                    k = b % p
                    i2 = c % n
                    k2 = c // n
                    if i == i2 and j == j2 and k == k2:
                        continue
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > 1e-9:
                        return False
        return True
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p  # 72
    U = np.zeros((n * m, rank_school), dtype=float)
    V = np.zeros((m * p, rank_school), dtype=float)
    W = np.zeros((p * n, rank_school), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U, V, W
    
    # Try to find lower rank decompositions
    # Strategy: use structured random search with verification
    
    # For M_{3,3,8}, try ranks from 71 down
    # This is computationally expensive, so be strategic
    
    # Actually, let's try a smarter approach: 
    # The tensor M_{3,3,8} can be seen as 3 copies of M_{1,3,8} in parallel? No.
    # 
    # Better: use the fact that we can combine terms.
    # 
    # For now, return the schoolbook decomposition which is guaranteed correct.
    # The rank is 72.
    
    # Check time
    if time.time() - start_time > deadline - 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try a few random structured attempts for lower rank
    # This is a heuristic; may not find improvement but won't hurt
    
    for target_rank in range(best_rank - 1, max(40, best_rank - 10), -1):
        if time.time() - start_time > deadline - 10:
            break
        
        # Random initialization with small values
        np.random.seed(target_rank)  # Reproducible
        U_rand = np.random.randn(n * m, target_rank) * 0.5
        V_rand = np.random.randn(m * p, target_rank) * 0.5
        W_rand = np.random.randn(p * n, target_rank) * 0.5
        
        # Verify
        if verify(U_rand, V_rand, W_rand, target_rank):
            best_rank = target_rank
            best_U, best_V, best_W = U_rand, V_rand, W_rand
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
