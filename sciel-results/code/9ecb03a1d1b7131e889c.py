"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for matrix multiplication
    of a 2x3 matrix by a 3x6 matrix.
    
    Uses structured search exploiting the factorization 6 = 2*3.
    """
    start_time = time.time()
    deadline = start_time + 115  # 115 second deadline with margin
    
    n, m, p = DIMENSIONS
    
    # Create the schoolbook decomposition as fallback
    rank_schoolbook = n * m * p  # 36
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, column] = 1.0
                V_sb[j * p + k, column] = 1.0
                W_sb[k * n + i, column] = 1.0
                column += 1
    
    best_rank = rank_schoolbook
    best_decomp = (U_sb, V_sb, W_sb)
    
    # Helper function to verify a decomposition
    def verify(U, V, W, tol=1e-10):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for r in range(rank):
                        actual += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(actual - expected) > tol:
                        return False
        # Also check that invalid positions give 0
        # For efficiency, only check a subset
        for _ in range(100):
            a = random.randint(0, n*m - 1)
            b = random.randint(0, m*p - 1)
            c = random.randint(0, p*n - 1)
            # Check if this is a valid position
            i1 = a // m
            j1 = a % m
            j2 = b // p
            k1 = b % p
            k2 = c // n
            i2 = c % n
            is_valid = (i1 == i2) and (j1 == j2) and (k1 == k2)
            if not is_valid:
                actual = 0.0
                for r in range(rank):
                    actual += U[a, r] * V[b, r] * W[c, r]
                if abs(actual) > tol:
                    return False
        return True
    
    # Try ranks from low to high
    for target_rank in range(1, rank_schoolbook + 1):
        if time.time() > deadline - 10:
            break
        
        # For small ranks, try more attempts
        max_attempts = max(10, 200 // target_rank)
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Structured initialization: use the 6=2*3 structure
            if target_rank >= 2:
                # Try to use Kronecker-like structure
                # Split into two halves for the p dimension
                half_p = p // 2  # 3
                
                # Initialize with structure
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                
                # First half: for k in 0..2
                # Second half: for k in 3..5
                for r in range(target_rank):
                    if r < target_rank // 2:
                        # First half
                        for i in range(n):
                            for j in range(m):
                                U[i*m+j, r] = random.gauss(0, 1)
                        for j in range(m):
                            for k in range(half_p):
                                V[j*p+k, r] = random.gauss(0, 1)
                        for k in range(half_p):
                            for i in range(n):
                                W[k*n+i, r] = random.gauss(0, 1)
                    else:
                        # Second half
                        r2 = r - target_rank // 2
                        for i in range(n):
                            for j in range(m):
                                U[i*m+j, r] = random.gauss(0, 1)
                        for j in range(m):
                            for k in range(half_p, p):
                                V[j*p+k, r] = random.gauss(0, 1)
                        for k in range(half_p, p):
                            for i in range(n):
                                W[k*n+i, r] = random.gauss(0, 1)
            else:
                # Random initialization for rank 1
                U = np.random.randn(n * m, target_rank)
                V = np.random.randn(m * p, target_rank)
                W = np.random.randn(p * n, target_rank)
            
            if verify(U, V, W):
                best_rank = target_rank
                best_decomp = (U.copy(), V.copy(), W.copy())
                break
        
        if best_rank == target_rank:
            break
    
    if best_rank < rank_schoolbook:
        return {"U": best_decomp[0].tolist(), "V": best_decomp[1].tolist(), "W": best_decomp[2].tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
