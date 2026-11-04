"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from fractions import Fraction

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # First, build the schoolbook decomposition as a baseline
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Try to find a lower-rank decomposition using known results
    # For 4x4 * 4x6, we can try various approaches:
    # 1. Use Strassen-like decompositions
    # 2. Use block matrix techniques
    
    # Known: 2x2 * 2x2 can be done in rank 7 (Strassen)
    # 4x4 * 4x4 might be doable in rank < 64
    
    # Let's try a systematic search for small rank improvements
    # We'll use a randomized approach with verification
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is correct."""
        # Check shapes
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        for row in U:
            if len(row) != rank:
                return False
        for row in V:
            if len(row) != rank:
                return False
        for row in W:
            if len(row) != rank:
                return False
        
        # Verify the tensor identity:
        # For all i,j,k: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # where T[i,j,k] = 1 if and only if the (i,k) entry of (A*B) depends on A[i,j]*B[j,k]
        # Actually, the multiplication tensor has T[i,j,k] = 1 for all valid i,j,k
        # and the decomposition must satisfy:
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    # Try a few known lower-rank constructions
    # For 4x4*4x6, one approach is to use the fact that 6 = 2*3
    # and decompose into blocks
    
    # Let's try rank values from 95 down to see if we can find something better
    # This is computationally expensive, so we'll be strategic
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try some specific constructions
    # Construction 1: Use 2x2 Strassen for the 4x4 part
    # 4x4 * 4x4: can be viewed as 2x2 blocks of 2x2 matrices
    # Each 2x2*2x2 multiplication takes rank 7 (Strassen)
    # 4x4*4x4 has 4 block multiplications, but they interact
    # Total rank for 4x4*4x4 might be around 7*4 = 28 if independent, but they're not
    
    # Actually, let's just try to verify if the schoolbook is the best we can do easily
    # and return it if we can't find better
    
    # For now, return the schoolbook decomposition
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
