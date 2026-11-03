"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s safety margin for 120s limit
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        rank = len(U[0]) if U else 0
        # Check shapes
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if any(len(row) != rank for row in U):
            return False
        if any(len(row) != rank for row in V):
            return False
        if any(len(row) != rank for row in W):
            return False
        
        # Verify the tensor decomposition
        # For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        # And for all other (i, j, k) combinations that don't correspond to the multiplication, should be 0
        # Actually, the matrix multiplication tensor T[i,j,k] = 1 if the indices correspond to 
        # (row i of A, col j of B) -> (row i of C, col k of C)
        # Wait, let me re-read the task.
        
        # The tensor for multiplying A (n x m) by B (m x p) to get C (n x p)
        # T[i, j, k] where i is row of A/C, j is col of B / row of C, k is col of A / row of B
        # Actually: C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor T has indices (i, j, k) where:
        # i = row of A and C (0..n-1)
        # j = col of A / row of B (0..m-1) 
        # k = col of B and C (0..p-1)
        # T[i,j,k] = 1 for all valid (i,j,k)
        
        # The factor matrices are:
        # U: shape (n*m, rank) - indexed by [i*m + j]
        # V: shape (m*p, rank) - indexed by [j*p + k]
        # W: shape (p*n, rank) - indexed by [k*n + i]  (note: k*n + i, not i*n + k)
        
        # The decomposition requires:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i in [0,n), j in [0,m), k in [0,p)
        
        # Let's verify this
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j], np.multiply(V_arr[j*p+k], W_arr[k*n+i]))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Start with known lower bounds and work up
    # For 4x4x6, the schoolbook rank is 4*4*6 = 96
    # Known bounds: rank >= max(n*m, m*p, p*n) = max(16, 24, 24) = 24
    # But actually for matrix multiplication, rank >= n*m*p / max(n,m,p) roughly
    # Let's try various ranks
    
    # Use random search with small values
    for target_rank in range(24, rank_schoolbook):
        if time.time() > deadline:
            break
        
        # Try a few random attempts for this rank
        attempts = 0
        max_attempts = 10 if target_rank < 50 else 5
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            # Generate random factor matrices with small integer/half-integer values
            # Use values from {-1, -0.5, 0, 0.5, 1}
            vals = [-1, -0.5, 0, 0.5, 1]
            
            U_rand = [[random.choice(vals) for _ in range(target_rank)] for _ in range(n * m)]
            V_rand = [[random.choice(vals) for _ in range(target_rank)] for _ in range(m * p)]
            W_rand = [[random.choice(vals) for _ in range(target_rank)] for _ in range(p * n)]
            
            if verify(U_rand, V_rand, W_rand):
                best_result = {"U": U_rand, "V": V_rand, "W": W_rand}
                best_rank = target_rank
                break
    
    if time.time() < deadline:
        # Try a more structured approach for small ranks
        # For rank r, we can try to find a decomposition using structured patterns
        pass
    
    return best_result
# EVOLVE-BLOCK-END
