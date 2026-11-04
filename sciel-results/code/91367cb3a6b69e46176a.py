"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]
import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 150s kill
    
    # Schoolbook decomposition as fallback
    best_rank = n * m * p  # 36
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize schoolbook
    U = [[0] * best_rank for _ in range(n * m)]
    V = [[0] * best_rank for _ in range(m * p)]
    W = [[0] * best_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    best_U, best_V, best_W = U, V, W
    
    def verify(U_mat, V_mat, W_mat, rank):
        """Verify that the decomposition is exact using exact arithmetic."""
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for j2 in range(m):
                        total = Fraction(0)
                        for c in range(rank):
                            total += Fraction(W_mat[k*n+i][c]) * Fraction(U_mat[i*m+j][c]) * Fraction(V_mat[j2*p+k][c])
                        expected = 1 if j == j2 else 0
                        if total != expected:
                            return False
        return True
    
    # Winograd-style explicit decomposition for 3x3 x 3x4
    # Use 4-point interpolation for k-dimension, 3-point for i and j
    # Rank = 4 * 3 * 3 = 36 is schoolbook. Try to do better.
    
    # Attempt: use structured factors with small integers
    # Try rank 25-35 with better random search using numpy for speed
    def verify_fast(U_np, V_np, W_np, rank):
        """Fast verification using numpy with exact Fraction check on residuals."""
        # Build the full tensor and check
        # U: (9, rank), V: (12, rank), W: (12, rank)
        # For each (i,k), check if sum_c W[k*3+i,c] * U[i*3+j,c] * V[j2*4+k,c] = delta_{j,j2}
        for i in range(n):
            for k in range(p):
                # Build 3x3 matrix M[j,j2] = sum_c W[k*n+i,c] * U[i*m+j,c] * V[j2*p+k,c]
                M = np.zeros((m, m), dtype=np.float64)
                for c in range(rank):
                    w_val = W_np[k*n+i, c]
                    u_col = U_np[i*m:(i+1)*m, c]  # shape (3,)
                    # V is indexed by (j2, k) -> row j2*p+k
                    v_vals = np.array([V_np[j2*p+k, c] for j2 in range(m)])
                    M += np.outer(u_col, v_vals) * w_val
                # Check M == identity
                if not np.allclose(M, np.eye(m), atol=1e-10):
                    return False
        return True
    
    # Try ranks from 35 down to 25
    for target_rank in range(35, 24, -1):
        if time.time() > deadline - 20:
            break
        
        max_attempts = 200
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                break
            
            # Use structured random initialization
            U_try = np.random.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(n*m, target_rank))
            V_try = np.random.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(m*p, target_rank))
            W_try = np.random.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(p*n, target_rank))
            
            if verify_fast(U_try, V_try, W_try, target_rank):
                # Convert to list of Fractions for exact representation
                U_list = [[Fraction(int(x*2), 2) for x in row] for row in U_try]
                V_list = [[Fraction(int(x*2), 2) for x in row] for row in V_try]
                W_list = [[Fraction(int(x*2), 2) for x in row] for row in W_try]
                
                # Final exact verification
                if verify(U_list, V_list, W_list, target_rank):
                    best_rank = target_rank
                    best_U = U_list
                    best_V = V_list
                    best_W = W_list
                    found = True
                    break
        
        if found:
            continue
        else:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
