"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

import numpy as np
import time

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # 20 second safety margin
    
    # Schoolbook decomposition
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
    
    best_result = (U_school, V_school, W_school)
    best_rank = rank_school
    
    # Try to find a lower rank decomposition using structured search
    # Strategy: try to find rank r < best_rank by solving linear systems
    # with carefully chosen U and V patterns
    
    np.random.seed(42)
    
    def try_find_rank(r, max_attempts=100):
        """Try to find a rank-r decomposition."""
        if time.time() > deadline:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Try different patterns for U and V
            # Pattern 1: sparse with structure
            U_rand = np.zeros((n*m, r))
            V_rand = np.zeros((m*p, r))
            
            # Assign non-zero entries in a structured way
            # For each column c, assign it to a specific (i,k) pair
            # and set U and V entries accordingly
            
            # Try: for each c, pick random (i_c, k_c) and set
            # U[i_c*m + j, c] and V[j*p + k_c, c] for all j
            # Then W[k_c*n + i_c, c] should make the equations work
            
            # Actually, let's try a simpler approach:
            # Randomly initialize U and V with entries from {-1, 0, 1}
            # but ensure that for each (i,k), the matrix A is full rank
            
            U_rand = np.random.randint(-1, 2, size=(n*m, r)).astype(float)
            V_rand = np.random.randint(-1, 2, size=(m*p, r)).astype(float)
            
            # Ensure no row is all zeros
            for idx in range(n*m):
                if np.all(U_rand[idx] == 0):
                    U_rand[idx, np.random.randint(r)] = 1.0
            for idx in range(m*p):
                if np.all(V_rand[idx] == 0):
                    V_rand[idx, np.random.randint(r)] = 1.0
            
            W_rand = np.zeros((p*n, r))
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build A: A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U_rand[i*m+j, :] * V_rand[j*p+k, :]
                    b = np.ones(m)
                    
                    # Check if system is consistent
                    try:
                        W_row, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                        if np.allclose(A @ W_row, b, atol=1e-10):
                            W_rand[k*n+i, :] = W_row
                        else:
                            valid = False
                            break
                    except:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if valid:
                # Verify
                U_list = U_rand.tolist()
                V_list = V_rand.tolist()
                W_list = W_rand.tolist()
                if _verify(U_list, V_list, W_list, n, m, p):
                    return (U_list, V_list, W_list)
        
        return None
    
    # Try ranks from 74 down to 50
    for r in range(best_rank - 1, 49, -1):
        if time.time() > deadline:
            break
        
        result = try_find_rank(r, max_attempts=50)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
