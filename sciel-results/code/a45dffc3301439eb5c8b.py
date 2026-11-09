"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_rank = rank_school
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Strategy: Try to find a decomposition with rank < 30
    # Theoretical lower bound is max(n*m, m*p, p*n) = max(6, 15, 10) = 15
    # But for 2x3x5, the rank is conjectured to be around 18-20
    
    # We'll use a structured search approach:
    # 1. Fix the structure of U and V to be "block diagonal" or similar
    # 2. Solve for W using linear algebra
    
    # Approach: Random search with verification, but smarter
    # We'll try ranks from 15 up to 29
    
    for target_rank in range(15, 30):
        if time.time() > deadline - 10:
            break
        
        # Number of attempts based on remaining time
        remaining_time = deadline - 10 - time.time()
        attempts = int(min(200, max(50, remaining_time * 5)))
        
        np.random.seed(42 + target_rank * 1000)
        
        for attempt in range(attempts):
            if time.time() > deadline - 10:
                break
            
            # Generate random half-integer matrices
            # Use a smaller coefficient range for better convergence
            coeff_range = 3 if target_rank < 20 else 5
            
            U = np.random.randint(-coeff_range, coeff_range + 1, size=(n*m, target_rank)) / 2.0
            V = np.random.randint(-coeff_range, coeff_range + 1, size=(m*p, target_rank)) / 2.0
            W = np.random.randint(-coeff_range, coeff_range + 1, size=(p*n, target_rank)) / 2.0
            
            if _verify_decomposition_fast(U, V, W, n, m, p):
                best_rank = target_rank
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break
        
        if best_rank == target_rank:
            break
    
    # If we found a rank < 30, try to verify and potentially improve
    if best_rank < rank_school:
        # Verify the best result one more time
        U_best = np.array(best_result["U"])
        V_best = np.array(best_result["V"])
        W_best = np.array(best_result["W"])
        
        if _verify_decomposition_fast(U_best, V_best, W_best, n, m, p):
            # Try to find an even lower rank
            for target_rank in range(15, best_rank):
                if time.time() > deadline - 5:
                    break
                
                remaining_time = deadline - 5 - time.time()
                if remaining_time < 1:
                    break
                    
                attempts = int(min(100, max(20, remaining_time * 3)))
                np.random.seed(999 + target_rank * 1000)
                
                for attempt in range(attempts):
                    if time.time() > deadline - 5:
                        break
                    
                    coeff_range = 3
                    U = np.random.randint(-coeff_range, coeff_range + 1, size=(n*m, target_rank)) / 2.0
                    V = np.random.randint(-coeff_range, coeff_range + 1, size=(m*p, target_rank)) / 2.0
                    W = np.random.randint(-coeff_range, coeff_range + 1, size=(p*n, target_rank)) / 2.0
                    
                    if _verify_decomposition_fast(U, V, W, n, m, p):
                        best_rank = target_rank
                        best_result = {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
                        break
                
                if best_rank == target_rank:
                    break
    
    return best_result

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_decomposition_fast(U, V, W, n, m, p):
    """
    Vectorized verification using NumPy.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    rank = U.shape[1]
    
    # Reshape for easier broadcasting
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    try:
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T, 1.0, atol=1e-9)
    except Exception:
        return False

# EVOLVE-BLOCK-END
