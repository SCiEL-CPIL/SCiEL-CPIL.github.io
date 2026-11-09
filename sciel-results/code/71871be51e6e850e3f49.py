"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Verify that the decomposition correctly computes matrix multiplication."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        
        # Test with random matrices
        rng = np.random.RandomState(42)
        for trial in range(10):
            A = rng.rand(n, m)
            B = rng.rand(m, p)
            C_expected = A @ B
            
            C_computed = np.zeros((n, p))
            for r in range(rank):
                u_val = np.dot(U_np[:, r], A.flatten())
                v_val = np.dot(V_np[:, r], B.flatten())
                # W index is k*n + i, so we need to map back to C[i, k]
                w_vec = W_np[:, r]
                for idx in range(p * n):
                    k = idx // n
                    i = idx % n
                    C_computed[i, k] += u_val * v_val * w_vec[idx]
            
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True

    def try_rank_decomposition(target_rank, max_time=10):
        """Try to find a decomposition of the given rank using random search."""
        if time.time() - start_time > deadline:
            return None
            
        # We search for integer solutions in {-1, 0, 1} first, then {-2, -1, 0, 1, 2}
        # This is a heuristic search.
        
        # The equation is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        
        # This is a system of n*m*p = 45 equations.
        # Variables: rank * (n*m + m*p + p*n) = rank * (9 + 15 + 15) = rank * 39.
        # For rank 40, 1560 variables.
        
        # We can try to satisfy the equations one by one or use a local search.
        # Given the complexity, we will try a few random initializations and local steps.
        
        rng = np.random.RandomState(42)
        
        for attempt in range(100):
            if time.time() - start_time > deadline:
                return None
                
            # Initialize with random small integers
            U = rng.choice([-1, 0, 1], size=(n * m, target_rank))
            V = rng.choice([-1, 0, 1], size=(m * p, target_rank))
            W = rng.choice([-1, 0, 1], size=(p * n, target_rank))
            
            # Check if it works
            if verify(U.tolist(), V.tolist(), W.tolist()):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
            # Local search: flip a random entry
            # This is very slow for large ranks.
            # Instead, we might try to construct a solution based on known structures.
            
        return None

    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try ranks from 30 to 44
    for target_rank in range(30, best_rank):
        if time.time() - start_time > deadline:
            break
            
        result = try_rank_decomposition(target_rank, max_time=5)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break # Found a valid one, stop searching for lower ranks if we are just looking for any improvement? 
                   # Actually, we want the minimal rank. So we should keep searching lower?
                   # But the loop goes 30, 31, ... so if we find 35, we should check 30-34?
                   # The loop is ascending. So if we find 30, we break. If we find 35, we continue to check 36? No, we want lower.
                   # So we should search from low to high? Or high to low?
                   # If we search 30, 31, ... 44, and find one, it's the lowest in that range.
                   # But what if 29 exists? We didn't check 29.
                   # Let's check 25 to 44.
                   
    return best_result
# EVOLVE-BLOCK-END
