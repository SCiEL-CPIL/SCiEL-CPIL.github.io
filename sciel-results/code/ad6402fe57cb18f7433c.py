import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 96
    
    # Schoolbook decomposition as fallback
    def _schoolbook():
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
    
    best_result = _schoolbook()
    best_rank = schoolbook_rank
    
    def _verify(U, V, W, rank, num_trials=3, seed=42):
        """Verify decomposition by checking random matrix products."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        np.random.seed(seed)
        for _ in range(num_trials):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            C_decomposed = np.zeros((n, p))
            for r in range(rank):
                Ar = U_arr[:, r].reshape(n, m)
                Br = V_arr[:, r].reshape(m, p)
                Cr = W_arr[:, r].reshape(p, n)
                temp = (Ar * A) @ (Br * B)
                C_decomposed += temp * Cr.T
            if not np.allclose(C_decomposed, C_expected, atol=1e-8):
                return False
        return True
    
    def _verify_tensor(U, V, W, rank):
        """Verify by checking the tensor equation directly."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def _try_random_search(rank, num_attempts=50):
        """Try to find a valid 0/1 decomposition at given rank."""
        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None
            
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            # Initialize with random 0/1
            for i in range(n * m):
                for r in range(rank):
                    U[i][r] = 1 if np.random.random() < 0.3 else 0
            for j in range(m * p):
                for r in range(rank):
                    V[j][r] = 1 if np.random.random() < 0.3 else 0
            for k in range(p * n):
                for r in range(rank):
                    W[k][r] = 1 if np.random.random() < 0.3 else 0
            
            # Quick check: for each (i,j,k), there must be at least one r where all three are 1
            # and the sum must be exactly 1
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0
                        for r in range(rank):
                            prod = U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                            total += prod
                        if abs(total - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try structured approach: use the fact that k can be split into k1 (0..1) and k2 (0..3)
    # where k = k1*4 + k2
    # C[i, k1*4+k2] = sum_j A[i,j] * B[j, k1*4+k2]
    # = sum_j A[i,j] * B1[j, k2] if k1=0, or sum_j A[i,j] * B2[j, k2] if k1=1
    # where B1 = B[:, 0:4], B2 = B[:, 4:8]
    #
    # If we have a rank-r decomposition for 3x4 by 4x4:
    #   C1[i,k2] = sum_r U[i*4+j, r] * A[i,j] * V[j*4+k2, r] * B1[j,k2] * W[k2*3+i, r]
    #   C2[i,k2] = sum_r U[i*4+j, r] * A[i,j] * V[j*4+k2, r] * B2[j,k2] * W2[k2*3+i, r]
    #
    # For the combined problem, we need:
    #   C[i, k1*4+k2] = sum_r U_comb[i*4+j, r] * A[i,j] * V_comb[j*8+k1*4+k2, r] * B[j, k1*4+k2] * W_comb[(k1*4+k2)*3+i, r]
    #
    # One approach: use rank 2r where r is the rank for 3x4 by 4x4
    # For k1=0: use the first r ranks, for k1=1: use the next r ranks
    # But we need to share U across both blocks.
    #
    # Actually, let's try: if we have a rank-r decomposition for 3x4 by 4x4,
    # we can use it for both B1 and B2 by duplicating the U and V rows appropriately.
    #
    # Let me try a different structured approach.
    
    # Try random search at various ranks below schoolbook
    # Start from rank 95 down to 32
    for rank in range(schoolbook_rank - 1, 31, -1):
        if time.time() > deadline:
            break
        
        # Try a few random attempts
        result = _try_random_search(rank, num_attempts=20)
        if result is not None:
            # Verify with tensor check
            if _verify_tensor(result["U"], result["V"], result["W"], rank):
                best_result = result
                best_rank = rank
                break  # Found a valid decomposition at this rank
    
    return best_result
# EVOLVE-BLOCK-END
