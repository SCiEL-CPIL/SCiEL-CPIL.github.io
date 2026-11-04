import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0
    n, m, p = DIMENSIONS
    
    # Target Rank: 41 (23 for 3x3x3 + 18 for 3x3x2)
    # Fallback Rank: 45 (Schoolbook)
    
    def verify(U, V, W, rank):
        """
        Verify the decomposition.
        U: (n*m, rank)
        V: (m*p, rank)
        W: (p*n, rank)
        """
        n, m, p = DIMENSIONS
        # Random test matrices
        np.random.seed(42)
        A = np.random.randint(0, 10, (n, m))
        B = np.random.randint(0, 10, (m, p))
        C_expected = A @ B
        
        # Compute C_decomposed
        # C[i, k] = sum_c W[k*n + i, c] * (sum_j U[i*m + j, c] * A[i, j]) * (sum_j V[j*p + k, c] * B[j, k])
        # Note: The indices in U, V, W are flattened.
        # U_idx = i*m + j
        # V_idx = j*p + k
        # W_idx = k*n + i
        
        C_decomposed = np.zeros((n, p), dtype=int)
        for c in range(rank):
            # Extract vectors
            u_vec = U[:, c] # size n*m
            v_vec = V[:, c] # size m*p
            w_val = W[:, c] # size p*n
            
            # Compute U * A part? No, the structure is:
            # term_c = (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k]) * W[k*n+i, c]
            # This is not a simple matrix mult.
            
            # Let's iterate over i, k
            for i in range(n):
                for k in range(p):
                    w_ck = W[k * n + i, c]
                    if w_ck == 0:
                        continue
                    
                    # Sum over j for U part: sum_j U[i*m + j, c] * A[i, j]
                    u_sum = 0
                    for j in range(m):
                        u_sum += U[i * m + j, c] * A[i, j]
                        
                    # Sum over j for V part: sum_j V[j*p + k, c] * B[j, k]
                    v_sum = 0
                    for j in range(m):
                        v_sum += V[j * p + k, c] * B[j, k]
                    
                    C_decomposed[i, k] += w_ck * u_sum * v_sum
                    
        if not np.allclose(C_expected, C_decomposed):
            return False
        return True

    # Bini Factors for 3x3x3 (23 terms)
    # Hardcoded from literature. 
    # Format: list of 23 tuples (U_flat, V_flat, W_flat)
    # Each flat is size 9.
    # U_flat[i*3+j], V_flat[j*3+k], W_flat[k*3+i]
    
    # I will use a known set. If I don't have it, I use 0s and rely on fallback.
    # Since I don't have the exact Bini factors memorized, I will use a placeholder 
    # that will likely fail verification, triggering the fallback to 45.
    # However, I can try to construct a valid 41 rank decomposition if I find a simpler one.
    
    # Let's try to use the "Bini" factors from a standard source if possible.
    # Otherwise, the fallback is robust.
    
    bini_factors = []
    # Placeholder: 23 zeros. This will fail verification.
    for _ in range(23):
        bini_factors.append(([0]*9, [0]*9, [0]*9))
        
    # To actually get a score of 41, I need the real factors.
    # Without them, I return 45.
    
    # Construct 45 rank solution (Schoolbook)
    rank = 45
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
                
    # Try to verify 45 (should pass)
    if verify(U, V, W, rank):
        return {"U": U, "V": V, "W": W}
        
    # If 45 fails (it shouldn't), return empty? No, 45 is correct.
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
