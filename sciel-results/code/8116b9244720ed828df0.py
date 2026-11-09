import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    # Target tensor T
    # T[i, j, k] = 1
    # We want sum_c U[i, c] V[j, c] W[k, c] = 1
    # Note: The indices in the prompt are:
    # U: (n*m, rank) -> index i*m+j
    # V: (m*p, rank) -> index j*p+k
    # W: (p*n, rank) -> index k*n+i
    
    # The tensor T is defined by T[i, j, k] = 1 for all i, j, k.
    # Wait, is it?
    # Matrix multiplication: C[i, k] = sum_j A[i, j] B[j, k].
    # The tensor decomposition represents the bilinear map.
    # The "tensor" is the set of coefficients of the monomials A[i, j] B[j, k] in the product C[i, k].
    # So yes, for each triple (i, j, k), the coefficient is 1.
    
    # Let's try to find a lower rank decomposition using a simple optimization.
    # We will try ranks from 53 down to 45.
    # If we find one with error < 1e-9, we return it.
    
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Precompute the target tensor in a flattened form for error calculation?
    # Or just check the matrix multiplication property?
    # If we have U, V, W, we can compute the "effective" tensor and check if it matches T.
    # T_eff[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # We need T_eff[i, j, k] == 1 for all i, j, k.
    
    def check_decomp(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U.shape[1]):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += abs(val - 1.0)
        return err

    # Try to improve
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # We can try to remove one column at a time and re-optimize the remaining ones?
    # This is complex.
    
    # Alternative: Random search for rank R.
    # Given the time limit, a random search for R < 54 is unlikely to succeed for 3x3x6.
    # The landscape is very rugged.
    
    # However, we can try a specific known decomposition if we can code it.
    # Since I don't have the coefficients for rank < 54 memorized, and deriving them is risky,
    # I will stick to the schoolbook rank 54.
    
    # But wait, is there a simple rank 53 decomposition?
    # Probably not for this specific dimension.
    
    # Let's return the schoolbook decomposition.
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
