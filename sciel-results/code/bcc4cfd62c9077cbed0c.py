"""Matrix multiplication 2x2x7 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    
    Strategy:
    - Use Strassen's rank-7 algorithm for column pairs (0,1), (2,3), (4,5)
    - Use schoolbook (rank 4) for column 6
    Total rank: 7 + 7 + 7 + 4 = 25
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = 2, 2, 7
    rank = 25
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Indexing conventions:
    # A[i,j] -> U[i*m + j] = U[2*i + j]
    # B[j,k] -> V[j*p + k] = V[7*j + k]
    # C[i,k] -> W[k*n + i] = W[2*k + i]
    
    def add_strassen(col_start, k_start):
        """Add Strassen rank-7 terms for 2x2 block at columns k_start, k_start+1."""
        
        # M1 = (A00 + A11) * (B0,ks + B1,ks+1)
        # C0,ks += M1, C1,ks+1 += M1
        U[0, col_start + 0] += 1
        U[3, col_start + 0] += 1
        V[7 * 0 + k_start, col_start + 0] += 1
        V[7 * 1 + (k_start + 1), col_start + 0] += 1
        W[2 * k_start + 0, col_start + 0] += 1
        W[2 * (k_start + 1) + 1, col_start + 0] += 1
        
        # M2 = (A10 + A11) * B0,ks
        # C1,ks += M2, C1,ks+1 -= M2
        U[2, col_start + 1] += 1
        U[3, col_start + 1] += 1
        V[7 * 0 + k_start, col_start + 1] += 1
        W[2 * k_start + 1, col_start + 1] += 1
        W[2 * (k_start + 1) + 1, col_start + 1] -= 1
        
        # M3 = A00 * (B0,ks+1 - B1,ks+1)
        # C0,ks+1 += M3, C1,ks+1 += M3
        U[0, col_start + 2] += 1
        V[7 * 0 + (k_start + 1), col_start + 2] += 1
        V[7 * 1 + (k_start + 1), col_start + 2] -= 1
        W[2 * (k_start + 1) + 0, col_start + 2] += 1
        W[2 * (k_start + 1) + 1, col_start + 2] += 1
        
        # M4 = A11 * (B1,ks - B0,ks)
        # C0,ks += M4, C1,ks += M4
        U[3, col_start + 3] += 1
        V[7 * 1 + k_start, col_start + 3] += 1
        V[7 * 0 + k_start, col_start + 3] -= 1
        W[2 * k_start + 0, col_start + 3] += 1
        W[2 * k_start + 1, col_start + 3] += 1
        
        # M5 = (A00 + A01) * B1,ks+1
        # C0,ks -= M5, C0,ks+1 += M5
        U[0, col_start + 4] += 1
        U[1, col_start + 4] += 1
        V[7 * 1 + (k_start + 1), col_start + 4] += 1
        W[2 * k_start + 0, col_start + 4] -= 1
        W[2 * (k_start + 1) + 0, col_start + 4] += 1
        
        # M6 = (A10 - A00) * (B0,ks + B0,ks+1)
        # C1,ks+1 += M6
        U[2, col_start + 5] += 1
        U[0, col_start + 5] -= 1
        V[7 * 0 + k_start, col_start + 5] += 1
        V[7 * 0 + (k_start + 1), col_start + 5] += 1
        W[2 * (k_start + 1) + 1, col_start + 5] += 1
        
        # M7 = (A01 - A11) * (B1,ks + B1,ks+1)
        # C0,ks += M7
        U[1, col_start + 6] += 1
        U[3, col_start + 6] -= 1
        V[7 * 1 + k_start, col_start + 6] += 1
        V[7 * 1 + (k_start + 1), col_start + 6] += 1
        W[2 * k_start + 0, col_start + 6] += 1
    
    # Add Strassen for column pairs (0,1), (2,3), (4,5)
    add_strassen(0, 0)
    add_strassen(7, 2)
    add_strassen(14, 4)
    
    # Add schoolbook for column 6 (4 terms)
    col_start = 21
    # A00 * B0,6 -> C0,6
    U[0, col_start + 0] += 1
    V[7 * 0 + 6, col_start + 0] += 1
    W[2 * 6 + 0, col_start + 0] += 1
    
    # A01 * B1,6 -> C0,6
    U[1, col_start + 1] += 1
    V[7 * 1 + 6, col_start + 1] += 1
    W[2 * 6 + 0, col_start + 1] += 1
    
    # A10 * B0,6 -> C1,6
    U[2, col_start + 2] += 1
    V[7 * 0 + 6, col_start + 2] += 1
    W[2 * 6 + 1, col_start + 2] += 1
    
    # A11 * B1,6 -> C1,6
    U[3, col_start + 3] += 1
    V[7 * 1 + 6, col_start + 3] += 1
    W[2 * 6 + 1, col_start + 3] += 1
    
    # Verify the decomposition with a random test (if time permits)
    if time.time() < deadline:
        rng = np.random.default_rng(42)
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C_expected = A @ B
        
        # Reconstruct C from the tensor decomposition
        # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] ... 
        # Actually: C[i,k] = sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * (W[k*n+i,r] should be 1 for valid)
        # The tensor identity: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta terms
        # More precisely: C[i,k] = sum_j (A[i,j] * B[j,k]) = sum_r (sum_j U[i*m+j,r]*A[i,j]) * (sum_k' V[j*p+k',r]*B[j,k']) ...
        # Let me just verify by computing the implied multiplication:
        # For each r, the term is U[:,r] outer V[:,r] outer W[:,r]
        # C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        C_reconstructed = np.zeros((n, p))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        C_reconstructed[i, k] += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
        
        if not np.allclose(C_reconstructed, C_expected, atol=1e-10):
            # Fallback to schoolbook rank 28 if verification fails
            rank_fb = n * m * p
            U_fb = np.zeros((n * m, rank_fb))
            V_fb = np.zeros((m * p, rank_fb))
            W_fb = np.zeros((p * n, rank_fb))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j, col] = 1
                        V_fb[j * p + k, col] = 1
                        W_fb[k * n + i, col] = 1
                        col += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
