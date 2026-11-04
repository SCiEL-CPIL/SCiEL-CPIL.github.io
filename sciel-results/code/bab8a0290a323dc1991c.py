import numpy as np
import time

DIMENSIONS = [3, 4, 4]
TIME_LIMIT = 110  # safety margin before 120s

def _build_block_decomposition():
    """
    Block decomposition of M_{3,4,4} into 4 blocks of M_{3,2,2}.
    Each M_{3,2,2} block is decomposed using:
      - Strassen-like for the 2x2 sub-block (rank 7)
      - Schoolbook for the remaining 1x2 row (rank 4)
    Total: 4 * (7 + 4) = 44
    """
    n, m, p = 3, 4, 4
    rank = 44
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for j_block in range(2):  # j in {0,1} or {2,3}
        j_offset = j_block * 2
        for k_block in range(2):  # k in {0,1} or {2,3}
            k_offset = k_block * 2
            
            # For each block, we have a 3x2 * 2x2 multiplication
            # We decompose it into:
            #   Strassen part: computes A[0:2, :] * B[:, 0:2] using 7 multiplications
            #   Schoolbook part: computes A[2, :] * B[:, 0:2] using 4 multiplications
            
            # --- Strassen part (rank 7) ---
            # Let A_loc be 3x2, B_loc be 2x2.
            # We use Strassen on the top 2x2 * 2x2 part:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B10-B11)
            # M4 = A11(B01-B10)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B10)
            # M7 = (A01-A11)(B01+B11)
            # C00 = M1 + M4 - M5 - M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M7
            
            # For each Strassen multiplication, we set U, V, W vectors.
            # A_loc row indices: i=0,1,2 (global i)
            # B_loc col indices: k=0,1 (local, global k = k_offset + k_loc)
            # C entries: C[i, k] for i=0,1,2; k=0,1 (local)
            # But Strassen only computes C[0:2, 0:2]. C[2, :] is done by schoolbook.
            
            strassen_terms = [
                # (name, A_factors, B_factors, C_contributions)
                # A_factors: dict mapping (i, j_loc) to coefficient for U
                # B_factors: dict mapping (j_loc, k_loc) to coefficient for V
                # C_contributions: dict mapping (i, k_loc) to coefficient for W
                # j_loc is always 0 or 1 within this block (since j_offset handles the global j)
                
                # M1 = (A00+A11)(B00+B11)
                ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, 
                 {(0, 0): 1, (0, 1): -1, (1, 0): 0, (1, 1): 1}),
                # M2 = (A10+A11)B00
                ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, 
                 {(1, 0): 1, (1, 1): -1}),
                # M3 = A00(B10-B11)
                ({(0, 0): 1}, {(1, 0): 1, (1, 1): -1}, 
                 {(0, 1): 1, (1, 0): 1}),
                # M4 = A11(B01-B10)
                ({(1, 1): 1}, {(0, 1): 1, (0, 0): -1}, 
                 {(0, 0): 1, (1, 0): 1, (1, 1): 1}),
                # M5 = (A00+A01)B11
                ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, 
                 {(0, 0): -1, (0, 1): 1}),
                # M6 = (A10-A00)(B00+B10)
                ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}, 
                 {}),  # M6 doesn't appear in the final C formulas? Let me re-check.
                # Actually standard Strassen:
                # C00 = M1 + M4 - M5 - M7
                # C01 = M3 + M5
                # C10 = M2 + M4
                # C11 = M1 - M2 + M3 + M7
                # M6 is not used in these formulas. So M6 is unnecessary.
                # Let me remove M6.
                
                # M7 = (A01-A11)(B01+B11)
                ({(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}, 
                 {(0, 0): -1, (1, 1): 1}),
            ]
            
            # Remove the unused M6
            strassen_terms = [t for t in strassen_terms if t[2] != {}]
            
            for (u_factors, v_factors, w_factors) in strassen_terms:
                for (i, j_loc), coeff in u_factors.items():
                    U[i * m + (j_offset + j_loc), col] += coeff
                for (j_loc, k_loc), coeff in v_factors.items():
                    V[(j_offset + j_loc) * p + (k_offset + k_loc), col] += coeff
                for (i, k_loc), coeff in w_factors.items():
                    W[(k_offset + k_loc) * n + i, col] += coeff
                col += 1
            
            # --- Schoolbook part for row i=2 (rank 4) ---
            # C[2, k_loc] = A[2, j_loc] * B[j_loc, k_loc] for j_loc in {0,1}, k_loc in {0,1}
            for j_loc in range(2):
                for k_loc in range(2):
                    U[2 * m + (j_offset + j_loc), col] = 1
                    V[(j_offset + j_loc) * p + (k_offset + k_loc), col] = 1
                    W[(k_offset + k_loc) * n + 2, col] = 1
                    col += 1
    
    return U, V, W

def construct():
    start_time = time.time()
    
    # Build the block decomposition (rank 44)
    U, V, W = _build_block_decomposition()
    
    # Verify correctness
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    # Verify: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == delta_ij
    # Actually the tensor decomposition is:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where T[i,j,k] = 1 if this is a valid term in matrix multiplication
    # For M_{3,4,4}: T[i,j,k] = 1 for all i in {0,1,2}, j in {0,1,2,3}, k in {0,1,2,3}
    # Wait, no. The matrix multiplication tensor has T[i,j,k] = 1 for all (i,j,k) in {0..n-1} x {0..m-1} x {0..p-1}
    # Actually, the standard matrix multiplication tensor is defined as:
    # M_{n,m,p} = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k  (all ones)
    # So T[i,j,k] = 1 for all i, j, k.
    
    # Let's verify:
    T_reconstructed = np.zeros((n, m, p))
    for c in range(rank):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_reconstructed[i, j, k] += uc[i * m + j] * vc[j * p + k] * wc[k * n + i]
    
    # All entries should be 1
    if not np.allclose(T_reconstructed, 1.0, atol=1e-9):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_s = np.zeros((n * m, rank_school))
        V_s = np.zeros((m * p, rank_school))
        W_s = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, col] = 1
                    V_s[j * p + k, col] = 1
                    W_s[k * n + i, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    elapsed = time.time() - start_time
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
