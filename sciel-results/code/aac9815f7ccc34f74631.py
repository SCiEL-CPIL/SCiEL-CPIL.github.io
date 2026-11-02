"""Matrix multiplication 2x4 x 4x4 tensor decomposition (Rank 28)."""

import time
import numpy as np

# Dimensions: n=2 (rows of A, rows of C), m=4 (cols of A, rows of B), p=4 (cols of B, cols of C)
N, M, P = 2, 4, 4

def _get_strassen_2x2_decomposition():
    """
    Returns U, V, W for 2x2 * 2x2 -> 2x2 multiplication using Strassen (Rank 7).
    U: (4, 7), V: (4, 7), W: (4, 7)
    Convention: T[i, k, j] = sum_r U[i+k, r] * V[k+j, r] * W[j+i, r] == 1
    Here i, k, j are in {0,1}.
    U index: i*2 + k
    V index: k*2 + j
    W index: j*2 + i
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen factors
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U coefficients for A entries (indices 0:A00, 1:A01, 2:A10, 3:A11)
    U_terms = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6 (A10 - A00)
        [0, 1, 0, -1]  # M7 (A01 - A11)
    ]
    
    # V coefficients for B entries (indices 0:B00, 1:B01, 2:B10, 3:B11)
    V_terms = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [0, 0, 1, -1], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # W coefficients for C entries (indices 0:C00, 1:C01, 2:C10, 3:C11)
    W_terms = [
        [1, 0, 0, 1], # M1 (C00 + C11)
        [0, 0, 1, -1], # M2 (C10 - C11)
        [0, 1, 0, 1], # M3 (C01 + C11)
        [1, 0, 1, 0], # M4 (C00 + C10)
        [-1, 1, 0, 0], # M5 (-C00 + C01)
        [0, 0, 0, 1], # M6 (C11)
        [1, 0, 0, 0]  # M7 (C00)
    ]
    
    for r in range(rank):
        U[:, r] = U_terms[r]
        V[:, r] = V_terms[r]
        W[:, r] = W_terms[r]
        
    return U, V, W

def _embed_2x2_into_2x4x4(U_small, V_small, W_small, 
                           a_row_start, a_col_start, 
                           b_row_start, b_col_start,
                           c_row_start, c_col_start,
                           offset_rank):
    """
    Embeds a 2x2*2x2 Strassen decomposition into the full 2x4*4x4 tensor.
    A is 2x4, B is 4x4, C is 2x4.
    
    U index in full: i*M + k. Here i in {0,1} relative to A block, k in {0,1} relative to A block.
    A block starts at row a_row_start (always 0 or 1? No, A has 2 rows. 
    Wait, A is 2x4. We split A columns into two 2x2 blocks: A_L (cols 0,1) and A_R (cols 2,3).
    So a_row_start is always 0 (since we only have 2 rows, we take rows 0,1 for both blocks).
    Actually, the decomposition sums over K.
    C = A_L B_L + A_R B_R.
    The terms are additive. We can just sum the tensors.
    
    The Strassen decomposition for A_L * B_L contributes to C_L.
    The Strassen decomposition for A_R * B_R contributes to C_R.
    
    But wait, the tensor T[i,k,j] = 1.
    The decomposition is a sum of rank-1 tensors.
    We have 4 blocks:
    1. A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    2. A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]  (Note: B rows 2,3)
    3. A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
    4. A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
    
    Wait, my previous analysis:
    C[:, 0:2] = A[:, 0:2] * B[0:2, 0:2] + A[:, 2:4] * B[2:4, 0:2]
    C[:, 2:4] = A[:, 0:2] * B[0:2, 2:4] + A[:, 2:4] * B[2:4, 2:4]
    
    Each product is (2x2) * (2x2) -> (2x2).
    There are 4 such products.
    Each requires rank 7.
    Total rank 28.
    """
    pass # Logic handled in construct

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = N, M, P
    rank = 28
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Get base Strassen 2x2 factors
    U_base, V_base, W_base = _get_strassen_2x2_decomposition()
    
    # Define the 4 blocks
    # Each block: (a_col_offset, b_row_offset, b_col_offset, c_col_offset)
    # A is 2x4. Rows are always 0,1. Cols are 0,1 or 2,3.
    # B is 4x4. Rows 0,1 or 2,3. Cols 0,1 or 2,3.
    # C is 2x4. Rows 0,1. Cols 0,1 or 2,3.
    
    blocks = [
        # (a_col_off, b_row_off, b_col_off, c_col_off)
        (0, 0, 0, 0), # A_L * B_TL -> C_L
        (2, 2, 0, 0), # A_R * B_BL -> C_L  (A cols 2,3; B rows 2,3; B cols 0,1)
        (0, 0, 2, 2), # A_L * B_TR -> C_R
        (2, 2, 2, 2)  # A_R * B_BR -> C_R
    ]
    
    current_rank = 0
    for a_coff, b_rtoff, b_coff, c_coff in blocks:
        for s in range(7):
            u_vec = U_base[:, s]
            v_vec = V_base[:, s]
            w_vec = W_base[:, s]
            
            # Map to U (n*m, rank)
            # U index: i*m + k
            # i in {0,1} (A rows)
            # k in {a_coff, a_coff+1}
            for i in range(2):
                for k_local in range(2):
                    k_global = a_coff + k_local
                    U[i*m + k_global, current_rank] += u_vec[i*2 + k_local]
            
            # Map to V (m*p, rank)
            # V index: k*p + j
            # k in {b_rtoff, b_rtoff+1}
            # j in {b_coff, b_coff+1}
            for k_local in range(2):
                for j_local in range(2):
                    k_global = b_rtoff + k_local
                    j_global = b_coff + j_local
                    V[k_global*p + j_global, current_rank] += v_vec[k_local*2 + j_local]
                    
            # Map to W (p*n, rank)
            # W index: j*n + i
            # j in {c_coff, c_coff+1}
            # i in {0,1}
            for j_local in range(2):
                for i_local in range(2):
                    j_global = c_coff + j_local
                    i_global = i_local # C rows are always 0,1
                    W[j_global*n + i_global, current_rank] += w_vec[j_local*2 + i_local]
            
            current_rank += 1
            
            if current_rank > rank:
                raise ValueError("Rank overflow")
                
    if current_rank != rank:
        raise ValueError("Rank mismatch")
        
    # Verification
    # T[i, k, j] should be 1
    # sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    
    # Optimize verification using numpy
    # We can't easily vectorize the triple sum without creating a huge tensor,
    # but 2*4*4 = 32 checks * 28 ranks is tiny.
    
    for i in range(n):
        for k in range(m):
            for j in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
                if not np.isclose(val, 1.0, atol=1e-9):
                    raise ValueError(f"Verification failed at i={i}, k={k}, j={j}, val={val}")
                    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # Should not happen, but safe guard
        pass
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    print("Constructed successfully.")
