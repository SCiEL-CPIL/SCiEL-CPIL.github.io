import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 exact tensor decomposition for 4x4 matrix multiplication.
    Uses recursive Strassen: outer 7 terms x inner 7 terms = 49 rank-1 tensors.
    Vectorized construction using Kronecker products.
    
    Convention:
      C[i,j] = sum_k A[i,k] * B[k,j]
      U index: i*4 + k  (A row i, A col k)
      V index: k*4 + j  (B row k, B col j)
      W index: j*4 + i  (C col j, C row i)  [trace convention]
    """
    t0 = time.time()
    deadline = 140.0

    n = m = p = 4
    R = 49

    # Define 2x2 Strassen basis as sparse index lists
    # Each term: list of (index, coeff) for A, B, C
    # 2x2 index: row*2+col
    # W2 index: col*2+row (trace convention for 2x2)
    
    # Precompute the 7 Strassen terms for 2x2 as flat coefficient vectors
    # A2: 4 elements indexed (r,c) -> r*2+c
    # B2: 4 elements indexed (r,c) -> r*2+c  
    # C2: 4 elements indexed (c,r) -> c*2+r (trace convention)
    
    strassen_terms = []
    
    # t0: (A00+A11)*(B00+B11) -> C00+=1, C11+=1
    a0 = np.array([1, 0, 0, 1], dtype=np.float64)  # A00, A01, A10, A11
    b0 = np.array([1, 0, 0, 1], dtype=np.float64)
    c0 = np.zeros(4, dtype=np.float64)
    c0[0] = 1  # C00 (index 0*2+0)
    c0[3] = 1  # C11 (index 1*2+1)
    strassen_terms.append((a0, b0, c0))
    
    # t1: (A10+A11)*B00 -> C10+=1, C11+=-1
    a1 = np.array([0, 0, 1, 1], dtype=np.float64)
    b1 = np.array([1, 0, 0, 0], dtype=np.float64)
    c1 = np.zeros(4, dtype=np.float64)
    c1[2] = 1   # C10 (index 0*2+1=2)
    c1[3] = -1  # C11 (index 1*2+1=3)
    strassen_terms.append((a1, b1, c1))
    
    # t2: A00*(B01-B11) -> C01+=1, C11+=1
    a2 = np.array([1, 0, 0, 0], dtype=np.float64)
    b2 = np.array([0, 1, 0, -1], dtype=np.float64)
    c2 = np.zeros(4, dtype=np.float64)
    c2[1] = 1  # C01 (index 1*2+0=1)
    c2[3] = 1  # C11 (index 1*2+1=3)
    strassen_terms.append((a2, b2, c2))
    
    # t3: A11*(B10-B00) -> C00+=1, C10+=1
    a3 = np.array([0, 0, 0, 1], dtype=np.float64)
    b3 = np.array([-1, 0, 1, 0], dtype=np.float64)
    c3 = np.zeros(4, dtype=np.float64)
    c3[0] = 1  # C00 (index 0*2+0)
    c3[2] = 1  # C10 (index 0*2+1=2)
    strassen_terms.append((a3, b3, c3))
    
    # t4: (A00+A01)*B11 -> C00+=-1, C01+=1
    a4 = np.array([1, 1, 0, 0], dtype=np.float64)
    b4 = np.array([0, 0, 0, 1], dtype=np.float64)
    c4 = np.zeros(4, dtype=np.float64)
    c4[0] = -1  # C00 (index 0*2+0)
    c4[1] = 1   # C01 (index 1*2+0=1)
    strassen_terms.append((a4, b4, c4))
    
    # t5: (A10-A00)*(B00+B01) -> C11+=1
    a5 = np.array([-1, 0, 1, 0], dtype=np.float64)
    b5 = np.array([1, 1, 0, 0], dtype=np.float64)
    c5 = np.zeros(4, dtype=np.float64)
    c5[3] = 1  # C11 (index 1*2+1=3)
    strassen_terms.append((a5, b5, c5))
    
    # t6: (A01-A11)*(B10+B11) -> C00+=1
    a6 = np.array([0, 1, 0, -1], dtype=np.float64)
    b6 = np.array([0, 0, 1, 1], dtype=np.float64)
    c6 = np.zeros(4, dtype=np.float64)
    c6[0] = 1  # C00 (index 0*2+0)
    strassen_terms.append((a6, b6, c6))
    
    # Build 4x4 decomposition using Kronecker structure
    # For each outer term t and inner term k, the rank-1 tensor is:
    # U_col = outer_A_kron_inner_A (reshaped to 16-vector)
    # V_col = outer_B_kron_inner_B (reshaped to 16-vector)  
    # W_col = outer_C_kron_inner_C (reshaped to 16-vector, with trace convention)
    
    U_mat = np.zeros((16, R), dtype=np.float64)
    V_mat = np.zeros((16, R), dtype=np.float64)
    W_mat = np.zeros((16, R), dtype=np.float64)
    
    col_idx = 0
    for t_idx in range(7):
        oa, ob, oc = strassen_terms[t_idx]
        for k_idx in range(7):
            ia, ib, ic = strassen_terms[k_idx]
            
            # Kronecker product gives 16-element vector for 4x4
            # For A: outer is 2x2 block structure, inner is 2x2 within block
            # Global A[i,k] where i = 2*br + lr, k = 2*bc + lc
            # A_global = kron(outer_A_2x2, inner_A_2x2) flattened in row-major
            ua = np.kron(oa, ia)  # 16 elements
            ub = np.kron(ob, ib)
            uc = np.kron(oc, ic)
            
            U_mat[:, col_idx] = ua
            V_mat[:, col_idx] = ub
            
            # W uses trace convention: W[j*4+i] for C[i,j]
            # The Kronecker gives us C in standard (i,j) ordering
            # We need to permute: C[i,j] -> W[j*4+i]
            # In 2x2: C[ci,cj] -> W2[cj*2+ci]
            # In 4x4: C[ci,cj] -> W4[cj*4+ci]
            # The kron(oc, ic) gives C in block structure:
            # C_block[br,bc, lr, lc] = oc[br,bc] * ic[lr,lc]
            # Global C[ci, cj] where ci=2*br+lr, cj=2*bc+lc
            # W4[cj*4+ci] = C[ci,cj]
            # So we need to permute the 16-element vector
            # uc currently has index: (br, bc, lr, lc) -> br*8 + bc*4 + lr*2 + lc
            # We need: (bc, br, lc, lr) -> bc*8 + br*4 + lc*2 + lr
            # which corresponds to cj*4 + ci where ci=br*2+lr, cj=bc*2+lc
            # So index = (bc*2+lc)*4 + (br*2+lr) = bc*8 + lc*4 + br*2 + lr
            # Original index = br*8 + bc*4 + lr*2 + lc
            # Permutation: new_index = bc*8 + lc*4 + br*2 + lr
            # old_index = br*8 + bc*4 + lr*2 + lc
            
            perm = np.zeros(16, dtype=np.int32)
            for br in range(2):
                for bc in range(2):
                    for lr in range(2):
                        for lc in range(2):
                            old_idx = br * 8 + bc * 4 + lr * 2 + lc
                            new_idx = bc * 8 + lc * 4 + br * 2 + lr
                            perm[old_idx] = new_idx
            
            W_mat[:, col_idx] = uc[perm]
            col_idx += 1
    
    # Verify decomposition: tensor[i,k,j] should equal 1 for all i,k,j
    # tensor[i,k,j] = sum_r U[i*4+k, r] * V[k*4+j, r] * W[j*4+i, r]
    max_err = 0.0
    for i in range(4):
        for k in range(4):
            for j in range(4):
                s = 0.0
                for r in range(R):
                    s += U_mat[i * 4 + k, r] * V_mat[k * 4 + j, r] * W_mat[j * 4 + i, r]
                max_err = max(max_err, abs(s - 1.0))
    
    if max_err > 1e-8:
        # Fallback: schoolbook rank-64
        U64 = np.zeros((16, 64), dtype=np.float64)
        V64 = np.zeros((16, 64), dtype=np.float64)
        W64 = np.zeros((16, 64), dtype=np.float64)
        idx = 0
        for i in range(4):
            for k in range(4):
                for j in range(4):
                    U64[i * 4 + k, idx] = 1.0
                    V64[k * 4 + j, idx] = 1.0
                    W64[j * 4 + i, idx] = 1.0
                    idx += 1
        return {"U": U64.tolist(), "V": V64.tolist(), "W": W64.tolist()}
    
    return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
# EVOLVE-BLOCK-END
