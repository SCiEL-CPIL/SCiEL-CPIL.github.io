import numpy as np
import time

_T0 = time.monotonic()
_CUTOFF = 135.0

def construct():
    n, m, p = 6, 6, 6
    
    # Build 2x2 Strassen factors as explicit arrays
    # Convention: for 2x2 matrices A[[a,b],[c,d]], B[[e,f],[g,h]], C[[i,j],[k,l]]
    # U factors extract from A (indexed by row*2+col of 2x2 block)
    # V factors extract from B (indexed by row*2+col of 2x2 block)  
    # W factors place into C (indexed by row*2+col of 2x2 block)
    
    # Strassen: 7 multiplications
    # M1 = (a+d)(e+h) -> contributes +i, +j, +k, -l ... let me use standard form
    # Actually let me use the standard Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # U: 4 entries (for 2x2 A: positions 0=A00, 1=A01, 2=A10, 3=A11)
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 1, 0, 0, 1, 0, -1],   # A01
        [0, 0, 1, 0, 0, 1, 0],    # A10
        [1, 0, 1, 1, 0, 0, -1],   # A11
    ], dtype=float)
    
    # V: 4 entries (for 2x2 B: positions 0=B00, 1=B01, 2=B10, 3=B11)
    V2 = np.array([
        [1, 0, 0, 1, 0, 1, 0],   # B00
        [0, 1, 0, -1, 0, 0, 1],  # B01
        [0, 0, 1, 0, 0, 1, 0],   # B10
        [1, 0, -1, 0, 1, 0, 1],  # B11
    ], dtype=float)
    
    # W: 4 entries (for 2x2 C: positions 0=C00, 1=C01, 2=C10, 3=C11)
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 1, 1, 0, 1, 0, 0],    # C01
        [0, 0, 1, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0],   # C11
    ], dtype=float)
    
    # Verify Strassen: C = sum_r U[:,r] * V[:,r] * W[:,r]^T (outer products)
    # Tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # For matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] should be 1 if (i,j,k) corresponds to A[i,j]*B[j,k]
    
    # Verify
    T2 = np.einsum('ir,jr,kr->ijk', U2, V2, W2)
    expected_T2 = np.zeros((4,4,4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                expected_T2[i*2+j, j*2+k, k*2+i] = 1  # A[i,j]*B[j,k] -> C[k,i]... 
    # Wait, need to be careful with indexing.
    # For 2x2: A is 2x2, B is 2x2, C is 2x2
    # C[i,k] = sum_j A[i,j] B[j,k]
    # U has 4 rows for A entries: row = i*2+j for A[i,j]
    # V has 4 rows for B entries: row = j*2+k for B[j,k]
    # W has 4 rows for C entries: row = k*2+i for C[k,i]? No...
    # W should index C entries: row = i*2+k for C[i,k]
    
    # Let me redo: the tensor for 2x2 matrix mult is T[a_idx, b_idx, c_idx]
    # where a_idx = i*2+j (A[i,j]), b_idx = j*2+k (B[j,k]), c_idx = i*2+k (C[i,k])
    # T[a_idx, b_idx, c_idx] = 1 if the indices are consistent
    # i.e., a_idx = i*2+j, b_idx = j*2+k, c_idx = i*2+k for some i,j,k in {0,1}
    
    expected2 = np.zeros((4,4,4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                a_idx = i*2+j
                b_idx = j*2+k
                c_idx = i*2+k
                expected2[a_idx, b_idx, c_idx] = 1.0
    
    # Check
    err2 = np.max(np.abs(T2 - expected2))
    if err2 > 1e-10:
        # Fallback to schoolbook for 2x2
        U2 = np.zeros((4,4))
        V2 = np.zeros((4,4))
        W2 = np.zeros((4,4))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    col = i*4+j*2+k
                    U2[i*2+j, col] = 1
                    V2[j*2+k, col] = 1
                    W2[i*2+k, col] = 1
    
    # 3x3 schoolbook
    S3 = 3
    U3 = np.zeros((S3*S3, S3*S3*S3))
    V3 = np.zeros((S3*S3, S3*S3*S3))
    W3 = np.zeros((S3*S3, S3*S3*S3))
    for i in range(S3):
        for j in range(S3):
            for k in range(S3):
                col = i*9+j*3+k
                U3[i*3+j, col] = 1
                V3[j*3+k, col] = 1
                W3[i*3+k, col] = 1
    
    # Combine: 6x6 = 3x3 blocks of 2x2
    # U_total: 36 rows, 7*27 cols
    # Row for U: block_row*2 + local_row, where block_row is 0-2 and local_row is 0-3
    # Actually: global index = (block_i * 2 + local_i) * 6 + (block_j * 2 + local_j)
    # = (block_i*2+local_i)*6 + block_j*2 + local_j
    
    rank_total = 7 * 27
    U = np.zeros((36, rank_total))
    V = np.zeros((36, rank_total))
    W = np.zeros((36, rank_total))
    
    # Vectorized construction
    # For each Strassen column s (0..6) and schoolbook column t (0..26):
    # combined col = s*27 + t
    # U[row, col] = U2[local_a, s] * U3[local_block_a, t]
    # where local_a = bi*2+li (0..3), local_block_a = bi*3+bj (0..8)
    # global row = (bi*2+li)*6 + (bj*2+lj)
    
    # Build index arrays
    bi = np.arange(3)
    bj = np.arange(3)
    li = np.arange(4)
    lj = np.arange(4)
    
    # U3 indices: bi*3+bj
    u3_idx = bi[:,None]*3 + bj[None,:]  # 3x3
    u3_idx_flat = u3_idx.ravel()  # 9
    
    # U2 indices: li
    # Global row for U: (bi*2+li)*6 + (bj*2+lj)
    # Let's build all (bi,bj,li,lj) -> global_row
    bi_grid, bj_grid, li_grid, lj_grid = np.meshgrid(bi, bj, li, lj, indexing='ij')
    global_rows_U = (bi_grid*2+li_grid)*6 + (bj_grid*2+lj_grid)
    global_rows_U_flat = global_rows_U.ravel()  # 36 entries... wait 3*3*4*4=144
    
    # Actually each combination (bi,bj,li,lj) gives one global row
    # But multiple (bi,bj,li,lj) can give same global row? No, they're all distinct.
    # 3*3*4*4 = 144 but we only have 36 rows. 
    # Wait: bi in 0..2, bj in 0..2, li in 0..3, lj in 0..3
    # global = (bi*2+li)*6 + (bj*2+lj)
    # bi*2+li ranges 0..7, bj*2+lj ranges 0..7
    # So global ranges 0..63, but we only want 0..35
    # The constraint is that bi*2+li < 6 and bj*2+lj < 6
    # i.e., bi*2+li in {0,1,2,3,4,5} and bj*2+lj in {0,1,2,3,4,5}
    # bi in {0,1,2}, li in {0,1,2,3}
    # bi=0: li in {0,1,2,3} -> 0,1,2,3 (all valid)
    # bi=1: li in {0,1,2,3} -> 2,3,4,5 (all valid)
    # bi=2: li in {0,1,2} -> 4,5,6,7 (only 4,5 valid)
    
    # So not all combinations are valid. Let me just iterate.
    
    for s in range(7):
        for t in range(27):
            c = s*27 + t
            # U2[:,s] has 4 entries (for li=0,1,2,3)
            # U3 has entries at positions bi*3+bj for the schoolbook part
            # For U: the "i" index of A is (bi, li), the "j" index is (bj, lj)
            # U3[bi*3+bj, t] gives the weight for the block
            # U2[li, s] gives the weight for the local row within the block
            
            # U[row, c]: row = (bi*2+li)*6 + (bj*2+lj)
            # = U2[li, s] * U3[bi*3+bj, t]
            # for valid (bi,bj,li,lj)
            
            for bi_val in range(3):
                for bj_val in range(3):
                    u3_val = U3[bi_val*3+bj_val, t]
                    if u3_val == 0:
                        continue
                    for li_val in range(4):
                        u2_val = U2[li_val, s]
                        if u2_val == 0:
                            continue
                        for lj_val in range(4):
                            row = (bi_val*2+li_val)*6 + (bj_val*2+lj_val)
                            if row >= 36:
                                continue
                            U[row, c] += u2_val * u3_val
            
            # V: V2[:,s] for V local, V3[bi*3+bj... wait
            # For V (B matrix): B is m x p = 6x6
            # B[j,k] where j = (bj*2+lj), k = (bk*2+lk)
            # V3 should be indexed by (j_block, k_block) = (bj, bk)
            # V3[bj*3+bk, t]
            # V2 indexed by local B entry: for 2x2 B, position = j_local*2+k_local
            # V2[j_local*2+k_local, s]
            
            # But wait - in my Strassen definition, V2 has 4 rows for B's 4 entries
            # B00 -> row 0, B01 -> row 1, B10 -> row 2, B11 -> row 3
            # So V2[local_b_idx, s] where local_b_idx = j_local*2+k_local
            
            for bj_val in range(3):
                for bk_val in range(3):
                    v3_val = V3[bj_val*3+bk_val, t]
                    if v3_val == 0:
                        continue
                    for lj_val in range(4):
                        for lk_val in range(4):
                            vb_idx = lj_val*2+lk_val
                            v2_val = V2[vb_idx, s]
                            if v2_val == 0:
                                continue
                            row = (bj_val*2+lj_val)*6 + (bk_val*2+lk_val)
                            if row >= 36:
                                continue
                            V[row, c] += v2_val * v3_val
            
            # W: W2[:,s] for C local, W3 indexed by (k_block, i_block) = (bk, bi)
            # C[i,k] where i=(bi*2+li), k=(bk*2+lk)
            # W3[bk*3+bi, t]
            # W2[local_c_idx, s] where local_c_idx = li*2+lk (C00,C01,C10,C11)
            
            for bk_val in range(3):
                for bi_val in range(3):
                    w3_val = W3[bk_val*3+bi_val, t]
                    if w3_val == 0:
                        continue
                    for lk_val in range(4):
                        for li_val in range(4):
                            wc_idx = li_val*2+lk_val
                            w2_val = W2[wc_idx, s]
                            if w2_val == 0:
                                continue
                            row = (bk_val*2+lk_val)*6 + (bi_val*2+li_val)
                            if row >= 36:
                                continue
                            W[row, c] += w2_val * w3_val
    
    # Verify quickly with random matrices
    if time.monotonic() - _T0 < _CUTOFF - 5:
        np.random.seed(42)
        A_test = np.random.randint(-5, 5, (6,6)).astype(float)
        B_test = np.random.randint(-5, 5, (6,6)).astype(float)
        C_expected = A_test @ B_test
        # Tensor reconstruction: C[i,k] = sum_{j,r} U[i*6+j, r] * V[j*6+k, r] * W[k*6+i, r]
        # Wait, the convention is W[k*n+i, r] where n=6
        # So W index = k*6+i for C[i,k]
        # C[i,k] = sum_{j=0}^{5} sum_{r=0}^{rank-1} U[i*6+j, r] * V[j*6+k, r] * W[k*6+i, r]
        
        C_recon = np.zeros((6,6))
        for i in range(6):
            for k in range(6):
                for j in range(6):
                    C_recon[i,k] += np.sum(U[i*6+j, :] * V[j*6+k, :] * W[k*6+i, :])
        
        max_err = np.max(np.abs(C_recon - C_expected))
        if max_err > 1e-6:
            # Something wrong, fall back to pure schoolbook
            rank_sb = 216
            U_sb = np.zeros((36, rank_sb))
            V_sb = np.zeros((36, rank_sb))
            W_sb = np.zeros((36, rank_sb))
            for i in range(6):
                for j in range(6):
                    for k in range(6):
                        col = i*36+j*6+k
                        U_sb[i*6+j, col] = 1
                        V_sb[j*6+k, col] = 1
                        W_sb[k*6+i, col] = 1
            return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
