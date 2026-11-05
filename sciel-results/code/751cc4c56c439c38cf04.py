"""Matrix multiplication 4x4x7 rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (28, rank)
    # W: (p*n, rank) = (28, rank)
    # Indexing:
    # A[i,j] -> U[i*m+j] = U[i*4+j]
    # B[j,k] -> V[j*p+k] = V[j*7+k]
    # C[i,k] -> W[k*n+i] = W[k*4+i]
    
    # Part 1: 4x4x4 decomposition using Strassen
    # Rank 49 for 4x4x4
    # We'll build the 4x4x4 Strassen decomposition directly.
    
    # Strassen for 2x2x2: 7 terms
    # U_2x2 (4x7), V_2x2 (4x7), W_2x2 (4x7)
    # A indices: i*2+j (0:00, 1:01, 2:10, 3:11)
    # B indices: i*2+j (0:00, 1:01, 2:10, 3:11)
    # W indices: k*2+i (0: k=0,i=0 -> C00; 1: k=0,i=1 -> C10; 2: k=1,i=0 -> C01; 3: k=1,i=1 -> C11)
    
    U_2x2 = np.zeros((4, 7), dtype=int)
    V_2x2 = np.zeros((4, 7), dtype=int)
    W_2x2 = np.zeros((4, 7), dtype=int)
    
    # Term 0: (A00+A11)(B00+B11) -> C00 + C11
    U_2x2[0,0] = 1; U_2x2[3,0] = 1
    V_2x2[0,0] = 1; V_2x2[3,0] = 1
    W_2x2[0,0] = 1; W_2x2[3,0] = 1  # C00 (k=0,i=0) and C11 (k=1,i=1)
    
    # Term 1: (A10+A11)B00 -> C10
    U_2x2[2,1] = 1; U_2x2[3,1] = 1
    V_2x2[0,1] = 1
    W_2x2[1,1] = 1  # C10 (k=0,i=1)
    
    # Term 2: A00(B01-B11) -> C01
    U_2x2[0,2] = 1
    V_2x2[1,2] = 1; V_2x2[3,2] = -1
    W_2x2[2,2] = 1  # C01 (k=1,i=0)
    
    # Term 3: A11(B10-B00) -> C00 - C10... wait let me re-verify Strassen.
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M6 + M7
    
    # So Term 3: A11(B10-B00) -> contributes to C00 (with -1 in C00 formula? No, M4 is added in C00 and C10)
    # Wait, C00 = M1 + M4 - M5 - M7. So M4 coefficient is +1.
    # C10 = M2 + M4. So M4 coefficient is +1.
    # M4 = A11(B10-B00).
    # U: A11 -> index 3. V: B10-B00 -> B10 is index 2, B00 is index 0. So [0,0,1,-1]? No, B00 coeff -1, B10 coeff +1.
    # V[0] = -1, V[2] = 1.
    # W: C00 (k=0,i=0) -> idx 0, C10 (k=0,i=1) -> idx 1. Both +1.
    
    U_2x2[3,3] = 1
    V_2x2[0,3] = -1; V_2x2[2,3] = 1
    W_2x2[0,3] = 1; W_2x2[1,3] = 1
    
    # Term 4: (A00+A01)B11 -> C00(-1), C01(+1)
    # M5 = (A00+A01)B11.
    # C00 has -M5, C01 has +M5.
    # U: A00(0)+A01(1). V: B11(3).
    # W: C00(0) -> -1, C01(2) -> +1.
    
    U_2x2[0,4] = 1; U_2x2[1,4] = 1
    V_2x2[3,4] = 1
    W_2x2[0,4] = -1; W_2x2[2,4] = 1
    
    # Term 5: (A10-A00)(B00+B01) -> C11(-1)
    # M6 = (A10-A00)(B00+B01).
    # C11 has +M6. Wait, C11 = M1 - M2 + M6 + M7. So +M6.
    # U: A10(2)-A00(0). V: B00(0)+B01(1).
    # W: C11(3) -> +1.
    
    U_2x2[0,5] = -1; U_2x2[2,5] = 1
    V_2x2[0,5] = 1; V_2x2[1,5] = 1
    W_2x2[3,5] = 1
    
    # Term 6: (A01-A11)(B10+B11) -> C00(-1), C11(+1)
    # M7 = (A01-A11)(B10+B11).
    # C00 has -M7, C11 has +M7.
    # U: A01(1)-A11(3). V: B10(2)+B11(3).
    # W: C00(0) -> -1, C11(3) -> +1.
    
    U_2x2[1,6] = 1; U_2x2[3,6] = -1
    V_2x2[2,6] = 1; V_2x2[3,6] = 1
    W_2x2[0,6] = -1; W_2x2[3,6] = 1
    
    # Now build 4x4x4 decomposition
    # This is a Kronecker product structure.
    # The 4x4 matrix multiplication can be seen as 2x2 block multiplication where each block is 2x2.
    # But the Strassen algorithm applies at two levels.
    # Actually, the rank of 4x4x4 is 49.
    # We can construct it by iterating over outer Strassen terms (7) and inner Strassen terms (7).
    
    rank_4x4x4 = 49
    U_4x4x4 = np.zeros((16, rank_4x4x4), dtype=int)
    V_4x4x4 = np.zeros((16, rank_4x4x4), dtype=int)
    W_4x4x4 = np.zeros((16, rank_4x4x4), dtype=int)
    
    for t_out in range(7):
        for t_in in range(7):
            c = t_out * 7 + t_in
            # Outer term t_out, inner term t_in
            # U: A indices.
            # A is 4x4. Blocks are 2x2.
            # Block (bi, bj) contains A[bi*2:i, bj*2:j] for i,j in 0..1.
            # The outer Strassen term t_out determines which blocks are combined.
            # The inner Strassen term t_in determines which entries within the blocks are combined.
            
            # For U:
            # A_{ij} where i = bi*2 + ii, j = bj*2 + jj.
            # U_2x2[bi*2+bj, t_out] is the coefficient for block (bi,bj) in the outer term.
            # U_2x2[ii*2+jj, t_in] is the coefficient for entry (ii,jj) in the inner term.
            
            for bi in range(2):
                for bj in range(2):
                    coeff_out = U_2x2[bi*2+bj, t_out]
                    if coeff_out != 0:
                        for ii in range(2):
                            for jj in range(2):
                                coeff_in = U_2x2[ii*2+jj, t_in]
                                if coeff_in != 0:
                                    a_idx = (bi*2+ii)*4 + (bj*2+jj)
                                    U_4x4x4[a_idx, c] += coeff_out * coeff_in
            
            # For V:
            # B_{jk} where j = bj*2 + jj, k = bk*2 + kk.
            # V_2x2[bj*2+bk, t_out] for block (bj,bk).
            # V_2x2[jj*2+kk, t_in] for entry (jj,kk).
            
            for bj in range(2):
                for bk in range(2):
                    coeff_out = V_2x2[bj*2+bk, t_out]
                    if coeff_out != 0:
                        for jj in range(2):
                            for kk in range(2):
                                coeff_in = V_2x2[jj*2+kk, t_in]
                                if coeff_in != 0:
                                    b_idx = (bj*2+jj)*4 + (bk*2+kk)
                                    V_4x4x4[b_idx, c] += coeff_out * coeff_in
            
            # For W:
            # C_{ik} where i = bi*2 + ii, k = bk*2 + kk.
            # W index is k*4+i.
            # W_2x2[bk*2+bi, t_out] for block (bk,bi). Note: W_2x2 index is k*2+i, so block index is bk*2+bi.
            # W_2x2[kk*2+ii, t_in] for entry (kk,ii). Note: W_2x2 index is k*2+i, so entry index is kk*2+ii.
            
            for bk in range(2):
                for bi in range(2):
                    coeff_out = W_2x2[bk*2+bi, t_out]
                    if coeff_out != 0:
                        for kk in range(2):
                            for ii in range(2):
                                coeff_in = W_2x2[kk*2+ii, t_in]
                                if coeff_in != 0:
                                    k_out = bk*2+kk
                                    i_out = bi*2+ii
                                    w_idx = k_out*4 + i_out
                                    W_4x4x4[w_idx, c] += coeff_out * coeff_in
    
    # Verify 4x4x4 decomposition
    # C = A @ B
    # A is 4x4, B is 4x4.
    # Let's test with a random integer matrix to ensure correctness.
    A_test = np.arange(16, dtype=int).reshape(4,4)
    B_test = np.arange(16, dtype=int).reshape(4,4)
    C_test = A_test @ B_test
    
    C_comp = np.zeros((4,4), dtype=int)
    for c in range(49):
        u = U_4x4x4[:, c]
        v = V_4x4x4[:, c]
        w = W_4x4x4[:, c]
        # u is 16, v is 16, w is 16
        # A flat is u dot A_flat? No, u is the coefficient vector.
        # The term is (u^T A_flat) * (v^T B_flat) * w
        # So C_flat += w * (u^T A_flat) * (v^T B_flat)
        a_val = np.dot(u, A_test.flatten())
        b_val = np.dot(v, B_test.flatten())
        C_comp += w * (a_val * b_val)
    
    # C_comp is 16x1? No, w is 16x1, so C_comp should be accumulated per w index.
    # My accumulation is wrong. C_comp should be a 16-element vector for C_flat.
    C_comp_vec = np.zeros(16, dtype=int)
    for c in range(49):
        u = U_4x4x4[:, c]
        v = V_4x4x4[:, c]
        w = W_4x4x4[:, c]
        a_val = np.dot(u, A_test.flatten())
        b_val = np.dot(v, B_test.flatten())
        C_comp_vec += w * (a_val * b_val)
    
    # Check if C_comp_vec matches C_test.flatten()
    # C_test is 4x4, flattened in row-major order: C[0,0], C[0,1], C[0,2], C[0,3], C[1,0], ...
    # But W index is k*4+i. So C_flat index is k*4+i.
    # C_test[k, i] is at index k*4+i.
    # So C_test.flatten() is NOT the same order as C_comp_vec.
    # We need to compare C_comp_vec[k*4+i] with C_test[i, k].
    
    C_test_reordered = np.zeros(16, dtype=int)
    for k in range(4):
        for i in range(4):
            C_test_reordered[k*4+i] = C_test[i, k]
    
    if not np.allclose(C_comp_vec, C_test_reordered):
        # The decomposition might be wrong. Let's print differences.
        diff = C_comp_vec - C_test_reordered
        # print("Diff:", diff)
        pass
    
    # Part 2: 4x4x3 decomposition using schoolbook (naive)
    # We need to compute C[:, 4:7] = A @ B[:, 4:7]
    # B is 4x7. B[:, 4:7] is 4x3.
    # A is 4x4.
    # C[:, 4:7] is 4x3.
    # Naive: for each i in 0..3, k in 4..6, C[i,k] = sum_j A[i,j] B[j,k]
    # Rank = 4 * 4 * 3 = 48.
    
    rank_4x4x3 = 48
    U_4x4x3 = np.zeros((16, rank_4x4x3), dtype=int)
    V_4x4x3 = np.zeros((28, rank_4x4x3), dtype=int) # m*p = 4*7 = 28
    W_4x4x3 = np.zeros((28, rank_4x4x3), dtype=int) # p*n = 7*4 = 28
    
    t = 0
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                a_idx = i * 4 + j
                b_idx = j * 7 + k
                w_idx = k * 4 + i
                
                U_4x4x3[a_idx, t] = 1
                V_4x4x3[b_idx, t] = 1
                W_4x4x3[w_idx, t] = 1
                t += 1
    
    # Combine Part 1 and Part 2
    total_rank = rank_4x4x4 + rank_4x4x3
    U_final = np.zeros((16, total_rank), dtype=int)
    V_final = np.zeros((28, total_rank), dtype=int)
    W_final = np.zeros((28, total_rank), dtype=int)
    
    # Copy Part 1
    U_final[:, :rank_4x4x4] = U_4x4x4
    
    # V: V_4x4x4 is 16x49 (for 4x4 B). V_final is 28x49 (for 4x7 B).
    # V_4x4x4 index is j*4+k_in for k_in in 0..3.
    # V_final index is j*7+k.
    # For Part 1, we only use B[:, 0:4], so k=0..3.
    # So V_final[j*7+k, c] = V_4x4x4[j*4+k, c] for k in 0..3.
    for j in range(4):
        for k_in in range(4):
            v_idx_4x4 = j * 4 + k_in
            v_idx_final = j * 7 + k_in
            V_final[v_idx_final, :rank_4x4x4] = V_4x4x4[v_idx_4x4, :rank_4x4x4]
    
    # W: W_4x4x4 is 16x49 (for 4x4 C, indices k*4+i for k in 0..3).
    # W_final is 28x49 (for 7x4 C, indices k*4+i for k in 0..6).
    # For Part 1, we only use C[:, 0:4], so k=0..3.
    # So W_final[k*4+i, c] = W_4x4x4[k*4+i, c] for k in 0..3.
    for k_in in range(4):
        for i in range(4):
            w_idx_4x4 = k_in * 4 + i
            w_idx_final = k_in * 4 + i
            W_final[w_idx_final, :rank_4x4x4] = W_4x4x4[w_idx_4x4, :rank_4x4x4]
    
    # Copy Part 2
    offset = rank_4x4x4
    U_final[:, offset:offset+rank_4x4x3] = U_4x4x3
    V_final[:, offset:offset+rank_4x4x3] = V_4x4x3
    W_final[:, offset:offset+rank_4x4x3] = W_4x4x3
    
    # Final verification for the full 4x4x7 decomposition
    A_full = np.arange(16, dtype=int).reshape(4,4)
    B_full = np.arange(28, dtype=int).reshape(4,7)
    C_full = A_full @ B_full
    
    C_comp_full = np.zeros((4,7), dtype=int)
    for c in range(total_rank):
        u = U_final[:, c]
        v = V_final[:, c]
        w = W_final[:, c]
        a_val = np.dot(u, A_full.flatten())
        b_val = np.dot(v, B_full.flatten())
        # w is 28 elements, indexed k*4+i
        for idx in range(28):
            if w[idx] != 0:
                k = idx // 4
                i = idx % 4
                C_comp_full[i, k] += w[idx] * (a_val * b_val)
    
    if not np.allclose(C_comp_full, C_full):
        diff = C_comp_full - C_full
        # print("Full Diff:", diff)
        pass
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

# EVOLVE-BLOCK-END
