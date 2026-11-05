import numpy as np
import time

def construct():
    n, m, p = 4, 4, 5
    start_time = time.time()
    DEADLINE = 135

    def elapsed():
        return time.time() - start_time

    def verify(U, V, W, seed=12345, trials=10):
        try:
            Ua = np.array(U, dtype=float)
            Va = np.array(V, dtype=float)
            Wa = np.array(W, dtype=float)
            rank = Ua.shape[1]
            rng = np.random.RandomState(seed)
            for _ in range(trials):
                A = rng.rand(n, m)
                B = rng.rand(m, p)
                C_expected = A @ B
                x = (A.reshape(n*m) @ Ua)
                y = (B.reshape(m*p) @ Va)
                prod = x * y
                C_computed = (Wa * prod[None, :]).sum(axis=1).reshape(p, n)
                C_expected_T = C_expected.T
                if not np.allclose(C_computed, C_expected_T, atol=1e-9, rtol=1e-9):
                    return False
            return True
        except Exception:
            return False

    def reduce_rank(U, V, W, max_iter=60):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        rank = Ua.shape[1]
        removed = 0
        for it in range(max_iter):
            if elapsed() > DEADLINE - 15:
                break
            if rank <= 1:
                break
            norms = [(Ua[:, c]**2).sum() + (Va[:, c]**2).sum() + (Wa[:, c]**2).sum()
                     for c in range(rank)]
            order = sorted(range(rank), key=lambda cc: norms[cc])
            found = False
            for c in order[:15]:
                if norms[c] < 1e-14:
                    Ua = np.delete(Ua, c, axis=1)
                    Va = np.delete(Va, c, axis=1)
                    Wa = np.delete(Wa, c, axis=1)
                    rank -= 1
                    removed += 1
                    found = True
                    break
                Uo = np.delete(Ua, c, axis=1)
                Vo = np.delete(Va, c, axis=1)
                Wo = np.delete(Wa, c, axis=1)
                try:
                    if (np.linalg.matrix_rank(Ua, tol=1e-9) == np.linalg.matrix_rank(Uo, tol=1e-9)
                        and np.linalg.matrix_rank(Va, tol=1e-9) == np.linalg.matrix_rank(Vo, tol=1e-9)
                        and np.linalg.matrix_rank(Wa, tol=1e-9) == np.linalg.matrix_rank(Wo, tol=1e-9)):
                        Ua, Va, Wa = Uo, Vo, Wo
                        rank -= 1
                        removed += 1
                        found = True
                        break
                except Exception:
                    continue
            if not found:
                break
        return Ua.tolist(), Va.tolist(), Wa.tolist(), removed

    # Build the decomposition using the Kronecker product structure
    # M_{4,4,5}: C_{ki} = sum_j A_{ij} B_{jk}
    # 
    # Decompose into 4 column-blocks of B: B[:, 0:4] and B[:, 4:5]
    # Part 1: A * B[:, 0:4] (4x4 * 4x4) -> use 2x2x2 Strassen recursively
    # Part 2: A * B[:, 4:5] (4x4 * 4x1) -> schoolbook rank 16
    #
    # For Part 1 (4x4x4), use the fact that:
    # Split A into 2x2 blocks of 2x2: A = [[A00,A01],[A10,A11]]
    # Split B[:,0:4] into 2x2 blocks of 2x2: B1 = [[B00,B01],[B10,B11]]
    # C1 = A * B1 has blocks: C00 = A00*B00+A01*B10, C01=A00*B01+A01*B11, etc.
    # Each 2x2*2x2 uses Strassen (7 mults), 4 output blocks * 7 = 28 mults for 4x4x4
    # Wait, that's not right. Each output 2x2 block needs 2 input 2x2 blocks each.
    # C00 needs A00*B00 and A01*B10 -> 2 Strassen 2x2 products? No.
    # Each product of two 2x2 matrices needs 7 multiplications via Strassen.
    # C1 has 4 2x2 blocks, each is sum of 2 products of 2x2 matrices.
    # So we need 8 Strassen decompositions = 56 multiplications. That's worse than schoolbook 16.
    # 
    # Actually for 4x4x4, the known best is 32 (using Strassen's algorithm).
    # Strassen for 4x4: treat as 2x2 blocks, each 2x2 product via 7 mults.
    # But 4x4*4x4 has 4 output 2x2 blocks, each requiring... 
    # Actually Strassen 4x4: you need to compute specific combinations.
    # Standard result: omega(4x4) <= 32.
    #
    # Let me just use a known 32-term 4x4x4 decomposition.
    # The 4x4x4 Strassen:
    # M1 = (A00+A11) * (B00+B11)
    # M2 = (A10+A11) * B00
    # M3 = A00 * (B01-B11)
    # M4 = A11 * (B10-B00)
    # M5 = (A00+A01) * B11
    # M6 = (A10-A00) * (B00+B01)
    # M7 = (A01-A11) * (B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    # Each Mi is a 2x2*2x2 product -> 7 mults each -> 49 mults.
    # That's worse than schoolbook 16!
    #
    # I'm overcomplicating this. Let me use a direct approach.
    # 
    # The rank of M_{4,4,5} is at most 4*5 = 20? No, that's for 1xn.
    # Rank of M_{n,m,p} <= n*m*p (schoolbook).
    # Known bounds: for M_{4,4,4}, rank <= 32, for M_{4,4,5} rank <= ?
    # 
    # Let me use the "slice" method: 
    # M_{4,4,5} can be seen as 5 slices of M_{4,4,1}
    # Rank of M_{4,4,1} = 16 (schoolbook, and it's actually minimal)
    # So naive bound: 5*16 = 80.
    # But there's cross-term sharing.
    #
    # Better approach: use the tensor as (n*p) x (m*p) ... no.
    #
    # Let me try: rank = n*p + m*p - something? 
    # Actually, C = A*B where A is 4x4, B is 4x5.
    # View C as: for each row i of A (4 choices), multiply by B (4x5) -> 4x5 output.
    # Each row of A times B: that's (1x4) * (4x5) = (1x5), rank 4 per row, total 4*4=16?
    # No: row_i of A is 1x4, times B (4x5) gives 1x5. The "rank" needed is 4 mults.
    # 4 rows * 4 mults = 16. But we need to handle all 5 columns of B.
    # 
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # For fixed i: C[i,:] = A[i,:] * B (row vector times matrix)
    # This needs 4 mults per output element, 5 outputs = 20? No, 4 mults total.
    # Because A[i,j]*B[j,k] for j=0..3 gives 4 products, but they contribute to all k.
    # 
    # The rank-4 decomposition for 1x4 x 4x5:
    # U: for j in 0..3, U[flat(i,j), term_j] ... 
    # Actually rank of (1x4 x 4x5) matmul tensor is 4.
    # M_r = A[i,j_r] * B[j_r, :] -> 1x5 vector
    # C[i,k] = sum_j A[i,j]*B[j,k] = sum_r M_r[k] where M_r = A[i,j_r]*B[j_r,k]
    # So W[i,k,r] = B[j_r,k]... but W must be fixed.
    # 
    # Hmm, the tensor T_{(i,k),j} = B[j,k] is not constant.
    # So this doesn't give a fixed decomposition.
    #
    # Let me just go with schoolbook 80 and rely on reduction.
    # But also try a smarter initial decomposition.
    
    # Approach: build terms that exploit the column structure of B
    # For each j (0..3), A[:,j] (4x1) * B[j,:] (1x5) contributes to C
    # C += outer(A[:,j], B[j,:])
    # This is a rank-1 update in the "outer product" sense, but for tensor rank:
    # We need: C_{ki} += A_{ij} * B_{jk}
    # For fixed j: define u_i = A_{ij}, v_k = B_{jk}
    # Then C_{ki} += u_i * v_k
    # As a tensor term: U[i,j] = 1 for this j, V[j,k] = B[j,k]... 
    # But V must be constant! It can't depend on B.
    # 
    # So for fixed j: we can't make a single term that works for all B.
    # We need: for fixed j, sum over k: U[i_m + j] * V[j_p + k] * W[k_n + i]
    # = A[i,j] * B[j,k] requires W[k_n+i, term] to encode the product structure.
    #
    # The only way is: U[i*m+j, term] = 1, V[j*p+k, term] = 1, W[k*n+i, term] = 1
    # for a specific (i,j,k) triple. That's schoolbook.
    #
    # OR: use Strassen-like tricks where combinations of A entries and B entries
    # are multiplied together.
    #
    # Let me try building a 32-term 4x4x4 decomposition properly.
    # 
    # 4x4x4 Strassen (from "How to multiply matrices faster"):
    # Divide A and B into 2x2 blocks of size 2x2.
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    #
    # Each Mx is a 2x2*2x2 product. Using schoolbook for each: 4 mults each.
    # 7 * 4 = 28 mults for 4x4x4.
    # 
    # Then 4x4x1 part: 4x4*4x1 = schoolbook 16 mults.
    # Total: 28 + 16 = 44. Then reduce.
    
    def build_2x2x2_schoolbook(A_blocks, B_blocks, W_target, base_col, n_block=2):
        """Build schoolbook terms for 2x2x2 within blocks."""
        terms = []
        for ii in range(2):
            for jj in range(2):
                for kk in range(2):
                    terms.append((A_blocks[ii][jj], B_blocks[jj][kk], W_target[ii][kk], base_col + len(terms)))
        return terms

    # I'll build the full decomposition directly using numpy
    rank = 44  # 28 for 4x4x4 + 16 for 4x4x1
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0

    # 4x4x4 part using Strassen at 2x2 block level
    # A block (br, bc): rows 2*br:2*br+2, cols 2*bc:2*bc+2
    # B block (br, bc): rows 2*br:2*br+2, cols 2*bc:2*bc+2
    # C block (br, bc): rows 2*br:2*br+2, cols 2*bc:2*bc+2
    #
    # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
    # M2 = (A21+A22)B11 -> C21 += M2
    # M3 = A11(B12-B22) -> C12 += M3
    # M4 = A22(B21-B11) -> C11 += M4, C21 += M4
    # M5 = (A11+A12)B22 -> C11 -= M5, C12 += M5
    # M6 = (A21-A11)(B11+B12) -> C22 += M6
    # M7 = (A12-A22)(B21+B22) -> C11 += M7
    
    strassen_defs = [
        # (A_block_coeffs, B_block_coeffs, C_block_coeffs)
        # Each is dict: block_idx -> coeff
        # A block: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),      # M1
        ({2:1, 3:1}, {0:1}, {2:1}),                  # M2
        ({0:1}, {1:1, 3:-1}, {1:1}),                 # M3
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),            # M4
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),            # M5
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),            # M6
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),            # M7
    ]

    for a_coeffs, b_coeffs, c_coeffs in strassen_defs:
        # For each pair of contributing A sub-entries and B sub-entries
        # A entry (ar, ac): actual row = 2*ar, col = 2*ac
        # B entry (br, bc): actual row = 2*br, col = 2*bc (but for 4x4x4, col is 0-3)
        # C entry (cr, cc): actual row = 2*cr, col = 2*cc
        a_entries = []
        for blk, coeff in a_coeffs.items():
            ar, ac = blk // 2, blk % 2
            a_entries.append((ar * 2, ac * 2, coeff))
        
        b_entries = []
        for blk, coeff in b_coeffs.items():
            br, bc = blk // 2, blk % 2
            b_entries.append((br * 2, bc * 2, coeff))
        
        c_entries = []
        for blk, coeff in c_coeffs.items():
            cr, cc = blk // 2, blk % 2
            c_entries.append((cr * 2, cc * 2, coeff))
        
        # For each combination, we need schoolbook within the 2x2x2
        # For each a_entry, b_entry pair, and for each 2x2 sub-element
        for (ar, ac, acf) in a_entries:
            for (br, bc, bcf) in b_entries:
                if br != ac:
                    continue  # A col must match B row
                coeff_ab = acf * bcf
                for sub_r in range(2):
                    for sub_c in range(2):
                        i = ar + sub_r
                        j = ac + sub_c
                        # B[j, bc + sub_c] ... wait, B row is j = br + sub_r
                        # B col is bc + sub_c
                        b_row = br + sub_r
                        b_col = bc + sub_c
                        for (cr, cc, ccf) in c_entries:
                            k = cc + sub_c  # C col
                            # C row = cr + sub_r
                            c_row = cr + sub_r
                            # W[k*n + i]
                            w_val = coeff_ab * ccf
                            U[i * m + j, col] += 1  # Will sum properly below
                            V[j * p + b_col, col] += 1
                            W[k * n + i, col] += w_val
        # This approach is getting complicated with the nested sums.
        # Let me just do it more carefully.
        col += 1

    # This is getting too complex. Let me just use schoolbook 80 and reduce.
    # Rebuild from scratch.
    U = np.zeros((n * m, 80))
    V = np.zeros((m * p, 80))
    W = np.zeros((p * n, 80))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * (m * p) + j * p + k
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    ok = verify(U_list, V_list, W_list, seed=1, trials=5)
    best_U, best_V, best_W = U_list, V_list, W_list

    if ok and elapsed() < DEADLINE - 25:
        Ur, Vr, Wr, rem = reduce_rank(U_list, V_list, W_list, max_iter=80)
        if rem > 0 and verify(Ur, Vr, Wr, seed=77, trials=8):
            best_U, best_V, best_W = Ur, Vr, Wr
            if elapsed() < DEADLINE - 12:
                Ur2, Vr2, Wr2, rem2 = reduce_rank(best_U, best_V, best_W, max_iter=50)
                if rem2 > 0 and verify(Ur2, Vr2, Wr2, seed=111, trials=8):
                    best_U, best_V, best_W = Ur2, Vr2, Wr2

    return {"U": best_U, "V": best_V, "W": best_W}
