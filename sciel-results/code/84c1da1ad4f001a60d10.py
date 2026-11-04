import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]
DEADLINE = 130.0

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36

    def check_time():
        return time.time() - t0 < DEADLINE

    def validate_fast(U, V, W, r):
        """Fast validation using numpy tensor contraction."""
        U_arr = np.array(U, dtype=np.int64)
        V_arr = np.array(V, dtype=np.int64)
        W_arr = np.array(W, dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    w_idx = k * n + i
                    s = np.sum(U_arr[a] * V_arr[b] * W_arr[w_idx])
                    if s != 1:
                        return False
        return True

    # Build the 4x4x4 Strassen decomposition (rank 49)
    # A is 4x4, B is 4x4, C is 4x4
    # We treat A, B, C as 2x2 blocks of 2x2 matrices
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = AB where C_ij = sum_k A_ik * B_kj
    # C is also 2x2 blocks: C = [[C00, C01], [C10, C11]]
    
    # The rank of 4x4x4 Strassen is 7 * 7 = 49
    # We need to build the terms for this decomposition
    
    def build_4x4x4_strassen_terms():
        """Build the 4x4x4 Strassen decomposition terms."""
        terms = []
        
        # Strassen formula for 2x2 block matrix multiplication
        # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
        # Each is 2x2 blocks of 2x2 matrices
        # C = AB where C_ij = sum_k A_ik * B_kj
        
        # The 7 Strassen products for the block level:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        # For each of the 7 outer Strassen products, we need to compute
        # a 2x2 * 2x2 multiplication, which itself has 7 terms (using Strassen).
        
        # Strassen for 2x2:
        # P1 = (X00+X11)(Y00+Y11)
        # P2 = (X10+X11)Y00
        # P3 = X00(Y01-Y11)
        # P4 = X11(Y10-Y00)
        # P5 = (X00+X01)Y11
        # P6 = (X10-X00)(Y00+Y01)
        # P7 = (X01-X11)(Y10+Y11)
        # Z00 = P1+P4-P5+P7
        # Z01 = P3+P5
        # Z10 = P2+P4
        # Z11 = P1-P2+P3+P6
        
        # For each of the 7 outer Strassen products, we need to compute
        # a 2x2 * 2x2 multiplication, which itself has 7 terms.
        # But we need to track the coefficients carefully.
        
        # Let's define the outer Strassen products with their A-block and B-block coefficients
        # A block: (ar, ac) where ar in {0,1} is block row, ac in {0,1} is block col
        # B block: (br, bc) where br in {0,1} is block row, bc in {0,1} is block col
        
        # Outer M1: A blocks (0,0) and (1,1), B blocks (0,0) and (1,1)
        # M1 = (A00+A11)(B00+B11)
        # This is a 2x2*2x2 multiplication where:
        # X = A00+A11 (2x2), Y = B00+B11 (2x2)
        # The result contributes to C00 and C11
        
        # Let me enumerate all terms systematically.
        
        # For the outer Strassen, each product term has:
        # - A coefficient vector over the 4 A blocks: [coeff for A00, A01, A10, A11]
        # - B coefficient vector over the 4 B blocks: [coeff for B00, B01, B10, B11]
        # - C output blocks it contributes to with signs
        
        outer_terms = [
            # (a_coeffs, b_coeffs, c_block_coeffs)
            # a_coeffs: [A00, A01, A10, A11]
            # b_coeffs: [B00, B01, B10, B11]
            # c_block_coeffs: [C00, C01, C10, C11]
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),    # M1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),     # M2
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),    # M3
            ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 0, 0]),    # M4
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 1]),    # M5
            ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),    # M6
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),    # M7
        ]
        
        # For each outer product, compute the inner 2x2*2x2 Strassen terms
        inner_terms = [
            # (x_coeffs, y_coeffs, z_coeffs)
            # x_coeffs: [X00, X01, X10, X11]
            # y_coeffs: [Y00, Y01, Y10, Y11]
            # z_coeffs: [Z00, Z01, Z10, Z11]
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),    # P1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),     # P2
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),    # P3
            ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 0, 0]),    # P4
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 1]),    # P5
            ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),    # P6
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),    # P7
        ]
        
        # Now build all terms
        # A global index: a_row * 4 + a_col where a_row = 2*ab_row + ab_r, a_col = 2*ab_col + ab_c
        # B global index: b_row * 4 + b_col where b_row = 2*bb_row + bb_r, b_col = 2*bb_col + bb_c
        # C global index: c_row * 4 + c_col where c_row = 2*cb_row + cb_r, c_col = 2*cb_col + cb_c
        
        term_dict = {}
        
        for oi, (a_coeffs, b_coeffs, c_block_coeffs) in enumerate(outer_terms):
            for ii, (x_coeffs, y_coeffs, z_coeffs) in enumerate(inner_terms):
                # For each A block with nonzero outer coeff
                for ab_idx, a_c in enumerate(a_coeffs):
                    if a_c == 0:
                        continue
                    ab_row, ab_col = divmod(ab_idx, 2)
                    for x_idx, x_c in enumerate(x_coeffs):
                        if x_c == 0:
                            continue
                        x_row, x_col = divmod(x_idx, 2)
                        # A element: row = 2*ab_row + x_row, col = 2*ab_col + x_col
                        a_row = 2 * ab_row + x_row
                        a_col = 2 * ab_col + x_col
                        a_global = a_row * 4 + a_col
                        
                        for bb_idx, b_c in enumerate(b_coeffs):
                            if b_c == 0:
                                continue
                            bb_row, bb_col = divmod(bb_idx, 2)
                            for y_idx, y_c in enumerate(y_coeffs):
                                if y_c == 0:
                                    continue
                                y_row, y_col = divmod(y_idx, 2)
                                # B element: row = 2*bb_row + y_row, col = 2*bb_col + y_col
                                b_row = 2 * bb_row + y_row
                                b_col = 2 * bb_col + y_col
                                b_global = b_row * 4 + b_col
                                
                                for cb_idx, cb_c in enumerate(c_block_coeffs):
                                    if cb_c == 0:
                                        continue
                                    cb_row, cb_col = divmod(cb_idx, 2)
                                    for z_idx, z_c in enumerate(z_coeffs):
                                        if z_c == 0:
                                            continue
                                        z_row, z_col = divmod(z_idx, 2)
                                        # C element: row = 2*cb_row + z_row, col = 2*cb_col + z_col
                                        c_row = 2 * cb_row + z_row
                                        c_col = 2 * cb_col + z_col
                                        c_global = c_row * 4 + c_col
                                        
                                        coeff = a_c * b_c * cb_c * x_c * y_c * z_c
                                        if coeff != 0:
                                            key = (a_global, b_global, c_global)
                                            if key in term_dict:
                                                term_dict[key] += coeff
                                            else:
                                                term_dict[key] = coeff
        
        # Remove zero-coefficient terms
        terms = [(a, b, c, coeff) for (a, b, c), coeff in term_dict.items() if coeff != 0]
        return terms
    
    # Build the 4x4x4 Strassen terms
    terms_4x4 = build_4x4x4_strassen_terms()
    print(f"4x4x4 Strassen terms: {len(terms_4x4)}")
    
    # For 4x4x9, we split B into three parts:
    # Part 1: B[:, 0:4] (first 4 columns) -> 4x4x4 Strassen (rank 49)
    # Part 2: B[:, 4:8] (next 4 columns) -> 4x4x4 Strassen (rank 49)
    # Part 3: B[:, 8:9] (last 1 column) -> 4x4x1 Schoolbook (rank 16)
    
    all_terms = []
    
    # Part 1: First 4 columns
    for (a, b, c, coeff) in terms_4x4:
        # b is in 0..15, representing B[j*4+k] where j=b//4, k=b%4
        # For the first 4 columns, b_new = b (no change)
        b_new = b
        all_terms.append((a, b_new, c, coeff))
    
    # Part 2: Next 4 columns (offset by 4 in B and C)
    for (a, b, c, coeff) in terms_4x4:
        # b is in 0..15, representing B[j*4+k] where j=b//4, k=b%4
        # For columns 4-8, we want B[j*9+(4+k)] = B[j*9+4+k]
        # In the original, B[j*4+k], so we map k -> 4+k
        # Actually, let's think more carefully.
        # B is 4x9. For the second block, we use columns 4-8 (indices 4,5,6,7).
        # In the 4x4x4 Strassen, B is 4x4 with indices 0-3.
        # We map B[j*4+k] -> B[j*9+(4+k)] for k=0..3.
        # So if b_orig = j*4+k, then b_new = j*9 + (4+k) = j*9 + 4 + k.
        # Since j = b_orig // 4 and k = b_orig % 4:
        # b_new = (b_orig // 4) * 9 + 4 + (b_orig % 4)
        j = b // 4
        k = b % 4
        b_new = j * 9 + (4 + k)
        
        # For C: c is in 0..15, representing C[c_row*4+c_col]
        # For the second block, we want C[(c_row)*9 + (4+c_col)]? No.
        # Wait, C is 9x4 (p*n = 9*4 = 36). The trace convention is W[k*n+i,r].
        # So C index = k*n + i where k is column of B (and row of C), i is row of A (and col of C).
        # For the first 4 columns: k in 0..3, i in 0..3, so C index = k*4 + i.
        # For the second 4 columns: k in 4..7, i in 0..3, so C index = k*4 + i.
        # But in the 4x4x4 Strassen, C is 4x4 with index = k' * 4 + i' where k'=0..3, i'=0..3.
        # We need to map k' -> k'+4 for the second block.
        # So c_orig = k'*4 + i', and c_new = (k'+4)*4 + i' = k'*4 + i' + 16 = c_orig + 16.
        c_new = c + 16
        all_terms.append((a, b_new, c_new, coeff))
    
    # Part 3: Last 1 column (index 8)
    # Schoolbook for 4x4 * 1x1 -> 1x1, repeated for each (i,j)
    # A is 4x4, B is 4x1, C is 1x4
    # For each i in 0..3, j in 0..3:
    #   C[8*4+i] += A[i*4+j] * B[j*9+8]
    # So terms are (a=i*4+j, b=j*9+8, c=8*4+i, coeff=1)
    for i in range(n):
        for j in range(m):
            a_global = i * m + j
            b_global = j * p + 8  # j*9 + 8
            c_global = 8 * n + i  # 8*4 + i
            all_terms.append((a_global, b_global, c_global, 1))
    
    r_best = len(all_terms)
    print(f"Initial rank: {r_best}")
    
    # Build U, V, W
    U = np.zeros((nm, r_best), dtype=np.int64)
    V = np.zeros((mp, r_best), dtype=np.int64)
    W = np.zeros((pn, r_best), dtype=np.int64)
    
    for c, (a, b, cg, coeff) in enumerate(all_terms):
        U[a, c] = coeff
        V[b, c] = 1
        W[cg, c] = 1
    
    # Validate
    if not validate_fast(U.tolist(), V.tolist(), W.tolist(), r_best):
        # Fallback: schoolbook
        r_fb = n * m * p
        U = np.zeros((nm, r_fb), dtype=np.int64)
        V = np.zeros((mp, r_fb), dtype=np.int64)
        W = np.zeros((pn, r_fb), dtype=np.int64)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U = U.tolist()
    best_V = V.tolist()
    best_W = W.tolist()
    best_rank = r_best
    
    # Try to merge terms to reduce rank
    # Check if any two terms can be merged (same a, b but different c, or same a, c but different b, etc.)
    # Actually, terms are (a, b, c, coeff). Two terms can be combined if they share the same
    # (a, b, c) but that's already handled. 
    # The real merging opportunity: if we have two terms where one has U[a1]=u1, U[a2]=u2
    # and we can find a linear combination... This is complex.
    
    # Alternative: try random perturbations and see if rank can be reduced
    if check_time():
        # Try to find rank reductions by checking if any term is linearly dependent
        # on others. This is a linear algebra problem over the reals.
        # For each term c, check if removing it still allows the equations to be satisfied
        # by adjusting other terms. This is hard in general.
        
        # Simpler: try small random modifications
        random.seed(123)
        for attempt in range(200):
            if not check_time():
                break
            if best_rank <= 50:
                break
            
            # Try removing one term and see if we can compensate
            # This is too complex. Instead, try a different approach:
            # Use a different Strassen variant or Winograd
            
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
