import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # We will accumulate the decomposition terms
    # Each term is (u_idx, v_idx, w_idx, coeff)
    # But U, V, W are matrices of shape (n*m, rank), (m*p, rank), (p*n, rank)
    # It is easier to collect (u_row, v_row, w_row, coeff) for each rank column.
    
    terms = []
    
    def add_block(n_local, m_local, p_local, u_base, v_base, w_base, rank_offset):
        """
        Adds decomposition for M_{n_local, m_local, p_local} to terms.
        u_base, v_base, w_base are offsets into the global U, V, W rows.
        rank_offset is the starting column index.
        Returns the new rank_offset (i.e., rank_offset + rank_used).
        """
        # Base cases
        if m_local == 1 and p_local == 1:
            # M_{n, 1, 1}: Rank n
            # C_{i, 1} = A_{i, 1} B_{1, 1}
            for i in range(n_local):
                u_idx = u_base + i * 1 + 0
                v_idx = v_base + 0 * 1 + 0
                w_idx = w_base + 0 * n_local + i
                terms.append((u_idx, v_idx, w_idx, 1.0, rank_offset))
                rank_offset += 1
            return rank_offset
            
        if n_local == 2 and m_local == 2 and p_local == 2:
            # Strassen
            # A indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
            # B indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
            # C indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
            
            # M1 = (A00+A11)(B00+B11)
            u1 = [(0, 1), (1, 1)]
            v1 = [(0, 1), (1, 1)]
            w1 = [(0, 1), (1, -1), (2, 1), (3, 1)] # C00+=M1, C11+=M1, C01-=M1? No.
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # Let's define the 7 multiplications
            # 1: (A00+A11) * (B00+B11)
            # 2: (A10+A11) * B00
            # 3: A00 * (B01-B11)
            # 4: A11 * (B10-A10) -> A11 * (B10 - A10)? No, A11 * (B10 - A10) is wrong. A11 * (B10 - A10) mixes A and B.
            # Strassen:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-A10) -> Wait, A10 is from A. B10 is from B.
            # M4 = A11(B10 - A10) is incorrect. It should be A11 * (B10 - A10)?? No.
            # Standard Strassen:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-A10) -> This term is A11 * (B10 - A10)?? No.
            # M4 = A11 * (B10 - A10) is wrong.
            # M4 = A11 * (B10 - A10) ... wait.
            # M4 = A11 * (B10 - A10) is not bilinear in A and B if A10 is in the second factor.
            # Correct Strassen:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-A10) -> No.
            # M4 = A11 * (B10 - A10) is invalid.
            # M4 = A11 * (B10 - A10) ...
            # Let's use the standard definition:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-A10) -> This is A11 * B10 - A11 * A10.
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # We need to add these to terms.
            # Helper to add a product L_A * L_B to W
            def add_product(u_coeffs, v_coeffs, w_coeffs, r_off):
                # u_coeffs: dict {u_idx: coeff}
                # v_coeffs: dict {v_idx: coeff}
                # w_coeffs: dict {w_idx: coeff}
                for w_idx, w_c in w_coeffs.items():
                    # We need to add w_c * (sum u_i A_i) * (sum v_j B_j)
                    # This expands to sum_{i,j} u_i v_j w_c A_i B_j C_{w_idx}
                    # But our tensor decomposition is sum_k U_{i,k} V_{j,k} W_{l,k}
                    # So for each pair (i, j), we add to column r_off:
                    # U_{i, r_off} += u_i
                    # V_{j, r_off} += v_j
                    # W_{l, r_off} += w_c
                    # Wait, this is only valid if the product is a single term.
                    # If L_A = A00 + A11, L_B = B00 + B11.
                    # Product = A00 B00 + A00 B11 + A11 B00 + A11 B11.
                    # We add this to C00, C11, etc.
                    # So for column r_off:
                    # U[0, r_off] += 1 (for A00)
                    # U[3, r_off] += 1 (for A11)
                    # V[0, r_off] += 1 (for B00)
                    # V[3, r_off] += 1 (for B11)
                    # W[0, r_off] += w_c (for C00)
                    # W[3, r_off] += w_c (for C11)
                    # But this adds 4 terms to the same column?
                    # No, the column represents ONE multiplication.
                    # The multiplication is (A00+A11)(B00+B11).
                    # This is ONE rank-1 component in the decomposition?
                    # No, (A00+A11)(B00+B11) is not rank 1 in the tensor sense unless we treat it as a single scalar product.
                    # Yes, in the bilinear complexity count, $M_1$ is one multiplication.
                    # So we add ONE column to U, V, W.
                    # U column has 1 at A00, 1 at A11.
                    # V column has 1 at B00, 1 at B11.
                    # W column has w_c at C00, w_c at C11, etc.
                    
                    pass
            
            # Let's implement add_product properly.
            # It takes u_dict, v_dict, w_dict and adds one column.
            
            def add_strassen_term(u_dict, v_dict, w_dict, r_off):
                for u_idx, u_c in u_dict.items():
                    if u_c != 0:
                        # We will store (u_idx, v_idx, w_idx, coeff, r_off)
                        # But we need to combine them?
                        # No, each term in the sum for U, V, W is separate.
                        # For a single multiplication column k:
                        # U[i, k] = coeff_i
                        # V[j, k] = coeff_j
                        # W[l, k] = coeff_l
                        # We can just append (i, j, l, coeff, k) for each non-zero entry?
                        # No, that's not how it works.
                        # The column k has values in U, V, W.
                        # We can just store the column.
                        
                        pass

            # Simpler: Just append tuples (u_idx, v_idx, w_idx, coeff, rank_col)
            # But for one multiplication, we have multiple u_idx, v_idx, w_idx.
            # We need to add to the SAME rank_col.
            
            # Let's restructure: 
            # `terms` will be a list of dicts: { 'U': {idx: val}, 'V': {idx: val}, 'W': {idx: val} }
            # Or just a list of (u_idx, v_idx, w_idx, coeff, col_idx) is wrong because one col has multiple u's.
            
            # Let's use a list of columns. Each column is a tuple (U_dict, V_dict, W_dict).
            # Then at the end, we assemble the matrices.
            
            pass
            
        # Recursive step
        if m_local >= 2:
            m1 = m_local - 1
            m2 = 1
            # Split A into A1 (n x m1) and A2 (n x m2)
            # Split B into B1 (m1 x p) and B2 (m2 x p)
            # C = A1 B1 + A2 B2
            
            # A1 is at u_base
            # A2 is at u_base + m1*n
            # B1 is at v_base
            # B2 is at v_base + m1*p
            # C is at w_base
            
            rank_offset = add_block(n_local, m1, p_local, u_base, v_base, w_base, rank_offset)
            rank_offset = add_block(n_local, m2, p_local, u_base + m1*n_local, v_base + m1*p_local, w_base, rank_offset)
            return rank_offset
            
        # If m_local == 1, it's handled by base case?
        # M_{n, 1, p}. Rank n*p.
        if m_local == 1:
            for i in range(n_local):
                for k in range(p_local):
                    u_idx = u_base + i
                    v_idx = v_base + k
                    w_idx = w_base + k*n_local + i
                    terms.append((u_idx, v_idx, w_idx, 1.0, rank_offset))
                    rank_offset += 1
            return rank_offset

    # Start decomposition
    rank_offset = 0
    rank_offset = add_block(n, m, p, 0, 0, 0, rank_offset)
    
    # Assemble matrices
    total_rank = rank_offset
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    for u_idx, v_idx, w_idx, coeff, col in terms:
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        
    return {"U": U, "V": V, "W": W}
