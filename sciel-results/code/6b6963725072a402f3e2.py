"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is n*m*p = 96
    # We aim for rank 81 (49 for 4x4x4 + 32 for 4x4x2)
    
    def get_strassen_2_terms():
        # Returns list of 7 tuples (a_coeffs, b_coeffs, c_coeffs)
        # a_coeffs, b_coeffs, c_coeffs are lists of 4 ints
        # Indices: 00, 01, 10, 11
        terms = []
        # M1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
        terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        # M2 = (A10+A11)B00 -> C10+=1
        terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))
        # M3 = A00(B01-B11) -> C01+=1, C11+=1
        terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
        # M4 = A11(B10-B00) -> C00+=1, C10+=1
        terms.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
        # M5 = (A00+A01)B11 -> C00-=1, C01+=1
        terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
        # M6 = (A10-A00)(B00+B01) -> C11+=1
        terms.append([[-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]])
        # M7 = (A01-A11)(B10+B11) -> C01+=1
        terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
        return terms

    def expand_4x4x4():
        terms_2 = get_strassen_2_terms()
        # We will build a list of terms for 4x4x4
        # Each term: (A_coeffs [16], B_coeffs [16], C_coeffs [16])
        raw_terms = []
        
        for s in range(7):
            top_a, top_b, top_c = terms_2[s]
            for t in range(7):
                bot_a, bot_b, bot_c = terms_2[t]
                
                A_coeffs = [0] * 16
                B_coeffs = [0] * 16
                C_coeffs = [0] * 16
                
                # Calculate A and B coefficients for the scalar product
                # X_pq = sum_ij top_a[ij] * (A_ij)_pq
                # A_index = (2*i + p) * 4 + (2*j + q) ? No, A is 4x4.
                # A_ij is block at row i, col j.
                # Element (p,q) of A_ij is at global row 2*i+p, col 2*j+q.
                # Index = (2*i+p)*4 + (2*j+q)
                
                # The scalar product is (sum_pq bot_a[pq] X_pq) * (sum_rs bot_b[rs] Y_rs)
                # So the total coeff for A_elem is sum over all i,j,p,q such that 
                # the term bot_a[pq] * top_a[ij] applies to the same A element?
                # No.
                # X_pq is a specific element.
                # X_pq = sum_{i,j} top_a[ij] * A[2*i+p, 2*j+q]
                # So the coefficient of A[2*i+p, 2*j+q] in X_pq is top_a[ij].
                # The term in the product is bot_a[pq] * X_pq.
                # So the coefficient of A[2*i+p, 2*j+q] in the first factor is bot_a[pq] * top_a[ij].
                
                for i in range(2):
                    for j in range(2):
                        for p in range(2):
                            for q in range(2):
                                idx = (2 * i + p) * 4 + (2 * j + q)
                                A_coeffs[idx] += top_a[i * 2 + j] * bot_a[p * 2 + q]
                                
                for k in range(2):
                    for l in range(2):
                        for r in range(2):
                            for s_idx in range(2):
                                idx = (2 * k + r) * 4 + (2 * l + s_idx)
                                B_coeffs[idx] += top_b[k * 2 + l] * bot_b[r * 2 + s_idx]
                
                # Calculate C coefficients
                # The product is a 2x2 matrix P_s.
                # bot_c defines which elements of P_s get this scalar product.
                # top_c defines which blocks of C get P_s.
                
                for bc in range(4):
                    if top_c[bc] != 0:
                        # Block bc corresponds to rows 2*(bc//2), cols 2*(bc%2)
                        row_block = bc // 2
                        col_block = bc % 2
                        
                        for lc in range(4):
                            val = bot_c[lc]
                            if val != 0:
                                # lc corresponds to row lc//2, col lc%2 within the block
                                r_local = lc // 2
                                c_local = lc % 2
                                
                                global_row = 2 * row_block + r_local
                                global_col = 2 * col_block + c_local
                                c_idx = global_row * 4 + global_col
                                C_coeffs[c_idx] += top_c[bc] * val
                                                    
                # Add term if non-zero
                if any(A_coeffs) and any(B_coeffs) and any(C_coeffs):
                    raw_terms.append((A_coeffs, B_coeffs, C_coeffs))
                    
        # Combine terms with identical A and B coefficients
        # Map key: (tuple(A), tuple(B)) -> C_coeffs
        combined = {}
        for a, b, c in raw_terms:
            key = (tuple(a), tuple(b))
            if key in combined:
                combined[key] = [x + y for x, y in zip(combined[key], c)]
            else:
                combined[key] = c
                
        final_terms = []
        for key, c in combined.items():
            a, b = key
            # Skip if C is all zero (shouldn't happen)
            if any(c):
                final_terms.append((list(a), list(b), c))
                
        return final_terms

    # Get the 49 terms for the first 4 columns of B
    terms_4x4x4 = expand_4x4x4()
    num_terms_1 = len(terms_4x4x4)
    
    # Prepare U, V, W for rank 81
    rank = num_terms_1 + 32  # 49 + 32
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(24)]
    W = [[0] * rank for _ in range(24)]
    
    col = 0
    # Add 4x4x4 terms
    for a_coeffs, b_coeffs, c_coeffs in terms_4x4x4:
        for i in range(16):
            if a_coeffs[i] != 0:
                U[i][col] = a_coeffs[i]
        for j in range(16):
            # B is 4x4 for this part. V index is j*6 + k.
            # Here we are processing columns 0..3 of B.
            # So for B element (row j, col k) where k in 0..3:
            # V[j*6 + k] gets the coeff.
            # b_coeffs is indexed 0..15 for 4x4.
            # Index r = j//4? No, b_coeffs index is (row*4 + col).
            for k in range(4):
                idx_b = j * 4 + k
                if b_coeffs[idx_b] != 0:
                    v_idx = j * 6 + k
                    V[v_idx][col] = b_coeffs[idx_b]
        
        for c_idx in range(16):
            if c_coeffs[c_idx] != 0:
                # W index is (k*n + i) -> (col*4 + row)
                # c_idx is (row*4 + col)
                row = c_idx // 4
                col_c = c_idx % 4
                w_idx = col_c * 4 + row
                W[w_idx][col] = c_coeffs[c_idx]
        
        col += 1
        
    # Add 4x4x2 schoolbook terms for columns 4,5 of B
    for i in range(4):
        for j in range(4):
            for k in range(2):
                k_global = 4 + k
                U[i * 4 + j][col] = 1
                V[j * 6 + k_global][col] = 1
                W[k_global * 4 + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
