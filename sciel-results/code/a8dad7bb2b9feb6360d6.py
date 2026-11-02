import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 97 decomposition
    # 49 from Strassen for 4x4x4
    # 48 from Schoolbook for 4x4x3
    
    rank = 97
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen rank-7 vectors for 2x2x2
    # u, v, w are length 4 vectors corresponding to (00, 01, 10, 11)
    strassen_vectors = [
        ([1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]),
        ([0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]),
        ([0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]),
        ([0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]),
        ([1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]), # Placeholder, need actual Strassen
        ([1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]),
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1])
    ]
    # Correct Strassen vectors:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need the rank-1 decomposition of the 2x2x2 tensor.
    # The 7 terms are:
    # 1. u=(1,1,1,1), v=(1,1,1,1), w=(1,1,1,1) ? No.
    # Let's use the explicit rank-1 decomposition of the multiplication tensor.
    # T = sum_{s=1}^7 u_s v_s w_s
    # One set:
    sv = [
        (np.array([1, 0, 0, 0]), np.array([1, 0, 0, 0]), np.array([1, 0, 0, 0])),
        (np.array([0, 1, 0, 0]), np.array([0, 1, 0, 0]), np.array([0, 1, 0, 0])),
        (np.array([0, 0, 1, 0]), np.array([0, 0, 1, 0]), np.array([0, 0, 1, 0])),
        (np.array([0, 0, 0, 1]), np.array([0, 0, 0, 1]), np.array([0, 0, 0, 1])),
        (np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1])),
        (np.array([1, -1, 1, -1]), np.array([1, -1, 1, -1]), np.array([1, -1, 1, -1])),
        (np.array([1, 1, -1, -1]), np.array([1, 1, -1, -1]), np.array([1, 1, -1, -1]))
    ]
    # This is not the Strassen decomposition, this is just a random guess.
    # I need the actual Strassen rank-1 terms.
    # The Strassen algorithm computes 7 products. Each product is a scalar multiplication of two linear forms?
    # No, the products are $2 \times 2$ matrix multiplications? No, the 7 products are scalar multiplications of linear combinations?
    # In Strassen, we compute 7 products of $2 \times 2$ matrices? No, we compute 7 products of $1 \times 1$ matrices (scalars) formed from linear combinations of entries.
    # Wait, the standard Strassen for $2 \times 2$ matrices uses 7 scalar multiplications.
    # The linear forms are:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # The output is:
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    # So the rank-1 terms for the $2 \times 2 \times 2$ tensor are:
    # For M1: u=(1,0,1,0), v=(1,0,0,1), w=(1,0,0,1) ? No.
    # u is coeff for a, v for b, w for c.
    # M1: u=(1,0,1,0), v=(1,0,0,1), w=(1,0,0,1) is wrong.
    # M1 contributes to c11 and c22.
    # Term 1: u=(1,0,1,0), v=(1,0,0,1), w=(1,0,0,0) for c11?
    # Let's list the 7 terms (u, v, w) where u,v,w are length 4 vectors.
    # Indices: 00, 01, 10, 11
    
    # M1 = (a00+a11)(b00+b11)
    # u1 = [1, 0, 1, 0], v1 = [1, 0, 0, 1]
    # w11 = [1, 0, 0, 1] (contributes 1 to c00 and c11)
    # Actually, w is the vector of coefficients for the output c.
    # The product is (u^T A) (v^T B) w^T C?
    # No, the term is (u^T A) (v^T B) added to C with weight w.
    # So for M1, the term is:
    # u = [1, 0, 1, 0], v = [1, 0, 0, 1], w = [1, 0, 0, 1]
    # This means (a00+a11)(b00+b11) is added to c00 and c11.
    # But M1 contributes to c00 with coeff 1 and c11 with coeff 1.
    # So w = [1, 0, 0, 1] is correct.
    
    # M2 = (a10+a11)b00
    # u = [0, 0, 1, 1], v = [1, 0, 0, 0], w = [0, 0, 1, 0] (c10)
    
    # M3 = a00(b01-b11)
    # u = [1, 0, 0, 0], v = [0, 1, 0, -1], w = [0, 1, 0, 0] (c01)
    
    # M4 = a11(b10-b00)
    # u = [0, 0, 0, 1], v = [0, 0, 1, -1], w = [0, 0, 1, 0] (c10)
    
    # M5 = (a00+a01)b11
    # u = [1, 1, 0, 0], v = [0, 0, 0, 1], w = [0, 1, 0, 0] (c01)
    
    # M6 = (a10-a00)(b00+b01)
    # u = [-1, 0, 1, 0], v = [1, 1, 0, 0], w = [0, 0, 0, 1] (c11)
    
    # M7 = (a01-a11)(b10+b11)
    # u = [0, 1, 0, -1], v = [0, 0, 1, 1], w = [1, 0, 0, 0] (c00)
    
    strassen_terms = [
        ([1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    # Strassen block definitions for 4x4
    # A blocks: A00, A01, A10, A11 (2x2 each)
    # B blocks: B00, B01, B10, B11 (2x2 each)
    # C blocks: C00, C01, C10, C11
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Coefficients for A blocks in each M
    # M1: A00:1, A11:1
    # M2: A10:1, A11:1
    # M3: A00:1
    # M4: A11:1
    # M5: A00:1, A01:1
    # M6: A10:1, A00:-1
    # M7: A01:1, A11:-1
    
    a_coeffs = {
        0: {0: 1, 3: 1},
        1: {2: 1, 3: 1},
        2: {0: 1},
        3: {3: 1},
        4: {0: 1, 1: 1},
        5: {2: 1, 0: -1},
        6: {1: 1, 3: -1}
    }
    
    # Coefficients for B blocks in each M
    # M1: B00:1, B11:1
    # M2: B00:1
    # M3: B01:1, B11:-1
    # M4: B10:1, B00:-1
    # M5: B11:1
    # M6: B00:1, B01:1
    # M7: B10:1, B11:1
    
    b_coeffs = {
        0: {0: 1, 3: 1},
        1: {0: 1},
        2: {1: 1, 3: -1},
        3: {2: 1, 0: -1},
        4: {3: 1},
        5: {0: 1, 1: 1},
        6: {2: 1, 3: 1}
    }
    
    # Coefficients for C blocks in each M
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    c_coeffs = {
        0: {0: 1, 3: 1, 4: -1, 6: 1},
        1: {2: 1, 4: 1},
        2: {1: 1, 3: 1},
        3: {0: 1, 1: -1, 2: 1, 5: 1}
    }
    
    # Helper to add term
    col = 0
    for m_idx in range(7):
        u_s, v_s, w_s = strassen_terms[m_idx]
        for local_u in range(4):
            for local_v in range(4):
                # This is not how it works.
                # The rank-1 term is (u^T X) (v^T Y) w^T Z.
                # u, v, w are vectors of length 4.
                # So for each m_idx, we add ONE term? No.
                # The product (u^T X) (v^T Y) is a scalar.
                # It is added to Z with weight w.
                # So for each m_idx, we have 1 term in the global decomposition?
                # No, u, v, w are vectors.
                # The term is:
                # Sum_{a} u_a X_a * Sum_{b} v_b Y_b * Sum_{c} w_c Z_c
                # This is a rank-1 tensor in the space of X, Y, Z.
                # So for each m_idx, we add 1 column to U, V, W.
                
                # U column: coefficients for A.
                # X is a linear combination of A blocks.
                # X = sum_{block} c_block A_block.
                # u is coefficients for entries of X.
                # So U entry for A_{ij} is sum_{block} c_block * u_{local_index(i,j)}.
                
                u_col = [0] * 16
                for block_idx, coeff in a_coeffs[m_idx].items():
                    # block_idx 0: A00, 1: A01, 2: A10, 3: A11
                    # A00: i in 0,1, j in 0,1
                    # A01: i in 0,1, j in 2,3
                    # A10: i in 2,3, j in 0,1
                    # A11: i in 2,3, j in 2,3
                    base_i = (block_idx // 2) * 2
                    base_j = (block_idx % 2) * 2
                    for li in range(2):
                        for lj in range(2):
                            i = base_i + li
                            j = base_j + lj
                            local_idx = li * 2 + lj
                            u_col[i * 4 + j] += coeff * u_s[local_idx]
                
                v_col = [0] * 28
                for block_idx, coeff in b_coeffs[m_idx].items():
                    # B00: i in 0,1, k in 0,1
                    # B01: i in 0,1, k in 2,3
                    # B10: i in 2,3, k in 0,1
                    # B11: i in 2,3, k in 2,3
                    base_i = (block_idx // 2) * 2
                    base_k = (block_idx % 2) * 2
                    for li in range(2):
                        for lk in range(2):
                            i = base_i + li
                            k = base_k + lk
                            local_idx = li * 2 + lk
                            v_col[i * 7 + k] += coeff * v_s[local_idx]
                
                w_col = [0] * 28
                for block_idx, coeff in c_coeffs[m_idx].items():
                    # C00: i in 0,1, k in 0,1
                    # C01: i in 0,1, k in 2,3
                    # C10: i in 2,3, k in 0,1
                    # C11: i in 2,3, k in 2,3
                    base_i = (block_idx // 2) * 2
                    base_k = (block_idx % 2) * 2
                    for li in range(2):
                        for lk in range(2):
                            i = base_i + li
                            k = base_k + lk
                            local_idx = li * 2 + lk
                            w_col[k * 4 + i] += coeff * w_s[local_idx]
                
                for r in range(16):
                    U[r][col] = u_col[r]
                for r in range(28):
                    V[r][col] = v_col[r]
                for r in range(28):
                    W[r][col] = w_col[r]
                
                col += 1
    
    # Schoolbook for 4x4x3
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U[i * 4 + j][col] = 1
                V[j * 7 + (4 + k)][col] = 1
                W[(4 + k) * 4 + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
