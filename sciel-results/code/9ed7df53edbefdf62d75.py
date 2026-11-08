import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS
    
    # Rank 9 decomposition for 2x2x3
    # A is 2x2, B is 2x3
    # Terms are (u, v, w) where u is 2x2, v is 2x3, w is 2x3
    # Actually, the tensor is T[i, j, k, l] = sum_r U[i, j, r] V[k, l, r] W[i, l, r]
    # Wait, the standard form is:
    # C[i, l] = sum_j sum_k A[i, j] B[k, l]
    # Decomposition: C[i, l] = sum_r (sum_j U[i, j, r] A[i, j]) (sum_k V[k, l, r] B[k, l])
    # No, the factors are:
    # U[r, i, j] A[i, j]
    # V[r, k, l] B[k, l]
    # W[r, i, l] C[i, l]
    
    # Let's define the 9 terms for 2x2x3.
    # Strassen for 2x2x2:
    # M1 = (A22+A11)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Indices for A: 00, 01, 10, 11
    # Indices for B (2x2 part): 00, 01, 10, 11
    # Indices for B (2x1 part): 02, 12
    
    terms_2x2x3 = []
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        # u_coeffs: dict (i,j) -> coeff
        # v_coeffs: dict (k,l) -> coeff
        # w_coeffs: dict (i,l) -> coeff
        terms_2x2x3.append((u_coeffs, v_coeffs, w_coeffs))

    # Strassen terms for B[:, 0:2]
    # A indices: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # B indices: (0,0)=e, (0,1)=f, (1,0)=h, (1,1)=i
    
    # M1
    add_term({(1,1):1, (0,0):1}, {(0,0):1, (1,1):1}, {(0,0):1, (0,1):1, (1,0):1, (1,1):-1})
    # M2
    add_term({(1,0):1, (1,1):1}, {(0,0):1, (1,0):1}, {(1,0):1, (1,1):1})
    # M3
    add_term({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1})
    # M4
    add_term({(1,1):1}, {(1,0):1, (0,0):-1}, {(1,1):1})
    # M5
    add_term({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1})
    # M6
    add_term({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,0):-1})
    # M7
    add_term({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (0,1):-1})
    
    # Schoolbook for B[:, 2]
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2]
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2]
    # Term 8: A[0,0] B[0,2] contributes to C[0,2]
    add_term({(0,0):1}, {(0,2):1}, {(0,2):1})
    # Term 9: A[0,1] B[1,2] contributes to C[0,2]
    # Wait, we can combine? No, each term is a rank-1 tensor.
    # We need to cover all products.
    # C[0,2] = A00 B02 + A01 B12
    # C[1,2] = A10 B02 + A11 B12
    # We can do:
    # Term 8: A00 B02 -> C02
    # Term 9: A01 B12 -> C02
    # But we also need C12.
    # So we need more terms?
    # Schoolbook for 2x2x1 has rank 4?
    # C[0,2] = A00 B02 + A01 B12
    # C[1,2] = A10 B02 + A11 B12
    # This is 4 products.
    # So total rank is 7 + 4 = 11?
    # Yes, 2x2x1 rank is 4.
    # So 2x2x3 rank is 7 + 4 = 11.
    # My previous estimate of 9 was wrong because I assumed we could share terms, but we can't easily.
    
    # Let's check rank of 2x2x1.
    # A is 2x2, B is 2x1.
    # C is 2x1.
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    # This is just 4 terms.
    # So rank(2x2x1) = 4.
    # Rank(2x2x3) <= 7 + 4 = 11.
    # Schoolbook is 12.
    # So we save 1 per 2x2x3 block.
    # For 2x2x9: 3 blocks of 2x2x3. Rank <= 3 * 11 = 33.
    # For 4x4x9: 4 blocks of 2x2x9. Rank <= 4 * 33 = 132.
    # 132 < 144.
    
    # So we add 4 more terms for the 2x2x1 part.
    add_term({(0,1):1}, {(1,2):1}, {(0,2):1})
    add_term({(1,0):1}, {(0,2):1}, {(1,2):1})
    add_term({(1,1):1}, {(1,2):1}, {(1,2):1})
    
    # Total 11 terms for 2x2x3.
    
    # Now build 4x4x9.
    # A is 4x4, B is 4x9.
    # Split A into 4 2x2 blocks: A00, A01, A10, A11.
    # Split B into 4 2x9 blocks: B00 (rows 0-1), B01 (rows 2-3)?
    # No, B is 4x9.
    # Rows of B correspond to columns of A.
    # A cols 0-1 multiply B rows 0-1.
    # A cols 2-3 multiply B rows 2-3.
    # So B_top is rows 0-1 (2x9). B_bot is rows 2-3 (2x9).
    
    # C_top (rows 0-1) = A00 B_top + A01 B_bot
    # C_bot (rows 2-3) = A10 B_top + A11 B_bot
    
    # Each term is 2x2x9.
    # We use the 11-term decomposition for 2x2x3, repeated 3 times for the 9 columns.
    
    # Total rank = 4 * 33 = 132.
    
    U = np.zeros((n*m, 132), dtype=float)
    V = np.zeros((m*p, 132), dtype=float)
    W = np.zeros((p*n, 132), dtype=float)
    
    col = 0
    # Block indices for A: (bi, bj) where bi, bj in {0,1}
    # bi: row block of A (0 for top, 1 for bot)
    # bj: col block of A (0 for left, 1 for right)
    # Block indices for B: (bk, bl) where bk in {0,1} (row block of B), bl in {0,1} (col block of B? No, B cols are not split in this way)
    # Wait, B is 4x9.
    # The "inner" dimension is 4.
    # A cols 0-1 -> B rows 0-1.
    # A cols 2-3 -> B rows 2-3.
    # So for each A block (bi, bj), we multiply by a B block.
    # If bj=0, we use B rows 0-1.
    # If bj=1, we use B rows 2-3.
    
    for bi in range(2):
        for bj in range(2):
            # A block is A[2*bi:2*bi+2, 2*bj:2*bj+2]
            # B block is B[2*bj:2*bj+2, :] (rows 2*bj to 2*bj+1)
            
            # Use the 33-term decomposition for 2x2x9
            # This is 3 sets of 11 terms for 2x2x3
            for chunk in range(3):
                # chunk 0: cols 0-2 of B
                # chunk 1: cols 3-5 of B
                # chunk 2: cols 6-8 of B
                for u_c, v_c, w_c in terms_2x2x3:
                    # Map indices
                    # A local (i, j) -> global (2*bi+i, 2*bj+j)
                    # B local (k, l) -> global (2*bj+k, chunk*3+l)
                    # C local (i, l) -> global (2*bi+i, chunk*3+l)
                    
                    for (i, j), coeff_u in u_c.items():
                        gi = 2*bi + i
                        gj = 2*bj + j
                        U[gi*m + gj, col] = coeff_u
                    
                    for (k, l), coeff_v in v_c.items():
                        gk = 2*bj + k
                        gl = chunk*3 + l
                        V[gk*p + gl, col] = coeff_v
                    
                    for (i, l), coeff_w in w_c.items():
                        gi = 2*bi + i
                        gl = chunk*3 + l
                        W[gl*n + gi, col] = coeff_w
                    
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
