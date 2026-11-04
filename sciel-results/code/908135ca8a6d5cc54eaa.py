import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    
    # U, V, W matrices
    # U: (n*m, rank) = (16, 49)
    # V: (m*p, rank) = (16, 49)
    # W: (p*n, rank) = (16, 49)
    # Note: The task says W[k*n+i, r]. 
    # C[i, k] is at index k*n + i? 
    # "output factor follows the official trace convention W[k*n+i,r]"
    # Usually C[i, k] is row i, col k.
    # If W is (p*n, rank), and index is k*n+i, then W[row, col] corresponds to C[i, k]?
    # Let's check: k is col index of C (0..3), i is row index of C (0..3).
    # k*n + i = k*4 + i.
    # So W[k*4+i, r] is the coefficient for C[i, k] in the r-th term.
    
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # Define Strassen coefficients for 2x2
    # S1 = (A00+A11)(B00+B11)
    # S2 = (A10+A11)B00
    # S3 = A00(B01-B11)
    # S4 = A11(B10-B00)
    # S5 = (A00+A01)B11
    # S6 = (A10-A00)(B00+B01)
    # S7 = (A01-A11)(B10+B11)
    
    # Inner Strassen:
    # For matrices X, Y (2x2)
    # Terms are indexed by l=0..6
    # x_coeffs[l] = 2x2 matrix of coeffs for X entries
    # y_coeffs[l] = 2x2 matrix of coeffs for Y entries
    
    # X entries: x00, x01, x10, x11
    # Y entries: y00, y01, y10, y11
    
    inner_x = [
        [[1, 0], [0, 1]], # S1: X00 + X11 -> x00+1, x11+1
        [[0, 0], [1, 1]], # S2: X10 + X11 -> x10+1, x11+1
        [[1, 0], [0, 0]], # S3: X00 -> x00+1
        [[0, 0], [0, 1]], # S4: X11 -> x11+1
        [[1, 1], [0, 0]], # S5: X00 + X01 -> x00+1, x01+1
        [[-1, 0], [1, 0]], # S6: X10 - X00 -> x00-1, x10+1
        [[0, -1], [0, 1]]  # S7: X01 - X11 -> x01-1, x11+1
    ]
    
    inner_y = [
        [[1, 0], [0, 1]], # S1: Y00 + Y11
        [[1, 0], [0, 0]], # S2: Y00
        [[0, 1], [0, -1]], # S3: Y01 - Y11
        [[0, -1], [1, 0]], # S4: Y10 - Y00
        [[0, 0], [0, 1]], # S5: Y11
        [[1, 1], [0, 0]], # S6: Y00 + Y01
        [[0, 0], [1, 1]]  # S7: Y10 + Y11
    ]
    
    # Outer Strassen:
    # X_k and Y_k are 2x2 matrices of 2x2 blocks.
    # We need to express X_k entries (which are 2x2 blocks) as linear combos of A blocks.
    # Then express A blocks as linear combos of A scalars.
    
    # A blocks: A00, A01, A10, A11
    # A00 is rows 0,1 cols 0,1
    # A01 is rows 0,1 cols 2,3
    # A10 is rows 2,3 cols 0,1
    # A11 is rows 2,3 cols 2,3
    
    # Outer X coeffs (for blocks):
    # X0 = A00 + A11
    # X1 = A10 + A11
    # X2 = A00
    # X3 = A11
    # X4 = A00 + A01
    # X5 = A10 - A00
    # X6 = A01 - A11
    
    outer_x_blocks = [
        [[1, 0], [0, 1]], # X0: A00+1, A11+1
        [[0, 0], [1, 1]], # X1: A10+1, A11+1
        [[1, 0], [0, 0]], # X2: A00+1
        [[0, 0], [0, 1]], # X3: A11+1
        [[1, 1], [0, 0]], # X4: A00+1, A01+1
        [[-1, 0], [1, 0]], # X5: A00-1, A10+1
        [[0, -1], [0, 1]]  # X6: A01-1, A11+1
    ]
    
    outer_y_blocks = [
        [[1, 0], [0, 1]], # Y0: B00+1, B11+1
        [[1, 0], [0, 0]], # Y1: B00+1
        [[0, 1], [0, -1]], # Y2: B01+1, B11-1
        [[0, -1], [1, 0]], # Y3: B00-1, B10+1
        [[0, 0], [0, 1]], # Y4: B11+1
        [[1, 1], [0, 0]], # Y5: B00+1, B01+1
        [[0, 0], [1, 1]]  # Y6: B10+1, B11+1
    ]
    
    # Mapping block indices to scalar indices
    # Block (r, c) contains scalars at rows 2r+i, cols 2c+j
    # A scalar index: i*4 + j
    # B scalar index: j*4 + k (since V is (m*p, rank), row is j*p+k? No, V is (m*p, rank). 
    # B is 4x4. Row j, Col k. Index j*4+k.
    # C scalar index: k*4 + i (per W[k*n+i, r])
    
    def get_block_scalars_A(r, c):
        # Returns list of (scalar_index, coeff) for block A[r,c]
        # We just need the scalar indices
        idxs = []
        for i in range(2):
            for j in range(2):
                row = 2*r + i
                col = 2*c + j
                idxs.append(row * 4 + col)
        return idxs
    
    def get_block_scalars_B(r, c):
        idxs = []
        for i in range(2):
            for j in range(2):
                row = 2*r + i
                col = 2*c + j
                idxs.append(row * 4 + col)
        return idxs
        
    col = 0
    for k in range(7):
        for l in range(7):
            # Get x_coeffs for inner Strassen
            # x_coeffs is 2x2 matrix of coeffs for X_k entries
            # X_k entries are blocks.
            # So we need to combine outer and inner coeffs.
            
            # For each entry (p, q) of X_k (which is a 2x2 block matrix? No, X_k is 2x2 matrix of blocks)
            # Wait. X_k is a 2x2 matrix. Its entries are 2x2 blocks.
            # So X_k[p, q] is a 2x2 block.
            # The inner Strassen operates on the 2x2 matrix X_k.
            # So inner_x[l][p, q] is the coeff for the block X_k[p, q].
            
            # So for a specific scalar in A:
            # It belongs to a block A[r, c].
            # That block contributes to X_k[p, q] with coeff outer_x_blocks[k][p, q].
            # Then X_k[p, q] contributes to S_{k,l} with coeff inner_x[l][p, q].
            # Total coeff for A[r, c] in S_{k,l} is sum over p,q of outer_x_blocks[k][p, q] * inner_x[l][p, q].
            
            # Let's compute the total coeff for each block A[r, c]
            coeff_A_blocks = [[0.0 for _ in range(2)] for _ in range(2)]
            coeff_B_blocks = [[0.0 for _ in range(2)] for _ in range(2)]
            
            for p in range(2):
                for q in range(2):
                    c_x = outer_x_blocks[k][p][q]
                    c_y = outer_y_blocks[k][p][q]
                    c_ix = inner_x[l][p][q]
                    c_iy = inner_y[l][p][q]
                    
                    # Wait, inner_x[l] is a 2x2 matrix of coeffs for the entries of X_k.
                    # X_k is a 2x2 matrix of blocks.
                    # So inner_x[l][p][q] multiplies the block X_k[p][q].
                    # And X_k[p][q] is a linear combo of A blocks.
                    # So coeff for A block [r, c] is:
                    # sum_{p,q} inner_x[l][p][q] * (coeff of A[r,c] in X_k[p][q])
                    
                    # But outer_x_blocks[k][p][q] is the coeff of A block [p, q]?
                    # No.
                    # X_k is a 2x2 matrix of blocks.
                    # X_k[p][q] is the block at position p, q of the matrix X_k.
                    # But X_k is formed from A blocks.
                    # X_k = sum_{r,c} outer_x_blocks[k][r][c] A[r][c].
                    # So X_k[p][q] is NOT a single block.
                    # X_k is a 2x2 matrix. Each entry of X_k is a 2x2 matrix (block).
                    # So X_k[p][q] is the block at position p, q of X_k.
                    # But X_k is a matrix of blocks.
                    # So X_k[p][q] is a 2x2 matrix of scalars.
                    # And X_k[p][q] = sum_{r,c} outer_x_blocks[k][r][c] * (A[r][c][p][q] ?)
                    # No.
                    # A is a 4x4 matrix.
                    # A[r][c] is a 2x2 block.
                    # X_k is a 4x4 matrix? No.
                    # Strassen works on 2x2 blocks.
                    # So X_k is a 2x2 matrix of blocks.
                    # So X_k has 4 blocks.
                    # Let's call the blocks of X_k as XB_00, XB_01, XB_10, XB_11.
                    # XB_pq = sum_{r,c} outer_x_blocks[k][r][c] * A_rc ?
                    # No.
                    # X_k = A00 + A11.
                    # This means XB_00 = A00_00 + A11_00, XB_01 = A00_01 + A11_01, etc.
                    # So XB_pq = sum_{r,c} outer_x_blocks[k][r][c] * A_rc_pq.
                    
                    # So the coeff of A_rc_pq in X_k_pq is outer_x_blocks[k][r][c].
                    
                    # Now, inner Strassen operates on the 2x2 matrix of blocks XB.
                    # S_{k,l} uses entries XB_pq.
                    # The coeff of XB_pq in S_{k,l} is inner_x[l][p][q].
                    
                    # So the total coeff of A_rc_pq in S_{k,l} is:
                    # sum_{p,q} inner_x[l][p][q] * outer_x_blocks[k][r][c] ?
                    # No.
                    # XB_pq depends on A_rc_pq.
                    # Specifically, XB_pq = sum_{r,c} outer_x_blocks[k][r][c] * A_rc_pq.
                    # So A_rc_pq appears in XB_pq with coeff outer_x_blocks[k][r][c].
                    # And XB_pq appears in S_{k,l} with coeff inner_x[l][p][q].
                    # So total coeff is outer_x_blocks[k][r][c] * inner_x[l][p][q].
                    
                    # But this is for a specific p, q.
                    # We need to sum over p, q?
                    # No.
                    # A_rc_pq is a specific scalar.
                    # It is part of block A_rc.
                    # It contributes to XB_pq.
                    # XB_pq is used in S_{k,l} with coeff inner_x[l][p][q].
                    # So yes, for a fixed scalar A_rc_pq, the coeff is outer_x_blocks[k][r][c] * inner_x[l][p][q].
                    
                    # Wait, is that right?
                    # S_{k,l} is a linear combo of XB_pq.
                    # XB_pq is a linear combo of A_rc_pq.
                    # So S_{k,l} is a linear combo of A_rc_pq.
                    # Coeff = inner_x[l][p][q] * outer_x_blocks[k][r][c].
                    
                    # So for each scalar A_rc_pq, we can compute its coeff in U.
                    
            # Let's iterate over all A scalars
            for r in range(2):
                for c in range(2):
                    for p in range(2):
                        for q in range(2):
                            # Scalar A[2r+p, 2c+q]
                            idx_A = (2*r + p) * 4 + (2*c + q)
                            coeff = outer_x_blocks[k][r][c] * inner_x[l][p][q]
                            U[idx_A, col] += coeff
                            
            for r in range(2):
                for c in range(2):
                    for p in range(2):
                        for q in range(2):
                            # Scalar B[2r+p, 2c+q]
                            idx_B = (2*r + p) * 4 + (2*c + q)
                            coeff = outer_y_blocks[k][r][c] * inner_y[l][p][q]
                            V[idx_B, col] += coeff
                            
            # Now for W
            # S_{k,l} contributes to M_k entries.
            # M_k_pq = sum_l inner_recon[l][p][q] * S_{k,l}
            # Then M_k contributes to C blocks.
            # C_rc = sum_k outer_recon[k][r][c] * M_k
            # So C_rc_pq = sum_k sum_l outer_recon[k][r][c] * inner_recon[l][p][q] * S_{k,l}
            
            # inner_recon[l][p][q] is the coeff of S_l in M_pq
            # M00 = S1 + S4 - S5 + S7
            # M01 = S3 + S5
            # M10 = S2 + S4
            # M11 = S1 - S2 + S3 + S6
            
            inner_recon = [
                [[1, 0], [0, 0]], # S1
                [[0, 0], [1, 0]], # S2
                [[0, 1], [0, 0]], # S3
                [[1, 0], [0, 0]], # S4 -- Wait. M00=S1+S4... M10=S2+S4
                # Let's build it properly
            ]
            
            # l=0 (S1): M00+1, M11+1
            # l=1 (S2): M10+1, M11-1
            # l=2 (S3): M01+1, M11+1
            # l=3 (S4): M00+1, M10+1
            # l=4 (S5): M00-1, M01+1
            # l=5 (S6): M11+1
            # l=6 (S7): M00+1
            
            # Matrix for each l: 2x2 coeffs for M00, M01, M10, M11
            inner_recon_coeffs = [
                [[1, 0], [0, 1]], # S1
                [[0, 0], [1, -1]], # S2
                [[0, 1], [0, 1]], # S3
                [[1, 0], [1, 0]], # S4
                [[-1, 1], [0, 0]], # S5
                [[0, 0], [0, 1]], # S6
                [[1, 0], [0, 0]]  # S7
            ]
            
            # Outer recon:
            # C00 = M0 + M3 - M4 + M6  (using 0-index: M0, M3, M4, M6)
            # C01 = M2 + M4
            # C10 = M1 + M3
            # C11 = M0 - M1 + M2 + M5
            
            outer_recon_coeffs = [
                [[1, 0], [0, 1]], # M0
                [[0, 0], [1, -1]], # M1
                [[0, 1], [0, 1]], # M2
                [[1, 0], [1, 0]], # M3
                [[-1, 1], [0, 0]], # M4
                [[0, 0], [0, 1]], # M5
                [[1, 0], [0, 0]]  # M6
            ]
            
            # Total coeff for C_rc_pq
            for r in range(2):
                for c in range(2):
                    for p in range(2):
                        for q in range(2):
                            # Scalar C[2r+p, 2c+q]
                            # W index: k*n + i. Here i is row, k is col.
                            # Row i = 2r+p, Col k = 2c+q.
                            idx_C = (2*c + q) * 4 + (2*r + p)
                            
                            coeff = 0.0
                            for k_idx in range(7):
                                for l_idx in range(7):
                                    # This is inefficient. We only need current k, l.
                                    pass
                            
                            # For current k, l:
                            # Coeff = outer_recon_coeffs[k][r][c] * inner_recon_coeffs[l][p][q]
                            coeff = outer_recon_coeffs[k][r][c] * inner_recon_coeffs[l][p][q]
                            W[idx_C, col] += coeff
                            
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
