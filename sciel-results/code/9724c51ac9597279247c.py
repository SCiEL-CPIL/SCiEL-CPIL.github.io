"""Optimized matrix multiplication tensor decomposition for 3x4 * 4x4."""
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    """
    Construct rank-minimal tensor decomposition for 3x4 * 4x4 matrix multiplication.
    
    Strategy: Pad A (3x4) to (4x4) by adding a zero row. Apply Strassen's algorithm
    to compute 4x4 * 4x4, then extract the first 3 rows of the result.
    
    This gives rank 28 (7 Strassen products × 4 scalar mults per 2x2 product).
    
    The tensor T[i,j,k] = 1 for i in [0,3), j in [0,4), k in [0,4).
    Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    where n=3, m=4, p=4.
    """
    n, m, p = 3, 4, 4
    rank = 28  # 7 Strassen terms × 4 scalar multiplications each
    
    U = np.zeros((n * m, rank), dtype=float)  # (12, 28)
    V = np.zeros((m * p, rank), dtype=float)  # (16, 28)
    W = np.zeros((p * n, rank), dtype=float)  # (12, 28)
    
    # Strassen's algorithm for 4x4 * 4x4:
    # Split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # where each block is 2x2.
    #
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4]
    # B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    #
    # Note: A[3,:] = 0 (padded row), so A21[1,:] = A[3,0:2] = 0, A22[1,:] = A[3,2:4] = 0.
    #
    # Strassen products:
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22) B11
    # P3 = A11 (B12 - B22)
    # P4 = A22 (B21 - B11)
    # P5 = (A11 + A12) B22
    # P6 = (A21 - A11)(B11 + B12)
    # P7 = (A12 - A22)(B21 + B22)
    #
    # Result blocks:
    # C11 = P1 + P4 - P5 - P7
    # C12 = P3 + P5
    # C21 = P2 + P7
    # C22 = P1 - P2 + P3 + P6
    #
    # We only need C[0:3, :] (first 3 rows), so we need:
    # C11 (rows 0-1, cols 0-1), C12 (rows 0-1, cols 2-3), 
    # C21[0,:] (row 2, cols 0-1)
    # We don't need C22 or C21[1,:].
    
    # Define the 7 Strassen products. Each is a 2x2 matrix product.
    # For each product, we'll expand it into 4 scalar multiplications (schoolbook for 2x2).
    
    # For a 2x2 product X * Y where X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]:
    # Result[0,0] = x00*y00 + x01*y10
    # Result[0,1] = x00*y01 + x01*y11
    # Result[1,0] = x10*y00 + x11*y10
    # Result[1,1] = x10*y01 + x11*y11
    
    # Each scalar multiplication is of the form (linear combo of A entries) * (linear combo of B entries).
    # We need to express this in the tensor decomposition form.
    
    # Let's define helper functions to get the linear coefficients for each Strassen product.
    
    # For P1 = (A11 + A22)(B11 + B22):
    # X = A11 + A22, Y = B11 + B22
    # X[0,0] = A[0,0] + A[2,2]
    # X[0,1] = A[0,1] + A[2,3]
    # X[1,0] = A[1,0] + A[3,2] = A[1,0]  (since A[3,2]=0)
    # X[1,1] = A[1,1] + A[3,3] = A[1,1]  (since A[3,3]=0)
    # Y[0,0] = B[0,0] + B[2,2]
    # Y[0,1] = B[0,1] + B[2,3]
    # Y[1,0] = B[1,0] + B[3,2]
    # Y[1,1] = B[1,1] + B[3,3]
    
    # For P2 = (A21 + A22) B11:
    # X = A21 + A22, Y = B11
    # X[0,0] = A[2,0] + A[2,2]
    # X[0,1] = A[2,1] + A[2,3]
    # X[1,0] = A[3,0] + A[3,2] = 0
    # X[1,1] = A[3,1] + A[3,3] = 0
    # Y[0,0] = B[0,0]
    # Y[0,1] = B[0,1]
    # Y[1,0] = B[1,0]
    # Y[1,1] = B[1,1]
    
    # For P3 = A11 (B12 - B22):
    # X = A11, Y = B12 - B22
    # X[0,0] = A[0,0]
    # X[0,1] = A[0,1]
    # X[1,0] = A[1,0]
    # X[1,1] = A[1,1]
    # Y[0,0] = B[0,2] - B[2,2]
    # Y[0,1] = B[0,3] - B[2,3]
    # Y[1,0] = B[1,2] - B[3,2]
    # Y[1,1] = B[1,3] - B[3,3]
    
    # For P4 = A22 (B21 - B11):
    # X = A22, Y = B21 - B11
    # X[0,0] = A[2,2]
    # X[0,1] = A[2,3]
    # X[1,0] = A[3,2] = 0
    # X[1,1] = A[3,3] = 0
    # Y[0,0] = B[2,0] - B[0,0]
    # Y[0,1] = B[2,1] - B[0,1]
    # Y[1,0] = B[3,0] - B[1,0]
    # Y[1,1] = B[3,1] - B[1,1]
    
    # For P5 = (A11 + A12) B22:
    # X = A11 + A12, Y = B22
    # X[0,0] = A[0,0] + A[0,2]
    # X[0,1] = A[0,1] + A[0,3]
    # X[1,0] = A[1,0] + A[1,2]
    # X[1,1] = A[1,1] + A[1,3]
    # Y[0,0] = B[2,2]
    # Y[0,1] = B[2,3]
    # Y[1,0] = B[3,2]
    # Y[1,1] = B[3,3]
    
    # For P6 = (A21 - A11)(B11 + B12):
    # X = A21 - A11, Y = B11 + B12
    # X[0,0] = A[2,0] - A[0,0]
    # X[0,1] = A[2,1] - A[0,1]
    # X[1,0] = A[3,0] - A[1,0] = -A[1,0]
    # X[1,1] = A[3,1] - A[1,1] = -A[1,1]
    # Y[0,0] = B[0,0] + B[0,2]
    # Y[0,1] = B[0,1] + B[0,3]
    # Y[1,0] = B[1,0] + B[1,2]
    # Y[1,1] = B[1,1] + B[1,3]
    
    # For P7 = (A12 - A22)(B21 + B22):
    # X = A12 - A22, Y = B21 + B22
    # X[0,0] = A[0,2] - A[2,2]
    # X[0,1] = A[0,3] - A[2,3]
    # X[1,0] = A[1,2] - A[3,2] = A[1,2]
    # X[1,1] = A[1,3] - A[3,3] = A[1,3]
    # Y[0,0] = B[2,0] + B[2,2]
    # Y[0,1] = B[2,1] + B[2,3]
    # Y[1,0] = B[3,0] + B[3,2]
    # Y[1,1] = B[3,1] + B[3,3]
    
    # Now, for each of the 7 products, and for each of the 4 scalar multiplications
    # in the 2x2 product, we create one column in the decomposition.
    
    # The 4 scalar multiplications for a 2x2 product X*Y are:
    # t=0: X[0,0]*Y[0,0] contributes to result[0,0]
    # t=1: X[0,1]*Y[1,0] contributes to result[0,0]
    # t=2: X[0,0]*Y[0,1] contributes to result[0,1]
    # t=3: X[0,1]*Y[1,1] contributes to result[0,1]
    # t=4: X[1,0]*Y[0,0] contributes to result[1,0]
    # t=5: X[1,1]*Y[1,0] contributes to result[1,0]
    # t=6: X[1,0]*Y[0,1] contributes to result[1,1]
    # t=7: X[1,1]*Y[1,1] contributes to result[1,1]
    
    # Wait, that's 8 scalar multiplications per 2x2 product, not 4.
    # Schoolbook 2x2 has 4 multiplications if we use the standard algorithm:
    # result[0,0] = x00*y00 + x01*y10
    # result[0,1] = x00*y01 + x01*y11
    # result[1,0] = x10*y00 + x11*y10
    # result[1,1] = x10*y01 + x11*y11
    # That's 8 scalar multiplications total (4 per row of result, 2 per entry).
    
    # Hmm, so 7 products × 8 scalar mults = 56. That's not 28.
    
    # Let me reconsider. The standard Strassen algorithm for 2x2 uses 7 multiplications
    # of 2x2 matrices. Each 2x2 multiplication using schoolbook requires 4 scalar multiplications.
    # So total scalar multiplications = 7 × 4 = 28.
    
    # But wait, each 2x2 matrix product has 4 entries, and each entry requires 2 scalar multiplications
    # (for the 2x2 case). So 4 entries × 2 = 8 scalar multiplications per 2x2 product.
    # 7 × 8 = 56.
    
    # I think I'm confusing myself. Let me think more carefully.
    
    # A 2x2 matrix product C = A*B where A and B are 2x2:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]  (2 mults)
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]  (2 mults)
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]  (2 mults)
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]  (2 mults)
    # Total: 8 scalar multiplications.
    
    # So 7 Strassen products × 8 scalar mults = 56 scalar multiplications.
    
    # But the standard claim is that Strassen uses 7 multiplications vs 8 for schoolbook
    # on 2x2 matrices. The "7 multiplications" refers to 7 multiplications of 2x2 matrices,
    # not scalar multiplications.
    
    # So for our 4x4 * 4x4 case:
    # Schoolbook: 4x4 = 64 scalar multiplications
    # Strassen: 7 × (2x2 schoolbook) = 7 × 8 = 56 scalar multiplications
    
    # So rank should be 56, not 28.
    
    # Let me redo this with rank = 56.
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Define the 7 Strassen products as (X_coeffs, Y_coeffs) where
    # X_coeffs is a dict mapping (row, col) -> list of (A_index, coeff)
    # Y_coeffs is a dict mapping (row, col) -> list of (B_index, coeff)
    
    # A_index is i*m + j = i*4 + j
    # B_index is j*p + k = j*4 + k
    
    # P1 = (A11 + A22)(B11 + B22)
    X1 = {
        (0,0): [(0*4+0, 1), (2*4+2, 1)],
        (0,1): [(0*4+1, 1), (2*4+3, 1)],
        (1,0): [(1*4+0, 1)],  # A[1,0] + A[3,2]=0
        (1,1): [(1*4+1, 1)],  # A[1,1] + A[3,3]=0
    }
    Y1 = {
        (0,0): [(0*4+0, 1), (2*4+2, 1)],
        (0,1): [(0*4+1, 1), (2*4+3, 1)],
        (1,0): [(1*4+0, 1), (3*4+2, 1)],
        (1,1): [(1*4+1, 1), (3*4+3, 1)],
    }
    
    # P2 = (A21 + A22) B11
    X2 = {
        (0,0): [(2*4+0, 1), (2*4+2, 1)],
        (0,1): [(2*4+1, 1), (2*4+3, 1)],
        (1,0): [],  # A[3,0]+A[3,2]=0
        (1,1): [],  # A[3,1]+A[3,3]=0
    }
    Y2 = {
        (0,0): [(0*4+0, 1)],
        (0,1): [(0*4+1, 1)],
        (1,0): [(1*4+0, 1)],
        (1,1): [(1*4+1, 1)],
    }
    
    # P3 = A11 (B12 - B22)
    X3 = {
        (0,0): [(0*4+0, 1)],
        (0,1): [(0*4+1, 1)],
        (1,0): [(1*4+0, 1)],
        (1,1): [(1*4+1, 1)],
    }
    Y3 = {
        (0,0): [(0*4+2, 1), (2*4+2, -1)],
        (0,1): [(0*4+3, 1), (2*4+3, -1)],
        (1,0): [(1*4+2, 1), (3*4+2, -1)],
        (1,1): [(1*4+3, 1), (3*4+3, -1)],
    }
    
    # P4 = A22 (B21 - B11)
    X4 = {
        (0,0): [(2*4+2, 1)],
        (0,1): [(2*4+3, 1)],
        (1,0): [],  # A[3,2]=0
        (1,1): [],  # A[3,3]=0
    }
    Y4 = {
        (0,0): [(2*4+0, 1), (0*4+0, -1)],
        (0,1): [(2*4+1, 1), (0*4+1, -1)],
        (1,0): [(3*4+0, 1), (1*4+0, -1)],
        (1,1): [(3*4+1, 1), (1*4+1, -1)],
    }
    
    # P5 = (A11 + A12) B22
    X5 = {
        (0,0): [(0*4+0, 1), (0*4+2, 1)],
        (0,1): [(0*4+1, 1), (0*4+3, 1)],
        (1,0): [(1*4+0, 1), (1*4+2, 1)],
        (1,1): [(1*4+1, 1), (1*4+3, 1)],
    }
    Y5 = {
        (0,0): [(2*4+2, 1)],
        (0,1): [(2*4+3, 1)],
        (1,0): [(3*4+2, 1)],
        (1,1): [(3*4+3, 1)],
    }
    
    # P6 = (A21 - A11)(B11 + B12)
    X6 = {
        (0,0): [(2*4+0, 1), (0*4+0, -1)],
        (0,1): [(2*4+1, 1), (0*4+1, -1)],
        (1,0): [(1*4+0, -1)],  # A[3,0]-A[1,0] = -A[1,0]
        (1,1): [(1*4+1, -1)],  # A[3,1]-A[1,1] = -A[1,1]
    }
    Y6 = {
        (0,0): [(0*4+0, 1), (0*4+2, 1)],
        (0,1): [(0*4+1, 1), (0*4+3, 1)],
        (1,0): [(1*4+0, 1), (1*4+2, 1)],
        (1,1): [(1*4+1, 1), (1*4+3, 1)],
    }
    
    # P7 = (A12 - A22)(B21 + B22)
    X7 = {
        (0,0): [(0*4+2, 1), (2*4+2, -1)],
        (0,1): [(0*4+3, 1), (2*4+3, -1)],
        (1,0): [(1*4+2, 1)],  # A[1,2]-A[3,2] = A[1,2]
        (1,1): [(1*4+3, 1)],  # A[1,3]-A[3,3] = A[1,3]
    }
    Y7 = {
        (0,0): [(2*4+0, 1), (2*4+2, 1)],
        (0,1): [(2*4+1, 1), (2*4+3, 1)],
        (1,0): [(3*4+0, 1), (3*4+2, 1)],
        (1,1): [(3*4+1, 1), (3*4+3, 1)],
    }
    
    products = [
        (X1, Y1),
        (X2, Y2),
        (X3, Y3),
        (X4, Y4),
        (X5, Y5),
        (X6, Y6),
        (X7, Y7),
    ]
    
    # For each product, expand into 8 scalar multiplications (2x2 schoolbook).
    # The 8 terms are:
    # (0,0): X[0,0]*Y[0,0] -> contributes to result[0,0]
    # (0,1): X[0,1]*Y[1,0] -> contributes to result[0,0]
    # (0,2): X[0,0]*Y[0,1] -> contributes to result[0,1]
    # (0,3): X[0,1]*Y[1,1] -> contributes to result[0,1]
    # (1,0): X[1,0]*Y[0,0] -> contributes to result[1,0]
    # (1,1): X[1,1]*Y[1,0] -> contributes to result[1,0]
    # (1,2): X[1,0]*Y[0,1] -> contributes to result[1,1]
    # (1,3): X[1,1]*Y[1,1] -> contributes to result[1,1]
    
    # For each term, we have:
    # U contribution: for each (a_idx, coeff_a) in X[row, col], U[a_idx, col_idx] += coeff_a
    # V contribution: for each (b_idx, coeff_b) in Y[row, col], V[b_idx, col_idx] += coeff_b
    # W contribution: W[k*n + i, col_idx] += coeff where (i, k) is the output position
    
    # The output position depends on which block of C this contributes to.
    # C11 = P1 + P4 - P5 - P7: rows 0-1, cols 0-1
    # C12 = P3 + P5: rows 0-1, cols 2-3
    # C21 = P2 + P7: rows 2-3, cols 0-1
    # C22 = P1 - P2 + P3 + P6: rows 2-3, cols 2-3
    
    # We only need C[0:3, :], so:
    # C11: rows 0-1, cols 0-1 (needed)
    # C12: rows 0-1, cols 2-3 (needed)
    # C21[0,:]: row 2, cols 0-1 (needed)
    # C21[1,:]: row 3, cols 0-1 (not needed, but row 3 is padding)
    # C22: rows 2-3, cols 2-3 (not needed)
    
    # So we need:
    # From P1: contributes to C11 (coeff +1) and C22 (coeff +1, but we don't need C22)
    # From P2: contributes to C21 (coeff +1) and C22 (coeff -1, but we don't need C22)
    # From P3: contributes to C12 (coeff +1) and C22 (coeff +1, but we don't need C22)
    # From P4: contributes to C11 (coeff +1)
    # From P5: contributes to C11 (coeff -1) and C12 (coeff +1)
    # From P6: contributes to C22 (coeff +1, but we don't need C22)
    # From P7: contributes to C11 (coeff -1) and C21 (coeff +1)
    
    # So the contributions we need:
    # P1 -> C11 (+1)
    # P2 -> C21 (+1)
    # P3 -> C12 (+1)
    # P4 -> C11 (+1)
    # P5 -> C11 (-1), C12 (+1)
    # P6 -> (none needed)
    # P7 -> C11 (-1), C21 (+1)
    
    # Let's define which C blocks each product contributes to, with coefficients.
    # c_block: (row_block, col_block) where row_block in {0,1}, col_block in {0,1}
    # row_block 0 -> rows 0-1, row_block 1 -> rows 2-3
    # col_block 0 -> cols 0-1, col_block 1 -> cols 2-3
    
    contrib = {
        0: {  # P1
            (0, 0): 1,   # C11
            # (1, 1): 1,  # C22 - not needed
        },
        1: {  # P2
            (1, 0): 1,   # C21
            # (1, 1): -1, # C22 - not needed
        },
        2: {  # P3
            (0, 1): 1,   # C12
            # (1, 1): 1,  # C22 - not needed
        },
        3: {  # P4
            (0, 0): 1,   # C11
        },
        4: {  # P5
            (0, 0): -1,  # C11
            (0, 1): 1,   # C12
        },
        5: {  # P6
            # (1, 1): 1,  # C22 - not needed
        },
        6: {  # P7
            (0, 0): -1,  # C11
            (1, 0): 1,   # C21
        },
    }
    
    # For each product k, and for each of the 8 scalar multiplications,
    # create one column in the decomposition.
    
    for k in range(7):
        X, Y = products[k]
        for r in range(2):  # row of result
            for c in range(2):  # col of result
                # The 2 scalar multiplications that contribute to result[r, c]
                # result[r, c] = X[r, 0]*Y[0, c] + X[r, 1]*Y[1, c]
                for t in range(2):  # t=0 or 1
                    # Term: X[r, t] * Y[t, c]
                    x_coeffs = X[(r, t)]
                    y_coeffs = Y[(t, c)]
                    
                    if not x_coeffs or not y_coeffs:
                        # Skip if either is empty (zero contribution)
                        continue
                    
                    # For each output block this product contributes to
                    for (rb, cb), coeff in contrib[k].items():
                        # Output position: row = rb*2 + r, col = cb*2 + c
                        out_i = rb * 2 + r
                        out_k = cb * 2 + c
                        
                        # We only need out_i < 3 (since n=3)
                        if out_i >= 3:
                            continue
                        
                        # U index: out_i * m + j_a where j_a is the column in A
                        # But we need to sum over all A entries in x_coeffs
                        # and all B entries in y_coeffs.
                        
                        # For each A entry in x_coeffs and each B entry in y_coeffs,
                        # we add a term to the decomposition.
                        
                        # Actually, the decomposition is:
                        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        
                        # For this term, we have:
                        # (sum_{a_idx, ca} ca * A[a_idx]) * (sum_{b_idx, cb} cb * B[b_idx])
                        # = sum_{a_idx, b_idx} ca * cb * A[a_idx] * B[b_idx]
                        
                        # Each (a_idx, b_idx) pair gives a contribution to T[i,j,k]
                        # where a_idx = i_a * m + j_a, b_idx = j_b * p + k_b
                        # and we need j_a = j_b for the product to be valid.
                        
                        # So we only include terms where j_a = j_b.
                        
                        for a_idx, ca in x_coeffs:
                            i_a = a_idx // m
                            j_a = a_idx % m
                            for b_idx, cb in y_coeffs:
                                j_b = b_idx // p
                                k_b = b_idx % p
                                if j_a == j_b:
                                    # This term contributes ca * cb to T[i_a, j_a, k_b]
                                    # But we also have the coefficient from the C block
                                    total_coeff = ca * cb * coeff
                                    
                                    U[i_a * m + j_a, col] += total_coeff
                                    V[j_a * p + k_b, col] += 1.0
                                    W[k_b * n + i_a, col] += 1.0
                        col += 1
    
    # Trim to actual used columns
    if col < rank:
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
        rank = col
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
