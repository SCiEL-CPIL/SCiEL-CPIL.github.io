"""Matrix multiplication 4x5x5 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 5
    nm, mp, pn = n * m, m * p, p * n  # 20, 25, 20

    # Strategy: Use Strassen's algorithm for the 4x4x4 sub-block (top-left 4x4 of A, 
    # top-left 4x4 of B, producing top-left 4x4 of C), and schoolbook for the remaining
    # border terms.
    #
    # A is 4x5, B is 5x5, C is 4x5.
    # Split A into A1 (4x4, cols 0-3) and A2 (4x1, col 4).
    # Split B into B1 (4x5, rows 0-3) and B2 (1x5, row 4).
    # B1 splits into B11 (4x4, cols 0-3) and B12 (4x1, col 4).
    # B2 is 1x5.
    #
    # C = A1*B1 + A1*B2 + A2*B1 + A2*B2
    #   = A1*(B11+B12) + A2*B1
    #   = A1*B11 + A1*B12 + A2*B11 + A2*B12 + A1*B2 + A2*B2
    #
    # Wait, let me be more careful.
    # A is 4x5: columns 0,1,2,3,4
    # B is 5x5: rows 0,1,2,3,4; columns 0,1,2,3,4
    #
    # Let's partition:
    # A = [A_left (4x4), A_right (4x1)]
    # B = [B_top (4x5); B_bottom (1x5)]
    # B_top = [B_tl (4x4), B_tr (4x1)]
    # B_bottom = [B_bl (1x4), B_br (1x1)]
    #
    # C = A_left * B_top + A_right * B_bottom
    #   = A_left * [B_tl, B_tr] + A_right * [B_bl, B_br]
    #
    # C[:, 0:4] = A_left * B_tl + A_right * B_bl
    # C[:, 4] = A_left * B_tr + A_right * B_br
    #
    # For C[:, 0:4]: This is a 4x4 times 4x4 plus 4x1 times 1x4.
    # The 4x4*4x4 part can use Strassen (rank 28 instead of 64).
    # The 4x1*1x4 part is outer product: rank 4 (each of 4 terms).
    #
    # For C[:, 4]: This is 4x1 * 4x1 + 4x1 * 1x1 = 4x1*4x1 + 4x1*1x1.
    # 4x1*4x1: This is like multiplying a 4-vector by a 4-vector to get... wait.
    # A_left is 4x4, B_tr is 4x1, so A_left*B_tr is 4x1. This is standard mat-vec: rank 16 (4*4*1).
    # A_right is 4x1, B_br is 1x1, so A_right*B_br is 4x1: rank 4 (each element is a product).
    #
    # Total rank estimate:
    # - Strassen for 4x4*4x4: rank 28
    # - 4x1 * 1x4 (outer product for C[:,0:4] border): rank 4
    # - 4x4 * 4x1 (mat-vec for C[:,4] from A_left*B_tr): rank 16
    # - 4x1 * 1x1 (for C[:,4] from A_right*B_br): rank 4
    # Total: 28 + 4 + 16 + 4 = 52
    #
    # Wait, I need to double-check the Strassen rank for 4x4*4x4.
    # Strassen for 2x2*2x2 has rank 7.
    # For 4x4*4x4, we can use block Strassen with 2x2 blocks:
    # 7 multiplications of 2x2*2x2, each with rank 4 (schoolbook) = 7*4 = 28.
    # Yes, rank 28 is achievable.
    
    # Let's build the decomposition step by step.
    
    U_dict = {}
    V_dict = {}
    W_dict = {}
    rank = 0
    
    def add_term(u_row, v_row, w_row, val=1):
        nonlocal rank
        U_dict[(u_row, rank)] = U_dict.get((u_row, rank), 0) + val
        V_dict[(v_row, rank)] = V_dict.get((v_row, rank), 0) + val
        W_dict[(w_row, rank)] = W_dict.get((w_row, rank), 0) + val
        rank += 1
    
    # Part 1: Strassen for A_left (4x4) * B_tl (4x4)
    # A_left[i,j] corresponds to U row i*5+j
    # B_tl[j,k] corresponds to V row j*5+k
    # C[i,k] corresponds to W row k*4+i
    
    # 2x2 block indices:
    # A00: A[0:2, 0:2] -> U rows 0*5+0, 0*5+1, 1*5+0, 1*5+1
    # A01: A[0:2, 2:4] -> U rows 0*5+2, 0*5+3, 1*5+2, 1*5+3
    # A10: A[2:4, 0:2] -> U rows 2*5+0, 2*5+1, 3*5+0, 3*5+1
    # A11: A[2:4, 2:4] -> U rows 2*5+2, 2*5+3, 3*5+2, 3*5+3
    
    # B00: B[0:2, 0:2] -> V rows 0*5+0, 0*5+1, 1*5+0, 1*5+1
    # B01: B[0:2, 2:4] -> V rows 0*5+2, 0*5+3, 1*5+2, 1*5+3
    # B10: B[2:4, 0:2] -> V rows 2*5+0, 2*5+1, 3*5+0, 3*5+1
    # B11: B[2:4, 2:4] -> V rows 2*5+2, 2*5+3, 3*5+2, 3*5+3
    
    # C00: C[0:2, 0:2] -> W rows 0*4+0, 0*4+1, 1*4+0, 1*4+1
    # C01: C[0:2, 2:4] -> W rows 2*4+0, 2*4+1, 3*4+0, 3*4+1
    # C10: C[2:4, 0:2] -> W rows 0*4+2, 0*4+3, 1*4+2, 1*4+3
    # C11: C[2:4, 2:4] -> W rows 2*4+2, 2*4+3, 3*4+2, 3*4+3
    
    # For each 2x2 product X*Y, I need to add 4 schoolbook terms.
    # X[i,k]*Y[k,j] contributes to C[i,j].
    
    def add_2x2_schoolbook(X_u_rows, Y_v_rows, C_w_rows, coeff=1):
        """
        X_u_rows: 2x2 list of U row indices for X
        Y_v_rows: 2x2 list of V row indices for Y
        C_w_rows: 2x2 list of W row indices for C
        """
        for i in range(2):
            for k in range(2):
                for j in range(2):
                    add_term(X_u_rows[i][k], Y_v_rows[k][j], C_w_rows[i][j], coeff)
    
    # Define the 2x2 blocks as 2x2 lists of indices
    def idx_A(i, j):
        return i * 5 + j
    
    def idx_B(j, k):
        return j * 5 + k
    
    def idx_C(i, k):
        return k * 4 + i  # W[k*n+i] per the convention
    
    A00_u = [[idx_A(0,0), idx_A(0,1)], [idx_A(1,0), idx_A(1,1)]]
    A01_u = [[idx_A(0,2), idx_A(0,3)], [idx_A(1,2), idx_A(1,3)]]
    A10_u = [[idx_A(2,0), idx_A(2,1)], [idx_A(3,0), idx_A(3,1)]]
    A11_u = [[idx_A(2,2), idx_A(2,3)], [idx_A(3,2), idx_A(3,3)]]
    
    B00_v = [[idx_B(0,0), idx_B(0,1)], [idx_B(1,0), idx_B(1,1)]]
    B01_v = [[idx_B(0,2), idx_B(0,3)], [idx_B(1,2), idx_B(1,3)]]
    B10_v = [[idx_B(2,0), idx_B(2,1)], [idx_B(3,0), idx_B(3,1)]]
    B11_v = [[idx_B(2,2), idx_B(2,3)], [idx_B(3,2), idx_B(3,3)]]
    
    C00_w = [[idx_C(0,0), idx_C(0,1)], [idx_C(1,0), idx_C(1,1)]]
    C01_w = [[idx_C(0,2), idx_C(0,3)], [idx_C(1,2), idx_C(1,3)]]
    C10_w = [[idx_C(2,0), idx_C(2,1)], [idx_C(3,0), idx_C(3,1)]]
    C11_w = [[idx_C(2,2), idx_C(2,3)], [idx_C(3,2), idx_C(3,3)]]
    
    # Strassen combinations:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # For each M, I need to add terms with appropriate coefficients.
    # Since we're using schoolbook for each 2x2 product, and the Strassen
    # combinations involve sums/differences of 2x2 matrices, I need to
    # expand these.
    
    # M1 = (A00+A11)*(B00+B11)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # Each of these is a 2x2*2x2 product, so 4 terms each = 16 terms for M1.
    # But wait, in Strassen, M1 is a single 2x2 matrix. The 4 schoolbook terms
    # compute C[i,j] = sum_k X[i,k]*Y[k,j].
    # Here X = A00+A11, Y = B00+B11.
    # So X[i,k] = A00[i,k] + A11[i,k], Y[k,j] = B00[k,j] + B11[k,j].
    # C[i,j] = sum_k (A00[i,k]+A11[i,k])*(B00[k,j]+B11[k,j])
    # = sum_k (A00[i,k]*B00[k,j] + A00[i,k]*B11[k,j] + A11[i,k]*B00[k,j] + A11[i,k]*B11[k,j])
    # = 4 products, each a 2x2*2x2 schoolbook.
    
    # So for M1, I add 4 schoolbook decompositions.
    add_2x2_schoolbook(A00_u, B00_v, C00_w, 1)
    add_2x2_schoolbook(A00_u, B11_v, C00_w, 1)
    add_2x2_schoolbook(A11_u, B00_v, C00_w, 1)
    add_2x2_schoolbook(A11_u, B11_v, C00_w, 1)
    # M1 contributes 16 terms.
    
    # M2 = (A10+A11)*B00
    # = A10*B00 + A11*B00
    add_2x2_schoolbook(A10_u, B00_v, C10_w, 1)
    add_2x2_schoolbook(A11_u, B00_v, C10_w, 1)
    # M2 contributes 8 terms.
    
    # M3 = A00*(B01-B11)
    # = A00*B01 - A00*B11
    add_2x2_schoolbook(A00_u, B01_v, C01_w, 1)
    add_2x2_schoolbook(A00_u, B11_v, C01_w, -1)
    # M3 contributes 8 terms.
    
    # M4 = A11*(B10-B00)
    # = A11*B10 - A11*B00
    add_2x2_schoolbook(A11_u, B10_v, C11_w, 1)
    add_2x2_schoolbook(A11_u, B00_v, C11_w, -1)
    # M4 contributes 8 terms.
    
    # M5 = (A00+A01)*B11
    # = A00*B11 + A01*B11
    add_2x2_schoolbook(A00_u, B11_v, C01_w, 1)
    add_2x2_schoolbook(A01_u, B11_v, C01_w, 1)
    # M5 contributes 8 terms.
    
    # M6 = (A10-A00)*(B00+B01)
    # = A10*B00 + A10*B01 - A00*B00 - A00*B01
    add_2x2_schoolbook(A10_u, B00_v, C10_w, 1)
    add_2x2_schoolbook(A10_u, B01_v, C10_w, 1)
    add_2x2_schoolbook(A00_u, B00_v, C10_w, -1)
    add_2x2_schoolbook(A00_u, B01_v, C10_w, -1)
    # M6 contributes 16 terms.
    
    # M7 = (A01-A11)*(B10+B11)
    # = A01*B10 + A01*B11 - A11*B10 - A11*B11
    add_2x2_schoolbook(A01_u, B10_v, C11_w, 1)
    add_2x2_schoolbook(A01_u, B11_v, C11_w, 1)
    add_2x2_schoolbook(A11_u, B10_v, C11_w, -1)
    add_2x2_schoolbook(A11_u, B11_v, C11_w, -1)
    # M7 contributes 16 terms.
    
    # Total for Strassen part: 16+8+8+8+8+16+16 = 80 terms.
    # Wait, that's more than 28! I think I'm double-counting.
    # 
    # Oh I see the issue. In Strassen, each M is a 2x2 matrix product.
    # The schoolbook for a 2x2*2x2 product has rank 4, not 16.
    # I was adding 4 schoolbook decompositions for M1, but each schoolbook
    # decomposition IS the 4 terms. So M1 should be 4 terms, not 16.
    #
    # Let me reconsider. For a 2x2*2x2 product X*Y:
    # C[0,0] = X[0,0]*Y[0,0] + X[0,1]*Y[1,0]
    # C[0,1] = X[0,0]*Y[0,1] + X[0,1]*Y[1,1]
    # C[1,0] = X[1,0]*Y[0,0] + X[1,1]*Y[1,0]
    # C[1,1] = X[1,0]*Y[0,1] + X[1,1]*Y[1,1]
    #
    # Schoolbook: 4 terms, one for each (i,k,j) combination... no wait.
    # Schoolbook for 2x2*2x2: sum over k of X[i,k]*Y[k,j].
    # For each (i,j), there are 2 terms (k=0,1). Total 4*(i,j) pairs * 2 = 8? No.
    # (i,j) has 4 combinations, each with 2 k values = 8 terms? That doesn't seem right.
    #
    # Actually, schoolbook for n x n * n x n has rank n^3.
    # For 2x2*2x2, rank = 8.
    # For 4x4*4x4, rank = 64.
    #
    # Strassen for 2x2*2x2 has rank 7.
    # For 4x4*4x4 using block Strassen with 2x2 blocks: 7 * rank(2x2*2x2) = 7*8 = 56?
    # No, that's not right either.
    #
    # Let me look this up mentally. Strassen reduces the number of multiplications.
    # For 2x2*2x2: 7 multiplications instead of 8. Each "multiplication" is of scalars.
    # So rank goes from 8 to 7.
    #
    # For 4x4*4x4 using Strassen recursively:
    # 7 multiplications of 2x2*2x2, each with rank 7 = 49.
    # Or 7 multiplications of 2x2*2x2, each with rank 8 (schoolbook) = 56.
    #
    # So rank(4x4*4x4) <= 49 using full Strassen, or 56 using one level.
    #
    # Let me use one level of Strassen for 4x4*4x4: 7 * 8 = 56 terms.
    # Plus the border terms.
    
    # Let me restart the construction more carefully.
    
    U_dict = {}
    V_dict = {}
    W_dict = {}
    rank = 0
    
    def add_term(u_row, v_row, w_row, val=1):
        nonlocal rank
        U_dict[(u_row, rank)] = U_dict.get((u_row, rank), 0) + val
        V_dict[(v_row, rank)] = V_dict.get((v_row, rank), 0) + val
        W_dict[(w_row, rank)] = W_dict.get((w_row, rank), 0) + val
        rank += 1
    
    def add_2x2_schoolbook(X_u, Y_v, C_w, coeff=1):
        """Add 8 schoolbook terms for a 2x2*2x2 product."""
        for i in range(2):
            for k in range(2):
                for j in range(2):
                    add_term(X_u[i][k], Y_v[k][j], C_w[i][j], coeff)
    
    # Strassen for 4x4*4x4 (A_left * B_tl):
    # 7 multiplications, each a 2x2*2x2 schoolbook (8 terms each) = 56 terms.
    
    # M1 = (A00+A11)*(B00+B11)
    # This is a 2x2*2x2 product where X=A00+A11, Y=B00+B11.
    # X[i,k] = A00[i,k] + A11[i,k], Y[k,j] = B00[k,j] + B11[k,j].
    # We can't directly add this as a single schoolbook because X and Y are sums.
    # We need to expand: X*Y = A00*B00 + A00*B11 + A11*B00 + A11*B11.
    # Each is a 2x2*2x2 schoolbook (8 terms). Total 32 terms for M1.
    #
    # Hmm, this is getting complex. Let me just use schoolbook for the entire
    # 4x4*4x4 part to keep things simple, then add the border terms.
    # Total rank would be 64 + border terms.
    
    # Actually, let me just do full schoolbook for simplicity and correctness.
    # Rank = 100. Then I'll try to optimize.
    
    # For now, let's return the schoolbook decomposition to ensure correctness.
    # The Strassen approach needs more careful implementation.
    
    U_dict = {}
    V_dict = {}
    W_dict = {}
    rank = 0
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                add_term(i * m + j, j * p + k, k * n + i, 1)
    
    # Convert dictionaries to arrays
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    for (row, col), val in U_dict.items():
        U[row, col] = val
    for (row, col), val in V_dict.items():
        V[row, col] = val
    for (row, col), val in W_dict.items():
        W[row, col] = val
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
