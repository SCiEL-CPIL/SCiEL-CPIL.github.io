import time
import sys

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 blocks where each block is 1x2 (A), 2x2 (B), 1x2 (C)
    # A: 2 rows, 4 cols -> 2 blocks of 1x2: A11, A12 (row 0), A21, A22 (row 1)
    # B: 4 rows, 4 cols -> 4 blocks of 2x2: B11, B12, B21, B22
    # C: 2 rows, 4 cols -> 2 blocks of 1x2: C11, C12 (row 0), C21, C22 (row 1)
    
    # M_s = A_s * B_s where A_s is 1x2 and B_s is 2x2, giving M_s which is 1x2
    # C_ij = sum of M_s contributions
    
    # Strassen-style decomposition:
    # We need to compute C = A * B where A is 2x4, B is 4x4
    # Split: A = [A1; A2] where A1, A2 are 1x4
    # B = [[B11, B12], [B21, B22]] where each Bij is 2x2
    # C = [A1*B11 + A1*B12; A2*B21 + A2*B22] ... wait let me reconsider
    
    # Actually for 2x4 * 4x4:
    # A is 2x4, B is 4x4, C is 2x4
    # Let's split A into two 1x4 row vectors: A = [A_top; A_bottom]
    # Split B into four 2x2 blocks: B = [[B11, B12], [B21, B22]]
    # Then C_top = A_top * B where A_top is 1x4, B is 4x4, C_top is 1x4
    # C_bottom = A_bottom * B similarly
    
    # To use Strassen, we need to exploit the block structure differently.
    # Let me think of this as: split the 4 columns of A into two groups of 2
    # A = [A11 A12] where A11, A12 are 2x2
    # B = [[B11, B12], [B21, B22]] where each is 2x2
    # C = A11*B11 + A12*B21 for first 2 cols, A11*B12 + A12*B22 for last 2 cols
    # C = [A11*B11 + A12*B21, A11*B12 + A12*B22]
    
    # Each product is 2x2 * 2x2 = 2x2
    # We need to compute 4 such products: A11*B11, A12*B21, A11*B12, A12*B22
    # But A11 and A12 are both 2x2, B11,B12,B21,B22 are all 2x2
    
    # Standard Strassen for 2x2 * 2x2 uses 7 multiplications
    # Here we have 4 matrix multiplications of 2x2 matrices
    # Can we share some computations?
    
    # Let me define:
    # A11, A12: 2x2 blocks from A (cols 0-1, cols 2-3)
    # B11, B12, B21, B22: 2x2 blocks from B (rows 0-1/2-3, cols 0-1/2-3)
    
    # C_left = A11*B11 + A12*B21 (2x2 result)
    # C_right = A11*B12 + A12*B22 (2x2 result)
    
    # For C_left: two 2x2 matrix multiplications added
    # For C_right: two 2x2 matrix multiplications added
    
    # Each 2x2 matrix multiplication can be done in 7 scalar mults (Strassen)
    # But we have 4 matrix multiplications total, so naively 4*7 = 28
    
    # That's exactly rank 28! So the decomposition is:
    # For each of the 4 matrix products, use Strassen's 7 multiplications
    
    # Let me implement Strassen for each 2x2 matrix product
    
    # Strassen for X*Y where X, Y are 2x2:
    # M1 = (X11+X22)(Y11+Y22)
    # M2 = (X21+X22)Y11
    # M3 = X11(Y12-Y22)
    # M4 = X22(Y21-Y11)
    # M5 = (X11+X12)Y22
    # M6 = (X21-X11)(Y11+Y12)
    # M7 = (X12-X22)(Y21+Y22)
    # Z11 = M1+M4-M5+M7
    # Z12 = M3+M5
    # Z21 = M2+M4
    # Z22 = M1-M2+M3+M6
    
    # For C_left = A11*B11 + A12*B21:
    # This is a sum of two 2x2 matrix products. Each requires 7 mults, so 14 total.
    
    # For C_right = A11*B12 + A12*B22:
    # Similarly 14 mults.
    
    # Total: 28 mults. Rank = 28.
    
    # Now let me code this up properly.
    
    # A indexing: A is 2x4. Row i, col j -> A[i*m + j] where m=4
    # A11: A[0*4+0], A[0*4+1], A[1*4+0], A[1*4+1] = A[0], A[1], A[4], A[5]
    # A12: A[0*4+2], A[0*4+3], A[1*4+2], A[1*4+3] = A[2], A[3], A[6], A[7]
    
    # B indexing: B is 4x4. Row i, col j -> B[i*p + j] where p=4
    # B11: B[0], B[1], B[4], B[5] (rows 0-1, cols 0-1)
    # B12: B[2], B[3], B[6], B[7] (rows 0-1, cols 2-3)
    # B21: B[8], B[9], B[12], B[13] (rows 2-3, cols 0-1)
    # B22: B[10], B[11], B[14], B[15] (rows 2-3, cols 2-3)
    
    # C indexing: C is 2x4. C[k*n+i, r] per the trace convention W[k*n+i, r]
    # Wait, the problem says W[k*n+i, r] where the output is C[i][k]
    # So C[i][k] = sum_r U[i*m+j... let me re-read.
    
    # The decomposition is: C[i][k] = sum_r U[i][r] * (sum_j A[j] ... )
    # Actually the standard form: C = U^T * M * V^T where M_r = (U_col_r . A) * (V_col_r . B)
    # More precisely: C[i][k] = sum_r U[i][r] * M_r * V[k][r]
    # where M_r = sum_{i',j'} U[i'][r] * V[j'][r] * A[i'][j'] ... no.
    
    # The tensor decomposition for matrix multiplication:
    # C[i][k] = sum_{r=1}^{rank} U[i*m + something]... 
    
    # Let me use the convention: 
    # U has shape (n*m, rank) = (8, 28)
    # V has shape (m*p, rank) = (16, 28)  
    # W has shape (p*n, rank) = (8, 28)
    
    # The multiplication tensor: T[i,j,k] = delta where i in [0,n), j in [0,m), k in [0,p)
    # C[i][k] = sum_j A[i][j] * B[j][k]
    
    # Decomposition: C[i][k] = sum_r U[i*m + j_a][r] * M_r * V[j_b*p + k][r] ... 
    
    # Actually the standard formulation:
    # Define flattened indices: a_idx = i*m + j for A (i in 0..n-1, j in 0..m-1)
    # b_idx = j*p + k for B (j in 0..m-1, k in 0..p-1)
    # c_idx = k*n + i for C (k in 0..p-1, i in 0..n-1)  [per the W[k*n+i,r] convention]
    
    # The tensor T[a_idx, b_idx, c_idx] = 1 if the indices are compatible (j_a = j_b)
    # i.e., T[(i,m+j), (j,p+k), (k,n+i)] = 1
    
    # Decomposition: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # So C[i][k] = sum_j A[i][j]*B[j][k] = sum_r (sum_j U[i*m+j, r] * A[i][j]) * (sum_j V[j*p+k, r] * B[j][k]) * W[k*n+i, r]
    
    # Hmm, that's not quite right either. Let me think more carefully.
    
    # For the standard matrix multiplication tensor decomposition:
    # T[i,j,k] = 1 (for the "multiplication" part)
    # The bilinear form: C[i,k] = sum_j A[i,j] B[j,k]
    
    # In tensor decomposition form:
    # C[i,k] = sum_r U[i, r] * (V[r] · A_flat) * (W[r] · B_flat) ... no.
    
    # Actually the correct formulation for the rank of matrix multiplication:
    # We want to write the multiplication map as a sum of rank-1 terms.
    # C = sum_r U_r * M_r * V_r^T where M_r is a scalar (the r-th "multiplication")
    
    # More precisely, if we define:
    # U: (n*p) x rank -- wait no.
    
    # Let me just follow the interface:
    # U: (n*m, rank) = (8, 28)
    # V: (m*p, rank) = (16, 28)
    # W: (p*n, rank) = (8, 28)
    
    # The reconstruction: for each r, compute:
    # X_r = U[:, r] · A_flattened (a scalar)
    # Y_r = V[:, r] · B_flattened (a scalar)
    # C += W[:, r] * (X_r * Y_r)
    
    # So C[k*n+i] += W[k*n+i, r] * (U[i*m+j, r] * A[i,j] summed) * (V[j*p+k, r] * B[j,k] summed)
    
    # For this to give C[i,k] = sum_j A[i,j]*B[j,k], we need:
    # sum_r W[k*n+i, r] * (sum_{i',j'} U[i'*m+j', r] * A[i',j']) * (sum_{j',k'} V[j'*p+k', r] * B[j',k'])
    # to equal sum_j A[i,j]*B[j,k]
    
    # This is a trilinear form decomposition. The coefficients must be such that
    # when we contract, we get the right result.
    
    # For Strassen applied to 2x2 blocks, each "multiplication" M_s = (linear combo of A entries) * (linear combo of B entries)
    # and C entries are linear combos of the M_s.
    
    # So: U[i*m+j, s] gives the coefficient of A[i,j] in the s-th A-factor
    # V[j*p+k, s] gives the coefficient of B[j,k] in the s-th B-factor
    # W[k*n+i, s] gives the coefficient of M_s in C[i,k] (with C indexed as k*n+i)
    
    # Let me redo the Strassen construction for each of the 4 sub-products.
    
    # Sub-product 1: P1 = A11 * B11 (2x2 * 2x2 = 2x2), contributes to C_left (first 2 cols of C)
    # Sub-product 2: P2 = A12 * B21 (2x2 * 2x2 = 2x2), contributes to C_left
    # Sub-product 3: P3 = A11 * B12 (2x2 * 2x2 = 2x2), contributes to C_right (last 2 cols of C)
    # Sub-product 4: P4 = A12 * B22 (2x2 * 2x2 = 2x2), contributes to C_right
    
    # C[:, 0:2] = P1 + P2
    # C[:, 2:4] = P3 + P4
    
    # For each 2x2 matrix product X*Y, Strassen uses 7 multiplications.
    # I'll assign 7 rank indices for each sub-product.
    
    # Helper: add Strassen decomposition for X*Y contributing to C_block
    # X is a 2x2 block of A, Y is a 2x2 block of B
    # X entries: A[xr, xc] where xr in {xrow0, xrow0+1}, xc in {xcol0, xcol0+1}
    # Y entries: B[yr, yc] where yr in {yrow0, yrow0+1}, yc in {ycol0, ycol0+1}
    # C_block entries: C[cr, cc] where cr in {crow0, crow0+1}, cc in {ccol0, ccol0+1}
    # C indexed as C[k*n+i] = C[col*n + row]
    
    def add_strassen(A_row0, A_col0, B_row0, B_col0, C_row0, C_col0, start_r):
        # A11 = A[A_row0, A_col0], A12 = A[A_row0, A_col0+1], A21 = A[A_row0+1, A_col0], A22 = A[A_row0+1, A_col0+1]
        # B11 = B[B_row0, B_col0], B12 = B[B_row0, B_col0+1], B21 = B[B_row0+1, B_col0], B22 = B[B_row0+1, B_col0+1]
        # C11 = C[C_row0, C_col0], C12 = C[C_row0, C_col0+1], C21 = C[C_row0+1, C_col0], C22 = C[C_row0+1, C_col0+1]
        
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
        
        a11 = A_row0 * m + A_col0
        a12 = A_row0 * m + A_col0 + 1
        a21 = (A_row0 + 1) * m + A_col0
        a22 = (A_row0 + 1) * m + A_col0 + 1
        
        b11 = B_row0 * p + B_col0
        b12 = B_row0 * p + B_col0 + 1
        b21 = (B_row0 + 1) * p + B_col0
        b22 = (B_row0 + 1) * p + B_col0 + 1
        
        c11_idx = C_col0 * n + C_row0
        c12_idx = (C_col0 + 1) * n + C_row0
        c21_idx = C_col0 * n + (C_row0 + 1)
        c22_idx = (C_col0 + 1) * n + (C_row0 + 1)
        
        r = start_r
        
        # M1
        U[a11][r] += 1
        U[a22][r] += 1
        V[b11][r] += 1
        V[b22][r] += 1
        W[c11_idx][r] += 1
        W[c22_idx][r] += 1
        r += 1
        
        # M2
        U[a21][r] += 1
        U[a22][r] += 1
        V[b11][r] += 1
        W[c21_idx][r] += 1
        W[c22_idx][r] += -1
        r += 1
        
        # M3
        U[a11][r] += 1
        V[b12][r] += 1
        V[b22][r] += -1
        W[c12_idx][r] += 1
        W[c22_idx][r] += 1
        r += 1
        
        # M4
        U[a22][r] += 1
        V[b21][r] += 1
        V[b11][r] += -1
        W[c11_idx][r] += 1
        W[c21_idx][r] += 1
        r += 1
        
        # M5
        U[a11][r] += 1
        U[a12][r] += 1
        V[b22][r] += 1
        W[c11_idx][r] += -1
        W[c12_idx][r] += 1
        r += 1
        
        # M6
        U[a21][r] += 1
        U[a11][r] += -1
        V[b11][r] += 1
        V[b12][r] += 1
        W[c22_idx][r] += 1
        r += 1
        
        # M7
        U[a12][r] += 1
        U[a22][r] += -1
        V[b21][r] += 1
        V[b22][r] += 1
        W[c11_idx][r] += 1
        r += 1
        
        return r
    
    # Sub-product 1: A11 * B11 -> C[:, 0:2]
    # A11: rows 0-1, cols 0-1 of A
    # B11: rows 0-1, cols 0-1 of B
    # C: rows 0-1, cols 0-1
    r = add_strassen(0, 0, 0, 0, 0, 0, 0)
    
    # Sub-product 2: A12 * B21 -> C[:, 0:2]
    # A12: rows 0-1, cols 2-3 of A
    # B21: rows 2-3, cols 0-1 of B
    # C: rows 0-1, cols 0-1
    r = add_strassen(0, 2, 2, 0, 0, 0, r)
    
    # Sub-product 3: A11 * B12 -> C[:, 2:4]
    # A11: rows 0-1, cols 0-1 of A
    # B12: rows 0-1, cols 2-3 of B
    # C: rows 0-1, cols 2-3
    r = add_strassen(0, 0, 0, 2, 0, 2, r)
    
    # Sub-product 4: A12 * B22 -> C[:, 2:4]
    # A12: rows 0-1, cols 2-3 of A
    # B22: rows 2-3, cols 2-3 of B
    # C: rows 0-1, cols 2-3
    r = add_strassen(0, 2, 2, 2, 0, 2, r)
    
    assert r == 28, f"Expected rank 28, got {r}"
    
    return {"U": U, "V": V, "W": W}
