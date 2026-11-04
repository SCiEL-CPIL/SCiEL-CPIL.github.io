"""Matrix multiplication 2,4,4 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 2, 4, 4
    
    # We'll use a block-based approach with Strassen for 2x2 blocks.
    # A is 2x4, B is 4x4, C is 2x4.
    # Split A into two 2x2 blocks: A = [A_left, A_right] where A_left is columns 0,1 and A_right is columns 2,3
    # Split B into four 2x2 blocks: B = [[B00, B01], [B10, B11]]
    # C = A * B = [A_left, A_right] * [[B00, B01], [B10, B11]]
    # C = [A_left*B00 + A_right*B10, A_left*B01 + A_right*B11]
    # So C_left = A_left*B00 + A_right*B10 (2x2)
    #    C_right = A_left*B01 + A_right*B11 (2x2)
    
    # Each 2x2 = 2x2 * 2x2 product can be done with rank 7 (Strassen).
    # We have 4 such products: A_left*B00, A_right*B10, A_left*B01, A_right*B11
    # Total rank = 4 * 7 = 28.
    
    # Strassen's algorithm for 2x2 matrix multiplication:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each 2x2 product, we need 7 terms.
    # Let's define the Strassen coefficients for a generic 2x2 product X*Y = Z:
    # Each term r has (x_row, y_col, z_row, z_col, coeff) entries.
    
    # Strassen terms for X*Y=Z where X, Y, Z are 2x2:
    # Term 0: M1 = (X00+X11)(Y00+Y11)
    # Term 1: M2 = (X10+X11)Y00
    # Term 2: M3 = X00(Y10-Y11)
    # Term 3: M4 = X11(Y01-Y00)
    # Term 4: M5 = (X00+X01)Y11
    # Term 5: M6 = (X10-X00)(Y00+Y01)
    # Term 6: M7 = (X01-X11)(Y10+Y11)
    
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # Let's define the factor matrices for one 2x2 Strassen product.
    # For a product X*Y=Z, the factors are:
    # U factors (for X): shape (4, 7) where row idx = x_row*2+x_col
    # V factors (for Y): shape (4, 7) where row idx = y_row*2+y_col
    # W factors (for Z): shape (4, 7) where row idx = z_col*2+z_row (note: output index convention)
    
    # Actually, let me be more careful with the indexing.
    # In the overall problem:
    # A has shape (n, m) = (2, 4), indexed by A[i, j] with i in [0,1], j in [0,3]
    # B has shape (m, p) = (4, 4), indexed by B[j, k] with j in [0,3], k in [0,3]
    # C has shape (n, p) = (2, 4), indexed by C[i, k] with i in [0,1], k in [0,3]
    
    # The factor matrices are:
    # U: shape (n*m, rank) = (8, rank), indexed by U[i*m+j, r] = U[i*4+j, r]
    # V: shape (m*p, rank) = (16, rank), indexed by V[j*p+k, r] = V[j*4+k, r]
    # W: shape (p*n, rank) = (8, rank), indexed by W[k*n+i, r] = W[k*2+i, r]
    
    # For the block decomposition:
    # A_left = A[:, 0:2] = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # A_right = A[:, 2:4] = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]]
    # B00 = B[0:2, 0:2], B01 = B[0:2, 2:4], B10 = B[2:4, 0:2], B11 = B[2:4, 2:4]
    # C_left = C[:, 0:2] = [[C[0,0], C[0,1]], [C[1,0], C[1,1]]]
    # C_right = C[:, 2:4] = [[C[0,2], C[0,3]], [C[1,2], C[1,3]]]
    
    # C_left = A_left * B00 + A_right * B10
    # C_right = A_left * B01 + A_right * B11
    
    # For each of the 4 products, we use Strassen with 7 terms.
    
    # Let's define a helper to get Strassen factors for a 2x2 product.
    # The product is X * Y = Z where X, Y, Z are 2x2.
    # X is indexed by (x_row, x_col) -> flat index x_row*2+x_col
    # Y is indexed by (y_row, y_col) -> flat index y_row*2+y_col
    # Z is indexed by (z_row, z_col) -> flat index z_col*2+z_row (output convention)
    
    # Strassen factors for one 2x2 product:
    def strassen_2x2_factors():
        """
        Returns U, V, W factors for a 2x2 Strassen product.
        U: (4, 7) for X
        V: (4, 7) for Y
        W: (4, 7) for Z
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Term 0: M1 = (X00+X11)(Y00+Y11)
        U[0, 0] = 1  # X00
        U[3, 0] = 1  # X11
        V[0, 0] = 1  # Y00
        V[3, 0] = 1  # Y11
        # Z00 += M1, Z11 += M1
        W[0*2+0, 0] = 1  # Z00 (z_col=0, z_row=0)
        W[1*2+1, 0] = 1  # Z11 (z_col=1, z_row=1)
        
        # Term 1: M2 = (X10+X11)Y00
        U[2, 1] = 1  # X10
        U[3, 1] = 1  # X11
        V[0, 1] = 1  # Y00
        # Z10 += M2, Z11 -= M2
        W[0*2+1, 1] = 1  # Z10 (z_col=0, z_row=1)
        W[1*2+1, 1] = -1  # Z11 (z_col=1, z_row=1)
        
        # Term 2: M3 = X00(Y10-Y11)
        U[0, 2] = 1  # X00
        V[2, 2] = 1  # Y10
        V[3, 2] = -1  # Y11
        # Z01 += M3, Z11 += M3
        W[1*2+0, 2] = 1  # Z01 (z_col=1, z_row=0)
        W[1*2+1, 2] = 1  # Z11 (z_col=1, z_row=1)
        
        # Term 3: M4 = X11(Y01-Y00)
        U[3, 3] = 1  # X11
        V[1, 3] = 1  # Y01
        V[0, 3] = -1  # Y00
        # Z00 += M4, Z10 += M4
        W[0*2+0, 3] = 1  # Z00
        W[0*2+1, 3] = 1  # Z10
        
        # Term 4: M5 = (X00+X01)Y11
        U[0, 4] = 1  # X00
        U[1, 4] = 1  # X01
        V[3, 4] = 1  # Y11
        # Z00 -= M5, Z01 += M5
        W[0*2+0, 4] = -1  # Z00
        W[1*2+0, 4] = 1  # Z01
        
        # Term 5: M6 = (X10-X00)(Y00+Y01)
        U[2, 5] = 1  # X10
        U[0, 5] = -1  # X00
        V[0, 5] = 1  # Y00
        V[1, 5] = 1  # Y01
        # Z11 += M6
        W[1*2+1, 5] = 1  # Z11
        
        # Term 6: M7 = (X01-X11)(Y10+Y11)
        U[1, 6] = 1  # X01
        U[3, 6] = -1  # X11
        V[2, 6] = 1  # Y10
        V[3, 6] = 1  # Y11
        # Z00 += M7
        W[0*2+0, 6] = 1  # Z00
        
        return U, V, W
    
    # Get the Strassen factors
    S_U, S_V, S_W = strassen_2x2_factors()
    
    # Now build the full factor matrices for the 2x4x4 problem.
    # We have 4 products, each with 7 Strassen terms.
    rank = 4 * 7
    U = np.zeros((8, rank))
    V = np.zeros((16, rank))
    W = np.zeros((8, rank))
    
    # Product 1: A_left * B00 -> contributes to C_left
    # A_left indices: A[0,0], A[0,1], A[1,0], A[1,1] -> flat: 0, 1, 4, 5
    # B00 indices: B[0,0], B[0,1], B[1,0], B[1,1] -> flat: 0, 1, 4, 5
    # C_left indices: C[0,0], C[0,1], C[1,0], C[1,1] -> flat: 0, 2, 1, 3 (W[k*n+i])
    
    # Mapping from local 2x2 indices to global indices:
    # A_left: local (r,c) -> global (r, c) -> flat r*4+c
    # B00: local (r,c) -> global (r, c) -> flat r*4+c
    # C_left: local (r,c) -> global (r, c) -> flat c*2+r
    
    for t in range(7):
        col = 0 * 7 + t
        # U factors
        for local_idx in range(4):
            val = S_U[local_idx, t]
            if val != 0:
                lr, lc = divmod(local_idx, 2)
                global_idx = lr * 4 + lc
                U[global_idx, col] += val
        # V factors
        for local_idx in range(4):
            val = S_V[local_idx, t]
            if val != 0:
                lr, lc = divmod(local_idx, 2)
                global_idx = lr * 4 + lc
                V[global_idx, col] += val
        # W factors
        for local_idx in range(4):
            val = S_W[local_idx, t]
            if val != 0:
                # local_idx = z_col*2 + z_row
                z_col, z_row = divmod(local_idx, 2)
                global_idx = z_col * 2 + z_row
                W[global_idx, col] += val
    
    # Product 2: A_right * B10 -> contributes to C_left
    # A_right indices: A[0,2], A[0,3], A[1,2], A[1,3] -> flat: 2, 3, 6, 7
    # B10 indices: B[2,0], B[2,1], B[3,0], B[3,1] -> flat: 8, 9, 12, 13
    # C_left indices: same as above
    
    for t in range(7):
        col = 1 * 7 + t
        # U factors
        for local_idx in range(4):
            val = S_U[local_idx, t]
            if val != 0:
                lr, lc = divmod(local_idx, 2)
                global_idx = lr * 4 + (2 + lc)
                U[global_idx, col] += val
        # V factors
        for local_idx in range(4):
            val = S_V[local_idx, t]
            if val != 0:
                lr, lc = divmod(local_idx, 2)
                global_idx = (2 + lr) * 4 + lc
                V[global_idx, col] += val
        # W factors
        for local_idx in range(4):
            val = S_W[local_idx, t]
            if val != 0:
                z_col, z_row = divmod(local_idx, 2)
                global_idx = z_col * 2 + z_row
                W[global_idx, col] += val
    
    # Product 3: A_left * B01 -> contributes to C_right
    # A_left indices: same as product 1
    # B01 indices: B[0,2], B[0,3], B[1,2], B[1,3] -> flat: 2, 3, 6, 7
    # C_right indices: C[0,2], C[0,3], C[1,2], C[1,3] -> flat: 4, 6, 5, 7
    
    for t in range(7):
        col = 2 * 7 + t
        # U factors
        for local_idx in range(4):
            val = S_U[local_idx, t]
            if val != 0:
                lr, lc = divmod(local_idx, 2)
                global_idx = lr * 4 + lc
                U[global_idx, col] += val
        # V factors
        for local_idx in range(4):
            val = S_V[local_idx, t]
            if val != 0:
                lr, lc = divmod(local_idx, 2)
                global_idx = lr * 4 + (2 + lc)
                V[global_idx, col] += val
        # W factors
        for local_idx in range(4):
            val = S_W[local_idx, t]
            if val != 0:
                z_col, z_row = divmod(local_idx, 2)
                global_idx = (2 + z_col) * 2 + z_row
                W[global_idx, col] += val
    
    # Product 4: A_right * B11 -> contributes to C_right
    # A_right indices: same as product 2
    # B11 indices: B[2,2], B[2,3], B[3,2], B[3,3] -> flat: 10, 11, 14, 15
    # C_right indices: same as product 3
    
    for t in range(7):
        col = 3 * 7 + t
        # U factors
        for local_idx in range(4):
            val = S_U[local_idx, t]
            if val != 0:
                lr, lc = divmod(local_idx, 2)
                global_idx = lr * 4 + (2 + lc)
                U[global_idx, col] += val
        # V factors
        for local_idx in range(4):
            val = S_V[local_idx, t]
            if val != 0:
                lr, lc = divmod(local_idx, 2)
                global_idx = (2 + lr) * 4 + (2 + lc)
                V[global_idx, col] += val
        # W factors
        for local_idx in range(4):
            val = S_W[local_idx, t]
            if val != 0:
                z_col, z_row = divmod(local_idx, 2)
                global_idx = (2 + z_col) * 2 + z_row
                W[global_idx, col] += val
    
    # Verify the decomposition
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Should equal 1 iff a=i*4+j, b=j*4+k, c=k*2+i for some i,j,k
    
    # Let's verify numerically
    T_decomposed = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Build the expected tensor
    T_expected = np.zeros((8, 16, 8))
    for i in range(2):
        for j in range(4):
            for k in range(4):
                a_idx = i * 4 + j
                b_idx = j * 4 + k
                c_idx = k * 2 + i
                T_expected[a_idx, b_idx, c_idx] = 1
    
    # Check if they match
    if not np.allclose(T_decomposed, T_expected, atol=1e-10):
        # If verification fails, fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
