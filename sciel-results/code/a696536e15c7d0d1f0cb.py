import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130
    
    def schoolbook():
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        for k2 in range(p):
                            for i2 in range(n):
                                R = 0.0
                                for t in range(rank):
                                    R += U[i*m+j, t] * V[j2*p+k, t] * W[k2*n+i2, t]
                                expected = 1.0 if (i==i2 and j==j2 and k==k2) else 0.0
                                if abs(R - expected) > tol:
                                    return False
        return True
    
    # Strategy: Use block structure.
    # A is 2x4, B is 4x4.
    # Split A into two 2x2 blocks: A = [A_top; A_bottom] where each is 2x2
    # Split B into four 2x2 blocks: B = [[B11, B12], [B21, B22]]
    # C = A*B = [A_top * B11 + A_bottom * B21, A_top * B12 + A_bottom * B22]
    # 
    # For each 2x2 * 2x2 multiplication, we can use Strassen (rank 7)
    # instead of schoolbook (rank 4).
    #
    # We have 4 such block multiplications:
    # C_top = A_top * B11 + A_bottom * B21  (two 2x2*2x2 products added)
    # C_bot = A_top * B12 + A_bottom * B22  (two 2x2*2x2 products added)
    #
    # Wait, that's not quite right. Let me reconsider.
    #
    # A is 2x4: rows 0,1; cols 0,1,2,3
    # B is 4x4: rows 0,1,2,3; cols 0,1,2,3
    # C is 2x4: rows 0,1; cols 0,1,2,3
    #
    # C[0,:] = A[0,:] * B = A[0,0]*B[0,:] + A[0,1]*B[1,:] + A[0,2]*B[2,:] + A[0,3]*B[3,:]
    # C[1,:] = A[1,:] * B = A[1,0]*B[0,:] + A[1,1]*B[1,:] + A[1,2]*B[2,:] + A[1,3]*B[3,:]
    #
    # Let's group:
    # C[0,:] = A[0,0:2] * B[0:2,:] + A[0,2:4] * B[2:4,:]
    # C[1,:] = A[1,0:2] * B[0:2,:] + A[1,2:4] * B[2:4,:]
    #
    # So we have 4 block multiplications of size 2x2 * 2x2:
    # P1 = A[0,0:2] * B[0:2,:]   (2x2 * 2x4 -> 2x4)
    # P2 = A[0,2:4] * B[2:4,:]   (2x2 * 2x4 -> 2x4)
    # P3 = A[1,0:2] * B[0:2,:]   (2x2 * 2x4 -> 2x4)
    # P4 = A[1,2:4] * B[2:4,:]   (2x2 * 2x4 -> 2x4)
    #
    # C[0,:] = P1 + P2
    # C[1,:] = P3 + P4
    #
    # Each P is a 2x2 * 2x4 multiplication.
    # A 2x2 * 2x4 can be seen as: left 2x2, right 2x4 (which is two 2x2 blocks side by side).
    # Actually, 2x2 * 2x4: the result is 2x4.
    # We can split the right operand into two 2x2 blocks:
    # P = L * R where L is 2x2, R is 2x4 = [R_left | R_right]
    # P = [L * R_left | L * R_right]
    # So P requires two 2x2 * 2x2 multiplications.
    #
    # Total: 4 * 2 = 8 scalar 2x2*2x2 multiplications.
    # With Strassen, each 2x2*2x2 costs 7 bilinear operations.
    # Total rank = 8 * 7 = 56.
    #
    # But wait, schoolbook for 2x4x4 is 2*4*4 = 32.
    # So 56 > 32. That's worse!
    #
    # Let me reconsider. The tensor rank of 2x4x4 multiplication.
    # Schoolbook gives rank 32.
    # 
    # Alternative: Think of it as 4 independent 2x1 * 1x4? No.
    #
    # Let me try a different block decomposition.
    # A is 2x4, B is 4x4.
    # Split A into 2x2 blocks: A = [A1 | A2] where A1, A2 are 2x2.
    # Split B into 2x2 blocks: B = [[B11, B12], [B21, B22]]
    # C = A*B = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    #
    # C is 2x4 = [C_left | C_right] where C_left, C_right are 2x2.
    # C_left = A1*B11 + A2*B21
    # C_right = A1*B12 + A2*B22
    #
    # So we need to compute 4 products of 2x2 matrices:
    # A1*B11, A2*B21, A1*B12, A2*B22
    # Then add pairs.
    #
    # Each 2x2*2x2 has tensor rank 7 (Strassen) or 4 (schoolbook).
    # With schoolbook: 4 * 4 = 16. Plus additions are free in tensor decomposition.
    # Wait, no. In tensor decomposition, additions are handled by the linearity.
    #
    # Actually, the tensor for C_left = A1*B11 + A2*B21 involves:
    # C_left[i,j] = sum_k A1[i,k]*B11[k,j] + sum_k A2[i,k]*B21[k,j]
    #
    # This is a sum of two 2x2*2x2 tensors. The rank of a sum is at most the sum of ranks.
    # So rank(C_left tensor) <= rank(A1*B11 tensor) + rank(A2*B21 tensor) <= 7 + 7 = 14.
    # Similarly for C_right.
    # Total rank <= 14 + 14 = 28.
    #
    # But can we do better by sharing? The four 2x2*2x2 products are independent
    # (different inputs), so no sharing is possible in the straightforward approach.
    #
    # However, 28 < 32, so this is an improvement!
    #
    # Let me construct this explicitly.
    
    def strassen_2x2():
        """
        Return the Strassen decomposition for 2x2 matrix multiplication.
        For matrices X, Y (each 2x2), compute Z = X*Y.
        Z[0,0] = M1 - M4 + M6 + M7
        Z[0,1] = M2 + M4
        Z[1,0] = M3 + M5
        Z[1,1] = M1 - M2 + M5 + M8
        
        where:
        M1 = (X00+X11)*(Y00+Y11)
        M2 = (X10+X11)*Y00
        M3 = X00*(Y01-Y11)
        M4 = X11*(Y10-Y00)
        M5 = (X00+X01)*Y11
        M6 = (X10-X00)*(Y00+Y01)
        M7 = (X01-X11)*(Y10+Y11)
        M8 = (X00+X10)*Y11
        
        Wait, let me use the standard Strassen formulas:
        M1 = (X00+X11)(Y00+Y11)
        M2 = (X10+X11)Y00
        M3 = X00(Y01-Y11)
        M4 = X11(Y10-Y00)
        M5 = (X00+X01)Y11
        M6 = (X10-X00)(Y00+Y01)
        M7 = (X01-X11)(Y10+Y11)
        M8 = (X00+X10)Y11
        
        Z00 = M1 - M4 + M6 + M7
        Z01 = M2 + M4
        Z10 = M3 + M5
        Z11 = M1 - M2 + M5 + M8
        """
        # Indexing: X[i,j] for i,j in {0,1}, Y[i,j] for i,j in {0,1}
        # U tensor: maps input (X) to bilinear forms
        # V tensor: maps input (Y) to bilinear forms
        # W tensor: maps output (Z) to combinations
        
        # 8 bilinear terms
        # Each term t has a U coefficient (for X), V coefficient (for Y), W coefficient (for Z)
        
        terms = []
        # M1 = (X00+X11)(Y00+Y11)
        terms.append(([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(0,1,1),(1,0,1),(1,1,-1)]))
        # M2 = (X10+X11)Y00
        terms.append(([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)]))
        # M3 = X00(Y01-Y11)
        terms.append(([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1)]))
        # M4 = X11(Y10-Y00)
        terms.append(([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,-1),(0,1,1)]))
        # M5 = (X00+X01)Y11
        terms.append(([(0,0,1),(0,1,1)], [(1,1,1)], [(1,0,1),(1,1,1)]))
        # M6 = (X10-X00)(Y00+Y01)
        terms.append(([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(0,0,1)]))
        # M7 = (X01-X11)(Y10+Y11)
        terms.append(([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]))
        # M8 = (X00+X10)Y11
        terms.append(([(0,0,1),(1,0,1)], [(1,1,1)], [(1,1,1)]))
        
        return terms
    
    def build_block_decomposition():
        """
        Build the full decomposition using 4 Strassen 2x2*2x2 blocks.
        
        A is 2x4: A[i,j] for i in {0,1}, j in {0,1,2,3}
        B is 4x4: B[i,j] for i in {0,1,2,3}, j in {0,1,2,3}
        C is 2x4: C[i,j] for i in {0,1}, j in {0,1,2,3}
        
        A1 = A[:, 0:2], A2 = A[:, 2:4]
        B11 = B[0:2, 0:2], B12 = B[0:2, 2:4]
        B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
        
        C_left = A1*B11 + A2*B21  (2x2 result)
        C_right = A1*B12 + A2*B22 (2x2 result)
        
        C[:, 0:2] = C_left
        C[:, 2:4] = C_right
        """
        strassen_terms = strassen_2x2()
        
        # We'll build U, V, W as lists of (index, value) pairs
        # U: (n*m, rank) = (8, rank)
        # V: (m*p, rank) = (16, rank)  
        # W: (p*n, rank) = (8, rank)
        
        # Index conventions:
        # U[i*m+j, t] where i in 0..1, j in 0..3
        # V[j*p+k, t] where j in 0..3, k in 0..3
        # W[k*n+i, t] where k in 0..3, i in 0..1
        
        U_entries = []  # list of (row, col, value)
        V_entries = []
        W_entries = []
        
        col = 0
        
        # Block 1: A1 * B11 -> contributes to C_left
        # A1[i,k] = A[i, k] for i in {0,1}, k in {0,1}
        # B11[k,j] = B[k, j] for k in {0,1}, j in {0,1}
        # Result: C_left[i,j] for i in {0,1}, j in {0,1}
        for term in strassen_terms:
            u_coeffs, v_coeffs, w_coeffs = term
            # u_coeffs: list of (x_i, x_j, val) for X matrix
            # v_coeffs: list of (y_i, y_j, val) for Y matrix
            # w_coeffs: list of (z_i, z_j, val) for Z matrix
            
            for (xi, xj, xv) in u_coeffs:
                # X is A1, so X[xi, xj] = A[xi, xj]
                U_entries.append((xi*m + xj, col, xv))
            for (yi, yj, yv) in v_coeffs:
                # Y is B11, so Y[yi, yj] = B[yi, yj]
                V_entries.append((yi*p + yj, col, yv))
            for (zi, zj, zv) in w_coeffs:
                # Z is C_left, so Z[zi, zj] = C[zi, zj]
                W_entries.append((zj*n + zi, col, zv))
            col += 1
        
        # Block 2: A2 * B21 -> contributes to C_left
        # A2[i,k] = A[i, 2+k] for i in {0,1}, k in {0,1}
        # B21[k,j] = B[2+k, j] for k in {0,1}, j in {0,1}
        # Result: C_left[i,j] for i in {0,1}, j in {0,1}
        for term in strassen_terms:
            u_coeffs, v_coeffs, w_coeffs = term
            for (xi, xj, xv) in u_coeffs:
                # X is A2, so X[xi, xj] = A[xi, 2+xj]
                U_entries.append((xi*m + (2+xj), col, xv))
            for (yi, yj, yv) in v_coeffs:
                # Y is B21, so Y[yi, yj] = B[2+yi, yj]
                V_entries.append(((2+yi)*p + yj, col, yv))
            for (zi, zj, zv) in w_coeffs:
                # Z is C_left, so Z[zi, zj] = C[zi, zj]
                W_entries.append((zj*n + zi, col, zv))
            col += 1
        
        # Block 3: A1 * B12 -> contributes to C_right
        # A1[i,k] = A[i, k] for i in {0,1}, k in {0,1}
        # B12[k,j] = B[k, 2+j] for k in {0,1}, j in {0,1}
        # Result: C_right[i,j] for i in {0,1}, j in {0,1}
        for term in strassen_terms:
            u_coeffs, v_coeffs, w_coeffs = term
            for (xi, xj, xv) in u_coeffs:
                # X is A1, so X[xi, xj] = A[xi, xj]
                U_entries.append((xi*m + xj, col, xv))
            for (yi, yj, yv) in v_coeffs:
                # Y is B12, so Y[yi, yj] = B[yi, 2+yj]
                V_entries.append((yi*p + (2+yj), col, yv))
            for (zi, zj, zv) in w_coeffs:
                # Z is C_right, so Z[zi, zj] = C[zi, 2+zj]
                W_entries.append(((2+zj)*n + zi, col, zv))
            col += 1
        
        # Block 4: A2 * B22 -> contributes to C_right
        # A2[i,k] = A[i, 2+k] for i in {0,1}, k in {0,1}
        # B22[k,j] = B[2+k, 2+j] for k in {0,1}, j in {0,1}
        # Result: C_right[i,j] for i in {0,1}, j in {0,1}
        for term in strassen_terms:
            u_coeffs, v_coeffs, w_coeffs = term
            for (xi, xj, xv) in u_coeffs:
                # X is A2, so X[xi, xj] = A[xi, 2+xj]
                U_entries.append((xi*m + (2+xj), col, xv))
            for (yi, yj, yv) in v_coeffs:
                # Y is B22, so Y[yi, yj] = B[2+yi, 2+yj]
                V_entries.append(((2+yi)*p + (2+yj), col, yv))
            for (zi, zj, zv) in w_coeffs:
                # Z is C_right, so Z[zi, zj] = C[zi, 2+zj]
                W_entries.append(((2+zj)*n + zi, col, zv))
            col += 1
        
        rank = col
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        for (row, col_idx, val) in U_entries:
            U[row, col_idx] += val
        for (row, col_idx, val) in V_entries:
            V[row, col_idx] += val
        for (row, col_idx, val) in W_entries:
            W[row, col_idx] += val
        
        return U, V, W, rank
    
    # Try the block Strassen decomposition
    U_block, V_block, W_block, rank_block = build_block_decomposition()
    
    if verify(U_block, V_block, W_block):
        U, V, W = U_block, V_block, W_block
        best_rank = rank_block
    else:
        # Fallback to schoolbook
        U, V, W = schoolbook()
        best_rank = n*m*p
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
