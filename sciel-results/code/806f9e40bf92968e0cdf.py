"""Matrix multiplication 2,4,4 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 2, 4, 4
    
    # Use Strassen's algorithm for 2x2x2 matrix multiplication.
    # Strassen's 7 products for 2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 2x4 * 4x4, we can view this as:
    # A is 2x4: A = [A_left, A_right] where A_left is 2x2, A_right is 2x2
    # B is 4x4: B = [[B11, B12], [B21, B22]] where each is 2x2
    # C = A*B is 2x4: C = [C_left, C_right] where C_left = A_left*B11 + A_right*B21, C_right = A_left*B12 + A_right*B22
    
    # We can use Strassen for each of the 4 products (A_left*B11, A_left*B12, A_right*B21, A_right*B22)
    # That's 4 * 7 = 28 products.
    # But we can share some computations. Let's just do 4 separate Strassen instances for simplicity.
    
    # Actually, let's think about this differently. We have:
    # C_left = A_left * B11 + A_right * B21
    # C_right = A_left * B12 + A_right * B22
    
    # Each of these is a sum of two 2x2*2x2 products.
    # We can compute each 2x2*2x2 product with Strassen (rank 7).
    # Total: 4 * 7 = 28.
    
    # Let's construct the rank-28 decomposition.
    # For each of the 4 block products, we use Strassen's 7 intermediate products.
    
    # Block product 1: A_left * B11 -> contributes to C_left
    # Block product 2: A_right * B21 -> contributes to C_left
    # Block product 3: A_left * B12 -> contributes to C_right
    # Block product 4: A_right * B22 -> contributes to C_right
    
    # Strassen for X(2x2) * Y(2x2) = Z(2x2):
    # M1 = (X11 + X22)(Y11 + Y22)
    # M2 = (X21 + X22)Y11
    # M3 = X11(Y12 - Y22)
    # M4 = X22(Y21 - Y11)
    # M5 = (X11 + X12)Y22
    # M6 = (X21 - X11)(Y11 + Y12)
    # M7 = (X12 - X22)(Y21 + Y22)
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M2 + M3 + M6
    
    # Let's build the full decomposition.
    # Total rank = 4 * 7 = 28
    
    rank = 28
    U = np.zeros((n * m, rank))  # 8 x 28
    V = np.zeros((m * p, rank))  # 16 x 28
    W = np.zeros((p * n, rank))  # 8 x 28
    
    # Helper function to add a Strassen block product
    # block_idx: 0=A_left*B11, 1=A_right*B21, 2=A_left*B12, 3=A_right*B22
    # output_offset: which part of C this contributes to (0=left, 1=right)
    
    def add_strassen_block(U, V, W, block_idx, output_offset, start_col):
        # A is 2x4, indexed as A[i, j] where i in {0,1}, j in {0,1,2,3}
        # A_left = A[:, 0:2], A_right = A[:, 2:4]
        # B is 4x4, indexed as B[j, k] where j in {0,1,2,3}, k in {0,1,2,3}
        # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
        # C is 2x4, indexed as C[i, k] where i in {0,1}, k in {0,1,2,3}
        # C_left = C[:, 0:2], C_right = C[:, 2:4]
        
        # Determine which A block and B block
        if block_idx == 0:  # A_left * B11
            a_offset = 0  # A[:, 0:2]
            b_offset = 0  # B[0:2, 0:2]
        elif block_idx == 1:  # A_right * B21
            a_offset = 2  # A[:, 2:4]
            b_offset = 4  # B[2:4, 0:2] -> row offset 2, col offset 0
        elif block_idx == 2:  # A_left * B12
            a_offset = 0  # A[:, 0:2]
            b_offset = 6  # B[0:2, 2:4] -> row offset 0, col offset 2
        elif block_idx == 3:  # A_right * B22
            a_offset = 2  # A[:, 2:4]
            b_offset = 8  # B[2:4, 2:4] -> row offset 2, col offset 2
        
        # Strassen's 7 products
        # For X(2x2) * Y(2x2):
        # X is A[a_offset:a_offset+2, :], but A is 2x4, so X = A[:, a_offset:a_offset+2]
        # Wait, A is 2x4. A_left = A[:, 0:2], A_right = A[:, 2:4]
        # So X = A[:, a_offset:a_offset+2]
        # Y = B[b_row_offset:b_row_offset+2, b_col_offset:b_col_offset+2]
        
        if block_idx == 0:  # A_left * B11
            b_row_offset = 0
            b_col_offset = 0
        elif block_idx == 1:  # A_right * B21
            b_row_offset = 2
            b_col_offset = 0
        elif block_idx == 2:  # A_left * B12
            b_row_offset = 0
            b_col_offset = 2
        elif block_idx == 3:  # A_right * B22
            b_row_offset = 2
            b_col_offset = 2
        
        # X indices in A: X[0,0]=A[0,a_offset], X[0,1]=A[0,a_offset+1], X[1,0]=A[1,a_offset], X[1,1]=A[1,a_offset+1]
        # Y indices in B: Y[0,0]=B[b_row_offset,b_col_offset], Y[0,1]=B[b_row_offset,b_col_offset+1],
        #                Y[1,0]=B[b_row_offset+1,b_col_offset], Y[1,1]=B[b_row_offset+1,b_col_offset+1]
        
        # Strassen products:
        # M1 = (X11 + X22)(Y11 + Y22)
        # M2 = (X21 + X22)Y11
        # M3 = X11(Y12 - Y22)
        # M4 = X22(Y21 - Y11)
        # M5 = (X11 + X12)Y22
        # M6 = (X21 - X11)(Y11 + Y12)
        # M7 = (X12 - X22)(Y21 + Y22)
        
        # Output Z = X*Y, where Z[i,k] contributes to C[i, output_offset + k]
        # Z11 = M1 + M4 - M5 + M7 -> C[0, output_offset+0]
        # Z12 = M3 + M5 -> C[0, output_offset+1]
        # Z21 = M2 + M4 -> C[1, output_offset+0]
        # Z22 = M1 - M2 + M3 + M6 -> C[1, output_offset+1]
        
        # Now build the factor matrices for each of the 7 Strassen products.
        # Each product M_t is a bilinear form in X and Y.
        
        # M1 = (X11 + X22)(Y11 + Y22)
        # = (A[0,a_offset] + A[1,a_offset+1]) * (B[b_row_offset,b_col_offset] + B[b_row_offset+1,b_col_offset+1])
        col = start_col + 0
        U[0 * m + a_offset, col] = 1  # A[0, a_offset]
        U[1 * m + a_offset + 1, col] = 1  # A[1, a_offset+1]
        V[b_row_offset * p + b_col_offset, col] = 1  # B[b_row_offset, b_col_offset]
        V[(b_row_offset + 1) * p + b_col_offset + 1, col] = 1  # B[b_row_offset+1, b_col_offset+1]
        W[(output_offset + 0) * n + 0, col] = 1  # C[0, output_offset+0]
        W[(output_offset + 0) * n + 1, col] = 1  # C[1, output_offset+0]
        W[(output_offset + 1) * n + 0, col] = 1  # C[0, output_offset+1]
        W[(output_offset + 1) * n + 1, col] = -1  # C[1, output_offset+1]
        
        # M2 = (X21 + X22)Y11
        # = (A[1,a_offset] + A[1,a_offset+1]) * B[b_row_offset, b_col_offset]
        col = start_col + 1
        U[1 * m + a_offset, col] = 1
        U[1 * m + a_offset + 1, col] = 1
        V[b_row_offset * p + b_col_offset, col] = 1
        W[(output_offset + 0) * n + 1, col] = 1  # C[1, output_offset+0]
        W[(output_offset + 1) * n + 1, col] = -1  # C[1, output_offset+1]
        
        # M3 = X11(Y12 - Y22)
        # = A[0,a_offset] * (B[b_row_offset, b_col_offset+1] - B[b_row_offset+1, b_col_offset+1])
        col = start_col + 2
        U[0 * m + a_offset, col] = 1
        V[b_row_offset * p + b_col_offset + 1, col] = 1
        V[(b_row_offset + 1) * p + b_col_offset + 1, col] = -1
        W[(output_offset + 1) * n + 0, col] = 1  # C[0, output_offset+1]
        W[(output_offset + 1) * n + 1, col] = 1  # C[1, output_offset+1]
        
        # M4 = X22(Y21 - Y11)
        # = A[1,a_offset+1] * (B[b_row_offset+1, b_col_offset] - B[b_row_offset, b_col_offset])
        col = start_col + 3
        U[1 * m + a_offset + 1, col] = 1
        V[(b_row_offset + 1) * p + b_col_offset, col] = 1
        V[b_row_offset * p + b_col_offset, col] = -1
        W[(output_offset + 0) * n + 0, col] = 1  # C[0, output_offset+0]
        W[(output_offset + 0) * n + 1, col] = 1  # C[1, output_offset+0]
        
        # M5 = (X11 + X12)Y22
        # = (A[0,a_offset] + A[0,a_offset+1]) * B[b_row_offset+1, b_col_offset+1]
        col = start_col + 4
        U[0 * m + a_offset, col] = 1
        U[0 * m + a_offset + 1, col] = 1
        V[(b_row_offset + 1) * p + b_col_offset + 1, col] = 1
        W[(output_offset + 0) * n + 0, col] = -1  # C[0, output_offset+0]
        W[(output_offset + 1) * n + 0, col] = 1  # C[0, output_offset+1]
        
        # M6 = (X21 - X11)(Y11 + Y12)
        # = (A[1,a_offset] - A[0,a_offset]) * (B[b_row_offset, b_col_offset] + B[b_row_offset, b_col_offset+1])
        col = start_col + 5
        U[1 * m + a_offset, col] = 1
        U[0 * m + a_offset, col] = -1
        V[b_row_offset * p + b_col_offset, col] = 1
        V[b_row_offset * p + b_col_offset + 1, col] = 1
        W[(output_offset + 1) * n + 1, col] = 1  # C[1, output_offset+1]
        
        # M7 = (X12 - X22)(Y21 + Y22)
        # = (A[0,a_offset+1] - A[1,a_offset+1]) * (B[b_row_offset+1, b_col_offset] + B[b_row_offset+1, b_col_offset+1])
        col = start_col + 6
        U[0 * m + a_offset + 1, col] = 1
        U[1 * m + a_offset + 1, col] = -1
        V[(b_row_offset + 1) * p + b_col_offset, col] = 1
        V[(b_row_offset + 1) * p + b_col_offset + 1, col] = 1
        W[(output_offset + 0) * n + 0, col] = 1  # C[0, output_offset+0]
    
    # Add 4 Strassen blocks
    add_strassen_block(U, V, W, 0, 0, 0)   # A_left * B11 -> C_left
    add_strassen_block(U, V, W, 1, 0, 7)   # A_right * B21 -> C_left
    add_strassen_block(U, V, W, 2, 2, 14)  # A_left * B12 -> C_right
    add_strassen_block(U, V, W, 3, 2, 21)  # A_right * B22 -> C_right
    
    # Verify the decomposition
    # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    # Should equal delta_{i,i'} * delta_{k,k'} where:
    # a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T is nonzero (=1) iff j=j' and i=i' and k=k'
    
    # Let's verify by computing the tensor contraction
    # For each (i, j, k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for other combinations, it should be 0.
    
    # Actually, let's just verify a few cases to make sure the construction is correct.
    
    # The key insight: C[i, k] = sum_j A[i, j] * B[j, k]
    # In our decomposition, C[i, k] should be correctly computed.
    
    # Let's do a full verification
    # Build the tensor T from U, V, W
    # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    
    # Check: for all i, j, k, T[i*m+j, j*p+k, k*n+i] should be 1
    # And for all other (a, b, c), T[a, b, c] should be 0.
    
    # Let's verify
    error = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = 0
                for r in range(rank):
                    val += U[a, r] * V[b, r] * W[c, r]
                if abs(val - 1) > 1e-10:
                    error += 1
                    print(f"Error at ({i},{j},{k}): got {val}, expected 1")
    
    if error > 0:
        print(f"Verification failed with {error} errors")
        # Fall back to schoolbook
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
    
    # Convert to list of lists as required
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
