import numpy as np

def strassen_decomposition(n):
    """
    Returns U, V, W matrices for n x n x n matrix multiplication
    using recursive Strassen's algorithm.
    Base case n=1: Rank 1, U=[[1]], V=[[1]], W=[[1]].
    """
    if n == 1:
        return np.array([[1.0]]), np.array([[1.0]]), np.array([[1.0]])
    
    # Split n into two halves n/2
    # This assumes n is a power of 2. 4 is 2^2.
    n_half = n // 2
    
    # Get decomposition for n_half x n_half
    U_half, V_half, W_half = strassen_decomposition(n_half)
    
    # Strassen's 7 products for 2x2 block matrix multiplication
    # P1 = (A00 + A11) * (B00 + B11)
    # P2 = (A10 + A11) * B00
    # P3 = A00 * (B01 - B11)
    # P4 = A11 * (B10 - B00)
    # P5 = (A00 + A01) * B11
    # P6 = (A10 - A00) * (B00 + B01)
    # P7 = (A01 - A11) * (B10 + B11)
    
    # We need to construct U, V, W for size n.
    # U has shape (n*n, rank_total)
    # rank_total = 7 * rank_half
    
    rank_half = U_half.shape[1]
    rank_total = 7 * rank_half
    
    U = np.zeros((n*n, rank_total))
    V = np.zeros((n*n, rank_total))
    W = np.zeros((n*n, rank_total))
    
    # Helper to get index in flattened n x n matrix
    def idx(i, j):
        return i * n + j
        
    # Helper to get index in flattened n_half x n_half matrix
    def idx_half(i, j):
        return i * n_half + j
        
    # Strassen linear forms for A (U)
    # A is split into A00, A01, A10, A11
    # A00 is rows 0..n_half-1, cols 0..n_half-1
    # A01 is rows 0..n_half-1, cols n_half..n-1
    # A10 is rows n_half..n-1, cols 0..n_half-1
    # A11 is rows n_half..n-1, cols n_half..n-1
    
    # For each of the 7 Strassen products, we define the linear combination of A blocks
    # and B blocks, then multiply by the half-decomposition.
    
    # The 7 products correspond to columns 0..6 in the "block" rank dimension.
    # For each block product k (0..6), we have rank_half sub-products.
    
    # Define the block linear forms
    # U_block[k] is a 2x2 matrix of coefficients for A00, A01, A10, A11
    # V_block[k] is a 2x2 matrix of coefficients for B00, B01, B10, B11
    # W_block[k] is a 2x2 matrix of coefficients for C00, C01, C10, C11
    
    # P1
    U_b0 = np.array([[1, 0], [0, 1]], dtype=float) # A00 + A11
    V_b0 = np.array([[1, 0], [0, 1]], dtype=float) # B00 + B11
    W_b0 = np.array([[1, 0], [0, 1]], dtype=float) # C00 += P1, C11 += P1
    
    # P2
    U_b1 = np.array([[0, 0], [1, 1]], dtype=float) # A10 + A11
    V_b1 = np.array([[1, 0], [0, 0]], dtype=float) # B00
    W_b1 = np.array([[0, 0], [1, -1]], dtype=float) # C10 += P2, C11 -= P2
    
    # P3
    U_b2 = np.array([[1, 0], [0, 0]], dtype=float) # A00
    V_b2 = np.array([[0, 1], [0, -1]], dtype=float) # B01 - B11
    W_b2 = np.array([[0, 1], [0, 1]], dtype=float) # C01 += P3, C11 += P3
    
    # P4
    U_b3 = np.array([[0, 0], [0, 1]], dtype=float) # A11
    V_b3 = np.array([[0, 0], [1, -1]], dtype=float) # B10 - B00
    W_b3 = np.array([[1, 0], [1, 0]], dtype=float) # C00 += P4, C10 += P4
    
    # P5
    U_b4 = np.array([[1, 1], [0, 0]], dtype=float) # A00 + A01
    V_b4 = np.array([[0, 0], [0, 1]], dtype=float) # B11
    W_b4 = np.array([[-1, 1], [0, 0]], dtype=float) # C00 -= P5, C01 += P5
    
    # P6
    U_b5 = np.array([[0, 0], [1, -1]], dtype=float) # A10 - A00
    V_b5 = np.array([[1, 1], [0, 0]], dtype=float) # B00 + B01
    W_b5 = np.array([[0, 0], [0, 1]], dtype=float) # C11 += P6
    
    # P7
    U_b6 = np.array([[0, 1], [0, -1]], dtype=float) # A01 - A11
    V_b6 = np.array([[0, 0], [1, 1]], dtype=float) # B10 + B11
    W_b6 = np.array([[1, 0], [0, 0]], dtype=float) # C00 += P7
    
    U_blocks = [U_b0, U_b1, U_b2, U_b3, U_b4, U_b5, U_b6]
    V_blocks = [V_b0, V_b1, V_b2, V_b3, V_b4, V_b5, V_b6]
    W_blocks = [W_b0, W_b1, W_b2, W_b3, W_b4, W_b5, W_b6]
    
    for k in range(7):
        # Column offset for this Strassen product in the full rank
        col_offset = k * rank_half
        
        for r_half in range(rank_half):
            r = col_offset + r_half
            
            # U construction
            # U[i, j] for full matrix
            # U_half[i_half, j_half] is coefficient for A_half[i_half, j_half]
            # We need to map A_half indices to A indices based on U_blocks[k]
            
            # U_blocks[k] is 2x2.
            # U_blocks[k][bi, bj] is coefficient for A block (bi, bj)
            # Block (0,0) corresponds to A[0:n_half, 0:n_half]
            # Block (0,1) corresponds to A[0:n_half, n_half:n]
            # Block (1,0) corresponds to A[n_half:n, 0:n_half]
            # Block (1,1) corresponds to A[n_half:n, n_half:n]
            
            for bi in range(2):
                for bj in range(2):
                    coeff = U_blocks[k][bi, bj]
                    if coeff == 0:
                        continue
                    # Map half indices to full indices
                    # i_half in 0..n_half-1 maps to i = bi*n_half + i_half
                    # j_half in 0..n_half-1 maps to j = bj*n_half + j_half
                    for i_half in range(n_half):
                        for j_half in range(n_half):
                            u_val = U_half[i_half, j_half]
                            if u_val == 0:
                                continue
                            i_full = bi * n_half + i_half
                            j_full = bj * n_half + j_half
                            U[idx(i_full, j_full), r] += coeff * u_val
                            
            # V construction
            for bi in range(2):
                for bj in range(2):
                    coeff = V_blocks[k][bi, bj]
                    if coeff == 0:
                        continue
                    for i_half in range(n_half):
                        for j_half in range(n_half):
                            v_val = V_half[i_half, j_half]
                            if v_val == 0:
                                continue
                            i_full = bi * n_half + i_half
                            j_full = bj * n_half + j_half
                            V[idx(i_full, j_full), r] += coeff * v_val
                            
            # W construction
            # W is for C.
            # W_blocks[k][ci, cj] is coefficient for C block (ci, cj)
            # C block (0,0) is C[0:n_half, 0:n_half]
            # etc.
            for ci in range(2):
                for cj in range(2):
                    coeff = W_blocks[k][ci, cj]
                    if coeff == 0:
                        continue
                    for i_half in range(n_half):
                        for j_half in range(n_half):
                            w_val = W_half[i_half, j_half]
                            if w_val == 0:
                                continue
                            i_full = ci * n_half + i_half
                            j_full = cj * n_half + j_half
                            # W index is k*n + i (trace convention W[k*n+i, r])
                            # Wait, W shape is (p*n, rank).
                            # For n x n x n, p=n.
                            # Index is j_full * n + i_full ?
                            # Task says: W[k*n+i, r].
                            # In standard matrix mult C = AB, C[i,k] = sum_j A[i,j]B[j,k].
                            # Here i is row of C, k is col of C.
                            # So index is k * n + i.
                            W[j_full * n + i_full, r] += coeff * w_val

    return U, V, W

def construct():
    n, m, p = 4, 4, 8
    
    # Get 4x4x4 decomposition
    U_4, V_4, W_4 = strassen_decomposition(4)
    
    # U_4 is 16x49, V_4 is 16x49, W_4 is 16x49
    # We need U (16x98), V (32x98), W (32x98)
    
    rank = 98
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # First 49 columns: B[0:4]
    U[:, 0:49] = U_4
    V[0:16, 0:49] = V_4
    # V_4 corresponds to B[0:4, 0:4]
    # V index is j*p + k. j in 0..3, k in 0..3.
    # V_4 index is j*4 + k.
    # So V_4[j*4+k] maps to V[j*8+k].
    for j in range(4):
        for k in range(4):
            V[j*8 + k, 0:49] = V_4[j*4 + k, 0:49]
            
    # W_4 corresponds to C[0:4, 0:4]
    # W_4 index is k*4 + i.
    # W index is k*4 + i (since p=4 for this part, but p=8 total).
    # Wait, W shape is (p*n, rank). p=8, n=4.
    # For the first part, C is 4x4. k in 0..3, i in 0..3.
    # W index is k*4 + i.
    # This matches W_4 index k*4 + i.
    W[0:16, 0:49] = W_4
    
    # Last 49 columns: B[4:8]
    U[:, 49:98] = U_4
    # V_4 corresponds to B[0:4, 0:4] again, but now mapped to B[0:4, 4:8]
    for j in range(4):
        for k in range(4):
            # B[j, 4+k]
            # V index is j*8 + (4+k)
            V[j*8 + (4+k), 49:98] = V_4[j*4 + k, 0:49]
            
    # W_4 corresponds to C[0:4, 0:4] again, but now C is 4x4 at cols 4..7
    # C[i, 4+k]
    # W index is (4+k)*4 + i
    for k in range(4):
        for i in range(4):
            # W_4 index is k*4 + i
            W[(4+k)*4 + i, 49:98] = W_4[k*4 + i, 0:49]
            
    # Convert to lists as required by interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
