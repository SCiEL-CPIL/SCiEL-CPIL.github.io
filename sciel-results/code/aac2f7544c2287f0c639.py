import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    Uses a block decomposition strategy:
    - Partitions A (3x4) into 2x2, 2x2, 1x2, 1x2 blocks.
    - Partitions B (4x5) into 2x2, 2x2, 2x1, 2x2, 2x2, 2x1 blocks.
    - Partitions C (3x5) into corresponding blocks.
    - Uses Strassen (rank 7) for 2x2x2 blocks.
    - Uses Schoolbook (rank 4) for 2x2x1 and 1x2x2 blocks.
    - Uses Schoolbook (rank 2) for 1x2x1 blocks.
    
    Total Rank = 4*7 + 2*4 + 4*4 + 2*2 = 28 + 8 + 16 + 4 = 56.
    """
    n, m, p = DIMENSIONS
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Helper to add a block
    def add_block(u_start, u_end, v_start, v_end, w_start, w_end, U_blk, V_blk, W_blk):
        """
        u_start, u_end: slice of U indices (flat row-major of n x m)
        v_start, v_end: slice of V indices (flat row-major of m x p)
        w_start, w_end: slice of W indices (flat column-major of p x n, i.e., k*n+i)
        U_blk, V_blk, W_blk: factor matrices for the block
        """
        for r in range(U_blk.shape[1]):
            u_vec = np.zeros(n * m)
            v_vec = np.zeros(m * p)
            w_vec = np.zeros(p * n)
            
            # Map block indices to global indices
            # U is n x m. Block is u_rows x u_cols.
            # We assume the block is a submatrix defined by row range and col range.
            # The helper function arguments are simplified to start/end of the 1D array?
            # No, it's better to pass the row/col ranges explicitly to avoid confusion.
            pass

    # Let's refine the helper to take row/col ranges.
    def add_block(u_rows, u_cols, v_rows, v_cols, w_rows, w_cols, U_blk, V_blk, W_blk):
        """
        u_rows, u_cols: list of row indices and col indices in A (n x m)
        v_rows, v_cols: list of row indices and col indices in B (m x p)
        w_rows, w_cols: list of row indices and col indices in C (n x p)
        Note: W is indexed as W[k*n+i] where k is col of C, i is row of C.
        So w_rows corresponds to i (rows of C), w_cols corresponds to k (cols of C).
        """
        for r in range(U_blk.shape[1]):
            u_vec = np.zeros(n * m)
            v_vec = np.zeros(m * p)
            w_vec = np.zeros(p * n)
            
            for i, u_r in enumerate(u_rows):
                for j, u_c in enumerate(u_cols):
                    idx = u_r * m + u_c
                    u_vec[idx] = U_blk[i, j, r] if U_blk.ndim == 3 else U_blk[i*m//u_cols + j, r] # This is getting complicated.
                    # Let's just use 2D arrays for blocks and map indices directly.
            # Actually, U_blk is (u_rows_len * u_cols_len, rank)? No, usually (u_size, rank) where u_size is flattened.
            # Let's stick to the previous approach: U_blk is (num_u_elements, rank).
            # We need to map the linear index of the block to the global index.
            
            # Let's change strategy: generate the block matrices with explicit global indices.
            pass

    # This is getting too complex with generic helpers. I will write the specific blocks explicitly.
    
    # Strassen 2x2x2
    def get_strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        # M3 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        # M4 = A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        return U, V, W

    # Schoolbook 2x2x1
    def get_schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        # A00*B0 -> C0
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # A01*B1 -> C0
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        # A10*B0 -> C1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        # A11*B1 -> C1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        return U, V, W

    # Schoolbook 1x2x2
    def get_schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        # A0*B00 -> C0
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # A1*B10 -> C0
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        # A0*B01 -> C1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        # A1*B11 -> C1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        return U, V, W

    # Schoolbook 1x2x1
    def get_schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        # A0*B0 -> C0
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # A1*B1 -> C0
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        return U, V, W

    # Define Blocks
    # A (3x4)
    # B (4x5)
    # C (3x5)
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    U_b, V_b, W_b = get_strassen_2x2x2()
    u_rows = [0, 1]; u_cols = [0, 1]
    v_rows = [0, 1]; v_cols = [0, 1]
    w_rows = [0, 1]; w_cols = [0, 1]
    for r in range(7):
        u_vec = np.zeros(n*m); v_vec = np.zeros(m*p); w_vec = np.zeros(p*n)
        for i, ur in enumerate(u_rows):
            for j, uc in enumerate(u_cols):
                u_vec[ur*m+uc] = U_b[i*2+j, r]
        for i, vr in enumerate(v_rows):
            for j, vc in enumerate(v_cols):
                v_vec[vr*p+vc] = V_b[i*2+j, r]
        for i, wr in enumerate(w_rows):
            for j, wc in enumerate(w_cols):
                w_vec[wc*n+wr] = W_b[i*2+j, r]
        U_parts.append(u_vec); V_parts.append(v_vec); W_parts.append(w_vec)

    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    U_b, V_b, W_b = get_strassen_2x2x2()
    u_rows = [0, 1]; u_cols = [2, 3]
    v_rows = [2, 3]; v_cols = [0, 1]
    w_rows = [0, 1]; w_cols = [0, 1]
    for r in range(7):
        u_vec = np.zeros(n*m); v_vec = np.zeros(m*p); w_vec = np.zeros(p*n)
        for i, ur in enumerate(u_rows):
            for j, uc in enumerate(u_cols):
                u_vec[ur*m+uc] = U_b[i*2+j, r]
        for i, vr in enumerate(v_rows):
            for j, vc in enumerate(v_cols):
                v_vec[vr*p+vc] = V_b[i*2+j, r]
        for i, wr in enumerate(w_rows):
            for j, wc in enumerate(w_cols):
                w_vec[wc*n+wr] = W_b[i*2+j, r]
        U_parts.append(u_vec); V_parts.append(v_vec); W_parts.append(w_vec)

    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    U_b, V_b, W_b = get_strassen_2x2x2()
    u_rows = [0, 1]; u_cols = [0, 1]
    v_rows = [0, 1]; v_cols = [2, 3]
    w_rows = [0, 1]; w_cols = [2, 3]
    for r in range(7):
        u_vec = np.zeros(n*m); v_vec = np.zeros(m*p); w_vec = np.zeros(p*n)
        for i, ur in enumerate(u_rows):
            for j, uc in enumerate(u_cols):
                u_vec[ur*m+uc] = U_b[i*2+j, r]
        for i, vr in enumerate(v_rows):
            for j, vc in enumerate(v_cols):
                v_vec[vr*p+vc] = V_b[i*2+j, r]
        for i, wr in enumerate(w_rows):
            for j, wc in enumerate(w_cols):
                w_vec[wc*n+wr] = W_b[i*2+j, r]
        U_parts.append(u_vec); V_parts.append(v_vec); W_parts.append(w_vec)

    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    U_b, V_b, W_b = get_strassen_2x2x2()
    u_rows = [0, 1]; u_cols = [2, 3]
    v_rows = [2, 3]; v_cols = [2, 3]
    w_rows = [0, 1]; w_cols = [2, 3]
    for r in range(7):
        u_vec = np.zeros(n*m); v_vec = np.zeros(m*p); w_vec = np.zeros(p*n)
        for i, ur in enumerate(u_rows):
            for j, uc in enumerate(u_cols):
                u_vec[ur*m+uc] = U_b[i*2+j, r]
        for i, vr in enumerate(v_rows):
            for j, vc in enumerate(v_cols):
                v_vec[vr*p+vc] = V_b[i*2+j, r]
        for i, wr in enumerate(w_rows):
            for j, wc in enumerate(w_cols):
                w_vec[wc*n+wr] = W_b[i*2+j, r]
        U_parts.append(u_vec); V_parts.append(v_vec); W_parts.append(w_vec)

    # Block 5: A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5]
    U_b, V_b, W_b = get_schoolbook_2x2x1()
    u_rows = [0, 1]; u_cols = [0, 1]
    v_rows = [0, 1]; v_cols = [4]
    w_rows = [0, 1]; w_cols = [4]
    for r in range(4):
        u_vec = np.zeros(n*m); v_vec = np.zeros(m*p); w_vec = np.zeros(p*n)
        for i, ur in enumerate(u_rows):
            for j, uc in enumerate(u_cols):
                u_vec[ur*m+uc] = U_b[i*2+j, r]
        for i, vr in enumerate(v_rows):
            for j, vc in enumerate(v_cols):
                v_vec[vr*p+vc] = V_b[i*2+j, r]
        for i, wr in enumerate(w_rows):
            for j, wc in enumerate(w_cols):
                w_vec[wc*n+wr] = W_b[i*2+j, r]
        U_parts.append(u_vec); V_parts.append(v_vec); W_parts.append(w_vec)

    # Block 6: A[0:2, 2:4] * B[2:4, 4:5] -> C[0:2, 4:5]
    U_b, V_b, W_b = get_schoolbook_2x2x1()
    u_rows = [0, 1]; u_cols = [2, 3]
    v_rows = [2, 3]; v_cols = [4]
    w_rows = [0, 1]; w_cols = [4]
    for r in range(4):
        u_vec = np.zeros(n*m); v_vec = np.zeros(m*p); w_vec = np.zeros(p*n)
        for i, ur in enumerate(u_rows):
            for j, uc in enumerate(u_cols):
                u_vec[ur*m+uc] = U_b[i*2+j, r]
        for i, vr in enumerate(v_rows):
            for j, vc in enumerate(v_cols):
                v_vec[vr*p+vc] = V_b[i*2+j, r]
        for i, wr in enumerate(w_rows):
            for j, wc in enumerate(w_cols):
                w_vec[wc*n+wr] = W_b[i*2+j, r]
        U_parts.append(u_vec); V_parts.append(v_vec); W_parts.append(w_vec)

    # Block 7: A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2]
    U_b, V_b, W_b = get_schoolbook_1x2x2()
    u_rows = [2]; u_cols = [0, 1]
    v_rows = [0, 1]; v_cols = [0, 1]
    w_rows = [2]; w_cols = [0, 1]
    for r in range(4):
        u_vec = np.zeros(n*m); v_vec = np.zeros(m*p); w_vec = np.zeros(p*n)
        for i, ur in enumerate(u_rows):
            for j, uc in enumerate(u_cols):
                u_vec[ur*m+uc] = U_b[i*2+j, r]
        for i, vr in enumerate(v_rows):
            for j, vc in enumerate(v_cols):
                v_vec[vr*p+vc] = V_b[i*2+j, r]
        for i, wr in enumerate(w_rows):
            for j, wc in enumerate(w_cols):
                w_vec[wc*n+wr] = W_b[i*2+j, r]
        U_parts.append(u_vec); V_parts.append(v_vec); W_parts.append(w_vec)

    # Block 8: A[2:3, 2:4] * B[2:4, 0:2] -> C[2:3, 0:2]
    U_b, V_b, W_b = get_schoolbook_1x2x2()
    u_rows = [2]; u_cols = [2, 3]
    v_rows = [2, 3]; v_cols = [0, 1]
    w_rows = [2]; w_cols = [0, 1]
    for r in range(4):
        u_vec = np.zeros(n*m); v_vec = np.zeros(m*p); w_vec = np.zeros(p*n)
        for i, ur in enumerate(u_rows):
            for j, uc in enumerate(u_cols):
                u_vec[ur*m+uc] = U_b[i*2+j, r]
        for i, vr in enumerate(v_rows):
            for j, vc in enumerate(v_cols):
                v_vec[vr*p+vc] = V_b[i*2+j, r]
        for i, wr in enumerate(w_rows):
            for j, wc in enumerate(w_cols):
                w_vec[wc*n+wr] = W_b[i*2+j, r]
        U_parts.append(u_vec); V_parts.append(v_vec); W_parts.append(w_vec)

    # Block 9: A[2:3, 0:2] * B[0:2, 2:4] -> C[2:3, 2:4]
    U_b, V_b, W_b = get_schoolbook_1x2x2()
    u_rows = [2]; u_cols = [0, 1]
    v_rows = [0, 1]; v_cols = [2, 3]
    w_rows = [2]; w_cols = [2, 3]
    for r in range(4):
        u_vec = np.zeros(n*m); v_vec = np.zeros(m*p); w_vec = np.zeros(p*n)
        for i, ur in enumerate(u_rows):
            for j, uc in enumerate(u_cols):
                u_vec[ur*m+uc] = U_b[i*2+j, r]
        for i, vr in enumerate(v_rows):
            for j, vc in enumerate(v_cols):
                v_vec[vr*p+vc] = V_b[i*2+j, r]
        for i, wr in enumerate(w_rows):
            for j, wc in enumerate(w_cols):
                w_vec[wc*n+wr] = W_b[i*2+j, r]
        U_parts.append(u_vec); V_parts.append(v_vec); W_parts.append(w_vec)

    # Block 10: A[2:3, 2:4] * B[2:4, 2:4] -> C[2:3, 2:4]
    U_b, V_b, W_b = get_schoolbook_1x2x2()
    u_rows = [2]; u_cols = [2, 3]
    v_rows = [2, 3]; v_cols = [2, 3]
    w_rows = [2]; w_cols = [2, 3]
    for r in range(4):
        u_vec = np.zeros(n*m); v_vec = np.zeros(m*p); w_vec = np.zeros(p*n)
        for i, ur in enumerate(u_rows):
            for j, uc in enumerate(u_cols):
                u_vec[ur*m+uc] = U_b[i*2+j, r]
        for i, vr in enumerate(v_rows):
            for j, vc in enumerate(v_cols):
                v_vec[vr*p+vc] = V_b[i*2+j, r]
        for i, wr in enumerate(w_rows):
            for j, wc in enumerate(w_cols):
                w_vec[wc*n+wr] = W_b[i*2+j, r]
        U_parts.append(u_vec); V_parts.append(v_vec); W_parts.append(w_vec)

    # Block 11: A[2:3, 0:2] * B[0:2, 4:5] -> C[2:3, 4:5]
    U_b, V_b, W_b = get_schoolbook_1x2x1()
    u_rows = [2]; u_cols = [0, 1]
    v_rows = [0, 1]; v_cols = [4]
    w_rows = [2]; w_cols = [4]
    for r in range(2):
        u_vec = np.zeros(n*m); v_vec = np.zeros(m*p); w_vec = np.zeros(p*n)
        for i, ur in enumerate(u_rows):
            for j, uc in enumerate(u_cols):
                u_vec[ur*m+uc] = U_b[i*2+j, r]
        for i, vr in enumerate(v_rows):
            for j, vc in enumerate(v_cols):
                v_vec[vr*p+vc] = V_b[i*2+j, r]
        for i, wr in enumerate(w_rows):
            for j, wc in enumerate(w_cols):
                w_vec[wc*n+wr] = W_b[i*2+j, r]
        U_parts.append(u_vec); V_parts.append(v_vec); W_parts.append(w_vec)

    # Block 12: A[2:3, 2:4] * B[2:4, 4:5] -> C[2:3, 4:5]
    U_b, V_b, W_b = get_schoolbook_1x2x1()
    u_rows = [2]; u_cols = [2, 3]
    v_rows = [2, 3]; v_cols = [4]
    w_rows = [2]; w_cols = [4]
    for r in range(2):
        u_vec = np.zeros(n*m); v_vec = np.zeros(m*p); w_vec = np.zeros(p*n)
        for i, ur in enumerate(u_rows):
            for j, uc in enumerate(u_cols):
                u_vec[ur*m+uc] = U_b[i*2+j, r]
        for i, vr in enumerate(v_rows):
            for j, vc in enumerate(v_cols):
                v_vec[vr*p+vc] = V_b[i*2+j, r]
        for i, wr in enumerate(w_rows):
            for j, wc in enumerate(w_cols):
                w_vec[wc*n+wr] = W_b[i*2+j, r]
        U_parts.append(u_vec); V_parts.append(v_vec); W_parts.append(w_vec)

    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
