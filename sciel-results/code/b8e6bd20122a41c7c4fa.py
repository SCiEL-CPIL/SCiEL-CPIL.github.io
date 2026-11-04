    def _strassen_decomposition(n):
        if n == 2:
            return _strassen_2x2()
        
        # n is power of 2
        half = n // 2
        U_h, V_h, W_h = _strassen_decomposition(half)
        
        # The 7 block terms
        # We need to map the block indices to the full indices.
        # Block (r, c) of size half x half corresponds to:
        # A indices: i in r*half..(r+1)*half-1, j in c*half..(c+1)*half-1
        # B indices: j in c*half..(c+1)*half-1, k in d*half..(d+1)*half-1
        # C indices: k in d*half..(d+1)*half-1, i in r*half..(r+1)*half-1
        
        # The 7 operations:
        # 1. (A11 + A22) * (B11 + B22) -> C11, C22
        # 2. (A21 + A22) * B11 -> C21
        # 3. A11 * (B12 - B22) -> C12
        # 4. A22 * (B21 - B11) -> C21
        # 5. (A11 + A12) * B22 -> C11, C12
        # 6. (A21 - A11) * (B11 + B12) -> C22
        # 7. (A12 - A22) * (B21 + B22) -> C11
        
        # We will construct U, V, W of size (n*n, 7 * rank_h)
        rank_h = len(U_h[0])
        rank_n = 7 * rank_h
        
        U = np.zeros((n*n, rank_n))
        V = np.zeros((n*n, rank_n))
        W = np.zeros((n*n, rank_n))
        
        # Helper to get index range for block (br, bc)
        def get_indices(br, bc):
            start = br * half * half + bc * half
            # Wait, flattening is row-major.
            # Block (br, bc) covers rows br*half to (br+1)*half-1
            # and cols bc*half to (bc+1)*half-1.
            # Indices in flattened n*n matrix:
            # i, j.
            # idx = i * n + j.
            pass
        
        # This is getting complicated to map recursively.
        # It's easier to just hardcode the 49 terms or use a known library?
        # No library.
        
        # Alternative: Just use the 7 terms of 2x2 and "inflate" them?
        # No, the 7 terms of 2x2 are for scalar multiplication of 2x2 matrices.
        # For 4x4, we need to decompose the 2x2 matrix multiplication inside each term.
        
        # Let's stick to the plan:
        # 1. Get 4x4 decomposition.
        # 2. Add border.
        
        # How to get 4x4 decomposition?
        # I can write a specific function for 4x4.
    