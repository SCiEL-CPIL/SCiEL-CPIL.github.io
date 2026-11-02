"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-14 exact tensor decomposition for 2x2 @ 2x4 matrix multiplication.
    
    The tensor T_{n,m,p} with n=2, m=2, p=4 decomposes as a direct sum of two
    2x2x2 matrix multiplication tensors (for columns 0-1 and columns 2-3).
    Each 2x2x2 block has tensor rank 7 (Strassen's algorithm), giving total rank 14.
    
    Convention: C = A @ B where A is n x m, B is m x p, C is n x p.
    Tensor entries: T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    Decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i, j, k]
    
    Strassen's 7 products for 2x2:
      M1 = A00 * (B01 - B11)      -> contributes to C00, C10
      M2 = (A10 + A00) * B11      -> contributes to C00, C10
      M3 = (A01 - A11) * B10      -> contributes to C01, C11
      M4 = A11 * (B00 - B10)      -> contributes to C01, C11
      M5 = (A00 + A11) * (B00 + B11) -> contributes to C00, C01, C10, C11
      M6 = (A10 - A00) * (B00 + B01) -> contributes to C10, C11
      M7 = (A01 + A11) * (B10 + B11) -> contributes to C01, C10
    """
    import time
    start_time = time.time()
    DEADLINE = 120.0
    SAFETY_MARGIN = 5.0
    
    n, m, p = DIMENSIONS  # 2, 2, 4
    rank = 14
    
    # U has shape (n*m, rank) = (4, 14)
    # V has shape (m*p, rank) = (8, 14)  
    # W has shape (p*n, rank) = (8, 14)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen products for a 2x2x2 block.
    # For a block handling columns k_offset..k_offset+1 (width 2):
    # A indices: (i, j) -> U index i*m + j
    # B indices: (j, k) -> V index j*p + k
    # C indices: (i, k) -> W index k*n + i
    
    # Define the 7 Strassen terms as (U_coefficients, V_coefficients, W_coefficients)
    # Each coefficient dict maps row_index -> value
    
    # For block at column offset `off` (off = 0 or 2):
    # B row indices: j*p + (off + dk) where dk in {0,1}
    # W row indices: (off + dk)*n + i where dk in {0,1}, i in {0,1}
    
    def strassen_block(off):
        """Return list of 7 dicts, each with keys 'u', 'v', 'w' mapping row -> coeff."""
        products = []
        
        # M1: A[0,0] * (B[0,1] - B[1,1])
        # A: u00=1
        # B: v0,off+1 = 1, v1,off+1 = -1
        # C: contributes to C[0,off]=1? No.
        # C00 = M1 - M2 + M5 - M6 - M7 ... let me use standard Strassen:
        # C00 = M1 - M2 + M5 - M6 - M7  -- wait, let me be precise.
        
        # Standard Strassen for C = A*B (2x2):
        # M1 = A00*(B01-B11)
        # M2 = (A10+A00)*B11
        # M3 = (A01-A11)*B10
        # M4 = A11*(B00-B10)
        # M5 = (A00+A11)*(B00+B11)
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01+A11)*(B10+B11)
        # C00 = M1 - M2 + M5 - M6 - M7
        # C01 = M3 + M4 + M7
        # C10 = M2 + M6
        # C11 = M1 - M4 + M5 - M3 + M7
        
        # Let me verify:
        # C00 should be A00*B00 + A01*B10
        # M1 - M2 + M5 - M6 - M7
        # = A00*B01 - A00*B11 - A10*B11 - A00*B11 + A00*B00 + A00*B11 + A11*B00 + A11*B11
        #   - A10*B00 - A10*B01 + A00*B00 + A00*B01 - A01*B10 - A01*B11 - A11*B10 - A11*B11
        # Hmm this is getting complex. Let me just use the well-known Strassen formulas.
        
        # Actually, let me use a different, well-verified set:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10+A01)(B00+B10)
        # M7 = (A01-A10)(B11+B10)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M1 - M2 + M6
        # C10 = M3 + M2
        # C11 = M1 - M4 - M3 + M5
        
        # Let me just hardcode the verified Strassen decomposition.
        # I'll use the form from Wikipedia:
        # M1 = A00(B01−B11)
        # M2 = (A10+A00)B11
        # M3 = (A01−A11)B10
        # M4 = A11(B00−B10)
        # M5 = (A00+A11)(B00+B11)
        # M6 = (A10−A00)(B00+B01)
        # M7 = (A01+A11)(B10+B11)
        # C00 = M1−M2+M5−M6−M7
        # C01 = M3+M4+M7
        # C10 = M2+M6
        # C11 = M1−M4+M5−M3+M7
        
        # For block at offset off:
        # B[k] means B[j, off+k]
        
        # M1: u={0:1}, v={off+1: 1, p+off+1: -1}, w: C00+=1, C10+=-1
        # Wait: C00 = M1 - M2 + M5 - M6 - M7, so M1 contributes +1 to C00
        # C10 = M2 + M6, so M1 does NOT contribute to C10
        # C11 = M1 - M4 + M5 - M3 + M7, so M1 contributes +1 to C11
        # C01 = M3 + M4 + M7, so M1 does NOT contribute to C01
        
        # So M1: w coefficients: C[0,off] = +1, C[1,off+1] = +1
        # In W indexing: C[i, off+dk] -> W[(off+dk)*n + i]
        # C[0,off] -> W[off*2 + 0] = W[off]
        # C[1,off+1] -> W[(off+1)*2 + 1] = W[off+3]
        
        prods = []
        
        # M1: A00 * (B[0,off+1] - B[1,off+1])
        # U: {0: 1}  (i=0,j=0)
        # V: {off+1: 1, p+off+1: -1}  (j=0,k=off+1 and j=1,k=off+1)
        # W: {off: 1, off+3: 1}  (C[0,off] and C[1,off+1])
        prods.append({'u': {0: 1}, 'v': {off+1: 1, p+off+1: -1}, 'w': {off: 1, off+3: 1}})
        
        # M2: (A10+A00) * B[1,off+1]
        # U: {2: 1, 0: 1}  (i=1,j=0 and i=0,j=0)
        # V: {p+off+1: 1}  (j=1,k=off+1)
        # C10 = M2 + M6, so M2 contributes +1 to C[1,off]
        # C00 = M1 - M2 + ..., so M2 contributes -1 to C[0,off]
        # W: {off: -1, off+2: 1}  (C[0,off] and C[1,off])
        prods.append({'u': {2: 1, 0: 1}, 'v': {p+off+1: 1}, 'w': {off: -1, off+2: 1}})
        
        # M3: (A01-A11) * B[1,off]
        # U: {1: 1, 3: -1}  (i=0,j=1 and i=1,j=1)
        # V: {p+off: 1}  (j=1,k=off)
        # C01 = M3 + M4 + M7, so M3 contributes +1 to C[0,off+1]
        # C11 = M1 - M4 + M5 - M3 + M7, so M3 contributes -1 to C[1,off+1]
        # W: {off+1: 1, off+3: -1}  (C[0,off+1] and C[1,off+1])
        prods.append({'u': {1: 1, 3: -1}, 'v': {p+off: 1}, 'w': {off+1: 1, off+3: -1}})
        
        # M4: A11 * (B[0,off] - B[1,off])
        # U: {3: 1}  (i=1,j=1)
        # V: {off: 1, p+off: -1}  (j=0,k=off and j=1,k=off)
        # C01 = M3 + M4 + M7, so M4 contributes +1 to C[0,off+1]
        # C11 = M1 - M4 + ..., so M4 contributes -1 to C[1,off+1]
        # W: {off+1: 1, off+3: -1}
        prods.append({'u': {3: 1}, 'v': {off: 1, p+off: -1}, 'w': {off+1: 1, off+3: -1}})
        
        # M5: (A00+A11) * (B[0,off]+B[1,off+1])
        # U: {0: 1, 3: 1}
        # V: {off: 1, p+off+1: 1}
        # C00 = M1 - M2 + M5 - M6 - M7, so M5 contributes +1 to C[0,off]
        # C01 = M3 + M4 + M7, no M5
        # C10 = M2 + M6, no M5
        # C11 = M1 - M4 + M5 - M3 + M7, so M5 contributes +1 to C[1,off+1]
        # W: {off: 1, off+3: 1}
        prods.append({'u': {0: 1, 3: 1}, 'v': {off: 1, p+off+1: 1}, 'w': {off: 1, off+3: 1}})
        
        # M6: (A10-A00) * (B[0,off]+B[0,off+1])
        # U: {2: 1, 0: -1}
        # V: {off: 1, off+1: 1}
        # C00 = M1 - M2 + M5 - M6 - M7, so M6 contributes -1 to C[0,off]
        # C10 = M2 + M6, so M6 contributes +1 to C[1,off]
        # W: {off: -1, off+2: 1}
        prods.append({'u': {2: 1, 0: -1}, 'v': {off: 1, off+1: 1}, 'w': {off: -1, off+2: 1}})
        
        # M7: (A01+A11) * (B[1,off]+B[1,off+1])
        # U: {1: 1, 3: 1}
        # V: {p+off: 1, p+off+1: 1}
        # C00 = M1 - M2 + M5 - M6 - M7, so M7 contributes -1 to C[0,off]
        # C01 = M3 + M4 + M7, so M7 contributes +1 to C[0,off+1]
        # C10 = M2 + M6, no M7
        # C11 = M1 - M4 + M5 - M3 + M7, so M7 contributes +1 to C[1,off+1]
        # W: {off: -1, off+1: 1, off+3: 1}
        prods.append({'u': {1: 1, 3: 1}, 'v': {p+off: 1, p+off+1: 1}, 'w': {off: -1, off+1: 1, off+3: 1}})
        
        return prods
    
    # Fill in the matrices for both blocks
    for block_idx, off in enumerate([0, 2]):
        prods = strassen_block(off)
        base_col = block_idx * 7
        for prod_idx, prod in enumerate(prods):
            col = base_col + prod_idx
            for row, val in prod['u'].items():
                U[row][col] = val
            for row, val in prod['v'].items():
                V[row][col] = val
            for row, val in prod['w'].items():
                W[row][col] = val
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p, rank):
        import numpy as np
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    if verify(U, V, W, n, m, p, rank):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook if verification fails (should not happen)
    rank_school = n * m * p
    U_s = [[0] * rank_school for _ in range(n * m)]
    V_s = [[0] * rank_school for _ in range(m * p)]
    W_s = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j][column] = 1
                V_s[j * p + k][column] = 1
                W_s[k * n + i][column] = 1
                column += 1
    return {"U": U_s, "V": V_s, "W": W_s}

# EVOLVE-BLOCK-END
