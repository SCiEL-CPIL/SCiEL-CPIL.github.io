import time
import numpy as np

DIMENSIONS = [2, 4, 7]
DEADLINE = 130

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify if U, V, W provide an exact decomposition for the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    rank = U.shape[1]
    U_ = U.reshape(n, m, rank)
    V_ = V.reshape(m, p, rank)
    W_ = W.reshape(p, n, rank)
    
    # Check all valid entries (i, j, k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                val = 0.0
                # Efficiently compute dot product
                val = np.dot(U_[i, j, :], np.dot(V_[j, k, :], W_[k, i, :]))
                
                if abs(val - 1.0) > tol:
                    return False
                    
    # Check a sample of invalid entries to ensure they are 0
    # Total entries: (n*m) * (m*p) * (p*n)
    # Valid: n*m*p
    # We check a few random invalid ones
    rng = np.random.RandomState(42)
    total_u = n * m
    total_v = m * p
    total_w = p * n
    
    for _ in range(200):
        u = rng.randint(0, total_u)
        v = rng.randint(0, total_v)
        w = rng.randint(0, total_w)
        
        i1 = u // m
        j1 = u % m
        j2 = v // p
        k2 = v % p
        k3 = w // n
        i3 = w % n
        
        # If indices match, it's a valid entry (already checked)
        if i1 == i3 and j1 == j2 and k2 == k3:
            continue
            
        val = np.dot(U_[i1, j1, :], np.dot(V_[j2, k2, :], W_[k3, i3, :]))
        
        if abs(val) > tol:
            return False
            
    return True

def _construct_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices for U (A): 0:A00, 1:A01, 2:A10, 3:A11
    Indices for V (B): 0:B00, 1:B01, 2:B10, 3:B11
    Indices for W (C): 0:C00, 1:C01, 2:C10, 3:C11
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00 + A11)(B00 + B11) -> C00, C11
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10 + A11)B00 -> C10, C11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1 # C10 += M2, C11 -= M2
    
    # M3 = A00(B01 - B11) -> C01, C11
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1 # C01 += M3, C11 += M3
    
    # M4 = A11(B10 - B00) -> C00, C10
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1 # C00 += M4, C10 += M4
    
    # M5 = (A00 + A01)B11 -> C00, C01
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1 # C00 -= M5, C01 += M5
    
    # M6 = (A10 - A00)(B00 + B01) -> C11
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C11 += M6
    
    # M7 = (A01 - A11)(B10 + B11) -> C00
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C00 += M7
    
    return U, V, W

def _construct_2x2x3_decomp():
    """
    Returns U, V, W for 2x2x3 matrix multiplication.
    A is 2x2, B is 2x3, C is 3x2.
    Rank is 12.
    U: (4, 12), V: (6, 12), W: (6, 12)
    U indices (A): 0:A00, 1:A01, 2:A10, 3:A11
    V indices (B): 0:B00, 1:B01, 2:B02, 3:B10, 4:B11, 5:B12
    W indices (C): 0:C00, 1:C01, 2:C02, 3:C10, 4:C11, 5:C12
    """
    U = np.zeros((4, 12))
    V = np.zeros((6, 12))
    W = np.zeros((6, 12))
    
    # We can view this as two 2x2x2 multiplications sharing the A matrix?
    # No, B is 2x3.
    # C = A B.
    # C = [A B0, A B1, A B2] where B0, B1, B2 are columns of B.
    # This is essentially 3 separate 2x2x1 multiplications? No, that would be rank 12 (3*4).
    # But we can do better?
    # Actually, for 2x2x3, the rank is known to be 12? 
    # Let's check: 2x2x2 is 7. 2x2x1 is 4.
    # If we split B into B_left (2x2) and B_right (2x1).
    # C_left = A B_left (rank 7)
    # C_right = A B_right (rank 4)
    # Total rank 11?
    # Let's verify if rank 11 is possible.
    # The tensor for 2x2x3 is a sub-tensor of 2x2x4 (which has rank 14).
    # Actually, let's just use the block decomposition.
    # Split B into two 2x2 blocks? No, B is 2x3.
    # Split B into 2x2 and 2x1.
    # The 2x2x2 part has rank 7.
    # The 2x2x1 part has rank 4.
    # Total 11.
    
    # Let's construct the 2x2x2 part (columns 0,1 of B)
    # And the 2x2x1 part (column 2 of B)
    
    # Part 1: 2x2x2 for B[:, 0:2]
    # We need to embed the 2x2x2 decomposition.
    # U1: (4, 7) from Strassen
    # V1: (4, 7) from Strassen, but V indices are j*2+k.
    #   Here B is 2x3. Indices: 0:B00, 1:B01, 2:B02, 3:B10, 4:B11, 5:B12.
    #   For the first 2x2 block of B (cols 0,1), indices are 0,1,3,4.
    #   Strassen V indices: 0:B00, 1:B01, 2:B10, 3:B11.
    #   Mapping: Strassen V[0]->V[0], V[1]->V[1], V[2]->V[3], V[3]->V[4].
    # W1: (4, 7) from Strassen.
    #   C indices for first 2 cols: 0:C00, 1:C01, 3:C10, 4:C11.
    #   Strassen W indices: 0:C00, 1:C01, 2:C10, 3:C11.
    #   Mapping: Strassen W[0]->W[0], W[1]->W[1], W[2]->W[3], W[3]->W[4].
    
    U_s, V_s, W_s = _construct_strassen_2x2x2()
    
    # Fill Part 1
    U[:, 0:7] = U_s
    # Map V_s to V
    V[0, 0:7] = V_s[0, :]
    V[1, 0:7] = V_s[1, :]
    V[3, 0:7] = V_s[2, :]
    V[4, 0:7] = V_s[3, :]
    
    # Map W_s to W
    W[0, 0:7] = W_s[0, :]
    W[1, 0:7] = W_s[1, :]
    W[3, 0:7] = W_s[2, :]
    W[4, 0:7] = W_s[3, :]
    
    # Part 2: 2x2x1 for B[:, 2:3]
    # This is just schoolbook for the single column.
    # A (2x2) * B_col (2x1) -> C_col (2x1)
    # C02 = A00 B02 + A01 B12
    # C12 = A10 B02 + A11 B12
    # Rank 4.
    
    # r=7: A00 B02 -> C02
    U[0, 7] = 1
    V[2, 7] = 1
    W[2, 7] = 1
    
    # r=8: A01 B12 -> C02
    U[1, 8] = 1
    V[5, 8] = 1
    W[2, 8] = 1
    
    # r=9: A10 B02 -> C12
    U[2, 9] = 1
    V[2, 9] = 1
    W[5, 9] = 1
    
    # r=10: A11 B12 -> C12
    U[3, 10] = 1
    V[5, 10] = 1
    W[5, 10] = 1
    
    # Total rank 11.
    # We allocated 12 columns. We can just return the first 11 or keep 12 with zeros.
    # Let's return with rank 11 to be minimal.
    return U[:, :11], V[:, :11], W[:, :11]

def _embed_block(n, m, p, U_block, V_block, W_block, i_start, j_start, k_start, U_full, V_full, W_full, col_start):
    """
    Embed a block decomposition into the full matrices.
    n, m, p are full dimensions.
    i_start, j_start, k_start are the starting indices of the block.
    U_block, V_block, W_block are the block decompositions.
    U_full, V_full, W_full are the full matrices to fill.
    col_start is the starting column index in the full matrices.
    Returns the next column index.
    """
    rank_block = U_block.shape[1]
    
    # U indices: i in [i_start, i_start + n_block), j in [j_start, j_start + m_block)
    # U_block index: (i - i_start) * m_block + (j - j_start)
    # U_full index: i * m + j
    
    # V indices: j in [j_start, j_start + m_block), k in [k_start, k_start + p_block)
    # V_block index: (j - j_start) * p_block + (k - k_start)
    # V_full index: j * p + k
    
    # W indices: k in [k_start, k_start + p_block), i in [i_start, i_start + n_block)
    # W_block index: (k - k_start) * n_block + (i - i_start)
    # W_full index: k * n + i
    
    n_block = U_block.shape[0] // m_block if m_block else 0 # This logic is fragile, better to pass block dims
    
    # Let's infer block dims from the start indices and the full dims?
    # Actually, we know the block dims from the context.
    # For 2x4x7:
    # Block 1: i=0..1, j=0..1, k=0..1 (2x2x2)
    # Block 2: i=0..1, j=2..3, k=0..1 (2x2x2)
    # Block 3: i=0..1, j=0..1, k=2..3 (2x2x2)
    # Block 4: i=0..1, j=2..3, k=2..3 (2x2x2)
    # Block 5: i=0..1, j=0..1, k=4..6 (2x2x3)
    # Block 6: i=0..1, j=2..3, k=4..6 (2x2x3)
    
    # We need to know the block sizes.
    # Let's pass them explicitly or infer.
    # Since U_block is (n_block * m_block, rank), etc.
    
    # For now, let's just do it manually in the main construct function to avoid generic embedding bugs.
    pass

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Total rank: 4 * 7 + 2 * 11 = 28 + 22 = 50.
    # Wait, 2x2x3 rank is 11.
    # 4 * 7 = 28.
    # 2 * 11 = 22.
    # Total 50.
    
    total_rank = 50
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    U_2x2, V_2x2, W_2x2 = _construct_strassen_2x2x2()
    U_2x3, V_2x3, W_2x3 = _construct_2x2x3_decomp()
    
    col = 0
    
    # Helper to embed a 2x2x2 block
    def embed_2x2_2x2_2x2(i0, j0, k0):
        nonlocal col
        # i0: 0 or 2? No, i is always 0..1 for n=2.
        # j0: 0 or 2.
        # k0: 0, 2.
        
        for r in range(7):
            r_global = col + r
            
            # U: i in 0..1, j in j0..j0+1
            for i in range(2):
                for dj in range(2):
                    val = U_2x2[i * 2 + dj, r]
                    if val != 0:
                        j = j0 + dj
                        U[i * m + j, r_global] = val
                        
            # V: j in j0..j0+1, k in k0..k0+1
            for dj in range(2):
                for dk in range(2):
                    val = V_2x2[dj * 2 + dk, r]
                    if val != 0:
                        j = j0 + dj
                        k = k0 + dk
                        V[j * p + k, r_global] = val
                        
            # W: k in k0..k0+1, i in 0..1
            for dk in range(2):
                for i in range(2):
                    val = W_2x2[dk * 2 + i, r]
                    if val != 0:
                        k = k0 + dk
                        W[k * n + i, r_global] = val
                        
        col += 7

    # Helper to embed a 2x2x3 block
    def embed_2x2_2x2_2x3(i0, j0, k0):
        nonlocal col
        # i0: 0.
        # j0: 0 or 2.
        # k0: 4.
        
        for r in range(11):
            r_global = col + r
            
            # U: i in 0..1, j in j0..j0+1
            for i in range(2):
                for dj in range(2):
                    val = U_2x3[i * 2 + dj, r]
                    if val != 0:
                        j = j0 + dj
                        U[i * m + j, r_global] = val
                        
            # V: j in j0..j0+1, k in k0..k0+2
            for dj in range(2):
                for dk in range(3):
                    val = V_2x3[dj * 3 + dk, r]
                    if val != 0:
                        j = j0 + dj
                        k = k0 + dk
                        V[j * p + k, r_global] = val
                        
            # W: k in k0..k0+2, i in 0..1
            for dk in range(3):
                for i in range(2):
                    val = W_2x3[dk * 2 + i, r]
                    if val != 0:
                        k = k0 + dk
                        W[k * n + i, r_global] = val
                        
        col += 11

    # Embed blocks:
    # j=0, k=0 (2x2x2)
    embed_2x2_2x2_2x2(0, 0, 0)
    # j=2, k=0 (2x2x2)
    embed_2x2_2x2_2x2(0, 2, 0)
    # j=0, k=2 (2x2x2)
    embed_2x2_2x2_2x2(0, 0, 2)
    # j=2, k=2 (2x2x2)
    embed_2x2_2x2_2x2(0, 2, 2)
    # j=0, k=4 (2x2x3)
    embed_2x2_2x2_2x3(0, 0, 4)
    # j=2, k=4 (2x2x3)
    embed_2x2_2x2_2x3(0, 2, 4)
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U_sb = np.zeros((n * m, rank))
        V_sb = np.zeros((m * p, rank))
        W_sb = np.zeros((p * n, rank))
        
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, c] = 1.0
                    V_sb[j * p + k, c] = 1.0
                    W_sb[k * n + i, c] = 1.0
                    c += 1
                    
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
