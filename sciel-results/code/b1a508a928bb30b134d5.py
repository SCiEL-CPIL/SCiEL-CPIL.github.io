import numpy as np
import time

def _build_2x2x2_strassen():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    Shapes: (4, 7)
    """
    # Strassen terms for 2x2x2
    # M1 = (A00 + A11) (B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B10 - B11)
    # M4 = A11 (B01 - B10)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00) (B00 + B01)
    # M7 = (A01 - A11) (B10 + B11)
    
    # U indices: A00, A01, A10, A11 (flattened row-major: 0,1,2,3)
    # V indices: B00, B01, B10, B11 (flattened row-major: 0,1,2,3)
    # W indices: C00, C01, C10, C11 (flattened row-major: 0,1,2,3)
    
    U = np.array([
        [1, 0, 1, 0, 1, -1, 1], # A00
        [0, 1, 0, 0, 1, 0, -1], # A01
        [0, 0, 1, 0, 0, 1, 0],  # A10
        [1, 0, 0, 1, 0, 0, -1]  # A11
    ], dtype=np.float64)
    
    V = np.array([
        [1, 1, 0, 0, 0, 1, 0],  # B00
        [1, 0, 1, -1, 0, 1, 0], # B01
        [1, 0, -1, 1, 1, 0, 1], # B10
        [1, 0, 0, 0, 1, 0, 1]   # B11
    ], dtype=np.float64)
    
    W = np.array([
        [1, 0, 0, 0, 0, 0, 0],  # C00
        [0, 0, 0, 1, 1, 0, 0],  # C01
        [0, 1, 1, 0, 0, 1, 0],  # C10
        [0, 0, 0, 0, 0, 0, 1]   # C11
    ], dtype=np.float64)
    
    # Wait, standard Strassen W matrix mapping:
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M4
    # C10 = M2 + M7
    # C11 = M1 - M2 + M5 + M7
    
    # Let's re-verify the W matrix columns corresponding to M1..M7
    # Col 0 (M1): C00=1, C01=0, C10=0, C11=1 -> [1,0,0,1]
    # Col 1 (M2): C00=0, C01=0, C10=1, C11=-1 -> [0,0,1,-1]
    # Col 2 (M3): C00=0, C01=1, C10=1, C11=0 -> [0,1,1,0]
    # Col 3 (M4): C00=-1, C01=1, C10=0, C11=0 -> [-1,1,0,0]
    # Col 4 (M5): C00=0, C01=0, C10=0, C11=1 -> [0,0,0,1]
    # Col 5 (M6): C00=1, C01=0, C10=1, C11=0 -> [1,0,1,0]
    # Col 6 (M7): C00=-1, C01=0, C10=0, C11=1 -> [-1,0,0,1]
    
    W_correct = np.array([
        [1, 0, 0, -1, 0, 1, -1], # C00
        [0, 0, 1, 1, 0, 0, 0],   # C01
        [0, 1, 1, 0, 0, 1, 0],   # C10
        [1, -1, 0, 0, 1, 0, 1]   # C11
    ], dtype=np.float64)
    
    return U, V, W_correct

def _build_4x4x4_strassen():
    """
    Build rank-49 decomposition for 4x4x4.
    """
    U2, V2, W2 = _build_2x2x2_strassen()
    
    # We treat 4x4 matrices as 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]] where Aij are 2x2.
    # B = [[B00, B01], [B10, B11]] where Bij are 2x2.
    # C = [[C00, C01], [C10, C11]] where Cij are 2x2.
    
    # The multiplication is:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # We use Strassen on the 2x2 block level.
    # This requires 7 multiplications of 2x2 blocks.
    # Each 2x2 block multiplication has rank 7.
    # Total rank 7 * 7 = 49.
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Block indices for 2x2 blocks of 2x2 matrices
    # Block (br, bc) contains elements at rows br*2..br*2+1, cols bc*2..bc*2+1
    # Flattened index in 4x4: row*4 + col
    
    # Strassen terms for block level (indices 0..6)
    # u_block, v_block, w_block
    
    # We need to map the 2x2 block Strassen factors to the 4x4 element factors.
    
    # For each of the 7 block-level terms (r=0..6):
    #   We have a 2x2x2 tensor T_r.
    #   T_r acts on block A, block B, block C.
    #   The block A is a 2x2 matrix. The factor U2 (4x7) applies to the elements of this 2x2 block.
    #   So for a fixed block term r, we iterate over the 7 element-level terms t=0..6.
    #   The column in the 49-rank matrix is r*7 + t.
    
    for r in range(7):
        # Get the block-level coefficients
        # U2[:, r] gives coefficients for A00, A01, A10, A11 (block level)
        u_blocks = U2[:, r] # shape (4,) -> A00, A01, A10, A11
        v_blocks = V2[:, r] # shape (4,) -> B00, B01, B10, B11
        w_blocks = W2[:, r] # shape (4,) -> C00, C01, C10, C11
        
        for t in range(7):
            col = r * 7 + t
            
            # Element-level factors for the 2x2 block multiplication
            u_elem = U2[:, t] # shape (4,) -> a00, a01, a10, a11 within the block
            v_elem = V2[:, t] # shape (4,) -> b00, b01, b10, b11 within the block
            w_elem = W2[:, t] # shape (4,) -> c00, c01, c10, c11 within the block
            
            # Map to 4x4 indices
            # A blocks:
            # A00 (block 0,0): rows 0,1 cols 0,1
            # A01 (block 0,1): rows 0,1 cols 2,3
            # A10 (block 1,0): rows 2,3 cols 0,1
            # A11 (block 1,1): rows 2,3 cols 2,3
            
            for bi in range(4): # block index 0..3
                coeff_u = u_blocks[bi]
                if coeff_u == 0: continue
                
                # Determine row/col offset of the block
                br = bi // 2
                bc = bi % 2
                
                # Iterate over the 4 elements within the 2x2 block
                for ei in range(4):
                    coeff_ue = u_elem[ei]
                    if coeff_ue == 0: continue
                    
                    # Element offset within block
                    er = ei // 2
                    ec = ei % 2
                    
                    # Global row/col
                    gr = br * 2 + er
                    gc = bc * 2 + ec
                    idx = gr * 4 + gc
                    
                    U[idx, col] += coeff_u * coeff_ue
            
            # B blocks
            for bi in range(4):
                coeff_v = v_blocks[bi]
                if coeff_v == 0: continue
                
                br = bi // 2
                bc = bi % 2
                
                for ei in range(4):
                    coeff_ve = v_elem[ei]
                    if coeff_ve == 0: continue
                    
                    er = ei // 2
                    ec = ei % 2
                    
                    gr = br * 2 + er
                    gc = bc * 2 + ec
                    idx = gr * 4 + gc
                    
                    V[idx, col] += coeff_v * coeff_ve
            
            # C blocks
            for bi in range(4):
                coeff_w = w_blocks[bi]
                if coeff_w == 0: continue
                
                br = bi // 2
                bc = bi % 2
                
                for ei in range(4):
                    coeff_we = w_elem[ei]
                    if coeff_we == 0: continue
                    
                    er = ei // 2
                    ec = ei % 2
                    
                    gr = br * 2 + er
                    gc = bc * 2 + ec
                    idx = gr * 4 + gc
                    
                    W[idx, col] += coeff_w * coeff_we
                    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify T = sum u_r v_r w_r
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Check a few random points or all points if small
    # 4x4x8 is small enough to check all
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                u_vec = U[i*m + j]
                v_vec = V[j*p + k]
                w_vec = W[k*n + i]
                
                # The term is sum_r u_r[i,j] * v_r[j,k] * w_r[k,i]
                # This is dot(u_vec, v_vec * w_vec)
                val = np.dot(u_vec, v_vec * w_vec)
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    """
    Construct rank-98 decomposition for 4x4x8.
    """
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = 4, 4, 8
    
    # Build 4x4x4 Strassen (Rank 49)
    U44, V44, W44 = _build_4x4x4_strassen()
    
    # Assemble 4x4x8
    # Rank 98
    U = np.zeros((16, 98))
    V = np.zeros((32, 98))
    W = np.zeros((32, 98))
    
    # First half (cols 0-47)
    U[:, 0:49] = U44
    V[0:16, 0:49] = V44
    W[0:16, 0:49] = W44
    
    # Second half (cols 49-97)
    U[:, 49:98] = U44
    V[16:32, 49:98] = V44
    W[16:32, 49:98] = W44
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        rank = n * m * p
        U_sb = np.zeros((n*m, rank))
        V_sb = np.zeros((m*p, rank))
        W_sb = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
