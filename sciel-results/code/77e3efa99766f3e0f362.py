import time
import numpy as np

DIMENSIONS = [4, 4, 7]
DEADLINE_SECONDS = 145.0

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Verify with random matrices
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            A_c = np.zeros((n, m))
            for idx in range(n * m):
                i, j = idx // m, idx % m
                A_c[i, j] = uc[idx]
            B_c = np.zeros((m, p))
            for idx in range(m * p):
                j, k = idx // p, idx % p
                B_c[j, k] = vc[idx]
            C_c = np.zeros((n, p))
            for idx in range(p * n):
                k, i = idx // n, idx % n
                C_c[i, k] = wc[idx]
            C_computed += C_c
        if not np.allclose(C_computed, C_expected, atol=tol, rtol=tol):
            return False
    return True

def _build_strassen_444(U, V, W):
    """
    Fills the first 49 columns of U, V, W with the rank-49 Strassen decomposition
    for 4x4 x 4x4 matrix multiplication.
    """
    # U, V, W are numpy arrays of shape (16, 97)
    # We will fill columns 0 to 48.
    
    # Define the Strassen steps for 2x2 x 2x2
    # Returns a list of 7 tuples: (u_coeff_dict, v_coeff_dict, w_coeff_dict)
    # u_coeff_dict maps (r, c) to coeff
    # This is for a single 2x2 block multiplication.
    
    def get_2x2_strassen():
        # Returns list of 7 (A_coeffs, B_coeffs, C_coeffs)
        # A_coeffs: dict {(r,c): coeff}
        # B_coeffs: dict {(r,c): coeff}
        # C_coeffs: dict {(r,c): coeff}
        
        terms = []
        
        # M1 = (A00+A11)(B00+B11)
        a1 = {(0,0):1, (1,1):1}
        b1 = {(0,0):1, (1,1):1}
        c1 = {(0,0):1, (1,1):1} # M1 contributes to C00 and C11
        terms.append((a1, b1, c1))
        
        # M2 = (A10+A11)B00
        a2 = {(1,0):1, (1,1):1}
        b2 = {(0,0):1}
        c2 = {(1,0):1} # M2 contributes to C10
        terms.append((a2, b2, c2))
        
        # M3 = A00(B01-B11)
        a3 = {(0,0):1}
        b3 = {(0,1):1, (1,1):-1}
        c3 = {(0,1):1, (1,1):1} # M3 contributes to C01 and C11
        terms.append((a3, b3, c3))
        
        # M4 = A11(B10-B00)
        a4 = {(1,1):1}
        b4 = {(1,0):1, (0,0):-1}
        c4 = {(0,0):1, (1,0):1} # M4 contributes to C00 and C10
        terms.append((a4, b4, c4))
        
        # M5 = (A00+A01)B11
        a5 = {(0,0):1, (0,1):1}
        b5 = {(1,1):1}
        c5 = {(0,0):-1, (0,1):1} # M5 contributes to C00 (-) and C01 (+)
        terms.append((a5, b5, c5))
        
        # M6 = (A10-A00)(B00+B01)
        a6 = {(1,0):1, (0,0):-1}
        b6 = {(0,0):1, (0,1):1}
        c6 = {(1,1):1} # M6 contributes to C11
        terms.append((a6, b6, c6))
        
        # M7 = (A01-A11)(B10+B11)
        a7 = {(0,1):1, (1,1):-1}
        b7 = {(1,0):1, (1,1):1}
        c7 = {(0,0):1} # M7 contributes to C00
        terms.append((a7, b7, c7))
        
        return terms

    strassen_2x2 = get_2x2_strassen()
    
    # Level 1: 4x4 x 4x4
    # 7 products P0..P6
    # Each P is a 2x2 matrix product.
    
    # Define the linear combinations for A and B blocks for each P
    # P0 = (A00+A11)(B00+B11)
    # P1 = (A10+A11)B00
    # P2 = A00(B01-B11)
    # P3 = A11(B10-B00)
    # P4 = (A00+A01)B11
    # P5 = (A10-A00)(B00+B01)
    # P6 = (A01-A11)(B10+B11)
    
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    
    # For each P, we need to map the 2x2 Strassen terms to the global 4x4 indices.
    
    # Helper to add a term
    col_idx = 0
    for p_idx in range(7):
        # Determine A and B block coefficients for this P
        # A_coeffs: dict {(block_r, block_c): coeff}
        # B_coeffs: dict {(block_r, block_c): coeff}
        
        if p_idx == 0:
            a_blocks = {(0,0):1, (1,1):1}
            b_blocks = {(0,0):1, (1,1):1}
        elif p_idx == 1:
            a_blocks = {(1,0):1, (1,1):1}
            b_blocks = {(0,0):1}
        elif p_idx == 2:
            a_blocks = {(0,0):1}
            b_blocks = {(0,1):1, (1,1):-1}
        elif p_idx == 3:
            a_blocks = {(1,1):1}
            b_blocks = {(1,0):1, (0,0):-1}
        elif p_idx == 4:
            a_blocks = {(0,0):1, (0,1):1}
            b_blocks = {(1,1):1}
        elif p_idx == 5:
            a_blocks = {(1,0):1, (0,0):-1}
            b_blocks = {(0,0):1, (0,1):1}
        else: # p_idx == 6
            a_blocks = {(0,1):1, (1,1):-1}
            b_blocks = {(1,0):1, (1,1):1}
            
        # For each of the 7 terms in the 2x2 Strassen of this P
        for t_idx in range(7):
            a_local, b_local, c_local = strassen_2x2[t_idx]
            
            # a_local maps (r,c) in 2x2 block to coeff
            # We need to expand this to global 4x4 indices
            a_global = {}
            for (br, bc), coeff_a_block in a_blocks.items():
                for (lr, lc), coeff_a_local in a_local.items():
                    # Global row = br*2 + lr
                    # Global col = bc*2 + lc
                    gr = br * 2 + lr
                    gc = bc * 2 + lc
                    a_global[(gr, gc)] = a_global.get((gr, gc), 0) + coeff_a_block * coeff_a_local
            
            b_global = {}
            for (br, bc), coeff_b_block in b_blocks.items():
                for (lr, lc), coeff_b_local in b_local.items():
                    gr = br * 2 + lr
                    gc = bc * 2 + lc
                    b_global[(gr, gc)] = b_global.get((gr, gc), 0) + coeff_b_block * coeff_b_local
            
            # c_local maps (r,c) in 2x2 result block to coeff
            # This P contributes to specific 2x2 blocks of the 4x4 result
            # P0 contributes to C00 (+1) and C11 (+1)
            # P1 contributes to C10 (+1)
            # P2 contributes to C01 (+1) and C11 (+1) -- Wait, let's re-verify P2
            # P2 = A00(B01-B11).
            # C01 = P2 + P4. So P2 contributes +1 to C01.
            # C11 = P0 - P1 + P2 + P5. So P2 contributes +1 to C11.
            # Correct.
            
            # Let's define the contribution of each P to the 4x4 result blocks
            # C blocks: 00, 01, 10, 11
            c_block_coeffs = {}
            if p_idx == 0:
                c_block_coeffs = {(0,0):1, (1,1):1}
            elif p_idx == 1:
                c_block_coeffs = {(1,0):1}
            elif p_idx == 2:
                c_block_coeffs = {(0,1):1, (1,1):1}
            elif p_idx == 3:
                c_block_coeffs = {(0,0):1, (1,0):1}
            elif p_idx == 4:
                c_block_coeffs = {(0,0):-1, (0,1):1}
            elif p_idx == 5:
                c_block_coeffs = {(1,1):1}
            else: # p_idx == 6
                c_block_coeffs = {(0,0):1}
                
            c_global = {}
            for (br, bc), coeff_c_block in c_block_coeffs.items():
                for (lr, lc), coeff_c_local in c_local.items():
                    gr = br * 2 + lr
                    gc = bc * 2 + lc
                    c_global[(gr, gc)] = c_global.get((gr, gc), 0) + coeff_c_block * coeff_c_local
            
            # Now fill U, V, W for this column
            for (i, j), val in a_global.items():
                U[i*4+j, col_idx] = val
            for (j, k), val in b_global.items():
                V[j*4+k, col_idx] = val
            for (i, k), val in c_global.items():
                # W index is k*4 + i
                W[k*4+i, col_idx] = val
            
            col_idx += 1

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Rank 97: 49 (Strassen 4x4x4) + 48 (Schoolbook for 3 cols)
    rank = 97
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Fill Strassen part
    _build_strassen_444(U, V, W)
    
    # Fill Schoolbook part for k=4,5,6
    col_idx = 49
    for k in range(4, 7):
        for i in range(n):
            for j in range(m):
                U[i*m+j, col_idx] = 1
                V[j*p+k, col_idx] = 1
                W[k*n+i, col_idx] = 1
                col_idx += 1
                
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        rank = n*m*p
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
                    if time.time() - start_time > DEADLINE_SECONDS:
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
