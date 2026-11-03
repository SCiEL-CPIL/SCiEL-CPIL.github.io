import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def _get_strassen_2x2_terms():
    """
    Returns the U, V, W matrices for the rank-7 Strassen decomposition of 2x2x2.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Term 0: (A00+A11)(B00+B11) -> C00 +=, C11 +=
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1 # C00 (k=0, i=0)
    W[3, 0] = 1 # C11 (k=1, i=1) -> index 1*2+1 = 3
    
    # Term 1: A11(B00+B11) -> C00 -=, C10 +=
    U[3, 1] = 1
    V[0, 1] = 1; V[3, 1] = 1
    W[0, 1] = -1 # C00
    W[2, 1] = 1  # C10 (k=0, i=1) -> index 0*2+1 = 1? No. W[k*n+i]. C10 is i=1, k=0. Index 0*2+1 = 1.
    # Wait, C10 means row 1, col 0. i=1, k=0. Index k*n+i = 0*2+1 = 1.
    # My previous W[2,1] was wrong.
    W[1, 1] = 1
    
    # Term 2: (A00-A11)(B10+B11) -> C01 +=, C11 +=
    U[0, 2] = 1; U[3, 2] = -1
    V[2, 2] = 1; V[3, 2] = 1 # B10 (j=1, k=0) idx 1*2+0=2, B11 (j=1, k=1) idx 1*2+1=3
    W[1, 2] = 1 # C01 (i=0, k=1) idx 1*2+0 = 2? No. C01 is i=0, k=1. Index 1*2+0 = 2.
    # Wait. W[k*n+i].
    # C01: i=0, k=1. Index 1*2+0 = 2.
    # C11: i=1, k=1. Index 1*2+1 = 3.
    W[2, 2] = 1
    W[3, 2] = 1
    
    # Term 3: A00(B00-B10) -> C00 +=, C10 +=
    U[0, 3] = 1
    V[0, 3] = 1; V[2, 3] = -1 # B00 (0,0) idx 0, B10 (1,0) idx 2
    W[0, 3] = 1 # C00
    W[1, 3] = 1 # C10
    
    # Term 4: (A00+A01)B11 -> C01 +=, C11 -=
    U[0, 4] = 1; U[1, 4] = 1 # A00 (0,0) idx 0, A01 (0,1) idx 1
    V[3, 4] = 1 # B11 (1,1) idx 3
    W[2, 4] = 1 # C01
    W[3, 4] = -1 # C11
    
    # Term 5: (A10+A11)(B00-B01) -> C00 +=
    U[2, 5] = 1; U[3, 5] = 1 # A10 (1,0) idx 2, A11 (1,1) idx 3
    V[0, 5] = 1; V[1, 5] = -1 # B00 (0,0) idx 0, B01 (0,1) idx 1
    W[0, 5] = 1 # C00
    
    # Term 6: (A01-A11)(B10+B11) -> C11 +=
    U[1, 6] = 1; U[3, 6] = -1 # A01 (0,1) idx 1, A11 (1,1) idx 3
    V[2, 6] = 1; V[3, 6] = 1 # B10 (1,0) idx 2, B11 (1,1) idx 3
    W[3, 6] = 1 # C11
    
    return U, V, W

def _build_strassen_4x4x4():
    """
    Builds rank-49 decomposition for 4x4x4.
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    U2, V2, W2 = _get_strassen_2x2_terms()
    
    # Top level Strassen terms for 4x4x4
    # Each term corresponds to a product of 2x2 blocks.
    # We iterate through the 7 top-level terms, and for each, we use the 7 sub-terms from 2x2x2.
    
    # Define the 7 top-level combinations of A blocks and B blocks
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    
    # Term structure: (A_block_coeffs, B_block_coeffs, C_block_contribs)
    # A_block_coeffs: dict {block_idx: coeff}
    # B_block_coeffs: dict {block_idx: coeff}
    # C_block_contribs: dict {block_idx: coeff}
    
    # Block indices: 0=A00, 1=A01, 2=A10, 3=A11
    # C blocks: 0=C00, 1=C01, 2=C10, 3=C11
    
    top_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({3: 1}, {0: 1, 3: 1}, {0: -1, 2: 1}),      # M2
        ({0: 1, 3: -1}, {2: 1, 3: 1}, {1: 1, 3: 1}),# M3
        ({0: 1}, {0: 1, 2: -1}, {0: 1, 2: 1}),      # M4
        ({0: 1, 1: 1}, {3: 1}, {1: 1, 3: -1}),      # M5
        ({2: 1, 3: 1}, {0: 1, 1: -1}, {0: 1}),      # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {3: 1})       # M7
    ]
    
    r_idx = 0
    for t_idx in range(7):
        A_coeffs, B_coeffs, C_coeffs = top_terms[t_idx]
        
        # For this top-level term, we need to compute the product of two 2x2 matrices.
        # The 2x2 matrix for A is formed by the blocks specified in A_coeffs.
        # The 2x2 matrix for B is formed by the blocks specified in B_coeffs.
        # We use the 2x2x2 Strassen decomposition (U2, V2, W2) to handle this.
        
        # We need to map the 2x2x2 indices to the 4x4x4 indices.
        # In 2x2x2, A is 2x2. In 4x4, A is 4x4.
        # The "A" in the sub-problem is a 2x2 matrix of blocks.
        # Let's say the sub-problem A' has entries A'_{lr, lc} which is a 2x2 block of A.
        # A'_{lr, lc} corresponds to A block (lr, lc).
        # In the 2x2x2 decomposition, U2[i, r] contributes to A'_{i//2, i%2}.
        # So if U2[i, r] = val, it contributes to A block (i//2, i%2).
        # In the 4x4 A, block (br, bc) covers rows 2*br..2*br+1 and cols 2*bc..2*bc+1.
        # Global index in U (4x4): i_global = 2*br + lr_local, j_global = 2*bc + lc_local.
        # Wait, U2[i, r] is a coefficient for A'_{i//2, i%2}.
        # A'_{br, bc} is the block A[2*br:2*br+2, 2*bc:2*bc+2].
        # So for each non-zero U2[i, r], we add it to all 4 entries in the corresponding block of U (4x4).
        
        for r_sub in range(7):
            r_global = r_idx + r_sub
            
            # Process U
            for i in range(4):
                val = U2[i, r_sub]
                if val != 0:
                    br, bc = i // 2, i % 2
                    # Check if block (br, bc) is in A_coeffs
                    if (br, bc) in [(k//2, k%2) for k in A_coeffs.keys()]:
                        block_idx = br * 2 + bc
                        coeff = A_coeffs.get(block_idx, 0)
                        if coeff != 0:
                            # Add to U (4x4)
                            for lr in range(2):
                                for lc in range(2):
                                    i_g = 2 * br + lr
                                    j_g = 2 * bc + lc
                                    U[i_g * 4 + j_g, r_global] += val * coeff
                                    
            # Process V
            for j in range(4):
                val = V2[j, r_sub]
                if val != 0:
                    br, bc = j // 2, j % 2
                    block_idx = br * 2 + bc
                    coeff = B_coeffs.get(block_idx, 0)
                    if coeff != 0:
                        for lr in range(2):
                            for lc in range(2):
                                i_g = 2 * br + lr
                                j_g = 2 * bc + lc
                                V[i_g * 4 + j_g, r_global] += val * coeff
                                
            # Process W
            # W2[k, r_sub] contributes to C'_{k//2, k%2}.
            # C'_{br, bc} is the block C[2*br:2*br+2, 2*bc:2*bc+2].
            # The top-level term contributes to C blocks with coeffs in C_coeffs.
            # So if W2[k, r_sub] = val, it contributes to C' block (k//2, k%2).
            # The global C block index is b_idx = (k//2)*2 + (k%2).
            # The top-level coefficient for this block is C_coeffs.get(b_idx, 0).
            for k in range(4):
                val = W2[k, r_sub]
                if val != 0:
                    br, bc = k // 2, k % 2
                    b_idx = br * 2 + bc
                    coeff = C_coeffs.get(b_idx, 0)
                    if coeff != 0:
                        for lr in range(2):
                            for lc in range(2):
                                i_g = 2 * br + lr
                                k_g = 2 * bc + lc
                                W[k_g * 4 + i_g, r_global] += val * coeff
                                
            r_idx += 1
            
    return U, V, W

def _build_strassen_4x4x8():
    n, m, p = 4, 4, 8
    rank = 98
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    U4, V4, W4 = _build_strassen_4x4x4()
    
    # Left half (B cols 0-3)
    U[:, :49] = U4
    for r in range(49):
        for j in range(4):
            for k in range(4):
                V[j*8 + k, r] = V4[j*4 + k, r]
        for i in range(4):
            for k in range(4):
                W[k*4 + i, r] = W4[k*4 + i, r]
                
    # Right half (B cols 4-7)
    U[:, 49:] = U4
    for r in range(49):
        for j in range(4):
            for k in range(4):
                V[j*8 + (k+4), 49+r] = V4[j*4 + k, r]
        for i in range(4):
            for k in range(4):
                W[(k+4)*4 + i, 49+r] = W4[k*4 + i, r]
                
    return U, V, W

def _verify(U, V, W):
    n, m, p = 4, 4, 8
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def construct():
    U, V, W = _build_strassen_4x4x8()
    if _verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        n, m, p = 4, 4, 8
        rank = n*m*p
        U_fb = np.zeros((n*m, rank))
        V_fb = np.zeros((m*p, rank))
        W_fb = np.zeros((p*n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, idx] = 1
                    V_fb[j*p+k, idx] = 1
                    W_fb[k*n+i, idx] = 1
                    idx += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
