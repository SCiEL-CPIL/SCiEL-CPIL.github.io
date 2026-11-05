import time
import numpy as np
import itertools

DIMENSIONS = [6, 6, 6]

# Strassen steps for 2x2 matrix multiplication
STRASSEN_STEPS = [
    ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
    ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,1)]),
    ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,0,-1),(0,1,1)]),
    ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,1,-1),(1,1,-1)]),
    ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,1,1),(1,1,1)]),
    ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,0,1)]),
    ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
]

def _build_strassen_2x2():
    """Build rank-7 decomposition for 2x2 matrix multiplication."""
    rank = 7
    U = np.zeros((4, rank), dtype=np.float64)
    V = np.zeros((4, rank), dtype=np.float64)
    W = np.zeros((4, rank), dtype=np.float64)
    
    for r, (a_terms, b_terms, c_terms) in enumerate(STRASSEN_STEPS):
        for i, j, coeff in a_terms:
            U[i * 2 + j, r] += coeff
        for j, k, coeff in b_terms:
            V[j * 2 + k, r] += coeff
        for k, i, coeff in c_terms:
            W[k * 2 + i, r] += coeff
    
    return U, V, W

def _build_strassen_4x4():
    """Build rank-49 decomposition for 4x4 using Strassen tensor square."""
    rank = 49
    U = np.zeros((16, rank), dtype=np.float64)
    V = np.zeros((16, rank), dtype=np.float64)
    W = np.zeros((16, rank), dtype=np.float64)
    
    col_idx = 0
    for step_a in STRASSEN_STEPS:
        a_terms_a, b_terms_a, c_terms_a = step_a
        for step_b in STRASSEN_STEPS:
            a_terms_b, b_terms_b, c_terms_b = step_b
            
            for I_outer, J_outer, ca_outer in a_terms_a:
                for I_inner, J_inner, ca_inner in a_terms_b:
                    row = I_outer * 2 + I_inner
                    col = J_outer * 2 + J_inner
                    coeff = ca_outer * ca_inner
                    U[row * 4 + col, col_idx] += coeff
            
            for J_outer, K_outer, cb_outer in b_terms_a:
                for J_inner, K_inner, cb_inner in b_terms_b:
                    row = J_outer * 2 + J_inner
                    col = K_outer * 2 + K_inner
                    coeff = cb_outer * cb_inner
                    V[row * 4 + col, col_idx] += coeff
            
            for I_outer, K_outer, cc_outer in c_terms_a:
                for I_inner, K_inner, cc_inner in c_terms_b:
                    row = K_outer * 2 + K_inner
                    col = I_outer * 2 + I_inner
                    coeff = cc_outer * cc_inner
                    W[row * 4 + col, col_idx] += coeff
            
            col_idx += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, sample_points=10):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = U.shape[1]
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    
    np.random.seed(42)
    for _ in range(sample_points):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        for i in range(n):
            for k in range(p):
                val = 0.0
                for j in range(m):
                    t_val = 0.0
                    for r in range(rank):
                        t_val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    val += A[i, j] * B[j, k] * t_val
                if abs(val - C_expected[i, k]) > 1e-8:
                    return False
    return True

def _build_direct_decomposition(n, m, p):
    """Build the direct (naive) rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, idx] = 1
                V[j*p + k, idx] = 1
                W[k*n + i, idx] = 1
                idx += 1
    
    return U, V, W

def _build_6x6_hybrid():
    """
    Build a 6x6 decomposition using 4x4 Strassen (rank 49) for the top-left
    4x4 block, and direct multiplication for remaining terms.
    
    Partition 6x6 as:
    A = [[A_TL(4x4), A_TR(4x2)], [A_BL(2x4), A_BR(2x2)]]
    B = [[B_TL(4x4), B_TR(4x2)], [B_BL(2x4), B_BR(2x2)]]
    
    C_TL = A_TL*B_TL + A_TR*B_BL  -> use Strassen for A_TL*B_TL (rank 49), direct for A_TR*B_BL
    C_TR = A_TL*B_TR + A_TR*B_BR  -> direct
    C_BL = A_BL*B_TL + A_BR*B_BL  -> direct  
    C_BR = A_BL*B_TR + A_BR*B_BR  -> direct
    """
    n, m, p = 6, 6, 6
    dim = 6
    
    # Rank for A_TL * B_TL using Strassen: 49
    # A_TR(4x2) * B_BL(2x4): 4*2*4 = 32
    # A_TL(4x4) * B_TR(4x2): 4*4*2 = 32
    # A_TR(4x2) * B_BR(2x2): 4*2*2 = 16
    # A_BL(2x4) * B_TL(4x4): 2*4*4 = 32
    # A_BR(2x2) * B_BL(2x4): 2*2*4 = 16
    # A_BL(2x4) * B_TR(4x2): 2*4*2 = 16
    # A_BR(2x2) * B_BR(2x2): 2*2*2 = 8
    
    total_rank = 49 + 32 + 32 + 16 + 32 + 16 + 16 + 8
    # = 49 + 32 + 32 + 16 + 32 + 16 + 16 + 8 = 201
    
    # Hmm, that's worse than 189. Let me try a different partition.
    # Actually, let's use the known 189 approach: 3x3 blocks with Strassen.
    # Partition into 3x3 blocks: 4 blocks of 3x3.
    # For each block pair, use a rank-? decomposition for 3x3.
    # Best known for 3x3 is 19 (Strassen), so 8 pairs * 19 = 152? No...
    # C11 = A11*B11 + A12*B21: two separate 3x3 multiplications
    # So 8 block products, each needing a 3x3 decomposition.
    # If we use rank-19 for 3x3: 8 * 19 = 152. But that's not standard.
    
    # Let me just use the verified 189 approach from the inspiration:
    # 3x3 block structure with 7-product Strassen per 2x2 sub-block.
    # Actually the inspiration says "3x3x3 block structure with 7-product Strassen per block"
    # giving 27 * 7 = 189.
    
    # 27 comes from: partition into 3x3 blocks (4 blocks), but the 27 is...
    # Actually 6/2 = 3, so 3x3 grid of 2x2 blocks = 9 blocks.
    # For each of the 9 block positions in C, we sum over 3 block products.
    # But using Strassen at the block level (3x3 blocks of 2x2):
    # The block matrix is 3x3, and we use a rank-? decomposition for 3x3 blocks.
    # If we use direct for 3x3 blocks: 3*3*3 = 27 block products, each of rank 7 (Strassen 2x2)
    # Total: 27 * 7 = 189.
    
    # So: partition 6x6 into 3x3 grid of 2x2 blocks.
    # Block (bi, bj) is at rows bi*2..bi*2+1, cols bj*2..bj*2+1.
    # C_block(ci, ck) = sum_{bj} A_block(bi, bj) * B_block(bj, ck)
    # For each (bi, bj, ck) triple, use Strassen rank-7 for the 2x2 multiplication.
    # Number of triples: 3*3*3 = 27. Total rank: 27*7 = 189.
    
    U_strassen_2x2, V_strassen_2x2, W_strassen_2x2 = _build_strassen_2x2()
    
    total_rank = 27 * 7
    U = np.zeros((36, total_rank), dtype=np.float64)
    V = np.zeros((36, total_rank), dtype=np.float64)
    W = np.zeros((36, total_rank), dtype=np.float64)
    
    col_idx = 0
    for bi in range(3):
        for bj in range(3):
            for ck in range(3):
                # This term computes A_block(bi,bj) * B_block(bj,ck) -> C_block(bi,ck)
                for r in range(7):
                    col = col_idx + r
                    # A-side: block (bi, bj)
                    for i_local in range(2):
                        for j_local in range(2):
                            global_i = bi * 2 + i_local
                            global_j = bj * 2 + j_local
                            U[global_i * 6 + global_j, col] = U_strassen_2x2[i_local * 2 + j_local, r]
                    
                    # B-side: block (bj, ck)
                    for j_local in range(2):
                        for k_local in range(2):
                            global_j = bj * 2 + j_local
                            global_k = ck * 2 + k_local
                            V[global_j * 6 + global_k, col] = V_strassen_2x2[j_local * 2 + k_local, r]
                    
                    # C-side: block (bi, ck)
                    for k_local in range(2):
                        for i_local in range(2):
                            global_k = bi * 2 + k_local
                            global_i = ck * 2 + i_local
                            W[global_k * 6 + global_i, col] = W_strassen_2x2[k_local * 2 + i_local, r]
                
                col_idx += 7
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    
    # Build the 6x6 decomposition using 3x3 grid of 2x2 blocks with Strassen
    U, V, W = _build_6x6_hybrid()
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p, sample_points=5):
        # Fall back to direct
        U_direct, V_direct, W_direct = _build_direct_decomposition(n, m, p)
        return {"U": U_direct.tolist(), "V": V_direct.tolist(), "W": W_direct.tolist()}
    
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
