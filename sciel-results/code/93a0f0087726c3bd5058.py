import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication via recursive Strassen.
    
    Uses the 7-product Strassen algorithm applied to 2x2 blocks of the 4x4 matrices,
    where each 2x2 block multiplication is also done via Strassen (7 products).
    Total rank = 7 * 7 = 49.
    
    Interface:
    U: shape (n*m, rank) = (16, 49), indexed by U[i*m+k, r] for A[i,k]
    V: shape (m*p, rank) = (16, 49), indexed by V[k*p+j, r] for B[k,j]
    W: shape (p*n, rank) = (16, 49), indexed by W[j*n+i, r] for C[i,j] (Trace convention)
    """
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill
    
    n = 4
    m = 4
    p = 4
    rank = 49
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get indices
    def u_idx(i, k):
        return i * m + k
        
    def v_idx(k, j):
        return k * p + j
        
    def w_idx(i, j):
        # Trace convention from current working code: W[j*n+i, r] for C[i,j]
        return j * n + i

    # Define Strassen terms for 2x2 matrices
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A*B
    # 
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We define the terms as (A_coeff, B_coeff, C_coeff)
    # A_coeff: dict {(row, col): coeff}
    # B_coeff: dict {(row, col): coeff}
    # C_coeff: dict {(row, col): coeff}
    
    # The 7 terms of Strassen
    strassen_terms = [
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # Outer level: 4x4 split into 2x2 blocks
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # 
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    #
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # Outer terms: (A_block_coeffs, B_block_coeffs, C_block_contribs)
    # A_block_coeffs: dict {block_idx: coeff} where block_idx is (br, bc)
    # B_block_coeffs: dict {block_idx: coeff}
    # C_block_contribs: dict {block_idx: coeff}
    
    outer_terms = [
        # P1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # P2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # P3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # P4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # P5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # P6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # P7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # Populate U, V, W
    for t_outer in range(7):
        a_blocks, b_blocks, c_blocks = outer_terms[t_outer]
        
        for t_inner in range(7):
            a_elems, b_elems, c_elems = strassen_terms[t_inner]
            r_idx = t_outer * 7 + t_inner
            
            # Fill U
            for (br, bc), coeff_a_block in a_blocks.items():
                for (er, ec), coeff_a_elem in a_elems.items():
                    global_row = 2 * br + er
                    global_col = 2 * bc + ec
                    u_idx_val = u_idx(global_row, global_col)
                    U[u_idx_val][r_idx] += coeff_a_block * coeff_a_elem
                    
            # Fill V
            for (br, bc), coeff_b_block in b_blocks.items():
                for (er, ec), coeff_b_elem in b_elems.items():
                    global_row = 2 * br + er
                    global_col = 2 * bc + ec
                    v_idx_val = v_idx(global_row, global_col)
                    V[v_idx_val][r_idx] += coeff_b_block * coeff_b_elem
                    
            # Fill W
            for (br, bc), coeff_c_block in c_blocks.items():
                for (er, ec), coeff_c_elem in c_elems.items():
                    global_row = 2 * br + er
                    global_col = 2 * bc + ec
                    w_idx_val = w_idx(global_row, global_col)
                    W[w_idx_val][r_idx] += coeff_c_block * coeff_c_elem
    
    # Verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    np.random.seed(42)
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 4)
    C_expected = A_test @ B_test
    
    # Reshape for einsum
    # U: (16, 49) -> (4, 4, 49) where U[i, k, r]
    U4 = U_np.reshape(4, 4, 49)
    # V: (16, 49) -> (4, 4, 49) where V[k, j, r]
    V4 = V_np.reshape(4, 4, 49)
    # W: (16, 49) -> (4, 4, 49) where W[j, i, r] (since index is j*4+i)
    W4 = W_np.reshape(4, 4, 49)
    
    # Compute C
    # u_vals[i, r] = sum_k U4[i, k, r] * A[i, k]
    u_vals = np.einsum('ikr,ik->ir', U4, A_test)
    # v_vals[j, r] = sum_k V4[k, j, r] * B[k, j]
    v_vals = np.einsum('kjr,kj->jr', V4, B_test)
    # C[i, j] = sum_r W4[j, i, r] * u_vals[i, r] * v_vals[j, r]
    C_computed = np.einsum('jir,ir,jr->ij', W4, u_vals, v_vals)
    
    max_err = np.max(np.abs(C_computed - C_expected))
    
    if max_err > 1e-9:
        # Fallback to rank 64
        rank_fallback = 64
        U_f = [[0] * rank_fallback for _ in range(16)]
        V_f = [[0] * rank_fallback for _ in range(16)]
        W_f = [[0] * rank_fallback for _ in range(16)]
        
        for i in range(4):
            for k in range(4):
                for j in range(4):
                    r = i * 16 + k * 4 + j
                    U_f[i*4+k][r] = 1
                    V_f[k*4+j][r] = 1
                    W_f[j*4+i][r] = 1
        
        return {"U": U_f, "V": V_f, "W": W_f}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
