import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n, m, p = DIMENSIONS
    
    # Strategy: 2x2 block decomposition (blocks of size 3x3)
    # Use Strassen (rank 7) on the 2x2 block matrix
    # Use Laderman (rank 23) for each 3x3 sub-multiplication
    # Total rank = 7 * 23 = 161
    
    # Laderman's algorithm for 3x3 matrix multiplication (rank 23)
    laderman_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(0,1): 1, (1,1): 1}, {(1,0): 1, (1,1): 1}, {(0,1): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,1): 1, (1,1): 1}, {(1,0): 1, (1,1): 1}),
        ({(0,0): 1, (1,1): -1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,1): 1, (0,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): -1}, {(0,1): 1, (1,0): -1}, {(1,0): 1, (1,1): 1}),
        ({(0,2): 1, (1,2): 1}, {(2,0): 1, (2,1): 1}, {(0,2): 1, (1,2): 1}),
        ({(0,2): 1, (1,2): 1}, {(2,1): 1, (2,2): 1}, {(0,2): 1, (1,2): 1}),
        ({(0,2): 1, (1,2): -1}, {(2,1): 1, (2,2): -1}, {(0,2): 1, (1,2): 1}),
        ({(0,0): 1, (2,2): 1}, {(0,2): 1, (2,0): 1}, {(0,2): 1, (2,0): 1}),
        ({(0,0): 1, (2,2): -1}, {(2,0): 1, (0,2): -1}, {(0,2): 1, (2,0): 1}),
        ({(0,1): 1, (2,1): 1}, {(1,2): 1, (2,2): 1}, {(0,2): 1, (2,2): 1}),
        ({(0,1): 1, (2,1): -1}, {(2,2): 1, (1,2): -1}, {(0,2): 1, (2,2): 1}),
        ({(0,2): 1, (2,0): 1}, {(0,0): 1, (2,2): 1}, {(0,0): 1, (2,2): 1}),
        ({(0,2): 1, (2,0): -1}, {(2,2): 1, (0,0): -1}, {(0,0): 1, (2,2): 1}),
        ({(1,0): 1, (2,0): 1}, {(0,2): 1, (2,2): 1}, {(1,2): 1, (2,2): 1}),
        ({(1,0): 1, (2,0): -1}, {(2,2): 1, (0,2): -1}, {(1,2): 1, (2,2): 1}),
        ({(1,2): 1, (2,2): 1}, {(2,0): 1, (0,0): 1}, {(1,0): 1, (2,0): 1}),
        ({(1,2): 1, (2,2): -1}, {(0,0): 1, (2,0): -1}, {(1,0): 1, (2,0): 1}),
        ({(0,0): 1, (1,2): 1, (2,2): 1}, {(0,0): 1, (2,2): 1}, {(0,0): 1, (2,2): 1}),
        ({(0,2): 1, (1,0): 1, (2,2): 1}, {(0,2): 1, (2,0): 1}, {(0,2): 1, (2,0): 1}),
        ({(0,1): 1, (1,0): 1, (2,1): 1}, {(1,0): 1, (1,2): 1}, {(0,1): 1, (1,2): 1}),
        ({(0,0): 1, (1,1): 1, (2,2): 1}, {(0,0): 1, (1,1): 1, (2,2): 1}, {(0,0): 1, (1,1): 1, (2,2): 1}),
    ]
    
    # Verify Laderman's algorithm
    def verify_laderman(terms, num_tests=20):
        np.random.seed(42)
        for _ in range(num_tests):
            A = np.random.randn(3, 3)
            B = np.random.randn(3, 3)
            C_expected = A @ B
            C_computed = np.zeros((3, 3))
            for u_coeffs, v_coeffs, w_coeffs in terms:
                u_val = sum(u_coeffs.get((i,j), 0) * A[i,j] for i in range(3) for j in range(3))
                v_val = sum(v_coeffs.get((i,j), 0) * B[i,j] for i in range(3) for j in range(3))
                prod = u_val * v_val
                for (i, j), val in w_coeffs.items():
                    C_computed[i, j] += val * prod
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        return True
    
    laderman_valid = verify_laderman(laderman_terms)
    
    if not laderman_valid:
        # Fall back to trivial 3x3 matmul
        laderman_terms = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    laderman_terms.append(
                        ({(i, k): 1}, {(k, j): 1}, {(i, j): 1})
                    )
        sub_rank = 27
    else:
        sub_rank = 23
    
    # Strassen's 7 terms for 2x2 matrix multiplication
    strassen_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    block_size = 3
    
    rank = 7 * sub_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
        for sub_u, sub_v, sub_w in laderman_terms:
            for (si, sj), su_val in u_coeffs.items():
                for (ti, tj), sub_u_val in sub_u.items():
                    r = si * block_size + ti
                    c = sj * block_size + tj
                    U[r * m + c][col_idx] += su_val * sub_u_val
            
            for (si, sj), sv_val in v_coeffs.items():
                for (ti, tj), sub_v_val in sub_v.items():
                    r = si * block_size + ti
                    c = sj * block_size + tj
                    V[r * p + c][col_idx] += sv_val * sub_v_val
            
            for (si, sj), sw_val in w_coeffs.items():
                for (ti, tj), sub_w_val in sub_w.items():
                    r = si * block_size + ti
                    c = sj * block_size + tj
                    W[r * n + c][col_idx] += sw_val * sub_w_val
            
            col_idx += 1
    
    # Final verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    np.random.seed(999)
    A_test = np.random.randn(6, 6)
    B_test = np.random.randn(6, 6)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((6, 6))
    for r in range(rank):
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    C_computed[i, j] += U_np[i * 6 + k, r] * V_np[k * 6 + j, r] * W_np[j * 6 + i, r]
    
    if not np.allclose(C_computed, C_expected, atol=1e-8):
        # Emergency fallback: use trivial 2x2 Strassen with 2x2 blocks (rank 189)
        block_size = 2
        blocks_per_dim = 3
        rank = 27 * 7  # 189
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col_idx = 0
        for I in range(blocks_per_dim):
            for J in range(blocks_per_dim):
                for K in range(blocks_per_dim):
                    for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
                        for (pi, pj), val in u_coeffs.items():
                            r = I * block_size + pi
                            c = J * block_size + pj
                            U[r * m + c][col_idx] += val
                        
                        for (pi, pj), val in v_coeffs.items():
                            r = J * block_size + pi
                            c = K * block_size + pj
                            V[r * p + c][col_idx] += val
                        
                        for (pi, pj), val in w_coeffs.items():
                            k = K * block_size + pj
                            i = I * block_size + pi
                            W[k * n + i][col_idx] += val
                        
                        col_idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
