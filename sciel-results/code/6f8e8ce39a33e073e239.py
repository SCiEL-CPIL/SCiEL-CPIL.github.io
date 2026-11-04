"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin

    # Strategy:
    # 1. Try Rank 161: Strassen (2x2 blocks) + Laderman (3x3 blocks)
    #    6x6 matrix viewed as 2x2 blocks of 3x3 matrices.
    #    Block mult: Strassen (7 terms).
    #    Inner mult: Laderman (23 terms).
    #    Total Rank: 7 * 23 = 161.
    
    # 2. Fallback Rank 189: Strassen (2x2 blocks) + Schoolbook (3x3 blocks)
    #    OR Schoolbook (3x3 blocks) + Strassen (2x2 blocks).
    #    Current program uses 3x3 blocks of 2x2 (27*7=189).

    # Helper: Verify decomposition
    def verify_decomp(U, V, W, n, m, p):
        rank = len(U[0])
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        
        U_reshaped = U_np.reshape(n, m, rank)
        V_reshaped = V_np.reshape(m, p, rank)
        W_reshaped = W_np.reshape(p, n, rank)
        
        T = np.einsum('ijt,jkt,kit->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        return np.allclose(T, 1.0, atol=1e-9)

    # --- Laderman 3x3 algorithm (23 terms) ---
    # Each term is (u, v, w) where u, v, w are 3x3 matrices (flat, row-major)
    # The tensor identity: sum_t u_t[i,j] * v_t[j,k] * w_t[k,i] = 1 for all i,j,k
    # Convention: T[i,j,k] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
    # For 3x3 inner: U[i,j,t] * V[j,k,t] * W[k,i,t] should give 1.
    
    # Laderman's algorithm for 3x3:
    # C = A * B
    # A = [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]
    # B = [[b11, b12, b13], [b21, b22, b23], [b31, b32, b33]]
    
    # 23 terms. I will use a known correct set.
    # Reference: "A 23-term algorithm for multiplying 3x3 matrices"
    # The terms are often listed as (x, y, z) such that sum x_t * y_t * z_t = C.
    # Here x corresponds to U, y to V, z to W.
    # U[i,j,t] is coeff of A[i,j]
    # V[j,k,t] is coeff of B[j,k]
    # W[k,i,t] is coeff for output C[i,k] (note indices k,i in W)
    
    # Let's use the terms from a reliable source.
    # I will hardcode the 23 terms.
    # If I get them wrong, verification will fail and I return 189.
    
    # Terms (u, v, w) as 3x3 flat arrays.
    # u: 3x3, v: 3x3, w: 3x3.
    # Note: W in the formula is W[k*n+i, t], which corresponds to W_mat[k, i] in my local 3x3 notation.
    # So the third element of the tuple is the matrix Z where Z[k,i] is the coefficient.
    
    laderman_terms = [
        # 1
        ([1,0,0, 0,1,0, 0,0,1], [1,0,0, 0,1,0, 0,0,1], [1,0,0, 0,1,0, 0,0,1]),
        # 2
        ([0,1,0, 1,0,0, 0,0,0], [0,1,0, 1,0,0, 0,0,0], [0,1,0, 1,0,0, 0,0,0]),
        # 3
        ([0,0,1, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 1,0,0]),
        # 4
        ([1,0,0, 0,0,0, 0,1,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
        # 5
        ([0,0,0, 0,1,0, 1,0,0], [0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
        # 6
        ([0,0,0, 0,0,1, 0,1,0], [0,0,0, 0,1,0, 1,0,0], [0,0,0, 0,0,0, 0,1,0]),
        # 7
        ([1,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 1,0,0]),
        # 8
        ([0,0,0, 0,0,0, 0,1,1], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,0,1]),
        # 9
        ([0,1,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,1]),
        # 10
        ([1,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,1]),
        # 11
        ([0,0,1, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,1]),
        # 12
        ([0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,0,1]),
        # 13
        ([1,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,1]),
        # 14
        ([0,1,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # 15
        ([0,0,0, 0,0,0, 1,1,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # 16
        ([0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # 17
        ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # 18
        ([0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # 19
        ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # 20
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # 21
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # 22
        ([1,1,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # 23
        ([0,0,0, 0,0,1, 1,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
    ]
    
    # Verify Laderman locally before using it
    def verify_laderman(terms):
        # Check sum_t u_t[i,j] v_t[j,k] w_t[k,i] = 1 for all i,j,k
        # u: (t, i, j)
        # v: (t, j, k)
        # w: (t, k, i)
        
        u_np = np.array([t[0] for t in terms], dtype=np.float64).reshape(len(terms), 3, 3)
        v_np = np.array([t[1] for t in terms], dtype=np.float64).reshape(len(terms), 3, 3)
        w_np = np.array([t[2] for t in terms], dtype=np.float64).reshape(len(terms), 3, 3)
        
        # T[i,j,k] = sum_t u[t,i,j] v[t,j,k] w[t,k,i]
        # einsum: 'tij,tjk,tki->ijk'
        T = np.einsum('tij,tjk,tki->ijk', u_np, v_np, w_np)
        
        return np.allclose(T, 1.0, atol=1e-9)
    
    laderman_ok = verify_laderman(laderman_terms)
    
    if laderman_ok:
        # Construct 6x6 decomposition using Strassen(2x2) of Laderman(3x3)
        # Block size 3, 2x2 blocks
        
        strassen_2x2 = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        rank = 7 * 23
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col_idx = 0
        for u_coeffs, v_coeffs, w_coeffs in strassen_2x2:
            for t in range(23):
                u_mat = np.array(laderman_terms[t][0]).reshape(3,3)
                v_mat = np.array(laderman_terms[t][1]).reshape(3,3)
                w_mat = np.array(laderman_terms[t][2]).reshape(3,3)
                
                # U
                for (bi, bj), u_val in u_coeffs.items():
                    for si in range(3):
                        for sj in range(3):
                            gr = bi * 3 + si
                            gc = bj * 3 + sj
                            U[gr * m + gc][col_idx] += u_val * u_mat[si, sj]
                            
                # V
                for (bk, bl), v_val in v_coeffs.items():
                    for sk in range(3):
                        for sl in range(3):
                            gr = bk * 3 + sk
                            gc = bl * 3 + sl
                            V[gr * p + gc][col_idx] += v_val * v_mat[sk, sl]
                            
                # W
                # w_coeffs keys are (i_block, k_block)
                for (bi, bk), w_val in w_coeffs.items():
                    for si in range(3):
                        for sk in range(3):
                            i = bi * 3 + si
                            k = bk * 3 + sk
                            W_flat_idx = k * n + i
                            W[W_flat_idx][col_idx] += w_val * w_mat[sk, si]
                            
                col_idx += 1
        
        # Verify
        if verify_decomp(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
            
    # Fallback: Rank 189
    block_size = 2
    blocks_per_dim = 3
    
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    rank = (blocks_per_dim ** 3) * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                for u_coeffs, v_coeffs, w_coeffs in strassen_2x2:
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
