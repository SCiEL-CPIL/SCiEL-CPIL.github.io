"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    _t0 = time.time()
    _deadline = _t0 + 145.0

    n, m, p = DIMENSIONS

    # Use 2x2 grid of 3x3 blocks.
    # Each 3x3 matmul uses Winograd rank-19 decomposition.
    # Total rank = 4 * 19 = 76.
    
    block_size = 3
    blocks_per_dim = 2  # 6 / 3
    sub_rank = 19
    rank = (blocks_per_dim ** 3) * sub_rank  # 4 * 19 = 76

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Winograd 3x3 matrix multiplication decomposition (rank 19)
    # C = A * B, all 3x3
    # U_3: 9x19, V_3: 9x19, W_3: 9x19
    # A[i,j] -> U_3[i*3+j, r]
    # B[j,k] -> V_3[j*3+k, r]
    # C[i,k] -> W_3[k*3+i, r]  (trace convention: W[k*n+i] with n=3)
    
    U_3 = [[0]*19 for _ in range(9)]
    V_3 = [[0]*19 for _ in range(9)]
    W_3 = [[0]*19 for _ in range(9)]
    
    # Winograd 3x3 factors (standard decomposition)
    # Product r = sum_{i,j} U_3[i*3+j, r] * A[i,j] * sum_{j,k} V_3[j*3+k, r] * B[j,k]
    # C[i,k] = sum_r W_3[k*3+i, r] * product_r
    
    # Using the standard Winograd 3x3 decomposition:
    # f0 = (A00+A11+A22)*(B00+B11+B22)
    # f1 = (A01+A11+A21)*(B00+B10+B20)
    # f2 = (A02+A12+A22)*(B00+B10+B20)  -- need distinct
    # Let me use a verified set:
    
    # Standard Winograd 3x3 (from literature):
    # The 19 products are formed as follows:
    
    # Define helper to set coefficients
    def set_u(U, i, j, r, val):
        U[i*3+j][r] += val
    def set_v(V, j, k, r, val):
        V[j*3+k][r] += val
    def set_w(W, i, k, r, val):
        W[k*3+i][r] += val  # trace convention: W[k*n+i]
    
    # Product 0: (A00+A11+A22)*(B00+B11+B22)
    set_u(U_3, 0,0, 0, 1); set_u(U_3, 1,1, 0, 1); set_u(U_3, 2,2, 0, 1)
    set_v(V_3, 0,0, 0, 1); set_v(V_3, 1,1, 0, 1); set_v(V_3, 2,2, 0, 1)
    set_w(W_3, 0,0, 0, 1); set_w(W_3, 1,1, 0, 1); set_w(W_3, 2,2, 0, 1)
    set_w(W_3, 0,1, 0, 1); set_w(W_3, 1,0, 0, 1)
    set_w(W_3, 0,2, 0, 1); set_w(W_3, 2,0, 0, 1)
    set_w(W_3, 1,2, 0, 1); set_w(W_3, 2,1, 0, 1)
    
    # Product 1: (A01+A11+A21)*(B00+B10+B20)
    set_u(U_3, 0,1, 1, 1); set_u(U_3, 1,1, 1, 1); set_u(U_3, 2,1, 1, 1)
    set_v(V_3, 0,0, 1, 1); set_v(V_3, 1,0, 1, 1); set_v(V_3, 2,0, 1, 1)
    set_w(W_3, 0,0, 1, -1); set_w(W_3, 1,0, 1, -1); set_w(W_3, 2,0, 1, -1)
    set_w(W_3, 0,1, 1, 1); set_w(W_3, 1,1, 1, 1); set_w(W_3, 2,1, 1, 1)
    
    # Product 2: (A02+A12+A22)*(B00+B10+B20)
    set_u(U_3, 0,2, 2, 1); set_u(U_3, 1,2, 2, 1); set_u(U_3, 2,2, 2, 1)
    set_v(V_3, 0,0, 2, 1); set_v(V_3, 1,0, 2, 1); set_v(V_3, 2,0, 2, 1)
    set_w(W_3, 0,0, 2, -1); set_w(W_3, 1,0, 2, -1); set_w(W_3, 2,0, 2, -1)
    set_w(W_3, 0,2, 2, 1); set_w(W_3, 1,2, 2, 1); set_w(W_3, 2,2, 2, 1)
    
    # Product 3: (A00+A11-A22)*(B00+B11-B22)
    set_u(U_3, 0,0, 3, 1); set_u(U_3, 1,1, 3, 1); set_u(U_3, 2,2, 3, -1)
    set_v(V_3, 0,0, 3, 1); set_v(V_3, 1,1, 3, 1); set_v(V_3, 2,2, 3, -1)
    set_w(W_3, 0,0, 3, 1); set_w(W_3, 1,1, 3, 1); set_w(W_3, 2,2, 3, -1)
    set_w(W_3, 0,1, 3, -1); set_w(W_3, 1,0, 3, -1)
    set_w(W_3, 0,2, 3, -1); set_w(W_3, 2,0, 3, -1)
    set_w(W_3, 1,2, 3, -1); set_w(W_3, 2,1, 3, -1)
    
    # Product 4: (A01+A11-A21)*(B00+B10-B20)
    set_u(U_3, 0,1, 4, 1); set_u(U_3, 1,1, 4, 1); set_u(U_3, 2,1, 4, -1)
    set_v(V_3, 0,0, 4, 1); set_v(V_3, 1,0, 4, 1); set_v(V_3, 2,0, 4, -1)
    set_w(W_3, 0,0, 4, -1); set_w(W_3, 1,0, 4, -1); set_w(W_3, 2,0, 4, 1)
    set_w(W_3, 0,1, 4, 1); set_w(W_3, 1,1, 4, 1); set_w(W_3, 2,1, 4, -1)
    
    # Product 5: (A02+A12-A22)*(B00+B10-B20)
    set_u(U_3, 0,2, 5, 1); set_u(U_3, 1,2, 5, 1); set_u(U_3, 2,2, 5, -1)
    set_v(V_3, 0,0, 5, 1); set_v(V_3, 1,0, 5, 1); set_v(V_3, 2,0, 5, -1)
    set_w(W_3, 0,0, 5, -1); set_w(W_3, 1,0, 5, -1); set_w(W_3, 2,0, 5, 1)
    set_w(W_3, 0,2, 5, 1); set_w(W_3, 1,2, 5, 1); set_w(W_3, 2,2, 5, -1)
    
    # Product 6: (A00-A11+A22)*(B00-B11+B22)
    set_u(U_3, 0,0, 6, 1); set_u(U_3, 1,1, 6, -1); set_u(U_3, 2,2, 6, 1)
    set_v(V_3, 0,0, 6, 1); set_v(V_3, 1,1, 6, -1); set_v(V_3, 2,2, 6, 1)
    set_w(W_3, 0,0, 6, 1); set_w(W_3, 1,1, 6, -1); set_w(W_3, 2,2, 6, 1)
    set_w(W_3, 0,1, 6, 1); set_w(W_3, 1,0, 6, 1)
    set_w(W_3, 0,2, 6, 1); set_w(W_3, 2,0, 6, 1)
    set_w(W_3, 1,2, 6, 1); set_w(W_3, 2,1, 6, 1)
    
    # Product 7: (A01-A11+A21)*(B00-B10+B20)
    set_u(U_3, 0,1, 7, 1); set_u(U_3, 1,1, 7, -1); set_u(U_3, 2,1, 7, 1)
    set_v(V_3, 0,0, 7, 1); set_v(V_3, 1,0, 7, -1); set_v(V_3, 2,0, 7, 1)
    set_w(W_3, 0,0, 7, -1); set_w(W_3, 1,0, 7, -1); set_w(W_3, 2,0, 7, -1)
    set_w(W_3, 0,1, 7, 1); set_w(W_3, 1,1, 7, -1); set_w(W_3, 2,1, 7, 1)
    
    # Product 8: (A02-A12+A22)*(B00-B10+B20)
    set_u(U_3, 0,2, 8, 1); set_u(U_3, 1,2, 8, -1); set_u(U_3, 2,2, 8, 1)
    set_v(V_3, 0,0, 8, 1); set_v(V_3, 1,0, 8, -1); set_v(V_3, 2,0, 8, 1)
    set_w(W_3, 0,0, 8, -1); set_w(W_3, 1,0, 8, -1); set_w(W_3, 2,0, 8, -1)
    set_w(W_3, 0,2, 8, 1); set_w(W_3, 1,2, 8, -1); set_w(W_3, 2,2, 8, 1)
    
    # Products 9-18: handle off-diagonal corrections
    # Product 9: (A00-A20)*(B20)
    set_u(U_3, 0,0, 9, 1); set_u(U_3, 2,0, 9, -1)
    set_v(V_3, 2,0, 9, 1)
    set_w(W_3, 0,0, 9, 1); set_w(W_3, 2,0, 9, -1)
    set_w(W_3, 0,1, 9, 0); set_w(W_3, 1,0, 9, 0)
    
    # This is getting too complex and error-prone from memory.
    # Let me fall back to the verified 2x2 Strassen approach (rank 189).
    
    # Fallback: use 2x2 Strassen blocks (rank 189)
    block_size = 2
    blocks_per_dim = 3
    rank = (blocks_per_dim ** 3) * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    offsets = [i * block_size for i in range(blocks_per_dim)]
    
    strassen_patterns = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        off_I = offsets[I]
        for J in range(blocks_per_dim):
            off_J = offsets[J]
            for K in range(blocks_per_dim):
                off_K = offsets[K]
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    for (pi, pj), val in u_coeffs.items():
                        r = off_I + pi
                        c = off_J + pj
                        U[r * m + c][col_idx] += val
                    for (pi, pj), val in v_coeffs.items():
                        r = off_J + pi
                        c = off_K + pj
                        V[r * p + c][col_idx] += val
                    for (pi, pj), val in w_coeffs.items():
                        k = off_K + pj
                        i = off_I + pi
                        W[k * n + i][col_idx] += val
                    col_idx += 1
                if time.time() > _deadline:
                    break
            if time.time() > _deadline:
                break
        if time.time() > _deadline:
            break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
