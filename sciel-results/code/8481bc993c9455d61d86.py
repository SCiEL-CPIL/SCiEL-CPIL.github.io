import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using Strassen for 2x2 blocks
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen(rank_offset, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        def u_idx(r, c):
            return (a_row_start + r) * m + (a_col_start + c)
            
        def v_idx(r, c):
            return (b_row_start + r) * p + (b_col_start + c)
            
        def w_idx(r, c):
            # W index: k*n + i. C[r, c] -> i=r, k=c
            return (c_col_start + c) * n + (c_row_start + r)

        # M1 = (X00 + X11)(Y00 + Y11)
        U[u_idx(0,0)][rank_offset] += 1
        U[u_idx(1,1)][rank_offset] += 1
        V[v_idx(0,0)][rank_offset] += 1
        V[v_idx(1,1)][rank_offset] += 1
        W[w_idx(0,0)][rank_offset] += 1
        W[w_idx(1,1)][rank_offset] += 1
        rank_offset += 1

        # M2 = (X10 + X11)Y00
        U[u_idx(1,0)][rank_offset] += 1
        U[u_idx(1,1)][rank_offset] += 1
        V[v_idx(0,0)][rank_offset] += 1
        W[w_idx(1,0)][rank_offset] += 1
        rank_offset += 1

        # M3 = X00(Y10 - Y11)
        U[u_idx(0,0)][rank_offset] += 1
        V[v_idx(1,0)][rank_offset] += 1
        V[v_idx(1,1)][rank_offset] -= 1
        W[w_idx(0,1)][rank_offset] += 1
        W[w_idx(1,1)][rank_offset] += 1
        rank_offset += 1

        # M4 = X11(Y01 - Y00)
        U[u_idx(1,1)][rank_offset] += 1
        V[v_idx(0,1)][rank_offset] += 1
        V[v_idx(0,0)][rank_offset] -= 1
        W[w_idx(0,0)][rank_offset] += 1
        W[w_idx(1,0)][rank_offset] += 1
        rank_offset += 1

        # M5 = (X00 + X10)Y11
        U[u_idx(0,0)][rank_offset] += 1
        U[u_idx(1,0)][rank_offset] += 1
        V[v_idx(1,1)][rank_offset] += 1
        W[w_idx(0,0)][rank_offset] -= 1
        W[w_idx(0,1)][rank_offset] += 1
        rank_offset += 1

        # M6 = (X11 - X00)(Y00 + Y10)
        U[u_idx(1,1)][rank_offset] += 1
        U[u_idx(0,0)][rank_offset] -= 1
        V[v_idx(0,0)][rank_offset] += 1
        V[v_idx(1,0)][rank_offset] += 1
        W[w_idx(1,1)][rank_offset] += 1
        rank_offset += 1

        # M7 = (X01 - X11)(Y11 + Y01)
        U[u_idx(0,1)][rank_offset] += 1
        U[u_idx(1,1)][rank_offset] -= 1
        V[v_idx(1,1)][rank_offset] += 1
        V[v_idx(0,1)][rank_offset] += 1
        W[w_idx(0,0)][rank_offset] += 1
        W[w_idx(1,1)][rank_offset] -= 1
        rank_offset += 1
        
        return rank_offset

    # 4 sub-products
    # 1. A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    r = add_strassen(0, 0, 0, 0, 0, 0, 0)
    # 2. A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    r = add_strassen(r, 0, 2, 2, 0, 0, 0)
    # 3. A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    r = add_strassen(r, 0, 0, 0, 2, 0, 2)
    # 4. A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    r = add_strassen(r, 0, 2, 2, 2, 0, 2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
