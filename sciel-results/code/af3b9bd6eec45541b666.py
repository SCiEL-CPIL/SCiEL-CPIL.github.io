"""Tensor decomposition for 2x2 by 2x4 matrix multiplication."""

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]  # 4 x 14
    V = [[0] * rank for _ in range(m * p)]  # 8 x 14
    W = [[0] * rank for _ in range(p * n)]  # 8 x 14

    # Strassen factors for 2x2 * 2x2
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices for block 1: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices for block 2: 4:(0,2), 5:(0,3), 6:(1,2), 7:(1,3)
    # W indices for block 1: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W indices for block 2: 4:(0,2), 5:(0,3), 6:(1,2), 7:(1,3)
    # Note: W index is k*n + i.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # k=2, i=0 -> 4
    # k=2, i=1 -> 5
    # k=3, i=0 -> 6
    # k=3, i=1 -> 7

    # Define Strassen terms
    # Each term is a list of (U_idx, coeff), (V_idx, coeff), (W_idx, coeff)
    # We will add these to the matrices
    
    # Term 1: M1
    # U: 0(1), 3(1)
    # V: 0(1), 3(1) for block 1; 4(1), 7(1) for block 2
    # W: 0(1), 3(1) for block 1; 4(1), 7(1) for block 2
    
    # Let's define a helper to add a term
    def add_term(rank_offset, v_offset, w_offset):
        # M1
        U[0][rank_offset] += 1
        U[3][rank_offset] += 1
        V[0 + v_offset][rank_offset] += 1
        V[3 + v_offset][rank_offset] += 1
        W[0 + w_offset][rank_offset] += 1
        W[3 + w_offset][rank_offset] += 1

        # M2
        U[2][rank_offset + 1] += 1
        U[3][rank_offset + 1] += 1
        V[0 + v_offset][rank_offset + 1] += 1
        W[1 + w_offset][rank_offset + 1] += 1
        W[3 + w_offset][rank_offset + 1] -= 1

        # M3
        U[0][rank_offset + 2] += 1
        V[1 + v_offset][rank_offset + 2] += 1
        V[3 + v_offset][rank_offset + 2] -= 1
        W[2 + w_offset][rank_offset + 2] += 1
        W[3 + w_offset][rank_offset + 2] += 1

        # M4
        U[3][rank_offset + 3] += 1
        V[2 + v_offset][rank_offset + 3] += 1
        V[0 + v_offset][rank_offset + 3] -= 1
        W[0 + w_offset][rank_offset + 3] += 1
        W[1 + w_offset][rank_offset + 3] += 1

        # M5
        U[0][rank_offset + 4] += 1
        U[1][rank_offset + 4] += 1
        V[3 + v_offset][rank_offset + 4] += 1
        W[0 + w_offset][rank_offset + 4] -= 1
        W[2 + w_offset][rank_offset + 4] += 1

        # M6
        U[2][rank_offset + 5] += 1
        U[0][rank_offset + 5] -= 1
        V[0 + v_offset][rank_offset + 5] += 1
        V[1 + v_offset][rank_offset + 5] += 1
        W[3 + w_offset][rank_offset + 5] += 1

        # M7
        U[1][rank_offset + 6] += 1
        U[3][rank_offset + 6] -= 1
        V[2 + v_offset][rank_offset + 6] += 1
        V[3 + v_offset][rank_offset + 6] += 1
        W[0 + w_offset][rank_offset + 6] += 1

    # Block 1: rank 0-6, v_offset 0, w_offset 0
    add_term(0, 0, 0)
    # Block 2: rank 7-13, v_offset 4, w_offset 4
    add_term(7, 4, 4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
