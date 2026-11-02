"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition for 2x2x8 matrix multiplication.
    # Rank 28 (4 blocks of Strassen rank 7).
    n, m, p = DIMENSIONS
    rank = 28
    
    # U shape: (n*m, rank) = (4, 28)
    # V shape: (m*p, rank) = (16, 28)
    # W shape: (p*n, rank) = (16, 28)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen indices for A (2x2):
    # U[0] -> a00
    # U[1] -> a01
    # U[2] -> a10
    # U[3] -> a11
    
    # For each of the 4 blocks of B (each 2x2), we add 7 rank-1 terms.
    for b in range(4):
        base_col = b * 7
        
        # Global k indices for this block
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # V indices
        # v_00 -> V[0*8 + k0] = V[k0]
        # v_01 -> V[0*8 + k1] = V[k1]
        # v_10 -> V[1*8 + k0] = V[8 + k0]
        # v_11 -> V[1*8 + k1] = V[8 + k1]
        
        # W indices
        # w_00 (C[0, k0]) -> W[k0*2 + 0] = W[4b]
        # w_01 (C[0, k1]) -> W[k1*2 + 0] = W[4b+2]
        # w_10 (C[1, k0]) -> W[k0*2 + 1] = W[4b+1]
        # w_11 (C[1, k1]) -> W[k1*2 + 1] = W[4b+3]
        
        w_c00 = 4 * b
        w_c01 = 4 * b + 2
        w_c10 = 4 * b + 1
        w_c11 = 4 * b + 3
        
        # Term 1: M1 = (a10 + a11) * (b00 + b01)
        col = base_col
        U[2][col] += 1  # a10
        U[3][col] += 1  # a11
        V[k0][col] += 1  # b00
        V[k1][col] += 1  # b01
        W[w_c01][col] += 1  # c01
        W[w_c11][col] += 1  # c11
        
        # Term 2: M2 = a11 * (b10 - b00)
        col = base_col + 1
        U[3][col] += 1  # a11
        V[8 + k0][col] += 1  # b10
        V[k0][col] -= 1  # b00
        W[w_c10][col] += 1  # c10
        W[w_c11][col] -= 1  # c11
        
        # Term 3: M3 = a00 * (b00 + b11)
        col = base_col + 2
        U[0][col] += 1  # a00
        V[k0][col] += 1  # b00
        V[8 + k1][col] += 1  # b11
        W[w_c00][col] += 1  # c00
        W[w_c11][col] += 1  # c11
        
        # Term 4: M4 = a11 * (b10 + b11)
        col = base_col + 3
        U[3][col] += 1  # a11
        V[8 + k0][col] += 1  # b10
        V[8 + k1][col] += 1  # b11
        W[w_c00][col] += 1  # c00
        W[w_c10][col] += 1  # c10
        
        # Term 5: M5 = (a00 + a10) * b11
        col = base_col + 4
        U[0][col] += 1  # a00
        U[2][col] += 1  # a10
        V[8 + k1][col] += 1  # b11
        W[w_c00][col] += 1  # c00
        
        # Term 6: M6 = (a01 - a11) * (b10 + b11)
        col = base_col + 5
        U[1][col] += 1  # a01
        U[3][col] -= 1  # a11
        V[8 + k0][col] += 1  # b10
        V[8 + k1][col] += 1  # b11
        W[w_c11][col] += 1  # c11
        
        # Term 7: M7 = (a00 - a10) * (b00 + b10)
        col = base_col + 6
        U[0][col] += 1  # a00
        U[2][col] -= 1  # a10
        V[k0][col] += 1  # b00
        V[8 + k0][col] += 1  # b10
        W[w_c00][col] -= 1  # c00
        W[w_c01][col] += 1  # c01

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
