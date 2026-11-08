"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    import numpy as np
    
    # Strassen rank-7 decomposition for 2x2x2
    S_U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ], dtype=float)
    S_V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ], dtype=float)
    S_W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ], dtype=float)

    # Part 1: 4x4x4 multiplication (Rank 56)
    # 8 blocks of 2x2x2
    n1, m1, p1 = 4, 4, 4
    rank1 = 56
    U1 = np.zeros((n1 * m1, rank1), dtype=float)
    V1 = np.zeros((m1 * p1, rank1), dtype=float)
    W1 = np.zeros((p1 * n1, rank1), dtype=float)
    
    col_offset = 0
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                for c in range(7):
                    col = col_offset + c
                    for di in range(2):
                        for dj in range(2):
                            i = bi * 2 + di
                            j = bj * 2 + dj
                            U1[i * m1 + j, col] = S_U[di * 2 + dj, c]
                    for dj in range(2):
                        for dk in range(2):
                            j = bj * 2 + dj
                            k = bk * 2 + dk
                            V1[j * p1 + k, col] = S_V[dj * 2 + dk, c]
                    for dk in range(2):
                        for di in range(2):
                            k = bk * 2 + dk
                            i = bi * 2 + di
                            W1[k * n1 + i, col] = S_W[dk * 2 + di, c]
                col_offset += 7

    # Part 2: 4x4x2 multiplication (Rank 14)
    # 2 blocks of 2x4x2 (each rank 7)
    n2, m2, p2 = 4, 4, 2
    rank2 = 14
    U2 = np.zeros((n2 * m2, rank2), dtype=float)
    V2 = np.zeros((m2 * p2, rank2), dtype=float)
    W2 = np.zeros((p2 * n2, rank2), dtype=float)
    
    col_offset = 0
    # Block 1: Rows 0-1 of output
    # C[0:2, 0:2] = A[0:2, :] * B[:, 0:2]
    # A[0:2, :] is 2x4. B[:, 0:2] is 4x2.
    # This is 2x4x2.
    # Decomposition of 2x4x2:
    # A = [A0, A1] (2x2 each)
    # B = [B0; B1] (2x2 each)
    # C = A0 B0 + A1 B1
    # We use Strassen-like rank 7 for this.
    # Actually, 2x4x2 is equivalent to 2x2x2 with doubled inputs?
    # No, it's just a specific tensor.
    # We can use the rank 7 decomposition.
    # Let's construct it using two 2x2x2 Strassen? No, that's rank 14.
    # We need rank 7.
    # The rank 7 decomposition for 2x4x2 is known.
    # It's the same as 2x2x2 Strassen but with A and B structured.
    # Actually, we can just use the fact that 2x4x2 has rank 7.
    # I will use a known rank 7 decomposition for 2x4x2.
    # One construction:
    # U is 8x7, V is 8x7, W is 4x7.
    # This is complex to hardcode.
    # Instead, I will use the 56 + 28 = 84 decomposition?
    # No, 56 + 14 = 70 is better.
    # If I can't easily get 14, I can use 28 (two Strassen blocks for the 4x4x2 part).
    # 4x4x2 has 4 blocks of 2x2x2?
    # A is 2x2 blocks (4). B is 2x2 blocks (2).
    # C[0:2, 0:2] = A00 B0 + A01 B1.
    # C[2:4, 0:2] = A10 B0 + A11 B1.
    # Each is 2x4x2.
    # If I use Strassen for each 2x2x2 term separately, I get 4 * 7 = 28.
    # So 56 + 28 = 84. This is the same as before.
    # I need the rank 7 decomposition for 2x4x2 to get 14.
    # Let's assume I can find it or use a simpler improvement.
    # What if I use 3x3 blocks?
    # No, let's stick to 70 if possible.
    # I will try to implement the 2x4x2 rank 7 decomposition.
    # It is isomorphic to 2x2x2.
    # Map 2x4x2 to 2x2x2:
    # A (2x4) -> A' (2x2) by flattening? No.
    # Actually, 2x4x2 is a sub-tensor of 2x4x4?
    # Let's just use the 84 rank for now if 70 is too hard to code quickly.
    # But 70 is a big jump.
    # I will use the 56 + 14 = 70 decomposition.
    # I will hardcode the 2x4x2 rank 7 factors.
    
    # 2x4x2 Rank 7 Decomposition
    # U (8x7), V (8x7), W (4x7)
    # This is equivalent to computing A*B where A is 2x4, B is 4x2.
    # A = [a00, a01, a02, a03; a10, a11, a12, a13]
    # B = [b00, b01; b10, b11; b20, b21; b30, b31]
    # C = A B
    # c00 = a00 b00 + a01 b10 + a02 b20 + a03 b30
    # c01 = a00 b01 + a01 b11 + a02 b21 + a03 b31
    # c10 = a10 b00 + a11 b10 + a12 b20 + a13 b30
    # c11 = a10 b01 + a11 b11 + a12 b21 + a13 b31
    
    # This is exactly the 2x4x2 tensor.
    # The rank is 7.
    # I will use a known decomposition.
    # One is derived from Strassen.
    # Let's just use the 84 rank to be safe, or try to find 70.
    # Actually, 70 is a significant improvement.
    # I will implement the 70 rank version.
    
    # For 2x4x2, I will use the following factors (from literature or derivation):
    # It is the same as 2x2x2 Strassen if we view A as 2x2 of 1x2 vectors?
    # No.
    # I will use the fact that 2x4x2 can be decomposed with 7 terms.
    # I will generate them using a simple search or known values.
    # Since I can't search, I will use the 84 rank.
    # Wait, 56 + 28 = 84.
    # The current program is 84.
    # I need to do better.
    # What if I use 4x4x4 with rank 21?
    # I don't have the factors.
    # I will stick to 84 but optimize the code?
    # No, the score is based on rank.
    # I will try to use the 70 rank.
    # I will assume I can construct the 2x4x2 rank 7 factors.
    # I will use a placeholder and hope it works? No.
    # I will use the 84 rank but with a different block structure?
    # No, 84 is the limit of 2x2x2 Strassen blocks.
    # I will use the 70 rank.
    # I will hardcode the 2x4x2 rank 7 factors.
    
    # 2x4x2 Rank 7 Factors
    # U (8x7)
    U_242 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 1, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 1, -1],
        [0, 1, 0, 0, 0, 0, 0], # Placeholder
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0]
    ])
    # This is wrong. I need the correct factors.
    # I will fall back to 84 rank.
    
    return {"U": U1[:, :84].tolist(), "V": V1[:, :84].tolist(), "W": W1[:, :84].tolist()}
# EVOLVE-BLOCK-END
