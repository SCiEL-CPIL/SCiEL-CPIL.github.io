"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 48 decomposition
    # Split A into two 2x2 blocks horizontally: A = [A1, A2]
    # Split B into two 2x8 blocks vertically: B = [B1; B2]
    # C = A1 B1 + A2 B2
    # Each term is 2x2 by 2x8.
    # Rank of 2x2 by 2x8 is 24 (using rank 3 for 2x2x1).
    # Total rank 48.
    
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define rank-3 decomposition for 2x2 by 2x1
    # u, v, w vectors
    # We need to verify the rank-3 decomposition.
    # If it fails, we fall back to rank 4 (schoolbook) which gives rank 64.
    
    # Rank 3 decomposition for 2x2x1:
    # z0 = ax + bz
    # z1 = cx + dz
    # m1 = (a+b)x
    # m2 = (c+d)z
    # m3 = (a+c)(x+z) ? No.
    
    # Let's use a known rank-3 decomposition:
    # u1 = (1, 1, 0, 0), v1 = (1, 0), w1 = (1, 0)
    # u2 = (0, 0, 1, 1), v2 = (0, 1), w2 = (0, 1)
    # u3 = (1, 0, 1, 0), v3 = (1, 1), w3 = (1, 1)
    
    u_vecs = [
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [1, 0, 1, 0]
    ]
    v_vecs = [
        [1, 0],
        [0, 1],
        [1, 1]
    ]
    w_vecs = [
        [1, 0],
        [0, 1],
        [1, 1]
    ]
    
    # Verify rank 3 decomposition for 2x2x1
    # T = sum u_k v_k w_k
    # Check if T matches the multiplication tensor
    # T[i, j, k] = sum u_k[i] v_k[j] w_k[k]
    # i, j are indices of X (2x2), j is index of y (2), k is index of z (2)
    # Wait, indices:
    # X is 2x2, so i is (row, col) of X.
    # y is 2, so j is index of y.
    # z is 2, so k is index of z.
    
    # Let's check the tensor values
    # T[0,0,0] = u1[0]v1[0]w1[0] + u2[0]v2[0]w2[0] + u3[0]v3[0]w3[0]
    # = 1*1*1 + 0 + 1*1*1 = 2. Should be 1 (a*x -> z0).
    # So this decomposition is incorrect.
    
    # I will use rank 4 decomposition for 2x2x1 (schoolbook)
    # This gives rank 64 total.
    # I will try to find a better decomposition later.
    
    # For now, return rank 64 to be safe.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
