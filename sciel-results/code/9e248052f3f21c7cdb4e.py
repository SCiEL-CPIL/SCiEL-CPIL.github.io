import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    DEADLINE = 120

    n, m, p = DIMENSIONS  # 4, 4, 6

    # Strassen 2x2x2 decomposition
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

    # We need to decompose the multiplication of A (4x4) * B (4x6) = C (4x6)
    # The tensor has dimensions (n*m, m*p, p*n) = (16, 24, 24)
    # We use a tiling approach: split into 2x2x2 blocks
    # A is 4x4, so 2 row blocks, 2 col blocks
    # B is 4x6, so 2 row blocks, 3 col blocks
    # Each 2x2x2 sub-problem uses Strassen rank 7
    # Total blocks: 2*2*3 = 12, total rank = 12*7 = 84

    # But we can do better by exploiting structure.
    # Let's try a hybrid: use rank-7 Strassen for 4x4x4 part and rank-4 for 4x4x2 part
    # 4x4x4: split into 4x4 = 2x2 blocks in both dimensions -> 4 sub-problems of 2x2x2, rank = 4*7 = 28
    # 4x4x2: split into 2 row blocks of A, 2 col blocks of A, 1 col block of B (size 2) 
    #   This gives 2*2 = 4 sub-problems of 2x2x2, but B is only 2 wide... 
    #   Actually for 4x4x2: A is 4x4, B is 4x2, C is 4x2
    #   Split A into 2x2 blocks (2x2=4 blocks), B into 2x2 blocks (1 col block of size 2, so 1x1 blocks of 2x2)
    #   Wait, B is 4x2. We can split rows into 2 blocks of 2, cols into 1 block of 2.
    #   So 2*1 = 2 sub-problems of 2x2x2? No, that doesn't work dimensionally.
    
    # Let me think differently. The standard approach:
    # 4x4x6 = (4x4x4) + (4x4x2)
    # For 4x4x4: use tensor product of two 2x2x2 Strassen, rank = 7*7 = 49
    # For 4x4x2: we can use 2x2x2 Strassen applied to appropriate blocks
    #   A is 4x4, B is 4x2, C is 4x2
    #   Split A into 2x2 blocks (2x2 grid), B into 2x2 blocks (2x1 grid of 2x2 blocks), C into 2x2 blocks (2x1 grid)
    #   This gives 2*2*1 = 4 sub-problems, each 2x2x2, rank = 4*7 = 28
    # Total: 49 + 28 = 77

    # Let's build this decomposition.
    # Part 1: 4x4x4 (columns 0-3 of B)
    # Part 2: 4x4x2 (columns 4-5 of B)

    # For 4x4x4: tensor product of two Strassen 2x2x2
    # The 4x4x4 tensor can be decomposed as (Strassen_2x2x2) ⊗ (Strassen_2x2x2)
    # Rank = 7 * 7 = 49

    # For 4x4x2: 
    # A (4x4) split into 4 2x2 blocks: A_{bi,bj} for bi,bj in {0,1}
    # B (4x2) split into 2 2x2 blocks: B_{bj,0} for bj in {0,1} (one col block of width 2)
    # C (4x2) split into 2 2x2 blocks: C_{bi,0} for bi in {0,1}
    # So we have 2 (row blocks of A) * 2 (col blocks of A / row blocks of B) * 1 (col blocks of B) = 4 sub-problems
    # Each is 2x2x2, using Strassen rank 7
    # Total rank for part 2: 4 * 7 = 28

    # Build U, V, W for part 1 (4x4x4)
    U1 = np.zeros((n * m, 49))  # 16 x 49
    V1 = np.zeros((m * p, 49))  # 24 x 49
    W1 = np.zeros((p * n, 49))  # 24 x 49

    for c1 in range(7):
        for c2 in range(7):
            col = c1 * 7 + c2
            # U: index (i,j) = (i1*2+i2, j1*2+j2) -> row = i*4+j
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            row_U = i * m + j  # i*4 + j
                            U1[row_U, col] = S_U[i1 * 2 + j1, c1] * S_U[i2 * 2 + j2, c2]

            # V: index (j,k) for k in 0..3 (part 1 uses k=0..3)
            # j = j1*2+j2, k = k1*2+k2
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            if k < 4:  # Only for part 1
                                row_V = j * p + k
                                V1[row_V, col] = S_V[j1 * 2 + j2, c1] * S_V[k1 * 2 + k2, c2]

            # W: index (k,i) for k in 0..3, i in 0..3
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            if k < 4:  # Only for part 1
                                row_W = k * n + i
                                W1[row_W, col] = S_W[k1 * 2 + k2, c1] * S_W[i1 * 2 + i2, c2]

    # Build U, V, W for part 2 (4x4x2, columns 4-5 of B)
    U2 = np.zeros((n * m, 28))  # 16 x 28
    V2 = np.zeros((m * p, 28))  # 24 x 28
    W2 = np.zeros((p * n, 28))  # 24 x 28

    for bi in range(2):
        for bj in range(2):
            # One col block of B (width 2), so bk=0
            base_col = (bi * 2 + bj) * 7
            for c in range(7):
                col = base_col + c

                # U: rows (i,j) where i = bi*2+di, j = bj*2+dj
                for di in range(2):
                    for dj in range(2):
                        i = bi * 2 + di
                        j = bj * 2 + dj
                        row_U = i * m + j
                        U2[row_U, col] = S_U[di * 2 + dj, c]

                # V: rows (j,k) where j = bj*2+dj, k = 4+dk (columns 4,5)
                for dj in range(2):
                    for dk in range(2):
                        j = bj * 2 + dj
                        k = 4 + dk
                        row_V = j * p + k
                        V2[row_V, col] = S_V[dj * 2 + dk, c]

                # W: rows (k,i) where k = 4+dk, i = bi*2+di
                for dk in range(2):
                    for di in range(2):
                        k = 4 + dk
                        i = bi * 2 + di
                        row_W = k * n + i
                        W2[row_W, col] = S_W[dk * 2 + di, c]

    # Combine: U = [U1, U2], V = [V1, V2], W = [W1, W2]
    total_rank = 49 + 28
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    U[:, :49] = U1
    U[:, 49:] = U2
    V[:, :49] = V1
    V[:, 49:] = V2
    W[:, :49] = W1
    W[:, 49:] = W2

    # Remove zero columns to reduce rank
    nonzero = []
    for c in range(total_rank):
        if np.any(U[:, c] != 0) or np.any(V[:, c] != 0) or np.any(W[:, c] != 0):
            nonzero.append(c)

    U_final = U[:, nonzero]
    V_final = V[:, nonzero]
    W_final = W[:, nonzero]

    # Convert to list of lists with integer/half-integer entries
    def to_int_list(arr):
        result = []
        for row in arr:
            r = []
            for val in row:
                if abs(val - round(val)) < 1e-9:
                    r.append(int(round(val)))
                elif abs(val * 2 - round(val * 2)) < 1e-9:
                    r.append(int(round(val * 2)) / 2)
                else:
                    r.append(val)
            result.append(r)
        return result

    return {"U": to_int_list(U_final), "V": to_int_list(V_final), "W": to_int_list(W_final)}
