import time
import random
import itertools

DIMENSIONS = [3, 4, 7]
_DEADLINE = 140.0

def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS  # n=3, m=4, p=7

    # Generate all possible 2x2x2 blocks
    # i pairs: (0,1), (1,2)
    # j pairs: (0,1), (1,2), (2,3)
    # k pairs: (0,1), (1,2), (2,3), (3,4), (4,5), (5,6)
    
    i_pairs = [(0, 1), (1, 2)]
    j_pairs = [(0, 1), (1, 2), (2, 3)]
    k_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    
    all_blocks = []
    for i0, i1 in i_pairs:
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # Block is defined by the tuple of ranges
                # We represent a block by its 6 indices
                all_blocks.append((i0, i1, j0, j1, k0, k1))
                
    def is_disjoint(b1, b2):
        # b1, b2 are (i0, i1, j0, j1, k0, k1)
        # They are disjoint if any of the ranges are disjoint
        # Range 1: [b1[0], b1[1]], Range 2: [b2[0], b2[1]]
        # Disjoint if b1[1] < b2[0] or b2[1] < b1[0]
        if b1[1] < b2[0] or b2[1] < b1[0]:
            return True
        if b1[3] < b2[2] or b2[3] < b2[2]: # Wait, j is index 2,3 in tuple
            # b1 j: b1[2], b1[3]. b2 j: b2[2], b2[3]
            if b1[3] < b2[2] or b2[3] < b1[2]:
                return True
        # k is index 4,5
        if b1[5] < b2[4] or b2[5] < b1[4]:
            return True
        return False

    # Greedy search for maximum disjoint set
    best_blocks = []
    
    # To ensure we find a good set, we run multiple random restarts
    # We also try a deterministic greedy that prioritizes "rare" indices or just random
    
    num_restarts = 0
    while time.time() - _t0 < _DEADLINE - 5.0: # Leave 5s for construction/verification
        num_restarts += 1
        random.shuffle(all_blocks)
        current_blocks = []
        for b in all_blocks:
            can_add = True
            for cb in current_blocks:
                if not is_disjoint(b, cb):
                    can_add = False
                    break
            if can_add:
                current_blocks.append(b)
        
        if len(current_blocks) > len(best_blocks):
            best_blocks = current_blocks
            # If we reach theoretical max or a very high number, we can break early
            # Max possible is 10 (84/8). 10 is unlikely due to parity/structure, but 7,8,9 are possible.
            if len(best_blocks) >= 9:
                break

    # Fallback to known good set if search fails or is slow (shouldn't happen)
    if len(best_blocks) < 6:
        best_blocks = [
            (0, 1, 0, 1, 0, 1),
            (0, 1, 0, 1, 2, 3),
            (0, 1, 0, 1, 4, 5),
            (0, 1, 2, 3, 0, 1),
            (0, 1, 2, 3, 2, 3),
            (0, 1, 2, 3, 4, 5),
        ]

    # Construct matrices
    B = len(best_blocks)
    rank = 7 * B + (84 - 8 * B)
    rank = 84 - B
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    covered = set()
    
    def add_strassen_block(col_start, i0, i1, j0, j1, k0, k1):
        # M1
        U[i0 * m + j0][col_start] = 1
        U[i1 * m + j1][col_start] = 1
        V[j0 * p + k0][col_start] = 1
        V[j1 * p + k1][col_start] = 1
        W[k0 * n + i0][col_start] = 1
        W[k1 * n + i1][col_start] = 1

        # M2
        U[i1 * m + j0][col_start + 1] = 1
        U[i1 * m + j1][col_start + 1] = 1
        V[j0 * p + k0][col_start + 1] = 1
        W[k0 * n + i1][col_start + 1] = 1
        W[k1 * n + i1][col_start + 1] = -1

        # M3
        U[i0 * m + j0][col_start + 2] = 1
        V[j0 * p + k1][col_start + 2] = 1
        V[j1 * p + k1][col_start + 2] = -1
        W[k1 * n + i0][col_start + 2] = 1
        W[k1 * n + i1][col_start + 2] = 1

        # M4
        U[i1 * m + j1][col_start + 3] = 1
        V[j1 * p + k0][col_start + 3] = 1
        V[j0 * p + k0][col_start + 3] = -1
        W[k0 * n + i0][col_start + 3] = 1
        W[k0 * n + i1][col_start + 3] = 1

        # M5
        U[i0 * m + j0][col_start + 4] = 1
        U[i0 * m + j1][col_start + 4] = 1
        V[j1 * p + k1][col_start + 4] = 1
        W[k0 * n + i0][col_start + 4] = -1
        W[k1 * n + i0][col_start + 4] = 1

        # M6
        U[i1 * m + j0][col_start + 5] = 1
        U[i0 * m + j0][col_start + 5] = -1
        V[j0 * p + k0][col_start + 5] = 1
        V[j0 * p + k1][col_start + 5] = 1
        W[k1 * n + i1][col_start + 5] = 1

        # M7
        U[i0 * m + j1][col_start + 6] = 1
        U[i1 * m + j1][col_start + 6] = -1
        V[j1 * p + k0][col_start + 6] = 1
        V[j1 * p + k1][col_start + 6] = 1
        W[k0 * n + i0][col_start + 6] = 1

        return col_start + 7

    for b in best_blocks:
        col = add_strassen_block(col, b[0], b[1], b[2], b[3], b[4], b[5])
        for ii in (b[0], b[1]):
            for jj in (b[2], b[3]):
                for kk in (b[4], b[5]):
                    covered.add((ii, jj, kk))

    # Schoolbook for remaining
    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                if (i0, j0, k0) in covered:
                    continue
                U[i0 * m + j0][col] = 1
                V[j0 * p + k0][col] = 1
                W[k0 * n + i0][col] = 1
                col += 1

    assert col == rank, f"Rank mismatch: expected {rank}, got {col}"
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = 0
                for c in range(rank):
                    total += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if total != 1:
                    raise ValueError(f"Decomposition failed at ({i},{j},{k}): got {total}, expected 1")

    return {"U": U, "V": V, "W": W}
