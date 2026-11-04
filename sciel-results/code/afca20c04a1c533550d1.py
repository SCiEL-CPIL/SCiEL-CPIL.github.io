"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS

    # Precompute all 2x2x2 blocks
    blocks = []
    for i0 in range(n):
        for i1 in range(i0 + 1, n):
            for j0 in range(m):
                for j1 in range(j0 + 1, m):
                    for k0 in range(p):
                        for k1 in range(k0 + 1, p):
                            terms = set()
                            for i in (i0, i1):
                                for j in (j0, j1):
                                    for k in (k0, k1):
                                        terms.add((i, j, k))
                            blocks.append(((i0, i1), (j0, j1), (k0, k1), terms))

    total_terms = n * m * p
    num_blocks_total = len(blocks)

    # Greedy with random restarts
    best_blocks = []
    best_count = 0

    num_restarts = 500
    rng = random.Random(42)

    for r in range(num_restarts):
        if time.time() > deadline - 15:
            break

        shuffled = blocks[:]
        rng.shuffle(shuffled)

        covered = set()
        current_blocks = []

        for block in shuffled:
            terms = block[3]
            if not (terms & covered):
                covered |= terms
                current_blocks.append(block)

        if len(current_blocks) > best_count:
            best_count = len(current_blocks)
            best_blocks = current_blocks[:]

    # Local search: try to swap out one block for two that cover more
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        covered = set()
        for b in best_blocks:
            covered |= b[3]

        for idx in range(len(best_blocks)):
            if time.time() > deadline - 10:
                break
            # Remove block idx
            new_covered = set()
            for j, b in enumerate(best_blocks):
                if j != idx:
                    new_covered |= b[3]

            removed_terms = best_blocks[idx][3] - new_covered

            # Try adding two blocks that cover removed terms
            for a in range(num_blocks_total):
                if time.time() > deadline - 10:
                    break
                terms_a = blocks[a][3]
                if terms_a & new_covered:
                    continue
                # Check if terms_a helps cover removed
                if not (terms_a & removed_terms):
                    continue
                new_covered2 = new_covered | terms_a
                for b in range(num_blocks_total):
                    if b == a:
                        continue
                    terms_b = blocks[b][3]
                    if terms_b & new_covered2:
                        continue
                    if not (terms_b & removed_terms):
                        continue
                    new_count = len(best_blocks) - 1 + 2
                    if new_count > len(best_blocks):
                        new_blocks = [x for j, x in enumerate(best_blocks) if j != idx]
                        new_blocks.append(blocks[a])
                        new_blocks.append(blocks[b])
                        best_blocks = new_blocks
                        best_count = new_count
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    num_blocks = len(best_blocks)
    remaining_terms = total_terms - num_blocks * 8
    rank = num_blocks * 7 + remaining_terms

    # Strassen coefficients for 2x2x2 block
    # Local indices: U[li*2+lj, t], V[lj*2+lk, t], W[lk*2+li, t]
    U_strassen = np.zeros((4, 7))
    V_strassen = np.zeros((4, 7))
    W_strassen = np.zeros((4, 7))

    # M1: (A00+A11)(B00+B11) -> C00+=1, C11+=1
    U_strassen[0, 0] = 1; U_strassen[3, 0] = 1
    V_strassen[0, 0] = 1; V_strassen[3, 0] = 1
    W_strassen[0, 0] = 1; W_strassen[3, 0] = 1

    # M2: (A10+A11)B00 -> C10+=1, C11-=1
    U_strassen[2, 1] = 1; U_strassen[3, 1] = 1
    V_strassen[0, 1] = 1
    W_strassen[1, 1] = 1; W_strassen[3, 1] = -1

    # M3: A00(B01-B11) -> C01+=1, C11+=1
    U_strassen[0, 2] = 1
    V_strassen[1, 2] = 1; V_strassen[3, 2] = -1
    W_strassen[2, 2] = 1; W_strassen[3, 2] = 1

    # M4: A11(B10-B00) -> C00+=1, C10+=1
    U_strassen[3, 3] = 1
    V_strassen[2, 3] = 1; V_strassen[0, 3] = -1
    W_strassen[0, 3] = 1; W_strassen[1, 3] = 1

    # M5: (A00+A01)B11 -> C00-=1, C01+=1
    U_strassen[0, 4] = 1; U_strassen[1, 4] = 1
    V_strassen[3, 4] = 1
    W_strassen[0, 4] = -1; W_strassen[2, 4] = 1

    # M6: (A10-A00)(B00+B01) -> C11+=1
    U_strassen[2, 5] = 1; U_strassen[0, 5] = -1
    V_strassen[0, 5] = 1; V_strassen[1, 5] = 1
    W_strassen[3, 5] = 1

    # M7: (A01-A11)(B10+B11) -> C00+=1
    U_strassen[1, 6] = 1; U_strassen[3, 6] = -1
    V_strassen[2, 6] = 1; V_strassen[3, 6] = 1
    W_strassen[0, 6] = 1

    # Build U, V, W
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0
    covered = set()

    for I, J, K, terms in best_blocks:
        i0, i1 = I
        j0, j1 = J
        k0, k1 = K

        for t in range(7):
            for li in range(2):
                for lj in range(2):
                    val = U_strassen[li * 2 + lj, t]
                    if val != 0:
                        gi = i0 if li == 0 else i1
                        gj = j0 if lj == 0 else j1
                        U[gi * m + gj, col] += val

            for lj in range(2):
                for lk in range(2):
                    val = V_strassen[lj * 2 + lk, t]
                    if val != 0:
                        gj = j0 if lj == 0 else j1
                        gk = k0 if lk == 0 else k1
                        V[gj * p + gk, col] += val

            for lk in range(2):
                for li in range(2):
                    val = W_strassen[lk * 2 + li, t]
                    if val != 0:
                        gk = k0 if lk == 0 else k1
                        gi = i0 if li == 0 else i1
                        W[gk * n + gi, col] += val

            col += 1
        covered |= terms

    # Add remaining schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

    # Trim
    if col < U.shape[1]:
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]

    # Verify
    def verify(U_mat, V_mat, W_mat):
        U_r = U_mat.reshape(n, m, -1)
        V_r = V_mat.reshape(m, p, -1)
        W_r = W_mat.reshape(p, n, -1)

        T = np.zeros((n, m, p))
        for c in range(U_mat.shape[1]):
            T += np.einsum('ij,jk,ki->ijk', U_r[:, :, c], V_r[:, :, c], W_r[:, :, c])

        return np.allclose(T, 1.0, atol=1e-9)

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb), dtype=float)
        V_sb = np.zeros((m * p, rank_sb), dtype=float)
        W_sb = np.zeros((p * n, rank_sb), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1.0
                    V_sb[j * p + k, col] = 1.0
                    W_sb[k * n + i, col] = 1.0
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
