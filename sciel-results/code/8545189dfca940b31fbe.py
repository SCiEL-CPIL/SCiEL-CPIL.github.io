"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations, product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill

    def validate(U, V, W):
        """Check if the decomposition is exact using vectorized numpy."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.einsum('ijt,jkt,kit->ijk', U3, V3, W3)
        return np.allclose(T, 1.0, atol=1e-9)

    def build_solution(blocks, schoolbook_terms):
        """Build U, V, W from Strassen blocks and schoolbook terms."""
        total_rank = len(blocks) * 7 + len(schoolbook_terms)
        U = [[0.0] * total_rank for _ in range(n * m)]
        V = [[0.0] * total_rank for _ in range(m * p)]
        W = [[0.0] * total_rank for _ in range(p * n)]

        col = 0
        for (i0, j0, k0) in blocks:
            # M1 = (A00+A11)(B00+B11)
            c = col
            U[i0*m + j0][c] += 1; U[(i0+1)*m + (j0+1)][c] += 1
            V[j0*p + k0][c] += 1; V[(j0+1)*p + (k0+1)][c] += 1
            W[k0*n + i0][c] += 1; W[(k0+1)*n + (i0+1)][c] += 1
            col += 1
            # M2 = (A10+A11)B00
            c = col
            U[(i0+1)*m + j0][c] += 1; U[(i0+1)*m + (j0+1)][c] += 1
            V[j0*p + k0][c] += 1
            W[k0*n + (i0+1)][c] += 1; W[(k0+1)*n + (i0+1)][c] += -1
            col += 1
            # M3 = A00(B01-B11)
            c = col
            U[i0*m + j0][c] += 1
            V[j0*p + (k0+1)][c] += 1; V[(j0+1)*p + (k0+1)][c] += -1
            W[(k0+1)*n + i0][c] += 1; W[(k0+1)*n + (i0+1)][c] += 1
            col += 1
            # M4 = A11(B10-B00)
            c = col
            U[(i0+1)*m + (j0+1)][c] += 1
            V[(j0+1)*p + k0][c] += 1; V[j0*p + k0][c] += -1
            W[k0*n + i0][c] += 1; W[k0*n + (i0+1)][c] += 1
            col += 1
            # M5 = (A00+A01)B11
            c = col
            U[i0*m + j0][c] += 1; U[i0*m + (j0+1)][c] += 1
            V[(j0+1)*p + (k0+1)][c] += 1
            W[k0*n + i0][c] += -1; W[(k0+1)*n + i0][c] += 1
            col += 1
            # M6 = (A10-A00)(B00+B01)
            c = col
            U[(i0+1)*m + j0][c] += 1; U[i0*m + j0][c] += -1
            V[j0*p + k0][c] += 1; V[j0*p + (k0+1)][c] += 1
            W[(k0+1)*n + (i0+1)][c] += 1
            col += 1
            # M7 = (A01-A11)(B10+B11)
            c = col
            U[i0*m + (j0+1)][c] += 1; U[(i0+1)*m + (j0+1)][c] += -1
            V[(j0+1)*p + k0][c] += 1; V[(j0+1)*p + (k0+1)][c] += 1
            W[k0*n + i0][c] += 1
            col += 1

        for (i, j, k) in schoolbook_terms:
            c = col
            U[i*m + j][c] += 1
            V[j*p + k][c] += 1
            W[k*n + i][c] += 1
            col += 1

        return U, V, W

    def get_schoolbook_for_blocks(blocks):
        """Get schoolbook terms not covered by blocks."""
        covered = set()
        for (i0, j0, k0) in blocks:
            for i in (i0, i0+1):
                for j in (j0, j0+1):
                    for k in (k0, k0+1):
                        covered.add((i, j, k))
        return [(i, j, k) for i in range(n) for j in range(m) for k in range(p) if (i, j, k) not in covered]

    def are_disjoint(b1, b2):
        """Check if two Strassen blocks are disjoint."""
        return not (b1[0] in (b2[0], b2[0]+1) and b2[0] in (b1[0], b1[0]+1) and
                    b1[1] in (b2[1], b2[1]+1) and b2[1] in (b1[1], b1[1]+1) and
                    b1[2] in (b2[2], b2[2]+1) and b2[2] in (b1[2], b1[2]+1))

    def is_set_disjoint(blocks):
        """Check if a set of blocks are pairwise disjoint."""
        for a in range(len(blocks)):
            for b in range(a + 1, len(blocks)):
                if not are_disjoint(blocks[a], blocks[b]):
                    return False
        return True

    # Generate all possible Strassen block positions
    all_possible_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                all_possible_blocks.append((i0, j0, k0))

    # Current best
    best_rank = 75  # Schoolbook
    best_solution = None

    # Try the known good configuration: 4 blocks
    initial_blocks = [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)]
    sb = get_schoolbook_for_blocks(initial_blocks)
    rank = len(initial_blocks) * 7 + len(sb)
    if rank < best_rank:
        U, V, W = build_solution(initial_blocks, sb)
        if validate(U, V, W):
            best_rank = rank
            best_solution = (U, V, W)

    # Search for better block configurations
    # Try all combinations of up to 5 blocks
    for size in range(1, 6):
        if time.time() > deadline - 30:
            break
        for combo in combinations(all_possible_blocks, size):
            if not is_set_disjoint(list(combo)):
                continue
            sb = get_schoolbook_for_blocks(list(combo))
            rank = size * 7 + len(sb)
            if rank < best_rank:
                U, V, W = build_solution(list(combo), sb)
                if validate(U, V, W):
                    best_rank = rank
                    best_solution = (U, V, W)

    # Try to improve by merging residual terms with Strassen factors
    # Key insight: some residual terms might be mergeable with existing Strassen factors
    # by adding corrections that cancel out in the covered region but contribute to residual

    if best_solution is not None:
        return {"U": best_solution[0], "V": best_solution[1], "W": best_solution[2]}

    # Fallback: pure schoolbook
    U, V, W = build_solution([], [(i, j, k) for i in range(n) for j in range(m) for k in range(p)])
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
