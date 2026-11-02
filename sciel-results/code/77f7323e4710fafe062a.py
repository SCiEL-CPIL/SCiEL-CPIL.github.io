"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [3, 4, 4]
DEADLINE_MARGIN = 10.0  # seconds before the hard kill we must return

def _time_left(start):
    return DEADLINE_MARGIN - (time.time() - start)

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    # Schoolbook fallback (rank 48)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    # The matrix multiplication tensor T[i,j,k] has support at (i,j,k) in [n]x[m]x[p].
    # A rank-1 term (u, v, w) contributes u[i,j]*v[j,k]*w[k,i] to T[i,j,k].
    # We need sum_r u_r[i,j]*v_r[j,k]*w_r[k,i] = 1 if (i,j,k) in support, else 0.
    # Since support is all (i,j,k), we need sum_r u_r[i,j]*v_r[j,k]*w_r[k,i] = 1 for all i,j,k.
    # This is exactly: the bilinear map (A,B) -> AB has tensor rank equal to the
    # minimum r such that there exist U,V,W with the product identity.

    # Known bounds for 3x4 x 4x4:
    # Schoolbook: 48
    # Strassen-like or other known decompositions may exist.
    # Let's try to find a rank < 48 decomposition via structured search.

    # Approach: try rank r from some lower bound up, and use a randomized/structured
    # search. For small dimensions, we can try to verify candidate decompositions.

    # A key insight: if we can find a decomposition of rank r, we need:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1

    # Let's try a few known low-rank constructions.
    # For 2x2x2, Strassen gives rank 7.
    # For 3x3x3, the best known is 23 (or lower with recent results).
    # For 3x4x4, we don't have a well-known closed form, so we'll search.

    # Strategy: try to find a decomposition using a greedy/iterative approach.
    # We'll use the fact that the tensor is "dense" (all entries are 1) in its support.

    # Let's try a simple approach: assume U, V, W have entries in {0, 1/2, 1} and
    # search for small rank. This is a constraint satisfaction problem.

    # For efficiency, let's try rank values and use a randomized local search.

    def verify(U, V, W, r):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Try to find a rank-1 decomposition of the "unit" tensor.
    # Actually, let's try a different approach: use the known fact that
    # matmul(3,4,4) has rank at most 48, and try to reduce it.

    # Let's try a structured search for small rank values.
    # We'll use a simple heuristic: try to find U, V, W with small entries.

    # For now, let's just return the schoolbook if we can't find better quickly.
    # But let's try a few candidate ranks.

    # Actually, let's try a randomized search for rank < 48.
    # We'll fix r and try to find U, V, W.

    # This is a hard problem. Let's try a few known decompositions.

    # For 3x4 x 4x4, one approach is to use the fact that 3x4 = 3x(2x2),
    # and use Strassen for the 2x2 parts. But this is complex.

    # Let's just try to verify if rank 47, 46, ... is possible with a simple search.
    # Given the time constraint, we'll do a limited search.

    # Try rank from 40 to 47 (since 48 is schoolbook)
    for r in range(40, 48):
        if _time_left(start) < 5:
            break
        # Try a few random initializations
        for trial in range(20):
            if _time_left(start) < 5:
                break
            random.seed(42 + r * 100 + trial)
            U = [[random.choice([0, 1, -1, 0.5, -0.5]) for _ in range(r)] for _ in range(n * m)]
            V = [[random.choice([0, 1, -1, 0.5, -0.5]) for _ in range(r)] for _ in range(m * p)]
            W = [[random.choice([0, 1, -1, 0.5, -0.5]) for _ in range(r)] for _ in range(p * n)]
            if verify(U, V, W, r):
                best = {"U": U, "V": V, "W": W}
                best_rank = r
                break

    # If we found a better rank, return it; otherwise return schoolbook.
    if best_rank < rank_sb:
        return best
    return best
# EVOLVE-BLOCK-END
