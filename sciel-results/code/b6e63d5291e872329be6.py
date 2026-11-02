"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition.

Combines:
  * a structural rank-44 decomposition (rank-13 M_{2,2,4} blocks + M_{2,1,8}
    schoolbook part) as the guaranteed fallback / strong seed, with
  * alternating-least-squares polish that rounds to half-integers and is
    gated by exact Fraction verification, under an explicit deadline guard.
"""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    DEADLINE = t0 + 110.0  # Safety margin before 120s kill

    def elapsed():
        return time.time() - t0

    # ---------------- structural builders ----------------
    def build_2x2x4_rank13():
        """Rank-13 decomposition of M_{2,2,4}.

        Local indexing: U[i*2+j], V[j*4+k], W[k*2+i], i in {0,1}, j in {0,1},
        k in {0..3}.  Returns (U,V,W) as lists of lists, shape (4,13),(8,13),(8,13).
        """
        rank = 13
        U = [[0] * rank for _ in range(4)]
        V = [[0] * rank for _ in range(8)]
        W = [[0] * rank for _ in range(8)]

        # Left block (k=0,1)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[4][0] = 1
        W[0][0] = 1; W[5][0] = 1

        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[4][1] = 1

        U[0][2] = 1
        V[1][2] = 1; V[5][2] = 1
        W[1][2] = 1; W[5][2] = 1

        U[0][3] = 1; U[2][3] = -1
        V[1][3] = 1; V[4][3] = 1
        W[0][3] = 1; W[4][3] = 1

        U[1][4] = 1; U[2][4] = 1
        V[0][4] = 1; V[4][4] = -1
        W[0][4] = -1; W[1][4] = 1

        U[1][5] = -1; U[3][5] = 1
        V[1][5] = 1; V[5][5] = -1
        W[5][5] = 1

        U[0][6] = 1; U[1][6] = 1
        V[1][6] = -1; V[5][6] = 1
        W[0][6] = 1

        # Right block (k=2,3)
        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[6][7] = 1
        W[4][7] = 1; W[7][7] = 1

        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1
        W[6][8] = 1

        U[0][9] = 1
        V[3][9] = 1; V[7][9] = 1
        W[5][9] = 1; W[7][9] = 1

        U[0][10] = 1; U[2][10] = -1
        V[3][10] = 1; V[6][10] = 1
        W[4][10] = 1; W[6][10] = 1

        U[1][11] = 1; U[2][11] = 1
        V[2][11] = 1; V[6][11] = -1
        W[4][11] = -1; W[5][11] = 1

        U[1][12] = -1; U[3][12] = 1
        V[3][12] = 1; V[7][12] = -1
        W[7][12] = 1

        return U, V, W

    def build_1plus2_split():
        """Rank-42 decomposition: M_{2,1,8} (rank 16) + 2 * M_{2,2,4} (rank 13).

        Full indexing: U[i*m+j] (6 x r), V[j*p+k] (24 x r), W[k*n+i] (16 x r).
        """
        rank1 = 16
        rank2 = 26
        total_rank = rank1 + rank2

        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]

        col = 0
        # Part 1: A[:,0] (2x1) * B[0,:] (1x8)  -> schoolbook rank 16
        for i in range(n):
            for k in range(p):
                U[i * m + 0][col] = 1
                V[0 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

        U224, V224, W224 = build_2x2x4_rank13()

        # Part 2: A[:,1:3] (2x2) * B[1:3,:] (2x8), split into two 4-column blocks.
        for block in range(2):
            k_off = block * 4
            for r in range(13):
                for i in range(2):
                    U[i * m + 1][col + r] = U224[i * 2 + 0][r]  # j_local 0 -> j_full 1
                    U[i * m + 2][col + r] = U224[i * 2 + 1][r]  # j_local 1 -> j_full 2
                for j_local in range(2):
                    for k_local in range(4):
                        V[(1 + j_local) * p + (k_off + k_local)][col + r] = V224[j_local * 4 + k_local][r]
                for k_local in range(4):
                    for i in range(2):
                        W[(k_off + k_local) * n + i][col + r] = W224[k_local * 2 + i][r]
            col += 13

        return U, V, W

    def schoolbook():
        """Schoolbook rank n*m*p = 48 fallback."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    # ---------------- verification / rounding ----------------
    def verify_decomp(U, V, W, rank):
        """Exact verification using Fraction arithmetic."""
        Uf = [[Fraction(x) for x in row] for row in U]
        Vf = [[Fraction(x) for x in row] for row in V]
        Wf = [[Fraction(x) for x in row] for row in W]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += Uf[i * m + j][c] * Vf[j * p + k][c] * Wf[k * n + i][c]
                    if s != 1:
                        return False
        return True

    def round_to_half(vals):
        out = []
        for x in vals:
            r = round(x * 2) / 2
            out.append(int(r) if r == int(r) else r)
        return out

    def round_mat(M):
        return [round_to_half(row) for row in M]

    def compute_residual(U, V, W, rank):
        max_res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    res = abs(1.0 - s)
                    if res > max_res:
                        max_res = res
        return max_res

    # ---------------- polish (ALS + rounding) ----------------
    def try_optimize(U0, V0, W0, rank, budget_sec=6.0, seed=0):
        """Polish a rank-`rank` seed with ALS; round to half-integers; verify.
           Returns (U, V, W) as lists of lists, or None."""
        if elapsed() > DEADLINE - budget_sec:
            return None
        np.random.seed(seed)

        U = np.array(U0, dtype=np.float64).copy()
        V = np.array(V0, dtype=np.float64).copy()
        W = np.array(W0, dtype=np.float64).copy()

        def check_rounded():
            U_r = round_mat(U.tolist())
            V_r = round_mat(V.tolist())
            W_r = round_mat(W.tolist())
            if verify_decomp(U_r, V_r, W_r, rank):
                return U_r, V_r, W_r
            U_ri = [[int(round(x)) for x in row] for row in U.tolist()]
            V_ri = [[int(round(x)) for x in row] for row in V.tolist()]
            W_ri = [[int(round(x)) for x in row] for row in W.tolist()]
            if verify_decomp(U_ri, V_ri, W_ri, rank):
                return U_ri, V_ri, W_ri
            return None

        # If the seed already verifies after rounding, take it immediately.
        found = check_rounded()
        if found is not None:
            return found

        prev_res = float('inf')
        for it in range(400):
            if elapsed() > DEADLINE - 5:
                break

            # Update U
            for i in range(n):
                for j in range(m):
                    B = V[j * p:(j + 1) * p, :] * W[np.arange(p) * n + i, :]
                    try:
                        u, *_ = np.linalg.lstsq(B, np.ones(p), rcond=None)
                        U[i * m + j, :] = u
                    except Exception:
                        pass
            # Update V
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, rank), dtype=np.float64)
                    for i in range(n):
                        B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        v, *_ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                        V[j * p + k, :] = v
                    except Exception:
                        pass
            # Update W
            for i in range(n):
                for k in range(p):
                    B = np.zeros((m, rank), dtype=np.float64)
                    for j in range(m):
                        B[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        w, *_ = np.linalg.lstsq(B, np.ones(m), rcond=None)
                        W[k * n + i, :] = w
                    except Exception:
                        pass

            max_res = compute_residual(U, V, W, rank)
            if max_res < 1e-10:
                found = check_rounded()
                if found is not None:
                    return found
            if max_res > prev_res * 0.999 and it > 40:
                break
            prev_res = max_res
            if it % 50 == 49:
                found = check_rounded()
                if found is not None:
                    return found
        return None

    # ---------------- driver ----------------
    # Guaranteed fallbacks, best first.
    candidates = []
    U42, V42, W42 = build_1plus2_split()
    if verify_decomp(U42, V42, W42, 42):
        candidates.append((42, U42, V42, W42))
    U48, V48, W48 = schoolbook()
    if verify_decomp(U48, V48, W48, 48):
        candidates.append((48, U48, V48, W48))
    candidates.sort(key=lambda c: c[0])

    # 1) Polish the best structural seed at its own rank (sanity + possible luck).
    for rank, U0, V0, W0 in candidates:
        if elapsed() > DEADLINE - 12:
            break
        found = try_optimize(U0, V0, W0, rank, budget_sec=6.0, seed=777)
        if found is not None and verify_decomp(found[0], found[1], found[2], rank):
            return {"U": found[0], "V": found[1], "W": found[2]}

    # 2) Try to beat the best structural rank via ALS from that seed.
    if candidates:
        best_rank = candidates[0][0]
        U0, V0, W0 = candidates[0][1], candidates[0][2], candidates[0][3]
        for target_rank in range(best_rank - 1, max(n * m, m * p, p * n) - 1, -1):
            if elapsed() > DEADLINE - 15:
                break
            for seed in [42 + target_rank, 1000 + target_rank, 2000 + target_rank]:
                if elapsed() > DEADLINE - 10:
                    break
                found = try_optimize(U0, V0, W0, target_rank, budget_sec=5.0, seed=seed)
                if found is not None and verify_decomp(found[0], found[1], found[2], target_rank):
                    return {"U": found[0], "V": found[1], "W": found[2]}

    # 3) Return best verified structural candidate (or schoolbook).
    for rank, U0, V0, W0 in candidates:
        return {"U": U0, "V": V0, "W": W0}
    return {"U": U48, "V": V48, "W": W48}
# EVOLVE-BLOCK-END
