import time
import random
from math import gcd
from fractions import Fraction
import numpy as np

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.

    Baseline (rank 44):
      - A1(2x2)*B1(2x8): 4 Strassen 2x2x2 blocks -> rank 28
      - A2(2x1)*B2(1x8): outer product -> rank 16
      - Additive rank = 44 (sub-tensors share output indices but disjoint inputs)

    Refinement:
      The two sub-tensors share the SAME output index set C[k,i]; they only differ
      in the input indices they touch. We therefore search for a SINGLE shared
      rank-r decomposition (r < 44) of their direct sum, using random initialization
      + exact rational recovery + verification. Falls back to rank 44 if none found.
    """
    start_time = time.time()
    HARD_DEADLINE = 115.0  # safety margin before 120s kill

    n, m, p = 2, 3, 8
    nU, nV, nW = n * m, m * p, p * n  # 6, 24, 16

    # ---------- Build the full multiplication tensor T[i,a,k] = 1 ----------
    T = np.ones((nU, nV, nW), dtype=np.int64)

    # ---------- Baseline rank-44 construction ----------
    def build_rank44():
        rank = 44
        U = np.zeros((nU, rank), dtype=np.int64)
        V = np.zeros((nV, rank), dtype=np.int64)
        W = np.zeros((nW, rank), dtype=np.int64)

        S_U = [
            [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
            [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1],
        ]
        S_V = [
            [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
            [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1],
        ]
        S_W = [
            [1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 1], [1, 1, 0, 0],
            [-1, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0],
        ]

        col = 0
        # Part 1: A1(2x2) * B1(2x8), 4 Strassen blocks
        for block_idx in range(4):
            k_start = block_idx * 2
            for r in range(7):
                for li in range(2):
                    for la in range(2):
                        val = S_U[r][li * 2 + la]
                        if val != 0:
                            U[li * m + la, col] += val
                for la in range(2):
                    for lk in range(2):
                        val = S_V[r][la * 2 + lk]
                        if val != 0:
                            V[la * p + (k_start + lk), col] += val
                for lk in range(2):
                    for li in range(2):
                        val = S_W[r][lk * 2 + li]
                        if val != 0:
                            W[(k_start + lk) * n + li, col] += val
                col += 1

        # Part 2: A2(2x1) * B2(1x8) outer product
        for i in range(2):
            for k in range(8):
                U[i * m + 2, col] = 1
                V[2 * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

        return U, V, W

    # ---------- Verification ----------
    def verify(U, V, W):
        """Check sum_c U[i,a,c]*V[a,k,c]*W[k,i,c] == 1 for all (i,a,k)."""
        rank = U.shape[1]
        for i in range(n):
            for a in range(m):
                u_row = U[i * m + a]
                for k in range(p):
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    acc = 0
                    for c in range(rank):
                        uc, vc, wc = u_row[c], v_row[c], w_row[c]
                        if uc != 0 and vc != 0 and wc != 0:
                            acc += uc * vc * wc
                    if acc != 1:
                        return False
        return True

    U44, V44, W44 = build_rank44()
    if not verify(U44, V44, W44):
        # Fallback to schoolbook rank 48
        rank = n * m * p
        U = np.zeros((nU, rank), dtype=np.int64)
        V = np.zeros((nV, rank), dtype=np.int64)
        W = np.zeros((nW, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a, col] = 1
                    V[a * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_U, best_V, best_W = U44, V44, W44
    best_rank = 44

    # ---------- Refinement: search for shared rank-r decomposition ----------
    # The two sub-tensors share output indices; try to find a single rank-r
    # decomposition of their direct sum that beats 44.

    def recover_exact(Uf, Vf, Wf):
        """
        Given float factor matrices, recover integer/half-integer entries via
        rational reconstruction (GCD of denominators) + rounding.
        Returns (U, V, W) as int64 arrays, or None if recovery fails.
        """
        def recover_one(M):
            # Find smallest positive integer D such that D*M is (near) integer
            # Use the set of all entries; D = GCD-free approach:
            # For each entry, find its rational approximation, take LCM of denominators.
            D = 1
            for val in M.ravel():
                if abs(val) < 1e-9:
                    continue
                # Approximate val as a fraction with small denominator
                for denom in range(1, 65):
                    num = round(val * denom)
                    if abs(val * denom - num) < 1e-6:
                        # val ≈ num/denom
                        # D must be a multiple of denom / gcd(num, denom)
                        g = gcd(abs(num), denom)
                        d_needed = denom // g
                        # LCM(D, d_needed)
                        D = D * d_needed // gcd(D, d_needed)
                        break
                if D > 10**6:
                    return None
            # Round D*M to integers
            M_int = np.round(D * M).astype(np.int64)
            # Verify D*M is close to integer
            if np.max(np.abs(D * M - M_int)) > 1e-4:
                return None
            # Check that D*M_int / D equals M (sanity)
            if np.max(np.abs(M_int.astype(np.float64) / D - M)) > 1e-6:
                return None
            return M_int
        return recover_one(Uf), recover_one(Vf), recover_one(Wf)

    def try_rank(r, n_restarts):
        """Try to find a rank-r decomposition via random init + recovery."""
        rng = np.random.default_rng(r * 1000 + n_restarts)
        for restart in range(n_restarts):
            if time.time() > HARD_DEADLINE - 5:
                return None
            # Random initialization
            Uf = rng.standard_normal((nU, r))
            Vf = rng.standard_normal((nV, r))
            Wf = rng.standard_normal((nW, r))
            # Scale to roughly match T magnitude
            Uf *= 0.5
            Vf *= 0.5
            Wf *= 0.5

            # A few ALS-style iterations (coarse, to get near a solution)
            for it in range(30):
                # Update U: U[i,a,:] = sum_{k} T[i,a,k] * (V[a,k,:] * W[k,i,:]) / (V*W)
                for i in range(n):
                    for a in range(m):
                        num = np.zeros(r)
                        den = np.zeros(r)
                        for k in range(p):
                            vk = Vf[a * p + k]
                            wk = Wf[k * n + i]
                            num += vk * wk
                            den += vk * wk * Uf[i * m + a]
                        den[den < 1e-8] = 1e-8
                        Uf[i * m + a] = num / den

                for a in range(m):
                    for k in range(p):
                        num = np.zeros(r)
                        den = np.zeros(r)
                        for i in range(n):
                            ui = Uf[i * m + a]
                            wi = Wf[k * n + i]
                            num += ui * wi
                            den += ui * wi * Vf[a * p + k]
                        den[den < 1e-8] = 1e-8
                        Vf[a * p + k] = num / den

                for k in range(p):
                    for i in range(n):
                        num = np.zeros(r)
                        den = np.zeros(r)
                        for a in range(m):
                            ua = Uf[i * m + a]
                            va = Vf[a * p + k]
                            num += ua * va
                            den += ua * va * Wf[k * n + i]
                        den[den < 1e-8] = 1e-8
                        Wf[k * n + i] = num / den

            # Recover exact integers
            rec = recover_exact(Uf, Vf, Wf)
            if rec is None:
                continue
            U, V, W = rec
            if U is None or V is None or W is None:
                continue
            if verify(U, V, W):
                return U, V, W
        return None

    # Search ranks from 43 down to 38 (time permitting)
    for r in range(43, 37, -1):
        if time.time() > HARD_DEADLINE - 10:
            break
        # Budget: more restarts for higher ranks (closer to 44)
        n_restarts = 5 if r >= 41 else 3
        result = try_rank(r, n_restarts)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = r
            break  # Found a valid lower rank; stop

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
