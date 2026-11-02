"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 matrix multiplication.
    Uses randomized search with linear system solving for W given random U, V.
    """
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n  # 9, 9, 9

    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def schoolbook():
        rank = n*m*p
        U = [[0]*rank for _ in range(nm)]
        V = [[0]*rank for _ in range(mp)]
        W = [[0]*rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank

    U_s, V_s, W_s, rank_s = schoolbook()
    best = (U_s, V_s, W_s, rank_s)

    np.random.seed(12345)

    for target_rank in range(19, 27):
        if time.time() > deadline - 30:
            break
        max_attempts = 500
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break

            U = np.random.randn(nm, target_rank)
            V = np.random.randn(mp, target_rank)

            W = np.zeros((pn, target_rank))
            ok = True
            for c in range(pn):
                M = np.zeros((nm*mp, target_rank))
                for a in range(nm):
                    for b in range(mp):
                        M[a*mp+b, :] = U[a, :] * V[b, :]
                rhs = T[:, :, c].flatten()
                try:
                    w, res, rk, sv = np.linalg.lstsq(M, rhs, rcond=None)
                    W[:, c] = w
                except:
                    ok = False
                    break
            if not ok:
                continue

            # Verify
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(recon, T, atol=1e-10):
                # Try to convert to exact rationals
                U_frac = [[Fraction(U[a, r]).limit_denominator(1000000) for r in range(target_rank)] for a in range(nm)]
                V_frac = [[Fraction(V[b, r]).limit_denominator(1000000) for r in range(target_rank)] for b in range(mp)]
                W_frac = [[Fraction(W[c, r]).limit_denominator(1000000) for r in range(target_rank)] for c in range(pn)]

                # Verify exactness
                recon_exact = np.zeros((nm, mp, pn), dtype=object)
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            s = Fraction(0)
                            for r in range(target_rank):
                                s += U_frac[a][r] * V_frac[b][r] * W_frac[c][r]
                            recon_exact[a, b, c] = s

                exact = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if recon_exact[i*m+j, j*p+k, k*n+i] != 1:
                                exact = False
                if exact:
                    # Check zeros
                    for a in range(nm):
                        for b in range(mp):
                            for c in range(pn):
                                found = False
                                for i in range(n):
                                    for j in range(m):
                                        for k in range(p):
                                            if i*m+j == a and j*p+k == b and k*n+i == c:
                                                found = True
                                if not found and recon_exact[a, b, c] != 0:
                                    exact = False
                if exact:
                    best = (U_frac, V_frac, W_frac, target_rank)
                    break
        if best[3] == target_rank:
            break

    U, V, W, rank = best
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
