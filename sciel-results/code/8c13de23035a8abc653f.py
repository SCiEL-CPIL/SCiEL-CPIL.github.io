"""Rank-minimal exact tensor decomposition for 2x3 x 3x4 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]
_START = time.time()
_DEADLINE = 150.0 - 12.0  # safety margin


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _schoolbook(n, m, p):
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


def _build_system(n, m, p, r):
    """
    Build the linear system for a rank-r decomposition.

    We want U (nm x r), V (mp x r), W (pn x r) such that
        sum_r U[a, r] V[b, r] W[c, r] = T[a, b, c]
    where T[a,b,c] = 1 if a=i*m+j, b=j'*p+k, c=k'*n+i' with j==j' and i==i',
    i.e. T[i*m+j, jp+k, kp+ip] = 1 if j==jp and i==ip else 0.

    Fix V and W to basis vectors (one-hot) to make the system linear in U.
    Strategy: assign each column r a distinct (v_idx, w_idx) pair from
    {0..mp-1} x {0..pn-1}. Then V[b, r] = delta(b, v_idx_r), W[c, r] = delta(c, w_idx_r).
    The equation becomes: for each (a, b, c),
        sum_r U[a, r] delta(b, v_r) delta(c, w_r) = T[a,b,c]
    Only terms where v_r == b and w_r == c contribute, so:
        sum_{r: v_r==b and w_r==c} U[a, r] = T[a,b,c]
    """
    nm, mp, pn = n * m, m * p, p * n
    # Assign pairs (v_idx, w_idx) to columns. We need at most min(mp, pn) distinct pairs
    # to cover all (b,c) combos that appear in T. T is nonzero only when
    # a=i*m+j, b=j*p+k, c=k*n+i with j==j' and i==i'.
    # So (b,c) = (j*p+k, k*n+i) where j in [0,m), k in [0,p), i in [0,n).
    # The set of (b,c) pairs that matter: for each j, k, i: (j*p+k, k*n+i).
    # There are m*p*n = 24 such pairs, but many may coincide.
    # We need r >= number of distinct (b,c) pairs that have T != 0 for some a.
    # Actually T[a,b,c] depends on a too. Let's just enumerate.
    
    # For a linear system, we fix the assignment of (v_r, w_r) for r in range(r).
    # We'll try different assignments and solve for U.
    # The system: for each (a, b, c) with T[a,b,c] != 0,
    #   sum_{r: v_r==b and w_r==c} U[a,r] = 1
    # And for T[a,b,c] == 0, the sum must be 0.
    
    # Number of equations: nm * mp * pn (all triples), but most are 0.
    # Number of unknowns: nm * r.
    
    # Build sparse-ish system. For efficiency, only include nonzero equations.
    # Nonzero T: T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k.
    # So equations: for each (i,j,k): sum_{r: v_r==j*p+k and w_r==k*n+i} U[i*m+j, r] = 1
    # Plus zero equations for all other (a,b,c).
    
    # Total equations: nm*mp*pn = 6*12*8 = 576. Unknowns: nm*r = 6r.
    # For r < 24, 6r < 144 < 576, so overdetermined. We need a consistent solution.
    
    # The key constraint: for each (b,c) pair, let R_bc = {r: v_r==b and w_r==c}.
    # Then for each a, sum_{r in R_bc} U[a,r] must equal T[a,b,c].
    # T[a,b,c] is 1 only when a = i*m+j, b=j*p+k, c=k*n+i for the SAME i,j,k.
    # So for a given (b=j*p+k, c=k*n+i), T[a,b,c] = 1 iff a = i*m+j (with that i,j,k).
    # There's exactly one such a: a = i*m+j where i = c//n, j = b//p, and we need
    # i == c//n and j == b//p. But b = j*p+k means j = b//p, k = b%p.
    # c = k*n+i means k = c//n, i = c%n.
    # So consistency requires: b%p == c//n (the k matches).
    # If b%p != c//n, then T[a,b,c] = 0 for all a, so sum_{r in R_bc} U[a,r] = 0 for all a.
    # If b%p == c//n, then T[i*m+j, b, c] = 1 where i=c%n, j=b//p, and 0 otherwise.
    
    # So the system decomposes by (b,c) pairs! For each (b,c):
    #   If b%p != c//n: for all a, sum_{r in R_bc} U[a,r] = 0
    #   If b%p == c//n: for a = (c%n)*m + (b//p), sum_{r in R_bc} U[a,r] = 1
    #                     for other a, sum_{r in R_bc} U[a,r] = 0
    
    # This means for each (b,c) with b%p==c//n, we need |R_bc| >= 1 and
    # U[(c%n)*m + (b//p), r] for r in R_bc must sum to 1, and for other a, sum to 0.
    # Simplest: |R_bc| = 1, U[target_a, r] = 1.
    
    # So we need an assignment of pairs (v_r, w_r) such that every (b,c) with
    # b%p==c//n has exactly one r with v_r==b, w_r==c.
    # Number of such (b,c) pairs: for each k in [0,p), i in [0,n), j in [0,m):
    #   b = j*p+k, c = k*n+i. Distinct pairs? (j*p+k, k*n+i).
    #   Different (j,k,i) give different pairs? b determines j,k. c determines k,i.
    #   So (b,c) determines (j,k,i). Thus m*p*n = 24 distinct pairs.
    # We need r >= 24 for a one-to-one assignment. That's the schoolbook rank!
    
    # To get r < 24, we'd need some (b,c) pairs to share an r, meaning
    # v_r==b1 and v_r==b2 simultaneously (impossible) or the equations are
    # compatible. Since each r has a single (v_r, w_r), two different (b,c) pairs
    # can't share the same r unless they're the same pair.
    
    # Therefore the linear (one-hot V,W) approach gives exactly rank 24 minimum.
    # To do better, we need V and W to NOT be one-hot, i.e., a nonlinear problem.
    pass
    return None


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check C = U V^T W reproduces the multiplication tensor."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    rank = U_np.shape[1]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            computed = np.sum(U_np[i*m+j] * V_np[jp*p+k] * W_np[kp*n+ip])
                            if abs(computed - expected) > tol:
                                return False
    return True


def _try_rank(n, m, p, r, time_limit):
    """
    Try to find a rank-r decomposition using nonlinear optimization.
    We parameterize U, V, W directly and minimize the residual.
    """
    nm, mp, pn = n * m, m * p, p * n
    num_vars = (nm + mp + pn) * r
    
    # Build target tensor
    T = np.zeros((pn, nm, mp), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k*n+i, i*m+j, j*p+k] = 1.0
    
    def unpack(x):
        U = x[:nm*r].reshape(nm, r)
        V = x[nm*r:(nm+mp)*r].reshape(mp, r)
        W = x[(nm+mp)*r:].reshape(pn, r)
        return U, V, W
    
    def residuals(x):
        U, V, W = unpack(x)
        T_rec = np.einsum('ar,br,cr->abc', W, U, V)
        return (T_rec - T).flatten()
    
    import scipy.optimize
    start = time.time()
    for attempt in range(3):
        if time.time() - start > time_limit:
            break
        np.random.seed(attempt * 1000 + r)
        x0 = np.random.randn(num_vars) * 0.5
        # Set first column to a schoolbook-like initialization
        try:
            result = scipy.optimize.least_squares(
                residuals, x0, method='trf', max_nfev=500,
                xtol=1e-10, ftol=1e-10, gtol=1e-10
            )
            x_opt = result.x
            U, V, W = unpack(x_opt)
            # Snap to half-integers
            U_s = np.round(U * 2) / 2.0
            V_s = np.round(V * 2) / 2.0
            W_s = np.round(W * 2) / 2.0
            if _verify(U_s.tolist(), V_s.tolist(), W_s.tolist(), n, m, p):
                return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
            # Also try integer snapping
            U_i = np.round(U)
            V_i = np.round(V)
            W_i = np.round(W)
            if _verify(U_i.tolist(), V_i.tolist(), W_i.tolist(), n, m, p):
                return {"U": U_i.tolist(), "V": V_i.tolist(), "W": W_i.tolist()}
        except Exception:
            pass
    return None


def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 24
    
    # The linear one-hot approach proves rank >= 24 for that family.
    # Try nonlinear search for r < 24, but it's very unlikely to succeed
    # quickly. Allocate a small time budget.
    best = None
    for r in range(18, rank_schoolbook):
        if _time_left() < 5:
            break
        t_lim = min(8, _time_left() - 5)
        if t_lim <= 0:
            break
        res = _try_rank(n, m, p, r, t_lim)
        if res is not None:
            best = res
            break
    
    if best is None:
        best = _schoolbook(n, m, p)
    
    return best
