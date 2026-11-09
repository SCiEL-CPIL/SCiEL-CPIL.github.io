"""Tensor decomposition search for 2x5 by 5x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def _verify(U, V, W, n, m, p, rank):
    """Exact check: for all i,k and random-ish integer inputs, (A@B)[i,k] == sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]."""
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    # Precompute: C_ik[j] = sum_c U[i*m+j,c]*V[j*p+k,c]
    # Then C_ik[j]*W[k*n+i,c] summed over j,c must equal identity for the (i,k) slot.
    # Actually the tensor: T[i,k, j, j2] where A has (i,j), B has (j2,k); product[i,k] = sum_j A[i,j]B[j,k].
    # Tensor component T[i,k,j,j2] = 1 if j==j2 else 0.
    # Decomposition: sum_c U[i*m+j, c] * V[j2*p+k, c] * W[k*n+i, c] = T[i,k,j,j2]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for j2 in range(m):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j2*p+k, c] * W[k*n+i, c]
                    expected = 1 if j == j2 else 0
                    if s != expected:
                        return False
    return True

def _to_halfint(x):
    """Return (numerator, denominator) as ints if x is integer or half-integer, else None."""
    # x is a float
    # check if 2*x is close to an integer
    twox = 2.0 * x
    nearest = round(twox)
    if abs(twox - nearest) < 1e-6:
        # nearest is integer; value = nearest/2
        return (int(nearest), 2)
    return None

def construct():
    start = time.time()
    DEADLINE = start + 140.0  # safety margin before 150s
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    # Build schoolbook as fallback
    def schoolbook():
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U":U, "V":V, "W":W}

    best = schoolbook()

    # Target: try to find rank below schoolbook.
    # Use randomized iterative approach: fix two factors, solve for third linearly, iterate.
    # We need exact integer/half-integer solutions.
    # Strategy: for a given rank r, initialize U,V,W randomly in {0,1,2} or small ints,
    # then do alternating least squares (linear) and check for exactness + half-integer-ness.
    # This is heuristic; try many seeds.

    rng = np.random.default_rng(12345)

    def try_rank(r, budget):
        # Number of constraints: n*p*m*m = 250
        # Unknowns: r*(n*m + m*p + p*n) = r*(10 + 25 + 10) = 45r
        # For r >= ceil(250/45)=6, overdetermined or underdetermined? 45r >= 250 -> r>=6.
        # But bilinear, so not linear. Use ALS.
        for seed in range(200):
            if time.time() > budget:
                break
            rng2 = np.random.default_rng(seed)
            # Initialize with small integers in {-2,-1,0,1,2}
            U = rng2.integers(-2, 3, size=(n*m, r)).astype(float)
            V = rng2.integers(-2, 3, size=(m*p, r)).astype(float)
            W = rng2.integers(-2, 3, size=(p*n, r)).astype(float)
            # ALS: iterate solving for each factor given the others.
            # For fixed V,W, solve for U: for each i,j and c, U[i*m+j,c] appears in
            # constraint T[i,k,j,j2] = sum_c U[i*m+j,c] V[j2*p+k,c] W[k*n+i,c]
            # For fixed i,j: we have m*p equations (over k,j2) for r unknowns U[i*m+j, 0..r-1].
            # Build matrix A_ij of size (m*p) x r: A_ij[(k,j2), c] = V[j2*p+k,c]*W[k*n+i,c]
            # b_ij[(k,j2)] = T[i,k,j,j2]
            converged = False
            for it in range(60):
                # Solve U
                U_new = np.zeros((n*m, r))
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((m*p, r))
                        b = np.zeros(m*p)
                        for k in range(p):
                            for j2 in range(m):
                                idx = k*m + j2
                                for c in range(r):
                                    A[idx, c] = V[j2*p+k, c] * W[k*n+i, c]
                                b[idx] = 1.0 if j == j2 else 0.0
                        # least squares
                        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                        U_new[i*m+j, :] = sol
                U = U_new
                # Solve V
                V_new = np.zeros((m*p, r))
                for j2 in range(m):
                    for k in range(p):
                        A = np.zeros((n*m, r))
                        b = np.zeros(n*m)
                        for i in range(n):
                            for j in range(m):
                                idx = i*m + j
                                for c in range(r):
                                    A[idx, c] = U[i*m+j, c] * W[k*n+i, c]
                                b[idx] = 1.0 if j == j2 else 0.0
                        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                        V_new[j2*p+k, :] = sol
                V = V_new
                # Solve W
                W_new = np.zeros((p*n, r))
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((n*m, r))
                        b = np.zeros(n*m)
                        for j in range(m):
                            for j2 in range(m):
                                idx = i*m + j
                                for c in range(r):
                                    A[idx, c] = U[i*m+j, c] * V[j2*p+k, c]
                                b[idx] = 1.0 if j == j2 else 0.0
                        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                        W_new[k*n+i, :] = sol
                W = W_new
                # Check exactness with current (float) values
                # Compute residual
                maxres = 0.0
                for i in range(n):
                    for k in range(p):
                        for j in range(m):
                            for j2 in range(m):
                                s = 0.0
                                for c in range(r):
                                    s += U[i*m+j, c]*V[j2*p+k, c]*W[k*n+i, c]
                                e = 1.0 if j==j2 else 0.0
                                r_ = abs(s - e)
                                if r_ > maxres:
                                    maxres = r_
                if maxres < 1e-4:
                    # Check half-integer-ness
                    ok = True
                    for arr in (U, V, W):
                        for row in arr:
                            for val in row:
                                if _to_halfint(val) is None:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    if ok:
                        # Convert to exact rationals (as floats that are exact half-integers)
                        def conv(arr):
                            out = []
                            for row in arr:
                                nr = []
                                for val in row:
                                    num, den = _to_halfint(val)
                                    nr.append(num/den)
                                out.append(nr)
                            return out
                        cand = {"U": conv(U), "V": conv(V), "W": conv(W)}
                        # Exact integer verify
                        if _verify(cand["U"], cand["V"], cand["W"], n, m, p, r):
                            return cand
                    else:
                        # Try rounding to nearest half-integer and verify
                        def round_half(arr):
                            out = []
                            for row in arr:
                                nr = []
                                for val in row:
                                    num, den = _to_halfint(round(val*2)/2.0)
                                    if num is None:
                                        num, den = int(round(val)), 1
                                    nr.append(num/den)
                                out.append(nr)
                            return out
                        cand = {"U": round_half(U), "V": round_half(V), "W": round_half(W)}
                        if _verify(cand["U"], cand["V"], cand["W"], n, m, p, r):
                            return cand
                # If residual is tiny but not exact, continue iterating
                if maxres < 1e-8:
                    # do a few more iterations then check
                    pass
        return None

    # Try ranks from just below schoolbook downward
    for r in range(rank_school - 1, 4, -1):
        if time.time() > DEADLINE:
            break
        budget = min(DEADLINE, time.time() + 30.0)
        cand = try_rank(r, budget)
        if cand is not None:
            best = cand
            # Found a valid lower rank; try to go lower
            # Continue loop to try even lower ranks
            continue
        else:
            # No solution at this rank with our heuristic; keep best (schoolbook)
            # We could stop early but keep trying lower ranks with limited budget
            pass

    return best
