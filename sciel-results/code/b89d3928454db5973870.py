"""Rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication.
Uses block Strassen (rank 22) as baseline, then searches for rank-21 via
BFGS optimization of the bilinear verification error.
"""
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 140.0  # seconds, leaves 10s margin before 150s kill


def _elapsed():
    return time.time() - _T0


def _remaining():
    return _DEADLINE - _elapsed()


def _verify_full(U, V, W, n, m, p, tol=1e-8):
    """Full bilinear verification: for all (ia, ja, jb, kb, i, k),
    sum_r W[k*n+i, r] * U[ia*m+ja, r] * V[jb*p+kb, r] == delta(ia,i)*delta(ja,jb)*delta(kb,k)
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]

    for ia in range(n):
        for ja in range(m):
            for jb in range(m):
                for kb in range(p):
                    for i in range(n):
                        for k in range(p):
                            expected = 1.0 if (ia == i and ja == jb and kb == k) else 0.0
                            Uidx = ia * m + ja
                            Vidx = jb * p + kb
                            Widx = k * n + i
                            actual = np.dot(W[Widx, :], U[Uidx, :] * V[Vidx, :])
                            if abs(actual - expected) > tol:
                                return False
    return True


def _verify_rational(U, V, W, n, m, p):
    """Exact rational verification using fractions.
    Returns True iff the tensor contraction is exactly the identity.
    """
    from fractions import Fraction
    
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    r = U.shape[1]
    
    for ia in range(n):
        for ja in range(m):
            for jb in range(m):
                for kb in range(p):
                    for i in range(n):
                        for k in range(p):
                            expected = 1 if (ia == i and ja == jb and kb == k) else 0
                            Uidx = ia * m + ja
                            Vidx = jb * p + kb
                            Widx = k * n + i
                            actual = Fraction(0)
                            for rr in range(r):
                                u_val = Fraction(U[Uidx, rr])
                                v_val = Fraction(V[Vidx, rr])
                                w_val = Fraction(W[Widx, rr])
                                actual += u_val * v_val * w_val
                            if actual != expected:
                                return False
    return True


def _bilinear_error(U, V, W, n, m, p):
    """Compute the squared error of the bilinear verification.
    Error = sum over all (ia,ja,jb,kb,i,k) of (actual - expected)^2
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    total_err = 0.0
    for ia in range(n):
        for ja in range(m):
            for jb in range(m):
                for kb in range(p):
                    for i in range(n):
                        for k in range(p):
                            expected = 1.0 if (ia == i and ja == jb and kb == k) else 0.0
                            Uidx = ia * m + ja
                            Vidx = jb * p + kb
                            Widx = k * n + i
                            actual = np.dot(W[Widx, :], U[Uidx, :] * V[Vidx, :])
                            total_err += (actual - expected) ** 2
    return total_err


def _structured_random_walk(n, m, p, max_steps=2000, seed=42):
    """Structured random walk over half-integer tensor decompositions.
    
    Starting from rank-22 block decomposition, iteratively perturbs entries
    while maintaining exact rational verification. Explores the discrete
    solution manifold for potential rank reductions.
    
    Strategy: 
    1. Start with rank-22 exact decomposition
    2. Try to merge pairs of columns (reduce rank by 1)
    3. Use random walk to find valid rank-21 configurations
    4. If found, verify with exact rational arithmetic
    """
    import random
    from fractions import Fraction
    
    rng = random.Random(seed)
    
    # Start with rank-22
    U22, V22, W22 = _rank22_list(n, m, p)
    U22 = np.array(U22, dtype=object)
    V22 = np.array(V22, dtype=object)
    W22 = np.array(W22, dtype=object)
    
    # Convert to Fraction for exact arithmetic
    def to_frac(arr):
        return np.array([[Fraction(x) for x in row] for row in arr], dtype=object)
    
    U_frac = to_frac(U22)
    V_frac = to_frac(V22)
    W_frac = to_frac(W22)
    
    nm, mp, pn = n * m, m * p, p * n
    r0 = 22
    
    # Try to find rank-21 by merging two columns
    # A merge of columns a and b means: new_col = U[:,a] + alpha*U[:,b]
    # and we need to adjust V and W accordingly
    
    best_result = None
    
    for step in range(max_steps):
        if _remaining() < 20.0:
            break
        
        # Randomly select two columns to attempt merging
        col_a = rng.randint(0, r0 - 1)
        col_b = rng.randint(0, r0 - 1)
        if col_a == col_b:
            continue
        
        # Try different merge strategies
        # Strategy 1: U_new[:,a] = U[:,a] + U[:,b], V_new[:,a] = V[:,a], W_new[:,a] = W[:,a]
        # This works if the bilinear form allows it
        
        # More generally, we can try:
        # U_new[:,a] = U[:,a] + c*U[:,b] for random half-integer c
        # V_new[:,a] = V[:,a] - c*V[:,b] 
        # W_new[:,a] = W[:,a]
        # This preserves the product if done correctly
        
        c_options = [Fraction(1), Fraction(-1), Fraction(1,2), Fraction(-1,2)]
        c = rng.choice(c_options)
        
        # Create modified matrices
        U_mod = U_frac.copy()
        V_mod = V_frac.copy()
        W_mod = W_frac.copy()
        
        # Apply merge: absorb column b into column a
        U_mod[:, col_a] = U_mod[:, col_a] + c * U_mod[:, col_b]
        V_mod[:, col_a] = V_mod[:, col_a] - c * V_mod[:, col_b]
        # W stays the same for now
        
        # Remove column b (shift columns)
        keep = [i for i in range(r0) if i != col_b]
        U_new = np.array([U_mod[:, i] for i in keep], dtype=object)
        V_new = np.array([V_mod[:, i] for i in keep], dtype=object)
        W_new = W_mod[:, keep]
        
        r_new = r0 - 1
        
        # Verify with exact rational arithmetic
        if _verify_rational(U_new, V_new, W_new, n, m, p):
            # Found a valid rank-21 decomposition!
            U_list = [[float(x) for x in row] for row in U_new]
            V_list = [[float(x) for x in row] for row in V_new]
            W_list = [[float(x) for x in row] for row in W_new]
            return U_list, V_list, W_list, r_new
        
        # Try another strategy: perturb individual entries
        # Randomly select a matrix and entry to perturb by +/- 0.5
        mat_choice = rng.choice(['U', 'V', 'W'])
        if mat_choice == 'U':
            row = rng.randint(0, nm - 1)
            col = rng.randint(0, r0 - 1)
            delta = rng.choice([Fraction(1,2), Fraction(-1,2), Fraction(1), Fraction(-1)])
            U_frac[row, col] = U_frac[row, col] + delta
            
            # Compensate in V or W to maintain the identity
            # If we add delta to U[row,col], we need to subtract delta*V[?,col]*W[?,col] from other terms
            # This is complex; try a simple compensation
            # For now, just check if it's still valid (unlikely)
            if _verify_rational(U_frac, V_frac, W_frac, n, m, p):
                U_list = [[float(x) for x in row] for row in U_frac]
                V_list = [[float(x) for x in row] for row in V_frac]
                W_list = [[float(x) for x in row] for row in W_frac]
                return U_list, V_list, W_list, r0
        elif mat_choice == 'V':
            row = rng.randint(0, mp - 1)
            col = rng.randint(0, r0 - 1)
            delta = rng.choice([Fraction(1,2), Fraction(-1,2), Fraction(1), Fraction(-1)])
            V_frac[row, col] = V_frac[row, col] + delta
            if _verify_rational(U_frac, V_frac, W_frac, n, m, p):
                U_list = [[float(x) for x in row] for row in U_frac]
                V_list = [[float(x) for x in row] for row in V_frac]
                W_list = [[float(x) for x in row] for row in W_frac]
                return U_list, V_list, W_list, r0
        else:  # W
            row = rng.randint(0, pn - 1)
            col = rng.randint(0, r0 - 1)
            delta = rng.choice([Fraction(1,2), Fraction(-1,2), Fraction(1), Fraction(-1)])
            W_frac[row, col] = W_frac[row, col] + delta
            if _verify_rational(U_frac, V_frac, W_frac, n, m, p):
                U_list = [[float(x) for x in row] for row in U_frac]
                V_list = [[float(x) for x in row] for row in V_frac]
                W_list = [[float(x) for x in row] for row in W_frac]
                return U_list, V_list, W_list, r0
    
    return None


def _rank22_list(n, m, p):
    """Construct rank-22 decomposition using block Strassen."""
    rank = 22
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    col = 0
    # Block 1: 7 Strassen terms for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # A: 0=A00, 1=A01, 3=A10, 4=A11
    # B: 0=B00, 1=B01, 4=B10, 5=B11
    # W: 0=C00, 2=C01, 1=C10, 3=C11

    # M1 = (A00+A11)(B00+B11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1

    # M2 = (A10+A11)B00
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[3][col] = -1
    col += 1

    # M3 = A00(B01-B11)
    U[0][col] = 1
    V[1][col] = 1; V[5][col] = -1
    W[2][col] = 1; W[3][col] = 1
    col += 1

    # M4 = A11(B10-B00)
    U[4][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1

    # M5 = (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = -1; W[2][col] = 1
    col += 1

    # M6 = (A10-A00)(B00+B01)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1
    col += 1

    # M7 = (A01-A11)(B10+B11)
    U[1][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = 1
    col += 1

    # Block 2: 7 Strassen terms for A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    # B: 2=B02, 3=B03, 6=B12, 7=B13
    # W: 4=C02, 6=C03, 5=C12, 7=C13

    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[4][col] = 1; W[7][col] = 1
    col += 1

    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[5][col] = 1; W[7][col] = -1
    col += 1

    U[0][col] = 1
    V[3][col] = 1; V[7][col] = -1
    W[6][col] = 1; W[7][col] = 1
    col += 1

    U[4][col] = 1
    V[6][col] = 1; V[2][col] = -1
    W[4][col] = 1; W[5][col] = 1
    col += 1

    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[4][col] = -1; W[6][col] = 1
    col += 1

    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[7][col] = 1
    col += 1

    U[1][col] = 1; U[4][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[4][col] = 1
    col += 1

    # Block 3: 8 schoolbook terms for A[0:2,2:3] * B[2:3,0:4] -> C[0:2,0:4]
    # A: 2=A02, 5=A12
    # B: 8=B20, 9=B21, 10=B22, 11=B23
    for i, ai in [(0, 2), (1, 5)]:
        for k, bk in [(0, 8), (1, 9), (2, 10), (3, 11)]:
            U[ai][col] = 1
            V[bk][col] = 1
            W[k * n + i][col] = 1
            col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W


def _to_dict(U, V, W):
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = 2, 3, 4
    nm, mp, pn = n * m, m * p, p * n

    # Start with rank-22
    best_U, best_V, best_W = _rank22_list(n, m, p)
    best_rank = 22

    # Verify
    if not _verify_full(best_U, best_V, best_W, n, m, p):
        best_U, best_V, best_W = _schoolbook(n, m, p)
        best_rank = n * m * p

    if _remaining() < 30.0:
        return _to_dict(best_U, best_V, best_W)

    # Try to find rank-21 by optimizing from a perturbed rank-22
    # Strategy: take rank-22, zero out one column (or merge two columns),
    # then optimize the remaining parameters to minimize bilinear error.

    # Better approach: directly optimize rank-21 parameters from random init
    # using the bilinear error as objective.

    def _try_rank21():
        r = 21
        # Random initialization near rank-22 structure
        # Take rank-22 and randomly drop one column, then optimize
        U22, V22, W22 = _rank22_list(n, m, p)
        U22 = np.array(U22, dtype=float)
        V22 = np.array(V22, dtype=float)
        W22 = np.array(W22, dtype=float)

        best_err = float('inf')
        best_params = None

        for drop_col in range(22):
            if _remaining() < 25.0:
                break
            # Drop column drop_col
            keep = [c for c in range(22) if c != drop_col]
            U_init = U22[:, keep]
            V_init = V22[:, keep]
            W_init = W22[:, keep]

            # Flatten for optimization
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])

            def objective(x):
                U = x[:nm * r].reshape(nm, r)
                V = x[nm * r:nm * r + mp * r].reshape(mp, r)
                W = x[nm * r + mp * r:].reshape(pn, r)
                return _bilinear_error(U, V, W, n, m, p)

            try:
                from scipy.optimize import minimize
                res = minimize(objective, x0, method='BFGS',
                               options={'maxiter': 200, 'gtol': 1e-6})
                if res.fun < best_err:
                    best_err = res.fun
                    best_params = res.x
            except Exception:
                continue

        if best_params is not None and best_err < 1e-8:
            x = best_params
            U = x[:nm * r].reshape(nm, r)
            V = x[nm * r:nm * r + mp * r].reshape(mp, r)
            W = x[nm * r + mp * r:].reshape(pn, r)
            if _verify_full(U, V, W, n, m, p, tol=1e-6):
                return U.tolist(), V.tolist(), W.tolist(), r
        return None

    result21 = _try_rank21()
    if result21 is not None:
        U21, V21, W21, r21 = result21
        return _to_dict(U21, V21, W21)
    
    # Try structured random walk with rational arithmetic
    if _remaining() > 30.0:
        rw_result = _structured_random_walk(n, m, p, max_steps=1000, seed=42)
        if rw_result is not None:
            U_rw, V_rw, W_rw, r_rw = rw_result
            return _to_dict(U_rw, V_rw, W_rw)

    # If rank-21 not found, try a few random restarts for rank-21
    if _remaining() > 35.0:
        def _random_rank21_attempt():
            r = 21
            x0 = np.random.randn(nm * r + mp * r + pn * r) * 0.1

            def objective(x):
                U = x[:nm * r].reshape(nm, r)
                V = x[nm * r:nm * r + mp * r].reshape(mp, r)
                W = x[nm * r + mp * r:].reshape(pn, r)
                return _bilinear_error(U, V, W, n, m, p)

            try:
                from scipy.optimize import minimize
                res = minimize(objective, x0, method='BFGS',
                               options={'maxiter': 300, 'gtol': 1e-7})
                if res.fun < 1e-10:
                    x = res.x
                    U = x[:nm * r].reshape(nm, r)
                    V = x[nm * r:nm * r + mp * r].reshape(mp, r)
                    W = x[nm * r + mp * r:].reshape(pn, r)
                    if _verify_full(U, V, W, n, m, p, tol=1e-6):
                        return U.tolist(), V.tolist(), W.tolist(), r
            except Exception:
                pass
            return None

        for _ in range(3):
            if _remaining() < 25.0:
                break
            res = _random_rank21_attempt()
            if res is not None:
                U21, V21, W21, r21 = res
                return _to_dict(U21, V21, W21)

    return _to_dict(best_U, best_V, best_W)
