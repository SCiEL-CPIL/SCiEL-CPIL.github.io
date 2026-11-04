import time
import itertools
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

def _build_terms(n, m, p, max_size=2):
    """Structured rank-1 terms: I subset of rows x J subset of cols x K subset of cols2.
    Each term covers monomials (i,j,k) with i in I, j in J, k in K."""
    terms = []
    for size in range(1, max_size + 1):
        for I in itertools.combinations(range(n), size):
            for J in itertools.combinations(range(m), size):
                for K in itertools.combinations(range(p), size):
                    covered = []
                    for i in I:
                        for j in J:
                            for k in K:
                                covered.append((i, j, k))
                    terms.append((frozenset(covered), I, J, K))
    return terms

def _solve_ip(n, m, p, terms, monomials, deadline):
    """Min rank s.t. every monomial covered. Returns list of chosen (I,J,K) or None."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    import numpy as np
    idx = {mono: t for t, mono in enumerate(monomials)}
    A = np.zeros((len(monomials), len(terms)))
    for ti, (cov, I, J, K) in enumerate(terms):
        for mono in cov:
            A[idx[mono], ti] = 1.0
    c = np.ones(len(terms))
    cons = LinearConstraint(A, np.ones(len(monomials)), np.inf)
    bounds = Bounds(0, 1)
    res = milp(c, constraints=cons, integrality=np.ones(len(terms)), bounds=bounds)
    if res.status != 0:
        return None
    chosen = [ti for ti in range(len(terms)) if res.x[ti] > 0.5]
    return [(terms[ti][1], terms[ti][2], terms[ti][3]) for ti in chosen]

def _solve_lp(n, m, p, terms, monomials, deadline):
    """Min rank s.t. every monomial covered (LP relaxation). Returns list of (I,J,K) or None."""
    from scipy.optimize import linprog
    import numpy as np
    idx = {mono: t for t, mono in enumerate(monomials)}
    A = np.zeros((len(monomials), len(terms)))
    for ti, (cov, I, J, K) in enumerate(terms):
        for mono in cov:
            A[idx[mono], ti] = 1.0
    c = np.ones(len(terms))
    res = linprog(c, A_ub=None, b_ub=None, A_eq=A, b_eq=np.ones(len(monomials)),
                  bounds=(0, np.inf), method='highs')
    if res.status != 0:
        return None
    chosen = [ti for ti in range(len(terms)) if res.x[ti] > 1e-6]
    return [(terms[ti][1], terms[ti][2], terms[ti][3]) for ti in chosen]

def _decompose(n, m, p, terms, chosen):
    """Given chosen (I,J,K) terms, solve exact coefficients. Returns (U,V,W) or None."""
    import numpy as np
    idx = {mono: t for t, mono in enumerate([(i, j, k) for i in range(n) for j in range(m) for k in range(p)])}
    N = len(chosen)
    # Variables: U[i,j,c], V[j,k,c], W[i,k,c] for c in chosen
    # Equation for (i,j,k): sum_c U[i,j,c]*V[j,k,c]*W[i,k,c] = 1
    # This is bilinear (nonlinear). Solve via iterative linearization (alternating).
    # Init
    U = {}
    V = {}
    W = {}
    for c in range(N):
        I, J, K = chosen[c]
        for i in I:
            for j in J:
                U[(i, j, c)] = Fraction(1)
        for j in J:
            for k in K:
                V[(j, k, c)] = Fraction(1)
        for i in I:
            for k in K:
                W[(i, k, c)] = Fraction(1)
    # Alternating linear solve
    for _ in range(50):
        # Fix V,W, solve for U
        # For each (i,j,c): U[i,j,c] = (1 - sum_{c' != c} U[i,j,c']*V[j,k,c']*W[i,k,c]) / (V[j,k,c]*W[i,k,c])
        # But k varies. Actually for a given (i,j), the equation over k:
        # For each k: sum_c U[i,j,c]*V[j,k,c]*W[i,k,c] = 1
        # This is a linear system in U[i,j,*] for fixed i,j.
        # Let's build the system for each (i,j).
        for i in range(n):
            for j in range(m):
                # unknowns: U[i,j,c] for c where i in I_c and j in J_c
                active = [c for c in range(N) if i in chosen[c][0] and j in chosen[c][1]]
                if not active:
                    continue
                # For each k where any active term has V[j,k,c] and W[i,k,c]:
                ks = set()
                for c in active:
                    for k in chosen[c][2]:
                        ks.add(k)
                # Build linear system: for each k, sum_{c in active} U[i,j,c]*V[j,k,c]*W[i,k,c] = 1
                # Coeff matrix A[k, c] = V[j,k,c]*W[i,k,c]
                A = np.zeros((len(ks), len(active)))
                b = np.zeros(len(ks))
                klist = list(ks)
                for ki, k in enumerate(klist):
                    b[ki] = 1
                    for ai, c in enumerate(active):
                        if k in chosen[c][2]:
                            A[ki, ai] = float(V[(j, k, c)]) * float(W[(i, k, c)])
                if len(ks) >= len(active):
                    try:
                        sol = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        continue
                    for ai, c in enumerate(active):
                        U[(i, j, c)] = Fraction(sol[ai]).limit_denominator(1000000)
        # Fix U,W, solve for V
        for j in range(m):
            for k in range(p):
                active = [c for c in range(N) if j in chosen[c][1] and k in chosen[c][2]]
                if not active:
                    continue
                is_ = set()
                for c in active:
                    for i in chosen[c][0]:
                        is_.add(i)
                ilist = list(is_)
                A = np.zeros((len(ilist), len(active)))
                b = np.zeros(len(ilist))
                for ii, i in enumerate(ilist):
                    b[ii] = 1
                    for ai, c in enumerate(active):
                        if i in chosen[c][0]:
                            A[ii, ai] = float(U[(i, j, c)]) * float(W[(i, k, c)])
                if len(ilist) >= len(active):
                    try:
                        sol = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        continue
                    for ai, c in enumerate(active):
                        V[(j, k, c)] = Fraction(sol[ai]).limit_denominator(1000000)
        # Fix U,V, solve for W
        for i in range(n):
            for k in range(p):
                active = [c for c in range(N) if i in chosen[c][0] and k in chosen[c][2]]
                if not active:
                    continue
                js = set()
                for c in active:
                    for j in chosen[c][1]:
                        js.add(j)
                jlist = list(js)
                A = np.zeros((len(jlist), len(active)))
                b = np.zeros(len(jlist))
                for jj, j in enumerate(jlist):
                    b[jj] = 1
                    for ai, c in enumerate(active):
                        if j in chosen[c][1]:
                            A[jj, ai] = float(U[(i, j, c)]) * float(V[(j, k, c)])
                if len(jlist) >= len(active):
                    try:
                        sol = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        continue
                    for ai, c in enumerate(active):
                        W[(i, k, c)] = Fraction(sol[ai]).limit_denominator(1000000)
    # Verify exactness
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(N):
                    if i in chosen[c][0] and j in chosen[c][1] and k in chosen[c][2]:
                        total += U[(i, j, c)] * V[(j, k, c)] * W[(i, k, c)]
                if total != 1:
                    return None
    # Build matrices
    U_mat = [[0] * N for _ in range(n * m)]
    V_mat = [[0] * N for _ in range(m * p)]
    W_mat = [[0] * N for _ in range(p * n)]
    for c in range(N):
        I, J, K = chosen[c]
        for i in I:
            for j in J:
                U_mat[i * m + j][c] = U[(i, j, c)]
        for j in J:
            for k in K:
                V_mat[j * p + k][c] = V[(j, k, c)]
        for i in I:
            for k in K:
                W_mat[k * n + i][c] = W[(i, k, c)]
    return (U_mat, V_mat, W_mat)

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

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 120
    monomials = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    # Try size-1 and size-2 terms
    terms = _build_terms(n, m, p, max_size=2)
    best = None
    # Try IP
    if time.time() < deadline - 10:
        chosen = _solve_ip(n, m, p, terms, monomials, deadline)
        if chosen:
            res = _decompose(n, m, p, terms, chosen)
            if res:
                best = {"U": res[0], "V": res[1], "W": res[2]}
    # Try LP relaxation if IP failed or not run
    if best is None and time.time() < deadline - 10:
        chosen = _solve_lp(n, m, p, terms, monomials, deadline)
        if chosen:
            res = _decompose(n, m, p, terms, chosen)
            if res:
                best = {"U": res[0], "V": res[1], "W": res[2]}
    if best is None:
        best = _schoolbook(n, m, p)
    return best
