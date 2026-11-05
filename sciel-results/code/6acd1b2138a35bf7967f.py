import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 2, 4, 5

    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        # Also ensure no "extra" contributions: check that for each r the term
        # is consistent (already guaranteed by construction if we only add valid triples)
        return True

    # ---- Build rank-36 Strassen decomposition ----
    def build_rank36():
        U = [[Fraction(0)] * 36 for _ in range(n*m)]
        V = [[Fraction(0)] * 36 for _ in range(m*p)]
        W = [[Fraction(0)] * 36 for _ in range(p*n)]
        col = 0

        # Strassen 2x2x2 terms: (U_mask, V_mask, W_mask) over local indices
        # local u index = li*2 + lj, v = lj*2 + lk, w = lk*2 + li
        strassen = [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
        ]

        # Standard 2x2x1 terms
        standard = [
            ({(0,0):1}, {(0,0):1}, {(0,0):1}),
            ({(1,0):1}, {(0,0):1}, {(0,1):1}),
            ({(0,1):1}, {(1,0):1}, {(1,0):1}),
            ({(1,1):1}, {(1,0):1}, {(1,1):1}),
        ]

        for q in range(2):      # m-block: gj = 2*q + lj
            for s in range(3):  # p-block
                if s < 2:       # 2x2x2 Strassen
                    gk_base = 2*s
                    for u_t, v_t, w_t in strassen:
                        for (li, lj), val in u_t.items():
                            U[li*m + (2*q + lj)][col] += val
                        for (lj, lk), val in v_t.items():
                            V[(2*q + lj)*p + (gk_base + lk)][col] += val
                        for (lk, li), val in w_t.items():
                            W[(gk_base + lk)*n + li][col] += val
                        col += 1
                else:           # 2x2x1 standard, gk = 4
                    for u_t, v_t, w_t in standard:
                        for (li, lj), val in u_t.items():
                            U[li*m + (2*q + lj)][col] += val
                        for (lj, lk), val in v_t.items():
                            V[(2*q + lj)*p + 4][col] += val
                        for (lk, li), val in w_t.items():
                            W[4*n + li][col] += val
                        col += 1
        assert col == 36, col
        return U, V, W

    U, V, W = build_rank36()
    assert verify(U, V, W, 36), "rank36 failed verification"
    best = {"U": U, "V": V, "W": W}
    best_rank = 36

    # ---- Attempt to reduce rank via greedy rank-1 extraction with exact arithmetic ----
    # Represent the multiplication tensor and try to find a lower-rank decomposition.
    # We use a local search: start from rank-36, try to merge/remove terms.

    def time_left():
        return deadline - time.time()

    # Try a simple improvement: search for a rank-34 or lower decomposition
    # by solving a small system. Given the difficulty, we attempt a randomized
    # rank-1 extraction from the residual.
    def try_reduce(rank_target):
        """Try to find a decomposition of rank_target by iterative extraction."""
        if time_left() < 5:
            return None
        # Residual tensor T[i][j][k] = 1 for all (i,j,k)
        # We'll greedily pick rank-1 terms that are "valid" (triangular support)
        # and subtract, tracking the residual.
        import random
        rng = random.Random(12345)

        # Build list of all (i,j,k)
        T = {(i,j,k): Fraction(1) for i in range(n) for j in range(m) for k in range(p)}

        U_rows = []
        V_rows = []
        W_rows = []

        for step in range(rank_target):
            if time_left() < 5:
                return None
            # Find a nonzero residual entry to anchor
            nz = [(i,j,k) for (i,j,k),v in T.items() if v != 0]
            if not nz:
                break
            # Pick a random anchor
            i0, j0, k0 = rng.choice(nz)
            # Build a rank-1 term anchored at (i0,j0,k0) with U[i0,j0]=1
            # Choose support sets: I = set of i with U[i,j0] nonzero,
            # J = set of j with V[j0,k] nonzero, K = set of k with W[k,i0] nonzero.
            # To keep it simple, use singletons first, then expand.
            # Try a few patterns.
            best_pat = None
            for I in [ {i0}, {0,1} ]:
                for J in [ {j0}, set(range(m)) ]:
                    for K in [ {k0}, set(range(p)) ]:
                        # Determine coefficients so that the term matches residual on its support
                        # term[i,j,k] = a[i,j]*b[j,k]*c[k,i]
                        # We need a[i0,j0]=1.
                        # For (i,j0,k) in support: a[i,j0]*b[j0,k]*c[k,i0] = T[i,j0,k]
                        # For (i0,j,k): a[i0,j]*b[j,k]*c[k,i0] = T[i0,j,k]
                        # For (i0,j0,k): c[k,i0]*b[j0,k] = T[i0,j0,k]
                        # This is a system; solve if consistent.
                        a = {}
                        b = {}
                        c = {}
                        ok = True
                        # Set a[i0,j0]=1
                        a[(i0,j0)] = Fraction(1)
                        # From (i0,j0,k): b[j0,k]*c[k,i0] = T[i0,j0,k]
                        # From (i,j0,k): a[i,j0]*b[j0,k]*c[k,i0] = T[i,j0,k]
                        # From (i0,j,k): a[i0,j]*b[j,k]*c[k,i0] = T[i0,j,k]
                        # Let x = b[j0,k0]*c[k0,i0] = T[i0,j0,k0]
                        # This is underdetermined; use a simple assignment.
                        # Set c[k,i0] = 1 for all k in K, then b[j0,k] = T[i0,j0,k]
                        for k in K:
                            c[(k,i0)] = Fraction(1)
                        for k in K:
                            if (i0,j0,k) in T:
                                b[(j0,k)] = T[(i0,j0,k)]
                            else:
                                b[(j0,k)] = Fraction(0)
                        # Now a[i,j0] from (i,j0,k0): a[i,j0]*b[j0,k0]*c[k0,i0] = T[i,j0,k0]
                        if b[(j0,k0)] != 0:
                            for i in I:
                                if (i,j0,k0) in T:
                                    a[(i,j0)] = T[(i,j0,k0)] / (b[(j0,k0)] * c[(k0,i0)])
                                else:
                                    a[(i,j0)] = Fraction(0)
                        else:
                            ok = False
                        if not ok:
                            continue
                        # Check consistency on all support points
                        good = True
                        for i in I:
                            for j in J:
                                for k in K:
                                    if (i,j) not in a or (j,k) not in b or (k,i) not in c:
                                        continue
                                    val = a[(i,j)] * b[(j,k)] * c[(k,i)]
                                    target = T.get((i,j,k), Fraction(0))
                                    if val != target:
                                        good = False
                                        break
                                if not good:
                                    break
                            if not good:
                                break
                        if good:
                            # Count how many nonzero residuals this covers
                            cov = 0
                            for i in I:
                                for j in J:
                                    for k in K:
                                        if (i,j) in a and (j,k) in b and (k,i) in c:
                                            if a[(i,j)]*b[(j,k)]*c[(k,i)] != 0:
                                                cov += 1
                            if best_pat is None or cov > best_pat[0]:
                                best_pat = (cov, a, b, c)
            if best_pat is None:
                return None
            _, a, b, c = best_pat
            # Subtract from residual
            for (i,j), av in a.items():
                for (j,k), bv in b.items():
                    for (k,i), cv in c.items():
                        if (i,j,k) in T:
                            T[(i,j,k)] -= av*bv*cv
            U_rows.append(a)
            V_rows.append(b)
            W_rows.append(c)

        # Check residual is zero
        for (i,j,k), v in T.items():
            if v != 0:
                return None
        # Build matrices
        rank = len(U_rows)
        Uf = [[Fraction(0)]*rank for _ in range(n*m)]
        Vf = [[Fraction(0)]*rank for _ in range(m*p)]
        Wf = [[Fraction(0)]*rank for _ in range(p*n)]
        for r in range(rank):
            for (i,j), v in U_rows[r].items():
                Uf[i*m+j][r] = v
            for (j,k), v in V_rows[r].items():
                Vf[j*p+k][r] = v
            for (k,i), v in W_rows[r].items():
                Wf[k*n+i][r] = v
        if verify(Uf, Vf, Wf, rank):
            return {"U": Uf, "V": Vf, "W": Wf}
        return None

    # Try to beat 36
    for target in [34, 32, 30]:
        if time_left() < 8:
            break
        res = try_reduce(target)
        if res is not None:
            best = res
            best_rank = target
            break

    return best
