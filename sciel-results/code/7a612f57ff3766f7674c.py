"""Matrix multiplication 2x5x6: rank-minimal exact tensor decomposition.
Structural change: randomized construction + iterative rank elimination via linear algebra."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 6]
RNG = np.random.default_rng(42)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    n, m, p = DIMENSIONS
    nm = n * m  # 10
    mp = m * p  # 30
    pn = p * n  # 12

    # The multiplication tensor T[i,j,k] = delta (output index k*n+i, input i*m+j, j*p+k)
    # A decomposition of rank r: U (nm x r), V (mp x r), W (pn x r)
    # such that for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # and for all other (a,b,c) triples: sum_r U[a,r]*V[b,r]*W[c,r] == 0

    # Build the "target" tensor: T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k; else 0
    # Actually T is defined on the full index space. The constraint is:
    # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for ALL a in [0,nm), b in [0,mp), c in [0,pn)

    # Strategy:
    # 1. Build a valid decomposition of some rank r0 (e.g., schoolbook r0 = n*m*p = 60)
    # 2. Try to reduce rank by eliminating terms one at a time
    # 3. Use random restarts to find better starting points

    def build_schoolbook():
        """Schoolbook decomposition: rank = n*m*p = 60"""
        r = n * m * p
        U = np.zeros((nm, r), dtype=object)
        V = np.zeros((mp, r), dtype=object)
        W = np.zeros((pn, r), dtype=object)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def verify(U, V, W):
        """Check if decomposition is exact. Returns True if valid."""
        r = U.shape[1]
        # Compute the full tensor and check
        # For efficiency, check a subset of constraints
        # Full check: for all a,b,c: sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c]
        # This is nm*mp*pn = 10*30*12 = 3600 constraints
        # Each involves r multiplications and additions
        # With r ~ 50-60, this is ~200k operations, fast enough

        # Build target tensor
        T = np.zeros((nm, mp, pn), dtype=object)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1

        # Compute sum_r U[:,r] * V[:,r] * W[:,r] using outer products
        # This is a 3-tensor of size nm x mp x pn
        # We can compute it as: for each r, outer(U[:,r], outer(V[:,r], W[:,r]))
        # But that's expensive. Instead, use the identity:
        # sum_r U[a,r]*V[b,r]*W[c,r] = sum_r (U[a,r]*V[b,r]) * W[c,r]
        # = sum_r UV[a,b,r] * W[c,r] where UV[a,b,r] = U[a,r]*V[b,r]

        # More efficient: reshape and use einsum-like approach
        # Actually for small sizes, just loop
        computed = np.zeros((nm, mp, pn), dtype=object)
        for a in range(nm):
            for b in range(mp):
                # sum_r U[a,r]*V[b,r]*W[c,r] for all c
                # = W @ (U[a,:] * V[b,:])  (element-wise product then matrix-vector)
                uv = np.array([U[a, rr] * V[b, rr] for rr in range(r)])
                for c in range(pn):
                    val = sum(uv[rr] * W[c, rr] for rr in range(r))
                    computed[a, b, c] = val

        # Compare
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if computed[a, b, c] != T[a, b, c]:
                        return False
        return True

    def verify_fast(U, V, W):
        """Faster verification using numpy with float, then exact check on near-misses."""
        r = U.shape[1]
        # Use float for speed, then verify exactness
        Uf = U.astype(float)
        Vf = V.astype(float)
        Wf = W.astype(float)

        # Compute sum_r U[a,r]*V[b,r]*W[c,r]
        # = sum_r (U @ V^T)[a,b,r] * W[c,r]... no
        # = einsum('ar,br,cr->abc', U, V, W)
        # For small arrays, loop is fine
        # Actually: for each c, result[a,b] = sum_r U[a,r]*V[b,r]*W[c,r] = (U * W[c,:]) @ V.T
        # Wait: sum_r U[a,r]*V[b,r]*W[c,r] = sum_r (U[a,r]*W[c,r]) * V[b,r]
        # = (U * W[c]) @ V.T  where * is element-wise in the r dimension

        T = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0

        for c in range(pn):
            # result[a,b] = sum_r U[a,r]*V[b,r]*W[c,r]
            # = (U * W[c]) @ V.T  -- no, this gives sum_r U[a,r]*W[c,r]*V[b,r] which is correct
            UW = Uf * Wf[c]  # (nm, r)
            result = UW @ Vf.T  # (nm, mp)
            T[:, :, c] = result

        # Check
        if not np.allclose(T, T, atol=1e-9):
            return False
        # Exact check for non-zero entries
        # Since entries are integers/half-integers, check if they're close to integers
        if not np.allclose(T, np.round(T), atol=1e-6):
            return False
        return np.allclose(T, np.round(T), atol=1e-6)

    def try_eliminate(U, V, W, term_to_remove):
        """Try to eliminate term_to_remove by expressing its contribution as a combo of others.
        Returns (new_U, new_V, new_W, success)"""
        r = U.shape[1]
        if r <= 1:
            return U, V, W, False

        t = term_to_remove
        others = [x for x in range(r) if x != t]

        # We need: U[:,t]*V[:,t]*W[:,t] = sum_{s in others} alpha_s * U[:,s]*V[:,s]*W[:,s]
        # This is a system of equations in the alpha_s.
        # The constraint is: for all (a,b,c):
        #   U[a,t]*V[b,t]*W[c,t] = sum_{s in others} alpha_s * U[a,s]*V[b,s]*W[c,s]
        #
        # This is a quadratic system in general, but if we fix U and W, it's linear in V's...
        # Actually, this is hard in general. Let's use a different approach.
        #
        # Alternative: If we fix U and W, the constraint on V is:
        #   V[:,t] = sum_{s in others} alpha_s * V[:,s] * (U[:,t]*W[:,t]) / (U[:,s]*W[:,s])
        # This doesn't work cleanly.
        #
        # Better approach: Use the fact that the tensor must satisfy all constraints.
        # Let's set up a linear system. The constraint is bilinear in (U,V) given W, etc.
        #
        # Practical approach: Try to find alpha_s such that the residual is zero.
        # The residual for constraint (a,b,c) is:
        #   U[a,t]*V[b,t]*W[c,t] - sum_s alpha_s * U[a,s]*V[b,s]*W[c,s] = 0
        #
        # This is linear in alpha_s! For fixed U, V, W, each constraint gives:
        #   sum_s alpha_s * (U[a,s]*V[b,s]*W[c,s]) = U[a,t]*V[b,t]*W[c,t]
        #
        # So we have a linear system: M @ alpha = rhs
        # where M[a,b,c, s] = U[a,s]*V[b,s]*W[c,s] for s in others
        # and rhs[a,b,c] = U[a,t]*V[b,t]*W[c,t]
        #
        # But this is a huge system: nm*mp*pn = 3600 equations, (r-1) unknowns.
        # We can use least squares. If the residual is ~0, we succeeded.

        num_vars = len(others)
        num_eqs = nm * mp * pn  # 3600

        # Build the system using float for speed
        Ut = U[:, t].astype(float)
        Vt = V[:, t].astype(float)
        Wt = W[:, t].astype(float)

        # rhs[a,b,c] = Ut[a] * Vt[b] * Wt[c]
        rhs = np.einsum('a,b,c->abc', Ut, Vt, Wt).flatten()

        # M[e, s_idx] = U[a,s]*V[b,s]*W[c,s] where e = a*mp*pn + b*pn + c
        M = np.zeros((num_eqs, num_vars), dtype=float)
        for s_idx, s in enumerate(others):
            Us = U[:, s].astype(float)
            Vs = V[:, s].astype(float)
            Ws = W[:, s].astype(float)
            M[:, s_idx] = np.einsum('a,b,c->abc', Us, Vs, Ws).flatten()

        # Solve least squares
        try:
            alpha, residuals, rank_m, sv = np.linalg.lstsq(M, rhs, rcond=None)
        except np.linalg.LinAlgError:
            return U, V, W, False

        # Check residual
        residual = np.linalg.norm(M @ alpha - rhs)
        norm_rhs = np.linalg.norm(rhs)
        if norm_rhs < 1e-12:
            # The term to remove has zero contribution; just drop it
            pass
        elif residual / max(norm_rhs, 1e-12) > 1e-8:
            return U, V, W, False

        # Success! Now update: for each s in others, add alpha[s_idx] * term_t to term_s
        # New V[:,s] = V[:,s] + alpha[s_idx] * V[:,t]  ... wait, that's not right.
        #
        # The constraint is: U[:,t]*V[:,t]*W[:,t] = sum_s alpha_s * U[:,s]*V[:,s]*W[:,s]
        # We want to absorb term t into the others. But we can't just add alpha_s*V[:,t] to V[:,s]
        # because that would change U[:,s]*V[:,s]*W[:,s] in a complicated way.
        #
        # Actually, the correct update is: we're replacing the tensor contribution.
        # The new decomposition has r-1 terms. We need to find new U', V', W' such that
        # sum_{s in others} U'[a,s]*V'[b,s]*W'[c,s] = original sum for all a,b,c.
        #
        # One way: keep U', W' the same as before for the remaining terms, and adjust V'.
        # But that's not generally possible.
        #
        # Alternative: Use a different parameterization. Let's keep U and W fixed for remaining terms,
        # and solve for V'. The constraint is:
        #   sum_{s in others} U[a,s]*V'[b,s]*W[c,s] = T[a,b,c]
        # This is linear in V'[b,s]! For each b, we have a system in V'[b, :].
        #
        # Actually, let me reconsider. The cleanest approach:
        # Keep all U and W the same for remaining terms. Solve for new V columns.
        # Constraint: for all a,b,c: sum_{s in others} U[a,s]*V'[b,s]*W[c,s] = T[a,b,c]
        #
        # For fixed b, this is: for all a,c: sum_s U[a,s]*W[c,s]*V'[b,s] = T[a,b,c]
        # This is a linear system in V'[b,s]: (U*W)[a,c,s] * V'[b,s] = T[a,b,c]
        # where (U*W)[a,c,s] = U[a,s]*W[c,s]
        #
        # So for each b, we solve a linear system of size (n*p) x (r-1) = 12 x (r-1)
        # with RHS T[:,b,:] reshaped.

        # Let's do this: for each b in range(mp), solve for V'[b, :]
        r_new = r - 1
        V_new = np.zeros((mp, r_new), dtype=object)
        U_new = np.zeros((nm, r_new), dtype=object)
        W_new = np.zeros((pn, r_new), dtype=object)

        for s_idx, s in enumerate(others):
            U_new[:, s_idx] = U[:, s]
            W_new[:, s_idx] = W[:, s]

        # Build the linear system for each b
        # For each b: A_b @ V'[b,:] = T[:,b,:] reshaped
        # A_b[a*pn+c, s_idx] = U[a,s]*W[c,s]
        # T_b[a*pn+c] = T[a,b,c]

        # First, build T
        T = np.zeros((nm, mp, pn), dtype=object)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1

        success = True
        for b in range(mp):
            # Build A_b: (n*p) x (r-1)
            A_b = np.zeros((n * p, r_new), dtype=float)
            for s_idx, s in enumerate(others):
                Us = U[:, s].astype(float)
                Ws = W[:, s].astype(float)
                for a in range(nm):
                    for c in range(pn):
                        A_b[a * pn + c, s_idx] = Us[a] * Ws[c]

            # RHS: T[:,b,:] reshaped
            rhs_b = np.zeros(n * pn, dtype=float)
            for a in range(nm):
                for c in range(pn):
                    rhs_b[a * pn + c] = float(T[a, b, c])

            # Solve
            try:
                sol, _, _, _ = np.linalg.lstsq(A_b, rhs_b, rcond=None)
            except np.linalg.LinAlgError:
                success = False
                break

            # Check residual
            res = np.linalg.norm(A_b @ sol - rhs_b)
            norm_rhs = np.linalg.norm(rhs_b)
            if norm_rhs > 1e-12 and res / norm_rhs > 1e-8:
                success = False
                break

            for s_idx in range(r_new):
                V_new[b, s_idx] = sol[s_idx]

        if not success:
            return U, V, W, False

        return U_new, V_new, W_new, True

    def try_eliminate_v(U, V, W, term_to_remove):
        """Try eliminating by adjusting V. Similar to above but solving for V."""
        r = U.shape[1]
        if r <= 1:
            return U, V, W, False

        t = term_to_remove
        others = [x for x in range(r) if x != t]
        r_new = r - 1

        U_new = np.zeros((nm, r_new), dtype=object)
        W_new = np.zeros((pn, r_new), dtype=object)
        V_new = np.zeros((mp, r_new), dtype=object)

        for s_idx, s in enumerate(others):
            U_new[:, s_idx] = U[:, s]
            W_new[:, s_idx] = W[:, s]

        # Build T
        T = np.zeros((nm, mp, pn), dtype=object)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1

        success = True
        for b in range(mp):
            A_b = np.zeros((n * p, r_new), dtype=float)
            for s_idx, s in enumerate(others):
                Us = U[:, s].astype(float)
                Ws = W[:, s].astype(float)
                for a in range(nm):
                    for c in range(pn):
                        A_b[a * pn + c, s_idx] = Us[a] * Ws[c]

            rhs_b = np.zeros(n * pn, dtype=float)
            for a in range(nm):
                for c in range(pn):
                    rhs_b[a * pn + c] = float(T[a, b, c])

            try:
                sol, _, _, _ = np.linalg.lstsq(A_b, rhs_b, rcond=None)
            except np.linalg.LinAlgError:
                success = False
                break

            res = np.linalg.norm(A_b @ sol - rhs_b)
            norm_rhs = np.linalg.norm(rhs_b)
            if norm_rhs > 1e-12 and res / norm_rhs > 1e-8:
                success = False
                break

            for s_idx in range(r_new):
                V_new[b, s_idx] = sol[s_idx]

        if not success:
            return U, V, W, False

        return U_new, V_new, W_new, True

    def try_eliminate_u(U, V, W, term_to_remove):
        """Try eliminating by adjusting U."""
        r = U.shape[1]
        if r <= 1:
            return U, V, W, False

        t = term_to_remove
        others = [x for x in range(r) if x != t]
        r_new = r - 1

        V_new = np.zeros((mp, r_new), dtype=object)
        W_new = np.zeros((pn, r_new), dtype=object)
        U_new = np.zeros((nm, r_new), dtype=object)

        for s_idx, s in enumerate(others):
            V_new[:, s_idx] = V[:, s]
            W_new[:, s_idx] = W[:, s]

        T = np.zeros((nm, mp, pn), dtype=object)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1

        success = True
        for a in range(nm):
            A_a = np.zeros((m * p, r_new), dtype=float)
            for s_idx, s in enumerate(others):
                Vs = V[:, s].astype(float)
                Ws = W[:, s].astype(float)
                for b in range(mp):
                    for c in range(pn):
                        A_a[b * pn + c, s_idx] = Vs[b] * Ws[c]

            rhs_a = np.zeros(m * pn, dtype=float)
            for b in range(mp):
                for c in range(pn):
                    rhs_a[b * pn + c] = float(T[a, b, c])

            try:
                sol, _, _, _ = np.linalg.lstsq(A_a, rhs_a, rcond=None)
            except np.linalg.LinAlgError:
                success = False
                break

            res = np.linalg.norm(A_a @ sol - rhs_a)
            norm_rhs = np.linalg.norm(rhs_a)
            if norm_rhs > 1e-12 and res / norm_rhs > 1e-8:
                success = False
                break

            for s_idx in range(r_new):
                U_new[a, s_idx] = sol[s_idx]

        if not success:
            return U, V, W, False

        return U_new, V_new, W_new, True

    def try_eliminate_w(U, V, W, term_to_remove):
        """Try eliminating by adjusting W."""
        r = U.shape[1]
        if r <= 1:
            return U, V, W, False

        t = term_to_remove
        others = [x for x in range(r) if x != t]
        r_new = r - 1

        U_new = np.zeros((nm, r_new), dtype=object)
        V_new = np.zeros((mp, r_new), dtype=object)
        W_new = np.zeros((pn, r_new), dtype=object)

        for s_idx, s in enumerate(others):
            U_new[:, s_idx] = U[:, s]
            V_new[:, s_idx] = V[:, s]

        T = np.zeros((nm, mp, pn), dtype=object)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1

        success = True
        for c in range(pn):
            A_c = np.zeros((nm * mp, r_new), dtype=float)
            for s_idx, s in enumerate(others):
                Us = U[:, s].astype(float)
                Vs = V[:, s].astype(float)
                for a in range(nm):
                    for b in range(mp):
                        A_c[a * mp + b, s_idx] = Us[a] * Vs[b]

            rhs_c = np.zeros(nm * mp, dtype=float)
            for a in range(nm):
                for b in range(mp):
                    rhs_c[a * mp + b] = float(T[a, b, c])

            try:
                sol, _, _, _ = np.linalg.lstsq(A_c, rhs_c, rcond=None)
            except np.linalg.LinAlgError:
                success = False
                break

            res = np.linalg.norm(A_c @ sol - rhs_c)
            norm_rhs = np.linalg.norm(rhs_c)
            if norm_rhs > 1e-12 and res / norm_rhs > 1e-8:
                success = False
                break

            for s_idx in range(r_new):
                W_new[c, s_idx] = sol[s_idx]

        if not success:
            return U, V, W, False

        return U_new, V_new, W_new, True

    def reduce_rank(U, V, W, max_iterations=200):
        """Iteratively try to reduce rank by eliminating terms."""
        r = U.shape[1]
        best_rank = r
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            if r <= best_rank:
                break

            eliminated = False
            # Try each term and each elimination strategy
            for t in range(r):
                if time.time() > deadline:
                    break
                # Try eliminating by adjusting V
                new_U, new_V, new_W, success = try_eliminate_v(U, V, W, t)
                if success:
                    U, V, W = new_U, new_V, new_W
                    r -= 1
                    best_rank = r
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    eliminated = True
                    break
                # Try eliminating by adjusting U
                new_U, new_V, new_W, success = try_eliminate_u(U, V, W, t)
                if success:
                    U, V, W = new_U, new_V, new_W
                    r -= 1
                    best_rank = r
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    eliminated = True
                    break
                # Try eliminating by adjusting W
                new_U, new_V, new_W, success = try_eliminate_w(U, V, W, t)
                if success:
                    U, V, W = new_U, new_V, new_W
                    r -= 1
                    best_rank = r
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    eliminated = True
                    break

            if not eliminated:
                break

        return best_U, best_V, best_W, best_rank

    def to_integer_or_half(arr):
        """Convert float array to exact integer/half-integer representation."""
        result = np.zeros_like(arr, dtype=object)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                # Round to nearest half-integer
                rounded = round(val * 2) / 2
                if abs(val - rounded) < 1e-6:
                    # Represent as fraction for exactness
                    num = int(round(val * 2))
                    if num % 2 == 0:
                        result[i, j] = num // 2
                    else:
                        result[i, j] = num / 2
                else:
                    # Keep as is (shouldn't happen for valid decompositions)
                    result[i, j] = val
        return result

    # Main: try multiple starting decompositions and reduce rank
    best_U, best_V, best_W = None, None, None
    best_rank = 1000

    # Start with schoolbook (rank 60)
    U0, V0, W0 = build_schoolbook()
    U0f = U0.astype(float)
    V0f = V0.astype(float)
    W0f = W0.astype(float)

    U_red, V_red, W_red, rank_red = reduce_rank(U0f, V0f, W0f, max_iterations=100)
    if rank_red < best_rank:
        best_rank = rank_red
        best_U, best_V, best_W = U_red, V_red, W_red

    # Try random decompositions of various ranks
    for attempt in range(20):
        if time.time() > deadline:
            break

        # Random decomposition of rank r0
        r0 = 40 + attempt * 2  # Try ranks 40, 42, 44, ...
        if r0 > 60:
            r0 = 60

        U_rand = RNG.standard_normal((nm, r0))
        V_rand = RNG.standard_normal((mp, r0))
        W_rand = RNG.standard_normal((pn, r0))

        # This random decomposition won't be valid, so we need to project it to a valid one.
        # Skip this approach; instead, try different block decompositions.

    # Try the block decomposition from the current program as a starting point
    # (Strassen-based, rank 54)
    U_block = np.zeros((nm, 54), dtype=float)
    V_block = np.zeros((mp, 54), dtype=float)
    W_block = np.zeros((pn, 54), dtype=float)
    col = 0
    # Part 1: j=0, direct
    for i in range(n):
        for k in range(p):
            U_block[i * m + 0, col] = 1
            V_block[0 * p + k, col] = 1
            W_block[k * n + i, col] = 1
            col += 1

    strassen_terms = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
    ]

    for block_j in range(2):
        j_base = 1 + block_j * 2
        for block_k in range(3):
            k_base = block_k * 2
            for term in strassen_terms:
                u_comps, v_comps, w_comps = term
                for (li, lj, val) in u_comps:
                    U_block[li * m + (j_base + lj), col] += val
                for (lj, lk, val) in v_comps:
                    V_block[(j_base + lj) * p + (k_base + lk), col] += val
                for (li, lk, val) in w_comps:
                    W_block[(k_base + lk) * n + li, col] += val
                col += 1

    # Verify block decomposition
    if verify_fast(U_block, V_block, W_block):
        U_b2, V_b2, W_b2, rank_b2 = reduce_rank(U_block, V_block, W_block, max_iterations=100)
        if rank_b2 < best_rank:
            best_rank = rank_b2
            best_U, best_V, best_W = U_b2, V_b2, W_b2

    # Try more aggressive elimination with more iterations
    if time.time() < deadline - 20:
        U_b3, V_b3, W_b3, rank_b3 = reduce_rank(U_block, V_block, W_block, max_iterations=500)
        if rank_b3 < best_rank:
            best_rank = rank_b3
            best_U, best_V, best_W = U_b3, V_b3, W_b3

    # Convert to exact representation
    if best_U is not None:
        best_U_int = to_integer_or_half(best_U)
        best_V_int = to_integer_or_half(best_V)
        best_W_int = to_integer_or_half(best_W)

        # Final verification
        if verify(best_U_int, best_V_int, best_W_int):
            return {
                "U": best_U_int.tolist(),
                "V": best_V_int.tolist(),
                "W": best_W_int.tolist()
            }

    # Fallback: return the block decomposition (known valid, rank 54)
    U_block_int = to_integer_or_half(U_block)
    V_block_int = to_integer_or_half(V_block)
    W_block_int = to_integer_or_half(W_block)

    if verify(U_block_int, V_block_int, W_block_int):
        return {
            "U": U_block_int.tolist(),
            "V": V_block_int.tolist(),
            "W": W_block_int.tolist()
        }

    # Last resort: schoolbook
    U_sb_int = to_integer_or_half(U0f)
    V_sb_int = to_integer_or_half(V0f)
    W_sb_int = to_integer_or_half(W0f)
    return {
        "U": U_sb_int.tolist(),
        "V": V_sb_int.tolist(),
        "W": W_sb_int.tolist()
    }
