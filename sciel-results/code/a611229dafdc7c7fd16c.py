import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5

    rank = 7 * p + 2 * p  # 35? no: 7*5 + 2*5 = 45 -> that's wrong
    # Correct: Strassen gives 7 products for the 2x2 block (applied to all 5 cols)
    # plus 2 products for the single j=2 row (applied to all 5 cols)
    rank = 7 * p + 2 * p  # = 35
    # Wait: that double counts. Re-derive:
    #   7 Strassen factors * p columns = 35? No, Strassen already handles all 5 columns.
    #   Actually each Strassen product M_r = (A combination)*(B combination) is a
    #   1x5 x 5x1 -> scalar? No, it's a full vector of 5 outputs.
    #   So 7 Strassen terms give 7*5 = 35? That is the schoolbook count for 2x2x5.
    #   That's not right either. Let me just build it directly below and count.

    # Build directly: index each factor by (strassen r in 0..6, k in 0..4) and (direct i in 0..1, k in 0..4)
    nU = n * m  # 6
    nV = m * p  # 15
    nW = p * n  # 10

    U = np.zeros((nU, 0))
    V = np.zeros((nV, 0))
    W = np.zeros((nW, 0))

    def add_factor(u, v, w):
        nonlocal U, V, W
        U = np.column_stack([U, u]) if U.shape[1] else u.reshape(-1, 1)
        V = np.column_stack([V, v]) if V.shape[1] else v.reshape(-1, 1)
        W = np.column_stack([W, w]) if W.shape[1] else w.reshape(-1, 1)

    # Strassen 7 combinations (i in 0..1, j in 0..1, k in 0..4)
    # A[i, j], B[j, k]
    strassen = [
        # (Ai_plus_Aj, Bj_plus_Bk, (i,j) coeff for A, (j,k) coeff for B)
        (0, 0, 1, 0), (1, 1, 1, 1), (0, 1, 1, 0), (1, 0, 0, 1),
        (0, 0, 1, 1), (1, 1, 1, 1), (0, 1, 1, 1),
    ]
    # Proper Strassen M values:
    # M1=(A00+A11)(B00+B11)
    # M2=(A10+A11)B00
    # M3=A00(B10-B11)
    # M4=A11(B01-B00)
    # M5=(A00+A01)B11
    # M6=(A10-A00)(B00+B10)
    # M7=(A01-A11)(B01+B11)
    # contribution to C:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6
    # We'll expand each M into (A-rows, B-cols) contributions.

    # Represent each M as a list of (i,j) A terms and (j,k) B terms.
    # M1 = (A00+A11)(B00+B11): A terms [(0,0),(1,1)], B terms [(0,0),(1,1)]
    #   => 4 products
    # M2 = (A10+A11)B00: A [(1,0),(1,1)], B [(0,0)] => 2
    # M3 = A00(B10-B11): A [(0,0)], B [(1,0),(1,1)] => 2
    # M4 = A11(B01-B00): A [(1,1)], B [(0,1),(0,0)] => 2
    # M5 = (A00+A01)B11: A [(0,0),(0,1)], B [(1,1)] => 2
    # M6 = (A10-A00)(B00+B10): A [(1,0),(0,0)], B [(0,0),(1,0)] => 4
    # M7 = (A01-A11)(B01+B11): A [(0,1),(1,1)], B [(0,1),(1,1)] => 4
    # total = 4+2+2+2+2+4+4 = 20 = 2*2*5 schoolbook. That's not Strassen-efficient!

    # The above is the expanded schoolbook. True Strassen needs 7 products where
    # each product is a (2x2 A block) x (2x2 B block) -> but here A,B are 2x2 (j=0,1)
    # and the k dimension is separate. So Strassen applies to the 2x2x2 core, and we
    # replicate over k. For each k, we get 7 products -> 7*5 = 35? No:
    # For each fixed k, A is 2x2 (rows i, cols j=0,1), B is 2x2 (rows j=0,1, cols k).
    # Strassen computes the 2x2 result C[:,k] using 7 scalar products.
    # So total = 7 (per k) * 5 (k values) = 35? That equals schoolbook 2*2*5.
    # Hmm, Strassen reduces 2x2x2 from 8 to 7. So for 2x2x5: 7*5=35? No:
    # 2x2x2 schoolbook = 8, Strassen = 7. For 2x2x5, treat as 5 independent 2x2x2:
    # schoolbook = 8*5=40? No, 2*2*5=20. Strassen = 7*5=35? That's worse!
    # I'm confusing myself. Let me reconsider.

    # For 2x2x2 (i,j,k all in {0,1}): schoolbook 8, Strassen 7.
    # For 2x2x5 (i in {0,1}, j in {0,1}, k in {0..4}): this is NOT a tensor product
    # of a 2x2x2. The k dimension is 5, not 2. So we can't directly apply Strassen.
    # The correct approach: for each k, we have a 2x2x2 problem? No, for each k,
    # A is 2x2 (i x j) and B is 2x1 (j x {k}), giving C 2x1 (i x {k}).
    # That's 2x2x1 per k, schoolbook = 4 per k, total 4*5=20. No Strassen benefit.

    # So the rank-28 must come from a different decomposition. Let me use the
    # known explicit rank-28 construction for 2x3x5.
    # 2x3x5: split j into {0,1} and {2}.
    #   Part 1: A[:,{0,1}] (2x2) * B[{0,1},:] (2x5) -> 2x5 result
    #   Part 2: A[:,{2}] (2x1) * B[{2},:] (1x5) -> 2x5 result
    # Part 1 is 2x2x5, Part 2 is 2x1x5.
    # 2x1x5 schoolbook = 2*1*5 = 10.
    # 2x2x5: we need rank < 20. Using a 2x2x2 Strassen on a subset...
    # Actually the known result: rank(2x3x5) = 28.
    # Let me just use the numerical search to find it, with a good init.

    # Fall back to a robust numerical search for rank 28.
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    import time
    start = time.time()
    deadline = start + 140

    def try_rank(target_rank):
        import scipy.optimize as opt
        nU = n * m * target_rank
        nV = m * p * target_rank
        nW = p * n * target_rank
        total_vars = nU + nV + nW

        def objective_and_gradient(x):
            U = x[:nU].reshape(n*m, target_rank)
            V = x[nU:nU+nV].reshape(m*p, target_rank)
            W = x[nU+nV:].reshape(p*n, target_rank)
            S = np.einsum('ar,br,cr->abc', U, V, W)
            error = S - T
            val = np.sum(error**2)
            gU = 2.0 * np.einsum('abc,br,cr->ar', error, V, W)
            gV = 2.0 * np.einsum('abc,ar,cr->br', error, U, W)
            gW = 2.0 * np.einsum('abc,ar,br->cr', error, U, V)
            grad = np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
            return val, grad

        # Initialize using schoolbook structure truncated + perturbation
        for r in range(40):
            if time.time() > deadline:
                return None
            np.random.seed(7 + r)
            # Start near schoolbook: pick first target_rank of the 30 schoolbook terms
            U0 = np.zeros((n*m, target_rank))
            V0 = np.zeros((m*p, target_rank))
            W0 = np.zeros((p*n, target_rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col >= target_rank:
                            break
                        U0[i*m+j, col] = 1
                        V0[j*p+k, col] = 1
                        W0[k*n+i, col] = 1
                        col += 1
            # Perturb to escape the schoolbook manifold
            U0 += np.random.randn(n*m, target_rank) * 0.3
            V0 += np.random.randn(m*p, target_rank) * 0.3
            W0 += np.random.randn(p*n, target_rank) * 0.3
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            try:
                res = opt.minimize(
                    objective_and_gradient, x0, jac=True, method='L-BFGS-B',
                    options={'maxiter': 2000, 'maxfun': 20000, 'ftol': 1e-16, 'gtol': 1e-14}
                )
            except Exception:
                continue
            if res.fun < 1e-12:
                U = res.x[:nU].reshape(n*m, target_rank)
                V = res.x[nU:nU+nV].reshape(m*p, target_rank)
                W = res.x[nU+nV:].reshape(p*n, target_rank)
                S = np.einsum('ar,br,cr->abc', U, V, W)
                if np.allclose(S, T, atol=1e-9):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return None

    result = try_rank(28)
    if result is not None:
        return result
    result = try_rank(29)
    if result is not None:
        return result
    return schoolbook()
