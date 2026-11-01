import numpy as np

def construct():
    n, m, p = 2, 3, 4
    # Index helpers
    # A indexed by (i, j): i in [0,n), j in [0,m)  -> flat a = i*m + j
    # B indexed by (j, k): j in [0,m), k in [0,p)  -> flat b = j*p + k
    # C (output) indexed by (i, k): i in [0,n), k in [0,p)
    # W indexed by (k, n + i) per convention W[k*n + i, s]
    # U flat index: i*m + j, size n*m
    # V flat index: j*p + k, size m*p
    # W flat index: k*n + i, size p*n

    nU = n * m  # 6
    nV = m * p  # 12
    nW = p * n  # 8

    # Build the tensor constraint matrix.
    # For each (i,k) and each (i',j',j'',k''), the equation:
    #   sum_s W[k*n+i, s] * U[i'*m+j', s] * V[j''*p+k'', s] = delta
    # where delta = 1 if i'==i and k''==k and j'==j'' else 0.
    # This is bilinear. We'll solve via ALS.

    # Precompute: for each s, the contribution. We'll set up ALS.

    def residual(U, V, W):
        # U: (nU, r), V: (nV, r), W: (nW, r)
        r = U.shape[1]
        # C[i,k] = sum_s W[k*n+i, s] * (sum_{i',j'} U[i'*m+j',s] A[i',j']) * (sum_{j'',k''} V[j''*p+k'',s] B[j'',k''])
        # Coefficient of A[i',j'] B[j'',k''] in C[i,k]:
        #   sum_s W[k*n+i,s] U[i'*m+j',s] V[j''*p+k'',s]
        # Must equal delta[i',i]*delta[k'',k]*delta[j',j'']
        err = 0.0
        for i in range(n):
            for k in range(p):
                for ip in range(n):
                    for jp in range(m):
                        for jp2 in range(m):
                            for kp in range(p):
                                # target
                                tgt = 1.0 if (ip == i and kp == k and jp == jp2) else 0.0
                                # computed coefficient
                                val = 0.0
                                for s in range(r):
                                    val += W[k*n + i, s] * U[ip*m + jp, s] * V[jp2*p + kp, s]
                                err += (val - tgt) ** 2
        return err

    # ALS: fix two, solve for third via least squares.
    # Equation for W[k*n+i, s] given U,V: for all (ip,jp,jp2,kp):
    #   sum_s W[k*n+i,s] U[ip*m+jp,s] V[jp2*p+kp,s] = tgt(i,k,ip,jp,jp2,kp)
    # This is linear in W. Build a linear system.

    # We'll vectorize: for a fixed (i,k), the unknowns are W[k*n+i, s] for s=0..r-1.
    # Each equation (ip,jp,jp2,kp) gives: sum_s W[k*n+i,s] * (U[ip*m+jp,s] * V[jp2*p+kp,s]) = tgt
    # So A_mat[e, s] = U[ip*m+jp,s] * V[jp2*p+kp,s], rhs[e] = tgt.
    # Solve A_mat @ W_col = rhs.

    def solve_W(U, V, r):
        W = np.zeros((nW, r))
        for i in range(n):
            for k in range(p):
                # collect equations
                A_mat = np.zeros((n*m*m*p, r))
                rhs = np.zeros(n*m*m*p)
                e = 0
                for ip in range(n):
                    for jp in range(m):
                        for jp2 in range(m):
                            for kp in range(p):
                                for s in range(r):
                                    A_mat[e, s] = U[ip*m + jp, s] * V[jp2*p + kp, s]
                                tgt = 1.0 if (ip == i and kp == k and jp == jp2) else 0.0
                                rhs[e] = tgt
                                e += 1
                # Solve least squares
                sol, *_ = np.linalg.lstsq(A_mat, rhs, rcond=None)
                W[k*n + i, :] = sol
        return W

    def solve_U(V, W, r):
        U = np.zeros((nU, r))
        for ip in range(n):
            for jp in range(m):
                # Unknowns: U[ip*m+jp, s] for s=0..r-1
                # Equation (i,k,jp2,kp): sum_s U[ip*m+jp,s] * V[jp2*p+kp,s] * W[k*n+i,s] = tgt
                # tgt = 1 if ip==i and kp==k and jp==jp2 else 0
                A_mat = np.zeros((n*p*m*p, r))
                rhs = np.zeros(n*p*m*p)
                e = 0
                for i in range(n):
                    for k in range(p):
                        for jp2 in range(m):
                            for kp in range(p):
                                for s in range(r):
                                    A_mat[e, s] = V[jp2*p + kp, s] * W[k*n + i, s]
                                tgt = 1.0 if (ip == i and kp == k and jp == jp2) else 0.0
                                rhs[e] = tgt
                                e += 1
                sol, *_ = np.linalg.lstsq(A_mat, rhs, rcond=None)
                U[ip*m + jp, :] = sol
        return U

    def solve_V(U, W, r):
        V = np.zeros((nV, r))
        for jp2 in range(m):
            for kp in range(p):
                A_mat = np.zeros((n*p*n*m, r))
                rhs = np.zeros(n*p*n*m)
                e = 0
                for i in range(n):
                    for k in range(p):
                        for ip in range(n):
                            for jp in range(m):
                                for s in range(r):
                                    A_mat[e, s] = U[ip*m + jp, s] * W[k*n + i, s]
                                tgt = 1.0 if (ip == i and kp == k and jp == jp2) else 0.0
                                rhs[e] = tgt
                                e += 1
                sol, *_ = np.linalg.lstsq(A_mat, rhs, rcond=None)
                V[jp2*p + kp, :] = sol
        return V

    def to_half(x):
        # Round to nearest half-integer
        return np.round(2.0 * x) / 2.0

    def is_exact(U, V, W, tol=1e-9):
        return residual(U, V, W) < tol

    # Try ranks from a lower bound up to 24.
    # Lower bound: at least max(n*m, m*p, p*n)? No. Use a simple bound: >= n*p? 
    # We'll just try from 16 to 24.
    rng = np.random.default_rng(0)
    for r in range(16, 25):
        found = False
        for trial in range(200):
            U = rng.standard_normal((nU, r))
            V = rng.standard_normal((nV, r))
            W = np.zeros((nW, r))
            # ALS iterations
            for it in range(60):
                W = solve_W(U, V, r)
                U = solve_U(V, W, r)
                V = solve_V(U, W, r)
            # Round to half-integers
            Ur = to_half(U)
            Vr = to_half(V)
            Wr = to_half(W)
            if is_exact(Ur, Vr, Wr, tol=1e-6):
                # Verify with exact rational check
                if residual(Ur, Vr, Wr) < 1e-9:
                    found = True
                    break
            # Also try unrounded
            if is_exact(U, V, W, tol=1e-9):
                # Convert to half-integers if possible
                Ur2 = to_half(U)
                Vr2 = to_half(V)
                Wr2 = to_half(W)
                if residual(Ur2, Vr2, Wr2) < 1e-9:
                    U, V, W = Ur2, Vr2, Wr2
                found = True
                break
        if found:
            # Convert to lists of Python numbers (int or half-int)
            def to_list(M):
                out = []
                for row in M:
                    rrow = []
                    for v in row:
                        h = round(v * 2)
                        # represent as int if whole, else float (x.5)
                        if h % 2 == 0:
                            rrow.append(int(h // 2))
                        else:
                            rrow.append(h / 2.0)
                    out.append(rrow)
                return out
            return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

    # Fallback: schoolbook rank 24
    rank = n * m * p
    U = np.zeros((nU, rank))
    V = np.zeros((nV, rank))
    W = np.zeros((nW, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    def to_list(M):
        out = []
        for row in M:
            rrow = []
            for v in row:
                h = round(v * 2)
                if h % 2 == 0:
                    rrow.append(int(h // 2))
                else:
                    rrow.append(h / 2.0)
            out.append(rrow)
        return out
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
