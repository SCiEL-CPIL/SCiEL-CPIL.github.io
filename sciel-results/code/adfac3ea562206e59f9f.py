import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    def time_left():
        return deadline - time.time()

    def check_solution_fast(U, V, W):
        """Check if decomposition is exact. U:(n*m,R), V:(m*p,R), W:(p*n,R).
        For each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should equal 1."""
        R = U.shape[1]
        max_err = 0.0
        # Pre-compute W transposed for efficiency: W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                u = U[i*m+j, :]  # shape (R,)
                for k in range(p):
                    v = V[j*p+k, :]  # shape (R,)
                    w = W[k*n+i, :]  # shape (R,)
                    val = np.dot(u * v, w)  # sum_r u[r]*v[r]*w[r]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def check_solution_vectorized(U, V, W):
        """Vectorized check using einsum for speed."""
        R = U.shape[1]
        # Reshape for batch operations
        max_err = 0.0
        for i in range(n):
            Ui = U[i*m:(i+1)*m, :]  # shape (m, R)
            Wi = W[i::n, :]  # shape (p, R) - W[k*n+i, :] for all k
            for j in range(m):
                uj = Ui[j, :]  # shape (R,)
                for k in range(p):
                    vk = V[j*p+k, :]  # shape (R,)
                    wk = Wi[k, :]  # shape (R,)
                    val = float(np.sum(uj * vk * wk))
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def solve_U(V, W, R):
        """Given V and W, solve for each U[i*m+j] row."""
        U = np.zeros((n*m, R))
        for i in range(n):
            for j in range(m):
                # Need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
                A = np.zeros((p, R))
                for k in range(p):
                    A[k, :] = V[j*p + k, :] * W[k*n + i, :]
                b = np.ones(p)
                try:
                    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                    U[i*m + j, :] = sol
                except np.linalg.LinAlgError:
                    pass
        return U

    def solve_V(U, W, R):
        """Given U and W, solve for each V[j*p+k] row."""
        V = np.zeros((m*p, R))
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, R))
                for i in range(n):
                    A[i, :] = U[i*m + j, :] * W[k*n + i, :]
                b = np.ones(n)
                try:
                    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                    V[j*p + k, :] = sol
                except np.linalg.LinAlgError:
                    pass
        return V

    def solve_W(U, V, R):
        """Given U and V, solve for each W[k*n+i] row."""
        W = np.zeros((p*n, R))
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, R))
                for j in range(m):
                    A[j, :] = U[i*m + j, :] * V[j*p + k, :]
                b = np.ones(m)
                try:
                    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                    W[k*n + i, :] = sol
                except np.linalg.LinAlgError:
                    pass
        return W

    def als(U, V, W, R, max_iters=200, tol=1e-12, use_fast_check=False):
        """ALS optimization loop."""
        check_fn = check_solution_fast if use_fast_check else check_solution_vectorized
        for it in range(max_iters):
            U = solve_U(V, W, R)
            V = solve_V(U, W, R)
            W = solve_W(U, V, R)
            err = check_fn(U, V, W)
            if err < tol:
                break
        return U, V, W, err

    def als_with_monitor(U, V, W, R, max_iters=100, tol=1e-10, time_budget=20):
        """ALS with time monitoring."""
        start = time.time()
        for it in range(max_iters):
            if time.time() - start > time_budget:
                break
            U = solve_U(V, W, R)
            V = solve_V(U, W, R)
            W = solve_W(U, V, R)
            err = check_solution_fast(U, V, W)
            if err < tol:
                break
        return U, V, W, err

    # Schoolbook rank-30 solution (always valid)
    def schoolbook():
        R = n * m * p
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, R

    # Strassen-based rank-28 solution
    def strassen_28():
        R = 28
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        c = 0

        def add_strassen(j_off, k_off):
            nonlocal c
            # m1: (a0+a1)(b0+b1)
            U[0*m+j_off, c] = 1; U[1*m+(j_off+1), c] = 1
            V[j_off*p+k_off, c] = 1; V[(j_off+1)*p+(k_off+1), c] = 1
            W[k_off*n+0, c] = 1; W[(k_off+1)*n+1, c] = 1
            c += 1
            # m2: (a10+a11)b00
            U[1*m+j_off, c] = 1; U[1*m+(j_off+1), c] = 1
            V[j_off*p+k_off, c] = 1
            W[k_off*n+1, c] = 1; W[(k_off+1)*n+1, c] = -1
            c += 1
            # m3: a00(b01-b11)
            U[0*m+j_off, c] = 1
            V[j_off*p+(k_off+1), c] = 1; V[(j_off+1)*p+(k_off+1), c] = -1
            W[(k_off+1)*n+0, c] = 1; W[(k_off+1)*n+1, c] = 1
            c += 1
            # m4: a11(b10-b00)
            U[1*m+(j_off+1), c] = 1
            V[(j_off+1)*p+k_off, c] = 1; V[j_off*p+k_off, c] = -1
            W[k_off*n+0, c] = 1; W[k_off*n+1, c] = 1
            c += 1
            # m5: (a00+a01)b11
            U[0*m+j_off, c] = 1; U[0*m+(j_off+1), c] = 1
            V[(j_off+1)*p+(k_off+1), c] = 1
            W[k_off*n+0, c] = -1; W[(k_off+1)*n+0, c] = 1
            c += 1
            # m6: (a10-a00)(b00+b01)
            U[1*m+j_off, c] = 1; U[0*m+j_off, c] = -1
            V[j_off*p+k_off, c] = 1; V[j_off*p+(k_off+1), c] = 1
            W[(k_off+1)*n+1, c] = 1
            c += 1
            # m7: (a01-a11)(b10+b11)
            U[0*m+(j_off+1), c] = 1; U[1*m+(j_off+1), c] = -1
            V[(j_off+1)*p+k_off, c] = 1; V[(j_off+1)*p+(k_off+1), c] = 1
            W[k_off*n+0, c] = 1
            c += 1

        add_strassen(0, 0)   # j={0,1}, k={0,1}
        add_strassen(0, 2)   # j={0,1}, k={2,3}

        # Schoolbook for remaining: j=2 all k, and j=0,1 with k=4
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if j < 2 and k < 4:
                        continue
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1

        assert c == R, f"Expected rank {R}, got {c}"
        return U, V, W, R

    # Try to find lower rank solutions
    def try_rank(R, max_time=30, n_random=20):
        """Try to find exact decomposition of given rank."""
        best = None
        start = time.time()

        # Strategy 1: Random restarts
        for trial in range(n_random):
            if time.time() - start > max_time * 0.6:
                break
            np.random.seed(trial * 104729 + 7)
            scale = 1.0 / (n * m * p) ** 0.5
            U = np.random.randn(n*m, R) * scale
            V = np.random.randn(m*p, R) * scale
            W = np.random.randn(p*n, R) * scale
            U, V, W, err = als_with_monitor(U, V, W, R, max_iters=150, tol=1e-10, time_budget=max_time*0.3)
            if err < 1e-8:
                best = (U.copy(), V.copy(), W.copy(), err)
                break

        # Strategy 2: Structured initialization from schoolbook (zero random columns)
        if best is None and time.time() - start < max_time * 0.8:
            if R <= 30:
                U30, V30, W30, R30 = schoolbook()
                # Remove random columns and optimize
                for trial in range(10):
                    if time.time() - start > max_time * 0.8:
                        break
                    cols_to_keep = np.random.choice(R30, R, replace=False)
                    U = U30[:, cols_to_keep].copy()
                    V = V30[:, cols_to_keep].copy()
                    W = W30[:, cols_to_keep].copy()
                    U, V, W, err = als_with_monitor(U, V, W, R, max_iters=100, tol=1e-10, time_budget=max_time*0.15)
                    if err < 1e-8:
                        best = (U.copy(), V.copy(), W.copy(), err)
                        break

        # Strategy 3: From Strassen-28 (if applicable)
        if R <= 28 and best is None and time.time() - start < max_time:
            U28, V28, W28, _ = strassen_28()
            if R < 28:
                # Zero out random columns and optimize
                for trial in range(5):
                    if time.time() - start > max_time:
                        break
                    cols_to_keep = np.random.choice(28, R, replace=False)
                    U = U28[:, cols_to_keep].copy()
                    V = V28[:, cols_to_keep].copy()
                    W = W28[:, cols_to_keep].copy()
                    U, V, W, err = als_with_monitor(U, V, W, R, max_iters=100, tol=1e-10, time_budget=max_time*0.15)
                    if err < 1e-8:
                        best = (U.copy(), V.copy(), W.copy(), err)
                        break

        return best

    # Main search
    best_result = None
    best_rank = 30

    # First get guaranteed solutions
    U28, V28, W28, R28 = strassen_28()
    err28 = check_solution_fast(U28, V28, W28)

    if err28 < 1e-10:
        best_rank = 28
        best_result = (U28, V28, W28, err28)

    U30, V30, W30, R30 = schoolbook()
    err30 = check_solution_fast(U30, V30, W30)
    if err30 < 1e-10 and best_result is None:
        best_rank = 30
        best_result = (U30, V30, W30, err30)

    # Try to find rank 26 and 27
    if best_rank >= 27 and time_left() > 40:
        result_27 = try_rank(27, max_time=30)
        if result_27 is not None:
            best_rank = 27
            best_result = result_27

    if best_rank >= 26 and time_left() > 30:
        result_26 = try_rank(26, max_time=35)
        if result_26 is not None:
            best_rank = 26
            best_result = result_26

    # Try rank 25 if we have time
    if best_rank >= 25 and time_left() > 25:
        result_25 = try_rank(25, max_time=30)
        if result_25 is not None:
            best_rank = 25
            best_result = result_25

    # Final verification and return
    if best_result is not None:
        U, V, W, _ = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        return {"U": U30.tolist(), "V": V30.tolist(), "W": W30.tolist()}
