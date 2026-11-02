import time
import numpy as np

DIMENSIONS = [2, 2, 3]

def _verify(U, V, W, n, m, p, tol=1e-9):
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _round_half(mat):
    return [[round(x * 2) / 2.0 for x in row] for row in mat]

def _rank11_solution():
    n, m, p = 2, 2, 3
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def set_u(i, j, r, val): U[i*m+j][r] = val
    def set_v(j, k, r, val): V[j*p+k][r] = val
    def set_w(k, i, r, val): W[k*n+i][r] = val

    # Strassen-like terms for k=0,1 (rank 7)
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1); set_w(1, 1, 0, 1)

    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1); set_w(1, 1, 1, -1)

    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1); set_w(1, 1, 2, 1)

    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1); set_w(0, 1, 3, 1)

    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1); set_w(1, 0, 4, 1)

    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    # k=2 terms (4 more)
    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)

    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)

    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)

    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}

def _als_step_U(V, W, n, m, p, r):
    U = [[0.0]*r for _ in range(n*m)]
    for i in range(n):
        for j in range(m):
            A = np.zeros((p, r))
            for k in range(p):
                for c in range(r):
                    A[k, c] = V[j*p+k][c] * W[k*n+i][c]
            b = np.ones(p)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            for c in range(r):
                U[i*m+j][c] = float(sol[c])
    return U

def _als_step_V(U, W, n, m, p, r):
    V = [[0.0]*r for _ in range(m*p)]
    for j in range(m):
        for k in range(p):
            A = np.zeros((n, r))
            for i in range(n):
                for c in range(r):
                    A[i, c] = U[i*m+j][c] * W[k*n+i][c]
            b = np.ones(n)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            for c in range(r):
                V[j*p+k][c] = float(sol[c])
    return V

def _als_step_W(U, V, n, m, p, r):
    W = [[0.0]*r for _ in range(p*n)]
    for k in range(p):
        for i in range(n):
            A = np.zeros((m, r))
            for j in range(m):
                for c in range(r):
                    A[j, c] = U[i*m+j][c] * V[j*p+k][c]
            b = np.ones(m)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            for c in range(r):
                W[k*n+i][c] = float(sol[c])
    return W

def _als_optimize(U0, V0, W0, n, m, p, r, iters, deadline):
    U, V, W = U0, V0, W0
    for _ in range(iters):
        if time.time() > deadline:
            break
        U = _als_step_U(V, W, n, m, p, r)
        V = _als_step_V(U, W, n, m, p, r)
        W = _als_step_W(U, V, n, m, p, r)
    return U, V, W

def _residual(U, V, W, n, m, p):
    r = len(U[0])
    res = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                res.append(s - 1.0)
    return np.array(res)

def _try_gf13_rank10(deadline):
    """
    Try to find a rank-10 decomposition over GF(13) by starting from rank-11
    and applying random linear combinations to eliminate one term.
    """
    n, m, p = 2, 2, 3
    r_target = 10
    r_start = 11
    mod = 13
    
    sol11 = _rank11_solution()
    U11 = np.array(sol11["U"], dtype=mod)
    V11 = np.array(sol11["V"], dtype=mod)
    W11 = np.array(sol11["W"], dtype=mod)
    
    rng = np.random.RandomState(12345)
    
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        # Try to merge two terms into one via random linear combination
        # Pick two random terms to try to merge
        c1 = rng.randint(0, r_start)
        c2 = rng.randint(0, r_start)
        if c1 == c2:
            continue
        
        # Create a new term that is a linear combination of terms c1 and c2
        alpha = rng.randint(1, mod)
        beta = rng.randint(1, mod)
        
        # New U, V, W for the merged term
        U_new = (alpha * U11[:, c1] + beta * U11[:, c2]) % mod
        V_new = (alpha * V11[:, c1] + beta * V11[:, c2]) % mod
        W_new = (alpha * W11[:, c1] + beta * W11[:, c2]) % mod
        
        # Build rank-10 matrices by replacing terms c1, c2 with the merged term
        # We'll keep all terms except c1 and c2, and add the new merged term
        keep_indices = [i for i in range(r_start) if i not in (c1, c2)]
        
        U10 = np.zeros((n*m, r_target), dtype=mod)
        V10 = np.zeros((m*p, r_target), dtype=mod)
        W10 = np.zeros((p*n, r_target), dtype=mod)
        
        for idx, orig_idx in enumerate(keep_indices):
            U10[:, idx] = U11[:, orig_idx]
            V10[:, idx] = V11[:, orig_idx]
            W10[:, idx] = W11[:, orig_idx]
        
        # Add the merged term as the last column
        U10[:, r_target-1] = U_new
        V10[:, r_target-1] = V_new
        W10[:, r_target-1] = W_new
        
        # Verify over GF(13)
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r_target):
                        s = (s + U10[i*m+j, c] * V10[j*p+k, c] * W10[k*n+i, c]) % mod
                    if s != 1 % mod:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if not valid:
            continue
        
        # Check if all entries are valid (they should be since we're in GF(13))
        # Convert to integers (0-12 range, but we need to interpret them)
        # For the final answer, we need entries that are integers or half-integers
        # GF(13) elements 0-12 can be directly used as integers
        U_int = U10.astype(int)
        V_int = V10.astype(int)
        W_int = W10.astype(int)
        
        # Verify over reals (since integers are valid)
        U_list = U_int.tolist()
        V_list = V_int.tolist()
        W_list = W_int.tolist()
        if _verify(U_list, V_list, W_list, n, m, p, tol=1e-9):
            return U_list, V_list, W_list
    
    return None

def _try_rank10_optimization(deadline):
    """
    Try to find a rank-10 decomposition via numerical optimization with
    structured initialization.
    """
    n, m, p = 2, 2, 3
    r = 10
    
    sol11 = _rank11_solution()
    U11 = np.array(sol11["U"], dtype=float)
    V11 = np.array(sol11["V"], dtype=float)
    W11 = np.array(sol11["W"], dtype=float)
    
    rng = np.random.RandomState(42)
    best_residual = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for attempt in range(300):
        if time.time() > deadline:
            break
        
        if attempt < 100:
            # Truncated rank-11 + perturbation
            U_init = (U11[:, :r] + rng.normal(0, 0.1, (n*m, r))).tolist()
            V_init = (V11[:, :r] + rng.normal(0, 0.1, (m*p, r))).tolist()
            W_init = (W11[:, :r] + rng.normal(0, 0.1, (p*n, r))).tolist()
        elif attempt < 200:
            # Structured: 7 Strassen + 3 random for k=2
            U_init = [[0.0]*r for _ in range(n*m)]
            V_init = [[0.0]*r for _ in range(m*p)]
            W_init = [[0.0]*r for _ in range(p*n)]
            
            for i in range(n*m):
                for c in range(7):
                    U_init[i][c] = U11[i][c]
            for i in range(m*p):
                for c in range(7):
                    V_init[i][c] = V11[i][c]
            for i in range(p*n):
                for c in range(7):
                    W_init[i][c] = W11[i][c]
            
            for c in range(7, r):
                for i in range(n*m):
                    U_init[i][c] = rng.uniform(-1, 1)
                for i in range(m*p):
                    V_init[i][c] = rng.uniform(-1, 1)
                for i in range(p*n):
                    k = i // n
                    if k < 2:
                        W_init[i][c] = 0.0
                    else:
                        W_init[i][c] = rng.uniform(-1, 1)
        else:
            # Fully random
            U_init = rng.uniform(-1, 1, (n*m, r)).tolist()
            V_init = rng.uniform(-1, 1, (m*p, r)).tolist()
            W_init = rng.uniform(-1, 1, (p*n, r)).tolist()
        
        U, V, W = _als_optimize(U_init, V_init, W_init, n, m, p, r, iters=80, deadline=deadline)
        
        Ur, Vr, Wr = _round_half(U), _round_half(V), _round_half(W)
        if _verify(Ur, Vr, Wr, n, m, p, tol=1e-9):
            return Ur, Vr, Wr
        
        res = _residual(U, V, W, n, m, p)
        res_norm = np.linalg.norm(res)
        if res_norm < best_residual:
            best_residual = res_norm
            best_U, best_V, best_W = U, V, W
    
    # Final polish on best result
    if best_U is not None and time.time() < deadline:
        try:
            from scipy.optimize import least_squares
            
            def _flatten(U, V, W):
                return np.concatenate([np.array(U).flatten(), np.array(V).flatten(), np.array(W).flatten()])
            
            def _unflatten(x):
                U = x[:n*m*r].reshape(n*m, r)
                V = x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                W = x[n*m*r+m*p*r:].reshape(p*n, r)
                return U, V, W
            
            def _objective(x):
                U, V, W = _unflatten(x)
                res = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                            res.append(s - 1.0)
                return res
            
            x0 = _flatten(best_U, best_V, best_W)
            result = least_squares(_objective, x0, method='lm', max_nfev=20000, xtol=1e-15, ftol=1e-15)
            U, V, W = _unflatten(result.x)
            Ur, Vr, Wr = _round_half(U.tolist()), _round_half(V.tolist()), _round_half(W.tolist())
            if _verify(Ur, Vr, Wr, n, m, p, tol=1e-9):
                return Ur, Vr, Wr
        except Exception:
            pass
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    n, m, p = DIMENSIONS
    
    best = _rank11_solution()
    
    # Try GF(13) approach first (fast, exact)
    if time.time() < deadline:
        result_gf13 = _try_gf13_rank10(deadline)
        if result_gf13 is not None:
            best = {"U": result_gf13[0], "V": result_gf13[1], "W": result_gf13[2]}
    
    # Try numerical optimization if GF(13) didn't work
    if time.time() < deadline:
        result_opt = _try_rank10_optimization(deadline)
        if result_opt is not None:
            best = {"U": result_opt[0], "V": result_opt[1], "W": result_opt[2]}
    
    return best
