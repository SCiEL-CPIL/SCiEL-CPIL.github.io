"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify the decomposition is exact."""
    r = U.shape[1]
    # Check: sum_c W[k*n+i, c] * U[i'*m+j', c] * V[j''*p+k'', c] = delta
    for i in range(n):
        for k in range(p):
            for ip in range(n):
                for jp in range(m):
                    for jpp in range(m):
                        for kpp in range(p):
                            s = 0.0
                            for c in range(r):
                                s += W[k*n+ip, c] * U[ip*m+jp, c] * V[jpp*p+kpp, c]
                            expected = 1.0 if (ip==i and jp==jpp and kpp==k) else 0.0
                            if abs(s - expected) > tol:
                                return False
    return True

def _solve_decomposition(n, m, p, r, max_iters=5000, lr=0.01, seed=42):
    """Try to find a rank-r decomposition."""
    np.random.seed(seed)
    U = np.random.randn(n*m, r) * 0.5
    V = np.random.randn(m*p, r) * 0.5
    W = np.random.randn(p*n, r) * 0.5
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Compute gradient via finite differences or analytical
        # err = sum over all (i,k,ip,jp,jpp,kpp) of (sum_c W[k*n+ip,c]*U[ip*m+jp,c]*V[jpp*p+kpp,c] - delta)^2
        
        # This is expensive, so let's use a simpler approach:
        # For each nonzero equation (i,j,k): sum_c W[3k+i,c]*U[3i+j,c]*V[8j+k,c] should = 1
        # For each zero equation: should = 0
        
        err = 0.0
        # Nonzero equations
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(W[k*n+i, :] * U[i*m+j, :] * V[j*p+k, :])
                    err += (s - 1.0)**2
        # Zero equations (sample a subset for speed)
        for i in range(n):
            for k in range(p):
                for ip in range(n):
                    for jp in range(m):
                        for jpp in range(m):
                            for kpp in range(p):
                                if ip==i and jp==jpp and kpp==k:
                                    continue
                                s = np.sum(W[k*n+ip, :] * U[ip*m+jp, :] * V[jpp*p+kpp, :])
                                err += s**2
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if best_err < 1e-10:
                break
        
        # Simple gradient descent
        # dU[ip*m+jp, c] = 2 * sum over (i,k,jpp,kpp) of (residual) * W[k*n+ip,c] * V[jpp*p+kpp,c]
        # This is complex, so use a random perturbation approach
        
        # Perturb one variable at a time
        for _ in range(3):
            mat = np.random.choice(['U', 'V', 'W'])
            if mat == 'U':
                idx = np.random.randint(n*m)
                c = np.random.randint(r)
                delta = np.random.randn() * lr
                old = U[idx, c]
                U[idx, c] = old + delta
                new_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.sum(W[k*n+i, :] * U[i*m+j, :] * V[j*p+k, :])
                            new_err += (s - 1.0)**2
                for i in range(n):
                    for k in range(p):
                        for ip in range(n):
                            for jp in range(m):
                                for jpp in range(m):
                                    for kpp in range(p):
                                        if ip==i and jp==jpp and kpp==k:
                                            continue
                                        s = np.sum(W[k*n+ip, :] * U[ip*m+jp, :] * V[jpp*p+kpp, :])
                                        new_err += s**2
                if new_err < err:
                    err = new_err
                else:
                    U[idx, c] = old
            elif mat == 'V':
                idx = np.random.randint(m*p)
                c = np.random.randint(r)
                delta = np.random.randn() * lr
                old = V[idx, c]
                V[idx, c] = old + delta
                new_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.sum(W[k*n+i, :] * U[i*m+j, :] * V[j*p+k, :])
                            new_err += (s - 1.0)**2
                for i in range(n):
                    for k in range(p):
                        for ip in range(n):
                            for jp in range(m):
                                for jpp in range(m):
                                    for kpp in range(p):
                                        if ip==i and jp==jpp and kpp==k:
                                            continue
                                        s = np.sum(W[k*n+ip, :] * U[ip*m+jp, :] * V[jpp*p+kpp, :])
                                        new_err += s**2
                if new_err < err:
                    err = new_err
                else:
                    V[idx, c] = old
            else:
                idx = np.random.randint(p*n)
                c = np.random.randint(r)
                delta = np.random.randn() * lr
                old = W[idx, c]
                W[idx, c] = old + delta
                new_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.sum(W[k*n+i, :] * U[i*m+j, :] * V[j*p+k, :])
                            new_err += (s - 1.0)**2
                for i in range(n):
                    for k in range(p):
                        for ip in range(n):
                            for jp in range(m):
                                for jpp in range(m):
                                    for kpp in range(p):
                                        if ip==i and jp==jpp and kpp==k:
                                            continue
                                        s = np.sum(W[k*n+ip, :] * U[ip*m+jp, :] * V[jpp*p+kpp, :])
                                        new_err += s**2
                if new_err < err:
                    err = new_err
                else:
                    W[idx, c] = old
    
    return best_U, best_V, best_W, best_err

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10-second margin
    
    # Start with schoolbook as fallback
    rank_fallback = n * m * p
    U_fb = [[0] * rank_fallback for _ in range(n * m)]
    V_fb = [[0] * rank_fallback for _ in range(m * p)]
    W_fb = [[0] * rank_fallback for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_fallback
    
    # Try to find better decompositions
    for r in range(24, min(72, 40)):
        if time.time() > deadline:
            break
        U, V, W, err = _solve_decomposition(n, m, p, r, max_iters=200, lr=0.1)
        if err < 1e-8:
            # Round to nearest half-integer
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            # Verify rounded version
            if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, tol=1e-6):
                if r < best_rank:
                    best_rank = r
                    best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                break
    
    return best_result
