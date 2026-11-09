import time
import numpy as np

DEADLINE = 150.0
START = time.time()

def _remaining():
    return DEADLINE - (time.time() - START)

def _build_seed_36():
    n, m, p = 2, 4, 5
    strassen_U = [
        [1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,1],
        [1,1,0,0],[-1,0,1,0],[0,1,0,-1],
    ]
    strassen_V = [
        [1,0,0,1],[1,0,0,0],[0,1,0,-1],[-1,0,1,0],
        [0,0,0,1],[1,1,0,0],[0,0,1,1],
    ]
    strassen_W = [
        [1,0,0,1],[0,1,0,-1],[0,0,1,1],[1,1,0,0],
        [-1,0,1,0],[0,0,0,1],[1,0,0,0],
    ]
    rank = 4*7 + 8
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for j0 in (0,2):
        for k0 in (0,2):
            for r in range(7):
                for li in range(2):
                    for lj in range(2):
                        U[li*m + (j0+lj)][col] = strassen_U[r][li*2+lj]
                for lj in range(2):
                    for lk in range(2):
                        V[(j0+lj)*p + (k0+lk)][col] = strassen_V[r][lj*2+lk]
                for lk in range(2):
                    for li in range(2):
                        W[(k0+lk)*n + li][col] = strassen_W[r][lk*2+li]
                col += 1
    for i in range(n):
        for j in range(m):
            U[i*m+j][col] = 1
            V[j*p+4][col] = 1
            W[4*n+i][col] = 1
            col += 1
    return U, V, W

def _build_candidate_22():
    # Strassen over (i=2, k=2) for k-blocks 0,2 ; j in 4 lanes handled by
    # distributing: use 2 j-blocks of 2, each with Strassen over (i,k)? That's 2*2*7=28.
    # Provide a verified 22 via: 2 Strassen (i,j-half) x ... not dimensionally clean.
    # Return None to fall back.
    return None

def _verify(U, V, W):
    n, m, p = 2, 4, 5
    rank = len(U[0])
    U = np.array(U, dtype=float); V = np.array(V, dtype=float); W = np.array(W, dtype=float)
    # Reconstruct tensor T[(i,j),(j,k),(k,i)] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # Build expected schoolbook tensor and compare.
    T = np.zeros((n*m, m*p, p*n))
    Texp = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[(i*m+j),(j*p+k),(k*n+i)] += u[i,j]*v[j,k]*w[k,i]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Texp[(i*m+j),(j*p+k),(k*n+i)] = 1.0
    return np.allclose(T, Texp, atol=1e-9)

def _try_reduce(U, V, W, budget_s=20.0):
    # Heuristic: remove any column r that is exactly zero in all three factors.
    rank = len(U[0])
    keep = []
    for r in range(rank):
        if all(U[i][r] == 0 for i in range(len(U))) and all(V[i][r] == 0 for i in range(len(V))) and all(W[i][r] == 0 for i in range(len(W))):
            continue
        keep.append(r)
    if len(keep) == rank:
        return U, V, W
    U2 = [ [row[r] for r in keep] for row in U ]
    V2 = [ [row[r] for r in keep] for row in V ]
    W2 = [ [row[r] for r in keep] for row in W ]
    if _verify(U2, V2, W2):
        return U2, V2, W2
    return U, V, W

def _try_border_rank_35(U36, V36, W36, budget_s=30.0):
    """
    Attempt to find a rank-35 decomposition by checking if the tensor lies in
    the closure of rank-35 decompositions. We use a numerical approach:
    start from the rank-36 decomposition and try to merge/eliminate one term
    through optimization.
    """
    if _remaining() < budget_s:
        return None
    
    n, m, p = 2, 4, 5
    U36 = np.array(U36, dtype=float)
    V36 = np.array(V36, dtype=float)
    W36 = np.array(W36, dtype=float)
    
    # Strategy: try to find a rank-35 decomposition by perturbing the rank-36
    # and using gradient descent to reduce the rank.
    # We'll try multiple random initializations near the rank-36 solution.
    
    from scipy.optimize import minimize
    
    def _objective(params, n, m, p, target_tensor):
        """
        params: flattened U (n*m*35), V (m*p*35), W (p*n*35)
        Returns the sum of squared errors between the reconstructed tensor
        and the target tensor.
        """
        r = 35
        U = params[:n*m*r].reshape(n*m, r)
        V = params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W = params[n*m*r + m*p*r :].reshape(p*n, r)
        
        # Compute tensor contraction
        T = np.zeros((n*m, m*p, p*n))
        for ri in range(r):
            u = U[:, ri].reshape(n, m)
            v = V[:, ri].reshape(m, p)
            w = W[:, ri].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[(i*m+j),(j*p+k),(k*n+i)] += u[i,j]*v[j,k]*w[k,i]
        
        # Compute error
        err = T - target_tensor
        return np.sum(err**2)
    
    def _objective_grad(params, n, m, p, target_tensor):
        """Gradient of the objective function."""
        r = 35
        U = params[:n*m*r].reshape(n*m, r)
        V = params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W = params[n*m*r + m*p*r :].reshape(p*n, r)
        
        # Compute tensor contraction
        T = np.zeros((n*m, m*p, p*n))
        for ri in range(r):
            u = U[:, ri].reshape(n, m)
            v = V[:, ri].reshape(m, p)
            w = W[:, ri].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[(i*m+j),(j*p+k),(k*n+i)] += u[i,j]*v[j,k]*w[k,i]
        
        err = T - target_tensor
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for ri in range(r):
            u = U[:, ri].reshape(n, m)
            v = V[:, ri].reshape(m, p)
            w = W[:, ri].reshape(p, n)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = (i*m+j, j*p+k, k*n+i)
                        grad = 2 * err[idx]
                        dU[i*m+j, ri] += grad * v[j*p+k, ri] * w[k*n+i, ri]
                        dV[j*p+k, ri] += grad * u[i*m+j, ri] * w[k*n+i, ri]
                        dW[k*n+i, ri] += grad * u[i*m+j, ri] * v[j*p+k, ri]
        
        return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
    
    # Build target tensor
    target_tensor = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_tensor[(i*m+j),(j*p+k),(k*n+i)] = 1.0
    
    # Try multiple random perturbations of the rank-36 solution
    # to find a rank-35 solution
    np.random.seed(42)
    
    best_params = None
    best_obj = float('inf')
    
    for trial in range(5):
        if _remaining() < 5.0:
            break
        
        # Start from rank-36, zero out one column, and add small random noise
        # to the remaining columns
        U_init = U36.copy()
        V_init = V36.copy()
        W_init = W36.copy()
        
        # Zero out the last column (column 35)
        U_init[:, 35] = 0
        V_init[:, 35] = 0
        W_init[:, 35] = 0
        
        # Add small random noise to encourage exploration
        noise_scale = 0.01
        U_init = U_init + np.random.randn(*U_init.shape) * noise_scale
        V_init = V_init + np.random.randn(*V_init.shape) * noise_scale
        W_init = W_init + np.random.randn(*W_init.shape) * noise_scale
        
        # Keep the last column as a small random vector
        U_init[:, 35] = np.random.randn(n*m) * 0.1
        V_init[:, 35] = np.random.randn(m*p) * 0.1
        W_init[:, 35] = np.random.randn(p*n) * 0.1
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        result = minimize(
            _objective, params_init,
            args=(n, m, p, target_tensor),
            method='L-BFGS-B',
            jac=_objective_grad,
            options={'maxiter': 100, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x
        
        if best_obj < 1e-10:
            break
    
    if best_params is None or best_obj > 1e-8:
        return None
    
    # Extract the rank-35 factors
    r = 35
    U35 = best_params[:n*m*r].reshape(n*m, r)
    V35 = best_params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
    W35 = best_params[n*m*r + m*p*r :].reshape(p*n, r)
    
    # Verify
    U35_list = U35.tolist()
    V35_list = V35.tolist()
    W35_list = W35.tolist()
    
    if _verify(U35_list, V35_list, W35_list):
        return U35_list, V35_list, W35_list
    
    return None

def construct():
    n, m, p = 2, 4, 5
    # Start from known-valid 36.
    U, V, W = _build_seed_36()
    best = (len(U[0]), U, V, W)

    # Try candidate 22 if available.
    cand = _build_candidate_22()
    if cand is not None:
        Uc, Vc, Wc = cand
        if _verify(Uc, Vc, Wc):
            best = (len(Uc[0]), Uc, Vc, Wc)

    # Heuristic reduction with time guard.
    if _remaining() > 3.0:
        Ur, Vr, Wr = _try_reduce(best[1], best[2], best[3])
        if len(Ur[0]) < best[0] and _verify(Ur, Vr, Wr):
            best = (len(Ur[0]), Ur, Vr, Wr)

    # Try border rank 35
    if _remaining() > 35.0:
        cand35 = _try_border_rank_35(best[1], best[2], best[3])
        if cand35 is not None:
            Uc35, Vc35, Wc35 = cand35
            if _verify(Uc35, Vc35, Wc35):
                best = (len(Uc35[0]), Uc35, Vc35, Wc35)

    U, V, W = best[1], best[2], best[3]
    return {"U": U, "V": V, "W": W}
