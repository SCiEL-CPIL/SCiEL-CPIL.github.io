"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 4, 6
    NM, MP, PN = n * m, m * p, p * n  # 12, 24, 18
    
    # Build target tensor T of shape (NM, MP, PN)
    # T[u, v, w] = 1 if u=i*M+j, v=j*P+k, w=k*N+i for some (i,j,k)
    T = np.zeros((NM, MP, PN))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((NM, rank))
        V = np.zeros((MP, rank))
        W = np.zeros((PN, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def check_exact(U, V, W):
        """Check if the decomposition exactly reconstructs T."""
        r = U.shape[1]
        T_rec = np.einsum('uc,vc,wc->uvw', U, V, W)
        return np.allclose(T_rec, T, atol=1e-9)
    
    def objective_and_grad(x, r):
        size_U = NM * r
        size_V = MP * r
        size_W = PN * r
        
        U_flat = x[:size_U]
        V_flat = x[size_U:size_U + size_V]
        W_flat = x[size_U + size_V:]
        
        U = U_flat.reshape(n, m, r)
        V = V_flat.reshape(m, p, r)
        W = W_flat.reshape(p, n, r)
        
        T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        T_3d = np.ones((n, m, p))
        err = T_rec - T_3d
        loss = np.sum(err ** 2)
        
        grad_U = 2 * np.einsum('ijk,jkc,kic->ijc', err, V, W)
        grad_V = 2 * np.einsum('ijk,ijc,kic->jkc', err, U, W)
        grad_W = 2 * np.einsum('ijk,ijc,jkc->kic', err, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad
    
    def try_rank(r, num_seeds=3, max_iter=200):
        """Try to find a decomposition of rank r."""
        size_U = NM * r
        size_V = MP * r
        size_W = PN * r
        total_size = size_U + size_V + size_W
        
        # Get schoolbook initialization
        sb = schoolbook()
        U_sb = np.array(sb["U"])
        V_sb = np.array(sb["V"])
        W_sb = np.array(sb["W"])
        
        for seed in range(num_seeds):
            if time.time() > deadline:
                return None
            
            if seed == 0:
                # Use first r columns of schoolbook
                U_init = U_sb[:, :r].flatten()
                V_init = V_sb[:, :r].flatten()
                W_init = W_sb[:, :r].flatten()
            else:
                np.random.seed(seed * 1000 + r)
                U_init = U_sb[:, :r].flatten() + np.random.randn(size_U) * 0.5
                V_init = V_sb[:, :r].flatten() + np.random.randn(size_V) * 0.5
                W_init = W_sb[:, :r].flatten() + np.random.randn(size_W) * 0.5
            
            x0 = np.concatenate([U_init, V_init, W_init])
            
            try:
                res = minimize(
                    objective_and_grad,
                    x0,
                    args=(r,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30}
                )
            except Exception:
                continue
            
            if res.fun < 1e-8:
                U_flat = res.x[:size_U]
                V_flat = res.x[size_U:size_U + size_V]
                W_flat = res.x[size_U + size_V:]
                
                U = U_flat.reshape(NM, r)
                V = V_flat.reshape(MP, r)
                W = W_flat.reshape(PN, r)
                
                # Round to nearest 0.5
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if check_exact(U_rounded, V_rounded, W_rounded):
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
        
        return None
    
    # Try ranks from just below schoolbook downward
    # Schoolbook is 72. Try 71, 70, ... down to some lower bound.
    # Lower bound: max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # But actual lower bound for tensor rank is higher.
    
    for r in range(71, 40, -1):
        if time.time() > deadline:
            break
        
        result = try_rank(r, num_seeds=3, max_iter=200)
        if result is not None:
            best_result = result
            best_rank = r
            break  # Found a valid decomposition, stop
    
    if best_rank >= n * m * p:
        return schoolbook()
    
    return best_result
# EVOLVE-BLOCK-END
