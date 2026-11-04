"""Matrix multiplication tensor decomposition search for 3x4x6."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 6]
START_TIME = time.time()
DEADLINE = 145.0  # Safety margin before 150s limit

def construct():
    global START_TIME
    start_time = time.time()
    n, m, p = DIMENSIONS
    N, M, P = n, m, p
    NM, MP, PN = N * M, M * P, P * N
    
    # Build target tensor T of shape (N*M, M*P, P*N)
    # T[u, v, w] = 1 if u=i*M+j, v=j*P+k, w=k*N+i
    T = np.zeros((NM, MP, PN))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                u = i * M + j
                v = j * P + k
                w = k * N + i
                T[u, v, w] = 1.0
    T_flat = T.flatten()
    
    # Schoolbook fallback
    def schoolbook():
        rank = N * M * P
        U = [[0] * rank for _ in range(NM)]
        V = [[0] * rank for _ in range(MP)]
        W = [[0] * rank for _ in range(PN)]
        col = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i * M + j][col] = 1
                    V[j * P + k][col] = 1
                    W[k * N + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = best_result["U"][0].__len__() if best_result["U"] else N * M * P
    
    def check_exact(U_list, V_list, W_list):
        """Check if the decomposition exactly reconstructs T."""
        if not U_list:
            return False
        r = len(U_list[0])
        U = np.array(U_list)
        V = np.array(V_list)
        W = np.array(W_list)
        
        # Compute reconstructed tensor T_rec[u, v, w] = sum_c U[u, c] V[v, c] W[w, c]
        T_rec = np.einsum('uc,vc,wc->uvw', U, V, W)
        
        # Compare with the target tensor T
        return np.allclose(T_rec, T, atol=1e-9)
    
    def objective_and_grad(x, r):
        size_U = N * M * r
        size_V = M * P * r
        size_W = P * N * r
        
        U_flat = x[:size_U]
        V_flat = x[size_U:size_U + size_V]
        W_flat = x[size_U + size_V:]
        
        U = U_flat.reshape(N, M, r)
        V = V_flat.reshape(M, P, r)
        W = W_flat.reshape(P, N, r)
        
        T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        T_3d = np.ones((N, M, P))
        err = T_rec - T_3d
        loss = np.sum(err ** 2)
        
        grad_U = 2 * np.einsum('ijk,jkc,kic->ijc', err, V, W)
        grad_V = 2 * np.einsum('ijk,ijc,kic->jkc', err, U, W)
        grad_W = 2 * np.einsum('ijk,ijc,jkc->kic', err, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad

    def try_rank(r, num_seeds=5, max_iter=300):
        """Try to find a decomposition of rank r."""
        size_U = N * M * r
        size_V = M * P * r
        size_W = P * N * r
        total_size = size_U + size_V + size_W
        
        # Initialize with schoolbook columns if possible
        # Schoolbook has N*M*P columns.
        # If r < N*M*P, take the first r columns.
        # This provides a good starting point close to a valid solution.
        
        for seed in range(num_seeds):
            if time.time() - start_time > DEADLINE:
                return None
            
            if seed == 0:
                # Use schoolbook initialization
                sb = schoolbook()
                U_sb = np.array(sb["U"])
                V_sb = np.array(sb["V"])
                W_sb = np.array(sb["W"])
                
                # Take first r columns
                U_init = U_sb[:, :r].flatten()
                V_init = V_sb[:, :r].flatten()
                W_init = W_sb[:, :r].flatten()
                x0 = np.concatenate([U_init, V_init, W_init])
            else:
                np.random.seed(seed * 1000 + r)
                # Perturb the schoolbook initialization
                sb = schoolbook()
                U_sb = np.array(sb["U"])
                V_sb = np.array(sb["V"])
                W_sb = np.array(sb["W"])
                
                U_init = U_sb[:, :r].flatten() + np.random.randn(size_U) * 0.1
                V_init = V_sb[:, :r].flatten() + np.random.randn(size_V) * 0.1
                W_init = W_sb[:, :r].flatten() + np.random.randn(size_W) * 0.1
                x0 = np.concatenate([U_init, V_init, W_init])
            
            try:
                res = minimize(
                    objective_and_grad,
                    x0,
                    args=(r,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
                )
            except Exception:
                continue
            
            if res.fun < 1e-4:
                # Extract and round to half-integers
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
                
                U_list = U_rounded.tolist()
                V_list = V_rounded.tolist()
                W_list = W_rounded.tolist()
                
                if check_exact(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from low to high, starting from just below schoolbook
    # Schoolbook is 72. Try 71, 70, ...
    # If no better found, return schoolbook.
    
    # Theoretical lower bound is not very tight, but we know 72 works.
    # We try to find if 71 or 70 works.
    
    for r in range(71, 50, -1):
        if time.time() - start_time > DEADLINE:
            break
        
        result = try_rank(r, num_seeds=5, max_iter=300)
        if result is not None:
            best_result = result
            best_rank = r
            break  # Found minimal rank, stop
    
    # If no better found, return schoolbook
    if best_rank >= N * M * P:
        return schoolbook()
    
    return best_result
