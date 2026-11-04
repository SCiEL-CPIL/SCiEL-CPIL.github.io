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
    best_rank = N * M * P
    
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
    
    def alternating_projection(r, num_iters=200, num_seeds=5):
        """
        Alternating projection: 
        1. Start with random continuous factors
        2. Alternate between:
           - Solving for one factor given the other two (linear least squares)
           - Rounding all factors to nearest half-integer
        """
        T_3d = np.ones((N, M, P))
        
        for seed in range(num_seeds):
            if time.time() - start_time > DEADLINE:
                return None
            
            np.random.seed(seed * 10000 + r * 7 + 42)
            
            # Initialize with sparse structure: each factor has mostly zeros
            U = np.zeros((N, M, r))
            V = np.zeros((M, P, r))
            W = np.zeros((P, N, r))
            
            # Random sparse initialization
            for c in range(r):
                # Each rank component touches a subset of entries
                n_u = max(1, int(np.random.exponential(1.0)))
                n_v = max(1, int(np.random.exponential(1.0)))
                n_w = max(1, int(np.random.exponential(1.0)))
                
                u_idx = np.random.choice(N * M, min(n_u, N * M), replace=False)
                v_idx = np.random.choice(M * P, min(n_v, M * P), replace=False)
                w_idx = np.random.choice(P * N, min(n_w, P * N), replace=False)
                
                U.flat[u_idx] = np.random.choice([-0.5, 0.5, 1.0], len(u_idx))
                V.flat[v_idx] = np.random.choice([-0.5, 0.5, 1.0], len(v_idx))
                W.flat[w_idx] = np.random.choice([-0.5, 0.5, 1.0], len(w_idx))
            
            best_local_loss = float('inf')
            best_local_factors = (U.copy(), V.copy(), W.copy())
            
            for it in range(num_iters):
                if time.time() - start_time > DEADLINE:
                    return None
                
                # Alternate: solve for U given V, W
                # For each (i,j): sum_c U[i,j,c] * V[j,:,c] * W[:,i,c] should equal 1 for the right k
                # This is complex; instead use gradient step with projection
                
                # Compute current tensor
                T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
                err = T_rec - T_3d
                loss = np.sum(err ** 2)
                
                if loss < best_local_loss:
                    best_local_loss = loss
                    best_local_factors = (U.copy(), V.copy(), W.copy())
                
                if loss < 1e-10:
                    break
                
                # Gradient step
                lr = 0.01 / (1 + it * 0.01)
                
                grad_U = 2 * np.einsum('ijk,jkc,kic->ijc', err, V, W)
                grad_V = 2 * np.einsum('ijk,ijc,kic->jkc', err, U, W)
                grad_W = 2 * np.einsum('ijk,ijc,jkc->kic', err, U, V)
                
                U = U - lr * grad_U
                V = V - lr * grad_V
                W = W - lr * grad_W
                
                # Project to half-integers (round to nearest 0.5)
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
            
            # Check best local solution
            U_best, V_best, W_best = best_local_factors
            U_list = U_best.reshape(NM, r).tolist()
            V_list = V_best.reshape(MP, r).tolist()
            W_list = W_best.reshape(PN, r).tolist()
            
            if check_exact(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    def try_rank_optim(r, num_seeds=3, max_iter=100):
        """Try to find a decomposition of rank r using optimization."""
        size_U = N * M * r
        size_V = M * P * r
        size_W = P * N * r
        total_size = size_U + size_V + size_W
        
        for seed in range(num_seeds):
            if time.time() - start_time > DEADLINE:
                return None
            
            np.random.seed(seed * 1000 + r)
            x0 = np.random.randn(total_size) * 0.1
            
            try:
                res = minimize(
                    objective_and_grad,
                    x0,
                    args=(r,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 10}
                )
            except Exception:
                continue
            
            if res.fun < 1e-4:
                U_flat = res.x[:size_U]
                V_flat = res.x[size_U:size_U + size_V]
                W_flat = res.x[size_U + size_V:]
                
                U = U_flat.reshape(NM, r)
                V = V_flat.reshape(MP, r)
                W = W_flat.reshape(PN, r)
                
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                U_list = U_rounded.tolist()
                V_list = V_rounded.tolist()
                W_list = W_rounded.tolist()
                
                if check_exact(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from low to high
    # For 3x4x6, known lower bounds suggest rank around 24-30 might be achievable
    min_rank_to_try = 20
    max_rank_to_try = 50
    
    for r in range(min_rank_to_try, max_rank_to_try + 1):
        if time.time() - start_time > DEADLINE:
            break
        
        # Try alternating projection first (faster, exploits structure)
        result = alternating_projection(r, num_iters=100, num_seeds=5)
        if result is not None:
            best_result = result
            best_rank = r
            break
        
        # Try optimization if time remains
        if time.time() - start_time < DEADLINE - 20:
            result = try_rank_optim(r, num_seeds=2, max_iter=50)
            if result is not None:
                best_result = result
                best_rank = r
                break
    
    # If no better found, return schoolbook
    if best_rank >= N * M * P:
        return schoolbook()
    
    return best_result
