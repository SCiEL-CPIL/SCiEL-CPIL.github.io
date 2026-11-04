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
        rank = NM * MP * PN // (N * M * P) * (N * M * P)  # n*m*p = 72
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
        r = len(U_list[0]) if U_list else 0
        U = np.array(U_list).reshape(N, M, r)
        V = np.array(V_list).reshape(M, P, r)
        W = np.array(W_list).reshape(P, N, r)
        T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        # Check against original T reshaped
        T_orig = T.reshape(N, M, P)  # Wait, T is (NM, MP, PN), need to map back
        # Actually T[u,v,w] where u=i*M+j, v=j*P+k, w=k*N+i
        # So T[i*M+j, j*P+k, k*N+i] = 1
        # Let's build T_3d properly
        T_3d = np.zeros((N, M, P))
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    T_3d[i, j, k] = 1.0
        return np.allclose(T_rec, T_3d, atol=1e-9)
    
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
        
        for seed in range(num_seeds):
            if time.time() - start_time > DEADLINE:
                return None
            
            np.random.seed(seed * 1000 + r)
            x0 = np.random.randn(total_size) * 0.5
            
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
            
            if res.fun < 1e-6:
                # Extract and round to half-integers
                U_flat = res.x[:size_U]
                V_flat = res.x[size_U:size_U + size_V]
                W_flat = res.x[size_U + size_V:]
                
                U = U_flat.reshape(N, M, r)
                V = V_flat.reshape(M, P, r)
                W = W_flat.reshape(P, N, r)
                
                # Round to nearest 0.5
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                U_list = U_rounded.reshape(NM, r).tolist()
                V_list = V_rounded.reshape(MP, r).tolist()
                W_list = W_rounded.reshape(PN, r).tolist()
                
                if check_exact(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from low to high
    # Theoretical lower bound: For matrix multiplication, rank >= max(n*m, m*p, p*n) / min(n,m,p) ... 
    # Actually for (n,m,p), the rank is at least max(n*m, m*p, p*n) but that's not tight.
    # For 3x4x6, schoolbook is 72. Let's try from 20 upward.
    min_rank_to_try = 15
    max_rank_to_try = 70
    
    for r in range(min_rank_to_try, max_rank_to_try + 1):
        if time.time() - start_time > DEADLINE:
            break
        
        result = try_rank(r, num_seeds=3, max_iter=200)
        if result is not None:
            best_result = result
            best_rank = r
            break  # Found minimal rank, stop
    
    # If no better found, return schoolbook
    if best_rank >= N * M * P:
        return schoolbook()
    
    return best_result
