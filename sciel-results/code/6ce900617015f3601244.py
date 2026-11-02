import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Schoolbook solution
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = schoolbook_rank

    # Try lower ranks
    for rank in range(schoolbook_rank - 1, 19, -1):
        if time.time() > deadline:
            break
            
        result = _try_rank(n, m, p, rank, deadline)
        if result is not None:
            best_result = result
            best_rank = rank
            # If we find a very low rank, we might want to keep searching for even lower?
            # But lower is better, so we continue.
            continue
            
    return best_result

def _try_rank(n, m, p, rank, deadline):
    num_params = (n * m + m * p + p * n) * rank
    u_size = n * m * rank
    v_size = m * p * rank
    w_size = p * n * rank
    
    # Precompute delta
    delta = np.eye(m).reshape(1, 1, m, m) # shape (1, 1, m, m)
    
    def residual_function(params):
        U = params[:u_size].reshape(n, m, rank)
        V = params[u_size:u_size + v_size].reshape(m, p, rank)
        W = params[u_size + v_size:].reshape(p, n, rank)
        
        # Compute sum_r U[i,j,r] V[j2,k,r] W[k,i,r]
        # U: (n, m, r)
        # V: (m, p, r) -> indices j2, k, r
        # W: (p, n, r) -> indices k, i, r
        # Result: (n, p, m, m)
        # einsum: 'ijr,j2kr,kinr->ikjj2'
        try:
            prod = np.einsum('ijr,j2kr,kinr->ikjj2', U, V, W)
        except Exception:
            # Fallback to loops if einsum fails (unlikely)
            prod = np.zeros((n, p, m, m))
            for i in range(n):
                for k in range(p):
                    for j1 in range(m):
                        for j2 in range(m):
                            val = 0.0
                            for r in range(rank):
                                val += U[i, j1, r] * V[j2, k, r] * W[k, i, r]
                            prod[i, k, j1, j2] = val
                            
        residual = prod - delta
        return residual.flatten()

    # Initializations
    inits = []
    
    # 1. Random
    for _ in range(3):
        np.random.seed(int(time.time() * 10000) % 100000)
        U0 = np.random.randn(n, m, rank)
        V0 = np.random.randn(m, p, rank)
        W0 = np.random.randn(p, n, rank)
        scale = 1.0 / np.sqrt(rank)
        U0 *= scale
        V0 *= scale
        W0 *= scale
        inits.append(np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()]))
        
    # 2. Perturbed Schoolbook (if rank is close to schoolbook)
    # This is tricky because schoolbook has rank 24.
    # We can take the first 'rank' columns of schoolbook and randomize the rest?
    # Or just use random.
    
    for init in inits:
        if time.time() > deadline:
            break
        
        try:
            result = least_squares(residual_function, init, method='trf', max_nfev=200)
        except Exception:
            continue
            
        if result.cost < 1e-8:
            U_opt = result.x[:u_size].reshape(n, m, rank)
            V_opt = result.x[u_size:u_size + v_size].reshape(m, p, rank)
            W_opt = result.x[u_size + v_size:].reshape(p, n, rank)
            
            # Snap to half-integers
            for denom in [1, 2, 3, 4]:
                U_snap = np.round(U_opt * denom) / denom
                V_snap = np.round(V_opt * denom) / denom
                W_snap = np.round(W_opt * denom) / denom
                
                # Verify
                max_err = 0.0
                try:
                    prod = np.einsum('ijr,j2kr,kinr->ikjj2', U_snap, V_snap, W_snap)
                    err = np.max(np.abs(prod - delta))
                except Exception:
                    err = 1.0
                    
                if err < 1e-9:
                    # Convert back to list of lists
                    U_flat = U_snap.flatten().tolist()
                    V_flat = V_snap.flatten().tolist()
                    W_flat = W_snap.flatten().tolist()
                    
                    # Reshape to (rows, rank)
                    U_out = [U_flat[i*rank:(i+1)*rank] for i in range(n*m)]
                    V_out = [V_flat[i*rank:(i+1)*rank] for i in range(m*p)]
                    W_out = [W_flat[i*rank:(i+1)*rank] for i in range(p*n)]
                    
                    return {"U": U_out, "V": V_out, "W": W_out}
                    
    return None
