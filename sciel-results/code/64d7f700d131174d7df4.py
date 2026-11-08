import numpy as np
import time
from scipy.optimize import minimize

def check_decomp(U, V, W, n, m, p):
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # Check if sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i,j,k
    # and 0 otherwise? No, T is 1 for all i,j,k.
    # Wait, the tensor T has T[i,j,k] = 1 for all i,j,k.
    # So we check if the sum equals 1 for all i,j,k.
    
    r = U.shape[1]
    # Compute the product tensor
    # We can do this by iterating over r and accumulating
    # But for small sizes, we can just check all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(r):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 2, 3, 9
    
    # Sub-problem: 2 x 3 x 3
    n_sub, m_sub, p_sub = 2, 3, 3
    
    # Target ranks for sub-problem
    sub_ranks = [14, 15, 16, 17]
    
    best_sub = None
    best_sub_rank = 18
    
    for r_sub in sub_ranks:
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            U_init = np.random.randn(n_sub*m_sub, r_sub)
            V_init = np.random.randn(m_sub*p_sub, r_sub)
            W_init = np.random.randn(p_sub*n_sub, r_sub)
            
            # Objective function
            def objective(params):
                U = params[:n_sub*m_sub*r_sub].reshape(n_sub*m_sub, r_sub)
                V = params[n_sub*m_sub*r_sub:n_sub*m_sub*r_sub + m_sub*p_sub*r_sub].reshape(m_sub*p_sub, r_sub)
                W = params[n_sub*m_sub*r_sub + m_sub*p_sub*r_sub:].reshape(p_sub*n_sub, r_sub)
                
                error = 0.0
                for i in range(n_sub):
                    for j in range(m_sub):
                        for k in range(p_sub):
                            val = 0
                            for c in range(r_sub):
                                val += U[i*m_sub+j, c] * V[j*p_sub+k, c] * W[k*n_sub+i, c]
                            error += (val - 1.0)**2
                return error
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100})
            except:
                continue
                
            U_opt = res.x[:n_sub*m_sub*r_sub].reshape(n_sub*m_sub, r_sub)
            V_opt = res.x[n_sub*m_sub*r_sub:n_sub*m_sub*r_sub + m_sub*p_sub*r_sub].reshape(m_sub*p_sub, r_sub)
            W_opt = res.x[n_sub*m_sub*r_sub + m_sub*p_sub*r_sub:].reshape(p_sub*n_sub, r_sub)
            
            # Round to half-integers
            U_round = np.round(U_opt * 2) / 2
            V_round = np.round(V_opt * 2) / 2
            W_round = np.round(W_opt * 2) / 2
            
            if check_decomp(U_round, V_round, W_round, n_sub, m_sub, p_sub):
                best_sub = (U_round, V_round, W_round)
                best_sub_rank = r_sub
                break
                
        if best_sub:
            break
            
    # Construct 2x3x9 decomposition
    if best_sub:
        U_sub, V_sub, W_sub = best_sub
        r_sub = best_sub_rank
        rank = 3 * r_sub
        
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        for b in range(3):
            offset = b * r_sub
            for i in range(n):
                for j in range(m):
                    U[i*m+j, offset:offset+r_sub] = U_sub[i*m_sub+j, :]
                    
            for j in range(m):
                for k in range(p_sub):
                    V[j*p + (b*p_sub + k), offset:offset+r_sub] = V_sub[j*m_sub + k, :]
                    
            for k in range(p_sub):
                for i in range(n):
                    W[(b*p_sub + k)*n + i, offset:offset+r_sub] = W_sub[k*n_sub + i, :]
                    
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
                    
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
