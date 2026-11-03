import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 4, 5, 5
    
    # Schoolbook
    rank_school = n * m * p
    U = np.zeros((n*m, rank_school))
    V = np.zeros((m*p, rank_school))
    W = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    best_rank = rank_school
    best_U, best_V, best_W = U, V, W
    
    # Target tensor
    # M[i, j, j', k] = 1 if j==j', else 0
    # Shape (n, m, m, p)
    target = np.zeros((n, m, m, p))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                target[i, j, j, k] = 1.0
                
    # Function to compute error and gradient
    def error_and_grad(x, R):
        # Reshape
        U = x[:n*m*R].reshape((n, m, R))
        V = x[n*m*R:n*m*R + m*p*R].reshape((m, p, R))
        W = x[n*m*R + m*p*R:].reshape((p, n, R))
        
        # Compute M
        # M[i, j, j', k] = sum_r W[k, i, r] U[i, j, r] V[j', k, r]
        M = np.einsum('kir,ijr,ar->ijja', W, U, V)
        
        # Error
        diff = M - target
        err = np.sum(diff**2)
        
        # Gradient
        # dE/dU[i,j,r] = 2 * sum_{k, j'} diff[i, j, j', k] * W[k, i, r] * V[j', k, r]
        # dE/dV[j',k,r] = 2 * sum_{i, j} diff[i, j, j', k] * W[k, i, r] * U[i, j, r]
        # dE/dW[k,i,r] = 2 * sum_{j, j'} diff[i, j, j', k] * U[i, j, r] * V[j', k, r]
        
        # Note: diff is M - target.
        # M[i, j, j', k]
        
        # Grad U
        # sum_{k, j'} diff[i, j, j', k] W[k, i, r] V[j', k, r]
        # We can use einsum for gradient too.
        # gU[i, j, r] = sum_{k, j'} diff[i, j, j', k] W[k, i, r] V[j', k, r]
        # Indices: diff(i,j,j',k), W(k,i,r), V(j',k,r)
        # Sum over k, j'. Output i, j, r.
        gU = 2 * np.einsum('ijjk,kir,jkr->ijr', diff, W, V)
        
        # gV[j', k, r] = sum_{i, j} diff[i, j, j', k] W[k, i, r] U[i, j, r]
        # Indices: diff(i,j,j',k), W(k,i,r), U(i,j,r)
        # Sum over i, j. Output j', k, r.
        gV = 2 * np.einsum('ijjk,kir,ijr->jkr', diff, W, U)
        
        # gW[k, i, r] = sum_{j, j'} diff[i, j, j', k] U[i, j, r] V[j', k, r]
        # Indices: diff(i,j,j',k), U(i,j,r), V(j',k,r)
        # Sum over j, j'. Output k, i, r.
        gW = 2 * np.einsum('ijjk,ijr,jkr->kir', diff, U, V)
        
        grad = np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
        
        return err, grad

    # Search for lower rank
    for R in range(25, rank_school):
        if time.time() > deadline - 5:
            break
            
        # Try a few restarts
        found = False
        for restart in range(3):
            if time.time() > deadline - 5:
                break
                
            # Initialize
            # Random small values
            x0 = np.random.randn(n*m*R + m*p*R + p*n*R) * 0.1
            
            res = minimize(error_and_grad, x0, args=(R,), jac=True, method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if res.success and res.fun < 1e-8:
                # Check rounding
                U_opt = res.x[:n*m*R].reshape((n, m, R))
                V_opt = res.x[n*m*R:n*m*R + m*p*R].reshape((m, p, R))
                W_opt = res.x[n*m*R + m*p*R:].reshape((p, n, R))
                
                # Round to half-integers
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                # Verify
                M_check = np.einsum('kir,ijr,ar->ijja', W_round, U_round, V_round)
                if np.allclose(M_check, target, atol=1e-6):
                    # Found valid solution
                    best_rank = R
                    best_U = U_round.flatten()
                    best_V = V_round.flatten()
                    best_W = W_round.flatten()
                    found = True
                    break
        
        if found:
            break
            
    # Reshape back to required format
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    if best_rank < rank_school:
        U_out = best_U.reshape((n*m, best_rank))
        V_out = best_V.reshape((m*p, best_rank))
        W_out = best_W.reshape((p*n, best_rank))
    else:
        U_out = best_U
        V_out = best_V
        W_out = best_W

    return {"U": U_out, "V": V_out, "W": W_out}
