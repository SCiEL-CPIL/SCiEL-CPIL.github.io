import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Build target tensor (sparse)
    # T[alpha, beta, gamma] where alpha in [0, n*m), beta in [0, m*p), gamma in [0, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    target_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_entries.append((i*m+j, j*p+k, k*n+i))
    
    # Schoolbook
    rank_sb = n*m*p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    for idx, (a, b, c) in enumerate(target_entries):
        U_sb[a, idx] = 1
        V_sb[b, idx] = 1
        W_sb[c, idx] = 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank using optimization
    def tensor_error(params, rank):
        U = params[:n*m*rank].reshape(n*m, rank)
        V = params[n*m*rank:2*n*m*rank + m*p*rank - n*m*rank].reshape(m*p, rank)
        # This indexing is wrong, let me fix
        ...
    
    # This is getting complicated. Let me try a simpler approach.
    
    # For each target rank, try random initialization + ALS
    for target_rank in range(85, 30, -1):
        if time.time() > deadline - 30:
            break
        
        found = False
        for trial in range(3):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(target_rank * 1000 + trial)
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # ALS iterations
            for it in range(100):
                if time.time() > deadline - 30:
                    break
                
                # Update U: for each column r, we want to minimize the error
                # The contribution of term r to the tensor is U[:,r] ⊗ V[:,r] ⊗ W[:,r]
                # We want sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = T
                # For fixed V, W: minimize ||sum_r U[:,r] ⊗ (V[:,r] ⊗ W[:,r]) - T||
                # This is a linear problem in U if we vectorize properly.
                
                # Construct the "effective" target for U:
                # For each (alpha, beta, gamma) in target, T[alpha,beta,gamma] = 1
                # The r-th term contributes U[alpha,r] * V[beta,r] * W[gamma,r]
                # So for fixed V,W, the equation is: sum_r U[alpha,r] * (V[beta,r] * W[gamma,r]) = T[alpha,beta,gamma]
                # This is a linear system in U for each (beta,gamma) pair.
                
                # But we only have equations for the nonzero entries of T.
                # Let's build the linear system.
                
                # For each target entry (a, b, c): sum_r U[a,r] * V[b,r] * W[c,r] = 1
                # And for all other (a',b',c'): sum_r U[a',r] * V[b',r] * W[c',r] = 0
                
                # This is a huge system. Let's just use the nonzero entries for the ALS step.
                
                # Build matrix M where M[(a, (b,c)), r] = V[b,r] * W[c,r]
                # Then M @ U_flat = t where t is the target vector
                # But this is over-determined and we only use nonzero entries.
                
                # For ALS, update one factor at a time:
                # Update U: for each row a, solve least squares
                # The equations involving U[a, :]: for each (b,c) such that (a,b,c) is a target entry:
                #   sum_r U[a,r] * V[b,r] * W[c,r] = 1
                # Plus for each (b,c) such that (a,b,c) is NOT a target entry:
                #   sum_r U[a,r] * V[b,r] * W[c,r] = 0
                
                # This is still huge. Let me just do a simple gradient step.
                
                # Compute current tensor
                # T_current[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # Error = T_current - T_target
                # Gradient w.r.t. U[a,r] = sum_{b,c} error[a,b,c] * V[b,r] * W[c,r]
                
                # This is too expensive to compute fully. Let me sample.
                
                # Actually, for small dimensions, let me just compute the full tensor.
                # n*m = 15, m*p = 30, p*n = 18, rank up to 85
                # Full tensor is 15*30*18 = 8100 entries, rank 85
                # Computing T_current: for each a,b,c: dot(U[a,:], V[b,:]*W[c,:]) 
                # That's 8100 * 85 = 688500 operations, feasible.
                
                # Build full tensor
                T_current = np.zeros((n*m, m*p, p*n))
                for r in range(target_rank):
                    T_current += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                
                # Build target tensor
                T_target = np.zeros((n*m, m*p, p*n))
                for (a, b, c) in target_entries:
                    T_target[a, b, c] = 1
                
                error = T_current - T_target
                err_norm = np.linalg.norm(error)
                
                if err_norm < 1e-10:
                    best_rank = target_rank
                    best_U, best_V, best_W = U, V, W
                    found = True
                    break
                
                # Gradient step
                lr = 0.01
                # grad_U[a,r] = sum_{b,c} error[a,b,c] * V[b,r] * W[c,r]
                grad_U = np.einsum('abc,br,cr->ar', error, V, W)
                grad_V = np.einsum('abc,ar,cr->br', error, U, W)
                grad_W = np.einsum('abc,ar,br->cr', error, U, V)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
            
            if found:
                break
        
        if found:
            break
    
    # Verify best solution
    U_final = best_U
    V_final = best_V
    W_final = best_W
    
    # Convert to lists
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
