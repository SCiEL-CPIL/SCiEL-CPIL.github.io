"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Try various ranks and find the smallest that works
    # Start with a reasonable upper bound and try to go lower
    
    def try_rank(rank, max_time=100):
        """Try to find a decomposition with given rank."""
        deadline = start_time + max_time
        if time.time() > deadline:
            return None
        
        # Random initialization
        np.random.seed(42)
        
        # We want entries to be half-integers, so let's work in units of 0.5
        # i.e., multiply everything by 2, find integer solution, then divide by 2
        
        # Actually, let's just optimize in real space and then round
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Initialize with random values
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            # Optimize to minimize sum of squared errors
            def objective(params):
                U = params[:n*m*rank].reshape(n*m, rank)
                V = params[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
                W = params[n*m*rank+m*p*rank:].reshape(p*n, rank)
                
                # Compute error
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                            # Actually: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                            val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                            error += (val - 1.0)**2
                return error
            
            # This is too slow for scipy optimizer with the full tensor.
            # Let's use a different approach: gradient descent with vectorized computation
            
            # Precompute indices
            ii = np.array([i for i in range(n) for j in range(m) for k in range(p)])
            jj = np.array([j for i in range(n) for j in range(m) for k in range(p)])
            kk = np.array([k for i in range(n) for j in range(m) for k in range(p)])
            
            U_idx = ii * m + jj  # index into U
            V_idx = jj * p + kk  # index into V
            W_idx = kk * n + ii  # index into W
            
            def obj_and_grad(params):
                U = params[:n*m*rank].reshape(n*m, rank)
                V = params[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
                W = params[n*m*rank+m*p*rank:].reshape(p*n, rank)
                
                # Compute values: val_c = U[U_idx,c] * V[V_idx,c] * W[W_idx,c]
                # We need sum_c val_c = 1 for each (i,j,k)
                
                # Vectorized: for each (i,j,k) and each c, compute the product
                # This is memory intensive but let's try
                # num_entries = n*m*p = 84
                # For each entry, we have rank products to sum
                
                U_sel = U[U_idx]  # (84, rank)
                V_sel = V[V_idx]  # (84, rank)
                W_sel = W[W_idx]  # (84, rank)
                
                # Product: (84, rank)
                prod = U_sel * V_sel * W_sel
                vals = np.sum(prod, axis=1)  # (84,)
                
                error = np.sum((vals - 1.0)**2)
                
                # Gradient
                dvals = 2 * (vals - 1.0)  # (84,)
                dprod = dvals[:, None] * V_sel * W_sel  # dU part
                dU = np.zeros_like(U)
                np.add.at(dU, U_idx, dprod)
                
                dprod = dvals[:, None] * U_sel * W_sel
                dV = np.zeros_like(V)
                np.add.at(dV, V_idx, dprod)
                
                dprod = dvals[:, None] * U_sel * V_sel
                dW = np.zeros_like(W)
                np.add.at(dW, W_idx, dprod)
                
                grad = np.concatenate([dU.ravel(), dV.ravel(), dW.ravel()])
                return error, grad
            
            # Use L-BFGS-B
            x0 = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
            
            try:
                res = minimize(obj_and_grad, x0, method='L-BFGS-B', jac=True, 
                              options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            except:
                continue
            
            if time.time() > deadline:
                break
            
            U_opt = res.x[:n*m*rank].reshape(n*m, rank)
            V_opt = res.x[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
            W_opt = res.x[n*m*rank+m*p*rank:].reshape(p*n, rank)
            
            # Round to nearest 0.5
            U_round = np.round(U_opt * 2) / 2
            V_round = np.round(V_opt * 2) / 2
            W_round = np.round(W_opt * 2) / 2
            
            # Verify
            U_sel = U_round[U_idx]
            V_sel = V_round[V_idx]
            W_sel = W_round[W_idx]
            vals = np.sum(U_sel * V_sel * W_sel, axis=1)
            
            if np.allclose(vals, 1.0, atol=1e-9):
                # Found a valid decomposition!
                # Convert to list of lists
                U_list = U_round.tolist()
                V_list = V_round.tolist()
                W_list = W_round.tolist()
                return U_list, V_list, W_list
        
        return None
    
    # Try ranks from low to high
    # Lower bound: at least max(n*m, m*p, p*n) = max(12, 28, 21) = 28? No, that's not right.
    # The rank must be at least the max of the three dimensions... actually no.
    # Let's try from a reasonable range
    
    best_result = None
    for rank in range(20, 85):
        if time.time() > start_time + 110:
            break
        result = try_rank(rank, max_time=110 - (time.time() - start_time))
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        U, V, W = best_result
        return {"U": U, "V": V, "W": W}
    else:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
