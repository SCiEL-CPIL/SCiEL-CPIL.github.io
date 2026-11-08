import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0

    start_time = time.time()
    deadline = start_time + 100 # 100s deadline to be safe
    
    # Function to check if a decomposition is exact and half-integer
    def check_solution(U, V, W):
        # Check if entries are close to half-integers
        # Multiply by 2, check if close to integer
        U2 = U * 2
        V2 = V * 2
        W2 = W * 2
        
        # Check distance to nearest integer
        if np.max(np.abs(U2 - np.round(U2))) > 1e-5: return False
        if np.max(np.abs(V2 - np.round(V2))) > 1e-5: return False
        if np.max(np.abs(W2 - np.round(W2))) > 1e-5: return False
        
        # Snap to half-integers
        U_snap = np.round(U2) / 2.0
        V_snap = np.round(V2) / 2.0
        W_snap = np.round(W2) / 2.0
        
        # Verify exactness
        T_calc = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
        if not np.allclose(T_calc, T, atol=1e-9):
            return False
            
        return (U_snap, V_snap, W_snap)

    def objective_and_grad(x):
        R = len(x) // (nm + mp + pn) # This is wrong, R is fixed per call
        # We pass R as a closure variable or calculate from x length?
        # Better to define nested function with R captured.
        pass

    # We will try ranks 46, 47, 48, 49
    # 46 is the known minimum.
    
    best_result = None
    
    for R in [46, 47, 48, 49]:
        if time.time() > deadline:
            break
            
        # Number of variables: nm*R + mp*R + pn*R = R*(nm+mp+pn) = R*(10+25+10) = 45R
        # For R=46, 46*45 = 2070 variables.
        
        # Random restarts
        max_restarts = 20 # Limit restarts due to L-BFGS cost
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization
            # Scale matters. Start with small random values?
            # Or start from schoolbook-like structure?
            # Random normal is fine for L-BFGS.
            np.random.seed(42 + restart + R)
            x0 = np.random.randn(R * (nm + mp + pn)) * 0.1
            
            def fun_grad(x):
                # Reshape x
                U = x[:nm*R].reshape(nm, R)
                V = x[nm*R:nm*R+mp*R].reshape(mp, R)
                W = x[nm*R+mp*R:].reshape(pn, R)
                
                # Compute Residual R_tensor
                T_calc = np.einsum('ar,br,cr->abc', U, V, W)
                Res = T_calc - T
                
                # Objective
                val = np.sum(Res**2)
                
                # Gradients
                # dE/dU = 2 * einsum('abc,br,cr->ar', Res, V, W)
                grad_U = 2 * np.einsum('abc,br,cr->ar', Res, V, W)
                grad_V = 2 * np.einsum('abc,ar,cr->br', Res, U, W)
                grad_W = 2 * np.einsum('abc,ar,br->cr', Res, U, V)
                
                grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
                return val, grad
            
            try:
                result = opt.minimize(fun_grad, x0, method='L-BFGS-B', jac=True, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            except Exception:
                continue
                
            if result.fun < 1e-6:
                # Extract U, V, W
                U = result.x[:nm*R].reshape(nm, R)
                V = result.x[nm*R:nm*R+mp*R].reshape(mp, R)
                W = result.x[nm*R+mp*R:].reshape(pn, R)
                
                snap_result = check_solution(U, V, W)
                if snap_result:
                    best_result = (snap_result[0], snap_result[1], snap_result[2], R)
                    found = True
                    break
        
        if found:
            break

    if best_result:
        U, V, W, R = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook rank 50
        R = 50
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    alpha = i * m + j
                    beta = j * p + k
                    gamma = k * n + i
                    U[alpha, col] = 1
                    V[beta, col] = 1
                    W[gamma, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
