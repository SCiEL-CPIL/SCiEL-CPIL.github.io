import time
import numpy as np
from fractions import Fraction
import scipy.optimize
import itertools

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36

    # Build schoolbook decomposition as fallback
    U_school = np.zeros((n * m, schoolbook_rank), dtype=object)
    V_school = np.zeros((m * p, schoolbook_rank), dtype=object)
    W_school = np.zeros((p * n, schoolbook_rank), dtype=object)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = Fraction(1)
                V_school[j * p + k, col] = Fraction(1)
                W_school[k * n + i, col] = Fraction(1)
                col += 1

    def verify_exact(U_mat, V_mat, W_mat, rank):
        """Verify exact decomposition using Fractions"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if s != Fraction(1):
                        return False
        return True

    def compute_tensor(U_f, V_f, W_f, rank):
        """Compute the resulting tensor T[i,j,k]"""
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    T[i,j,k] = np.dot(U_f[i*m+j, :], V_f[j*p+k, :] * W_f[k*n+i, :])
        return T

    def objective_and_grad(params, rank):
        """
        params: flat array of size (n*m + m*p + p*n) * rank
        Returns: error, gradients
        """
        U_f = params[:n*m*rank].reshape(n*m, rank)
        V_f = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W_f = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        T = compute_tensor(U_f, V_f, W_f, rank)
        err = np.sum((T - 1.0)**2)
        
        # Gradient computation
        dU = np.zeros_like(U_f)
        dV = np.zeros_like(V_f)
        dW = np.zeros_like(W_f)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    diff = 2.0 * (T[i,j,k] - 1.0)
                    dU[i*m+j, :] += diff * V_f[j*p+k, :] * W_f[k*n+i, :]
                    dV[j*p+k, :] += diff * U_f[i*m+j, :] * W_f[k*n+i, :]
                    dW[k*n+i, :] += diff * U_f[i*m+j, :] * V_f[j*p+k, :]
        
        grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        return err, grad

    def try_rank(R, seed):
        """Try to find a rank R decomposition"""
        np.random.seed(seed)
        # Initialize with small random values
        init_U = np.random.randn(n*m, R) * 0.5
        init_V = np.random.randn(m*p, R) * 0.5
        init_W = np.random.randn(p*n, R) * 0.5
        
        init_params = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
        
        # Use L-BFGS-B
        # Note: L-BFGS-B might be slow if we call the python gradient function too often.
        # But it's more robust than manual momentum.
        
        def fun(x):
            err, _ = objective_and_grad(x, R)
            return err
            
        def jac(x):
            _, grad = objective_and_grad(x, R)
            return grad

        # We need to be careful with time.
        # Let's limit iterations.
        
        # Check time
        if time.time() > deadline - 10:
            return None

        try:
            res = scipy.optimize.minimize(
                fun, init_params, jac=jac, method='L-BFGS-B', 
                options={'maxiter': 200, 'gtol': 1e-8, 'disp': False}
            )
        except Exception:
            return None

        if res.success and res.fun < 1e-6:
            # Snap to rationals
            U_f = res.x[:n*m*R].reshape(n*m, R)
            V_f = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
            W_f = res.x[n*m*R + m*p*R:].reshape(p*n, R)
            
            # Try to snap to fractions with small denominators
            # We'll try denominators 1, 2, 3, 4
            U_rat = np.zeros((n*m, R), dtype=object)
            V_rat = np.zeros((m*p, R), dtype=object)
            W_rat = np.zeros((p*n, R), dtype=object)
            
            # Helper to snap
            def snap(val):
                best_frac = None
                best_err = float('inf')
                for d in [1, 2, 3, 4, 5, 6]:
                    num = round(val * d)
                    frac = Fraction(num, d)
                    err = abs(float(frac) - val)
                    if err < best_err:
                        best_err = err
                        best_frac = frac
                return best_frac

            for i in range(n*m):
                for r in range(R):
                    U_rat[i,r] = snap(U_f[i,r])
            for i in range(m*p):
                for r in range(R):
                    V_rat[i,r] = snap(V_f[i,r])
            for i in range(p*n):
                for r in range(R):
                    W_rat[i,r] = snap(W_f[i,r])
            
            if verify_exact(U_rat, V_rat, W_rat, R):
                return U_rat, V_rat, W_rat, R
        
        return None

    best_result = None
    
    # Search ranks from low to high
    # 3x3x4 rank is likely around 25-30.
    # Try 25, 26, 27...
    for R in range(25, 36):
        if time.time() > deadline - 20:
            break
            
        found = False
        for seed in range(20): # 20 seeds per rank
            if time.time() > deadline - 5:
                break
                
            result = try_rank(R, seed)
            if result:
                U_rat, V_rat, W_rat, rank = result
                best_result = (U_rat, V_rat, W_rat, rank)
                found = True
                break
        
        if found:
            break

    if best_result:
        U_rat, V_rat, W_rat, rank = best_result
        return {"U": [[float(x) for x in row] for row in U_rat], 
                "V": [[float(x) for x in row] for row in V_rat], 
                "W": [[float(x) for x in row] for row in W_rat]}
    else:
        # Fallback to schoolbook
        return {"U": [[float(x) for x in row] for row in U_school], 
                "V": [[float(x) for x in row] for row in V_school], 
                "W": [[float(x) for x in row] for row in W_school]}
