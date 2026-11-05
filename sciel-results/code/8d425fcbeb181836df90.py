import time
import numpy as np
from fractions import Fraction
import scipy.optimize

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

    def compute_tensor_and_grad(params, rank):
        """
        params: flat array of size (n*m + m*p + p*n) * rank
        Returns: error, gradients
        """
        U_f = params[:n*m*rank].reshape(n*m, rank)
        V_f = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W_f = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        # Compute T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We can do this efficiently by iterating over r and using outer products?
        # Or just loops since n,m,p are small.
        
        T = np.zeros((n, m, p))
        for r in range(rank):
            u = U_f[:, r]
            v = V_f[:, r]
            w = W_f[:, r]
            # u is (n*m, 1), v is (m*p, 1), w is (p*n, 1)
            # We need to map indices
            # u[i*m+j], v[j*p+k], w[k*n+i]
            
            # Reshape u to (n, m)
            u_2d = u.reshape(n, m)
            # Reshape v to (m, p)
            v_2d = v.reshape(m, p)
            # Reshape w to (p, n)
            w_2d = w.reshape(p, n)
            
            # T contribution from r: T_r[i,j,k] = u_2d[i,j] * v_2d[j,k] * w_2d[k,i]
            # This is not a simple matrix multiplication.
            # We can loop over i, j, k or use einsum.
            # np.einsum('ij,jk,ki->ijk', u_2d, v_2d, w_2d)
            T += np.einsum('ij,jk,ki->ijk', u_2d, v_2d, w_2d)
            
        err = np.sum((T - 1.0)**2)
        
        # Gradients
        dU = np.zeros_like(U_f)
        dV = np.zeros_like(V_f)
        dW = np.zeros_like(W_f)
        
        # diff[i,j,k] = 2 * (T[i,j,k] - 1)
        diff = 2.0 * (T - 1.0)
        
        for r in range(rank):
            u = U_f[:, r]
            v = V_f[:, r]
            w = W_f[:, r]
            
            u_2d = u.reshape(n, m)
            v_2d = v.reshape(m, p)
            w_2d = w.reshape(p, n)
            
            # dU[i*m+j, r] = sum_{k} diff[i,j,k] * v_2d[j,k] * w_2d[k,i]
            # dV[j*p+k, r] = sum_{i} diff[i,j,k] * u_2d[i,j] * w_2d[k,i]
            # dW[k*n+i, r] = sum_{j} diff[i,j,k] * u_2d[i,j] * v_2d[j,k]
            
            # Using einsum for gradients
            # dU_2d[i,j] = sum_k diff[i,j,k] * v_2d[j,k] * w_2d[k,i]
            dU_2d = np.einsum('ijk,jk,ki->ij', diff, v_2d, w_2d)
            # dV_2d[j,k] = sum_i diff[i,j,k] * u_2d[i,j] * w_2d[k,i]
            dV_2d = np.einsum('ijk,ij,ki->jk', diff, u_2d, w_2d)
            # dW_2d[k,i] = sum_j diff[i,j,k] * u_2d[i,j] * v_2d[j,k]
            dW_2d = np.einsum('ijk,ij,jk->ki', diff, u_2d, v_2d)
            
            dU[:, r] = dU_2d.flatten()
            dV[:, r] = dV_2d.flatten()
            dW[:, r] = dW_2d.flatten()
            
        grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        return err, grad

    def try_rank(R, seed):
        """Try to find a rank R decomposition"""
        np.random.seed(seed)
        init_U = np.random.randn(n*m, R) * 0.5
        init_V = np.random.randn(m*p, R) * 0.5
        init_W = np.random.randn(p*n, R) * 0.5
        
        init_params = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
        
        def fun(x):
            err, _ = compute_tensor_and_grad(x, R)
            return err
            
        def jac(x):
            _, grad = compute_tensor_and_grad(x, R)
            return grad

        if time.time() > deadline - 10:
            return None

        try:
            res = scipy.optimize.minimize(
                fun, init_params, jac=jac, method='L-BFGS-B', 
                options={'maxiter': 300, 'gtol': 1e-8, 'disp': False}
            )
        except Exception:
            return None

        if res.fun < 1e-6:
            U_f = res.x[:n*m*R].reshape(n*m, R)
            V_f = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
            W_f = res.x[n*m*R + m*p*R:].reshape(p*n, R)
            
            U_rat = np.zeros((n*m, R), dtype=object)
            V_rat = np.zeros((m*p, R), dtype=object)
            W_rat = np.zeros((p*n, R), dtype=object)
            
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
    for R in range(30, 36):
        if time.time() > deadline - 20:
            break
            
        found = False
        for seed in range(50):
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
        return {"U": [[float(x) for x in row] for row in U_school], 
                "V": [[float(x) for x in row] for row in V_school], 
                "W": [[float(x) for x in row] for row in W_school]}
