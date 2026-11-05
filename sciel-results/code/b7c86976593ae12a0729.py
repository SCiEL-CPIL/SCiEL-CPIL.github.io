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
        """Verify exact decomposition using Fractions for the FULL tensor"""
        # U_mat: (n*m, rank)
        # V_mat: (m*p, rank)
        # W_mat: (p*n, rank)
        
        # We need to check T[i,j,j',k,k',i'] = sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r]
        # Target: 1 if i==i', j==j', k==k', else 0.
        
        # To speed up, we can compute the full tensor using loops or einsum on Fractions?
        # Fractions are slow.
        # Since rank is small, we can iterate.
        
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                s = Fraction(0)
                                for r in range(rank):
                                    s += U_mat[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r]
                                target = 1 if (i == ip and j == jp and k == kp) else 0
                                if s != target:
                                    return False
        return True

    def compute_full_residuals(params, rank):
        """
        Compute residuals for least_squares.
        Residuals are the errors for each entry in the full tensor.
        """
        U_f = params[:n*m*rank].reshape(n*m, rank)
        V_f = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W_f = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        # Compute T_full[i,j,jp,k,kp,ip]
        # T_full = sum_r U[i,j,r] V[jp,k,r] W[kp,ip,r]
        
        # We can build this tensor
        T_full = np.zeros((n, m, m, p, p, n))
        
        for r in range(rank):
            u_r = U_f[:, r].reshape(n, m)
            v_r = V_f[:, r].reshape(m, p)
            w_r = W_f[:, r].reshape(p, n)
            
            # T_r[i,j,jp,k,kp,ip] = u_r[i,j] * v_r[jp,k] * w_r[kp,ip]
            # We can use einsum
            # 'ij,jp,kp->ijk' doesn't work directly for 6D output from 3 inputs
            # We can do:
            # T_r = np.einsum('ij,jp,kp->ijk', u_r, v_r, w_r) ? No, indices don't match.
            
            # Let's just use broadcasting
            # u_r[i,j, 1, 1, 1, 1] * v_r[1, jp, k, 1, 1, 1] * w_r[1, 1, 1, kp, ip, 1]?
            # u_r shape (n, m)
            # v_r shape (m, p)
            # w_r shape (p, n)
            
            # u_r[i, j] -> (n, m, 1, 1, 1, 1)
            # v_r[jp, k] -> (1, m, p, 1, 1, 1)
            # w_r[kp, ip] -> (1, 1, 1, p, n, 1)
            
            term = np.einsum('ij,jp,kp->ijk', u_r, v_r, w_r) # This is wrong, it contracts.
            
            # Correct way:
            u_reshaped = u_r[:, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
            v_reshaped = v_r[np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis]
            w_reshaped = w_r[np.newaxis, np.newaxis, np.newaxis, :, :, np.newaxis]
            
            T_full += u_reshaped * v_reshaped * w_reshaped
            
        # Reshape T_full to 1D
        T_flat = T_full.flatten()
        
        # Target tensor
        target = np.zeros((n, m, m, p, p, n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i, j, j, k, k, i] = 1.0
        
        target_flat = target.flatten()
        
        residuals = T_flat - target_flat
        return residuals

    def try_rank(R, seed):
        """Try to find a rank R decomposition"""
        np.random.seed(seed)
        init_U = np.random.randn(n*m, R) * 0.5
        init_V = np.random.randn(m*p, R) * 0.5
        init_W = np.random.randn(p*n, R) * 0.5
        
        init_params = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
        
        if time.time() > deadline - 10:
            return None

        try:
            res = scipy.optimize.least_squares(
                compute_full_residuals, init_params, args=(R,), 
                method='lm', max_nfev=1000
            )
        except Exception:
            return None

        if res.cost < 1e-8:
            U_f = res.x[:n*m*R].reshape(n*m, R)
            V_f = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
            W_f = res.x[n*m*R + m*p*R:].reshape(p*n, R)
            
            U_rat = np.zeros((n*m, R), dtype=object)
            V_rat = np.zeros((m*p, R), dtype=object)
            W_rat = np.zeros((p*n, R), dtype=object)
            
            def snap(val):
                # Try to find a simple fraction
                # limit_denominator is a good heuristic
                return Fraction(val).limit_denominator(100)
            
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
    
    # Search ranks from low to high, but start close to schoolbook to be safe
    # If we find 35, we stop.
    for R in range(35, 19, -1):
        if time.time() > deadline - 20:
            break
            
        found = False
        for seed in range(20):
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
