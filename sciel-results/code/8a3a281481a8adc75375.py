import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s margin
    
    # Precompute indices for the tensor
    # i in 0..n-1, j in 0..m-1, k in 0..p-1
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    def residual(params):
        # params: flattened U, V, W
        # U: (n*m, R)
        # V: (m*p, R)
        # W: (p*n, R)
        R = len(params) // (n*m + m*p + p*n)
        if R == 0: return np.array([0])
        
        U = params[0:n*m*R].reshape(n*m, R)
        V = params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
        W = params[n*m*R + m*p*R:].reshape(p*n, R)
        
        # Compute sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # We want this to be 1 for all i,j,k
        
        # Efficient computation using einsum?
        # S[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # Note the indices are permuted in W.
        # W is indexed by (k, i) -> k*n + i.
        # So W[k,i,r] is W[k*n+i, r].
        
        # Let's compute T = U @ V.T @ W.T ? No.
        # For fixed i, j, k:
        # term = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # We can compute this for all i, j, k by iterating or using broadcasting.
        # Given small sizes, iteration is fine.
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    # dot product of U[u_idx,:], V[v_idx,:], W[w_idx,:]
                    # sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    # This is element-wise product then sum
                    res[idx] = np.dot(U[u_idx, :] * V[v_idx, :], W[w_idx, :])
                    idx += 1
        return res - 1.0

    best_solution = None
    
    # Try ranks from 53 down to 45
    for R in range(53, 44, -1):
        if time.time() > deadline: break
        
        # Number of variables
        num_vars = R * (n*m + m*p + p*n)
        
        # We will try a few random restarts
        for restart in range(20):
            if time.time() > deadline: break
            
            np.random.seed(restart * 100 + R)
            
            # Initialize
            U = np.random.randn(n*m, R) * 0.1
            V = np.random.randn(m*p, R) * 0.1
            W = np.random.randn(p*n, R) * 0.1
            
            # Gauge fixing: Set first element of U for each r to 1?
            # This might be too restrictive if the solution has U[0,0,r]=0.
            # Instead, let's just let the solver work, but the scaling symmetry is an issue.
            # We can add a small penalty for large norms?
            # Or fix the norm of each U[:, r] to 1?
            
            # Let's try fixing U[0, r] = 1 for all r.
            # If the optimal solution has U[0,r]=0, this will fail.
            # But for generic tensor decompositions, usually no entry is zero?
            # Actually, for matrix multiplication, the "standard" solution has sparse U, V, W.
            # The rank-deficient solutions might be denser.
            
            # Let's try without gauge fixing first, but with a regularization term?
            # No, least_squares minimizes the residual. The scaling symmetry means
            # if (U, V, W) is a solution, (c*U, V/c, W) is also a solution.
            # The residual is 0 for all scalings. So the gradient is 0 along the scaling direction.
            # This makes the Hessian singular.
            # We should fix the gauge.
            
            # Fix U[0, r] = 1 for all r.
            # This is valid if U[0, r] != 0.
            # If we get stuck, we can try other gauges.
            
            U[:, 0] = 1.0 # Fix first column of U? No, U is (nm, R). U[0, r] is row 0, col r.
            # Wait, U[i*m+j, r]. Row 0 is i=0, j=0.
            # So U[0, r] = 1.
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                sol = least_squares(residual, params, method='lm', max_nfev=1000)
            except:
                continue
                
            if sol.success or np.linalg.norm(sol.fun) < 1e-8:
                # Check if we can snap to rationals
                # Reconstruct U, V, W
                U_sol = sol.x[0:n*m*R].reshape(n*m, R)
                V_sol = sol.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W_sol = sol.x[n*m*R + m*p*R:].reshape(p*n, R)
                
                # Try to snap to nearest 0.5
                U_snap = np.round(U_sol * 2) / 2
                V_snap = np.round(V_sol * 2) / 2
                W_snap = np.round(W_sol * 2) / 2
                
                # Verify exact decomposition with snapped values
                # We need to check if sum_r U_snap[i,j,r] V_snap[j,k,r] W_snap[k,i,r] == 1
                # Use fractions for exactness? Or just high precision float?
                # Given the small size, float check with tight tolerance might suffice if we are confident.
                # But "exactly decompose" implies we should be careful.
                
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i*m + j
                            v_idx = j*p + k
                            w_idx = k*n + i
                            s = np.dot(U_snap[u_idx, :] * V_snap[v_idx, :], W_snap[w_idx, :])
                            if abs(s - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                
                if valid:
                    best_solution = (U_snap, V_snap, W_snap, R)
                    break # Found valid solution for this R, try lower R? 
                    # The task asks for rank-minimal. So we should try lower R.
                    # But we are iterating down. So we continue to lower R.
                    # Wait, if we found a solution for R=53, we should check R=52.
                    # So we don't break the outer loop.
            
            # If we found a valid solution, we can break the restart loop for this R?
            # No, we want the minimal rank. So we should check all restarts for this R?
            # Or just one? If one works, R is achievable.
            # We want the *minimal* R.
            # So if we find a valid solution for R, we can stop searching for higher R (which we already did)
            # and move to lower R.
            # So we break the restart loop.
            if best_solution and best_solution[3] == R:
                break

    if best_solution:
        U, V, W, R = best_solution
        # Convert to lists of lists as required
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
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
