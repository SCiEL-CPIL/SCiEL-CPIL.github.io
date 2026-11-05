import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 8]

def _build_constraint_matrices(n, m, p):
    """
    Build the constraint matrices for the tensor decomposition.
    
    The constraint is: for all (i', j', j'', k', i, k):
        sum_r U[i'*m+j', r] * V[j''*p+k', r] * W[k*n+i, r] = delta_{i',i} * delta_{j',j''} * delta_{k',k}
    
    We can rewrite this as: for each (i, k), we need:
        sum_{i',j',j'',k'} U[i'*m+j', r] * V[j''*p+k', r] * W[k*n+i, r] * A[i',j'] * B[j'',k'] = C[i,k]
    
    which means for the tensor T[i,k, i',j',j'',k'] = delta_{i',i}*delta_{j',j''}*delta_{k',k}:
        sum_r U[i'*m+j', r] * V[j''*p+k', r] * W[k*n+i, r] = T[i,k, i',j',j'',k']
    """
    # Number of constraints: n * p * n * m * m * p = n^2 * m^2 * p^2
    num_constraints = n * p * n * m * m * p
    # Number of unknowns per factor: 
    # U has n*m*R unknowns, V has m*p*R unknowns, W has p*n*R unknowns
    
    # For solving U given V, W:
    # For each (i', j'), we have R unknowns U[i'*m+j', 0:R]
    # The constraints involving U[i'*m+j', r] are for all (i, k, j'', k')
    # where the coefficient is V[j''*p+k', r] * W[k*n+i, r]
    # and the RHS is delta_{i',i} * delta_{j',j''} * delta_{k',k}
    
    # Let's build the system for U.
    # Unknowns: U flattened to (n*m*R,)
    # For constraint (i, k, j'', k'), and for each (i', j'):
    #   sum_r U[i'*m+j', r] * V[j''*p+k', r] * W[k*n+i, r] = delta_{i',i} * delta_{j',j''} * delta_{k',k}
    
    # Actually, let me think about this differently.
    # The full constraint is a 6-tensor T[i,k, i',j',j'',k'] that must equal
    # sum_r U[i'*m+j', r] * V[j''*p+k', r] * W[k*n+i, r]
    
    # This is a trilinear form. For alternating least squares:
    # Fix V and W, solve for U: this is linear in U.
    # For each (i', j'), the unknowns are U[i'*m+j', 0:R].
    # The constraints are: for all (i, k, j'', k'):
    #   sum_r U[i'*m+j', r] * V[j''*p+k', r] * W[k*n+i, r] = delta_{i',i} * delta_{j',j''} * delta_{k',k}
    
    # So for fixed (i', j'), we have a system of (n * p * m * p) equations in R unknowns.
    # But wait, that's not right either. Let me re-derive.
    
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We want: C[i,k] = sum_r f_r(A) * g_r(B) * W_r[k,i]
    # where f_r(A) = sum_{i',j'} U[i'*m+j', r] * A[i',j']
    # and g_r(B) = sum_{j'',k'} V[j''*p+k', r] * B[j'',k']
    
    # So: sum_r (sum_{i',j'} U[i'*m+j', r] * A[i',j']) * (sum_{j'',k'} V[j''*p+k', r] * B[j'',k']) * W_r[k,i]
    # = sum_{i',j',j'',k'} U[i'*m+j', r] * V[j''*p+k', r] * W_r[k,i] * A[i',j'] * B[j'',k']
    
    # This should equal sum_j A[i,j] * B[j,k] for all i,k.
    # = sum_{i',j',j'',k'} delta_{i',i} * delta_{j',j''} * delta_{k',k} * A[i',j'] * B[j'',k']
    
    # So we need: for all (i, k, i', j', j'', k'):
    #   sum_r U[i'*m+j', r] * V[j''*p+k', r] * W[k*n+i, r] = delta_{i',i} * delta_{j',j''} * delta_{k',k}
    
    # For ALS solving U:
    # Fix V and W. For each (i', j'), we need to find U[i'*m+j', 0:R] such that:
    #   for all (i, k, j'', k'):
    #     sum_r U[i'*m+j', r] * V[j''*p+k', r] * W[k*n+i, r] = delta_{i',i} * delta_{j',j''} * delta_{k',k}
    
    # This is a system of (n * p * m * p) equations in R unknowns for each (i', j').
    # But these equations are coupled across (i', j') because they share the same (i, k, j'', k').
    
    # Actually, no. For a fixed (i', j'), the equations only involve U[i'*m+j', :].
    # So we can solve each (i', j') independently!
    
    # For fixed (i', j'), the system is:
    #   for all (i, k, j'', k'):
    #     sum_r U[i'*m+j', r] * (V[j''*p+k', r] * W[k*n+i, r]) = delta_{i',i} * delta_{j',j''} * delta_{k',k}
    
    # Let M[i',j', i,k,j'',k', r] = V[j''*p+k', r] * W[k*n+i, r]
    # Then: sum_r U[i'*m+j', r] * M[i',j', i,k,j'',k', r] = b[i',j', i,k,j'',k']
    # where b[i',j', i,k,j'',k'] = delta_{i',i} * delta_{j',j''} * delta_{k',k}
    
    # This is: U[i'*m+j', :] @ M[i',j', i,k,j'',k', :] = b[i',j', i,k,j'',k']
    
    # So we can form a matrix A_{i',j'} of shape (n*p*m*p, R) where
    # A_{i',j'}[idx, r] = V[j''*p+k', r] * W[k*n+i, r]
    # and b_{i',j'} of shape (n*p*m*p,) where
    # b_{i',j'}[idx] = delta_{i',i} * delta_{j',j''} * delta_{k',k}
    
    # Then solve A_{i',j'} @ U[i'*m+j', :] = b_{i',j'}
    
    return n, m, p

def _solve_for_U(U, V, W, n, m, p, R):
    """Solve for U given V and W using least squares."""
    U_new = np.zeros((n*m, R))
    for ip in range(n):
        for jp in range(m):
            # Build the system for U[ip*m+jp, :]
            # Unknowns: R
            # Equations: for all (i, k, jpp, kp)
            num_eq = n * p * m * p
            A = np.zeros((num_eq, R))
            b = np.zeros(num_eq)
            
            idx = 0
            for i in range(n):
                for k in range(p):
                    for jpp in range(m):
                        for kp in range(p):
                            for r in range(R):
                                A[idx, r] = V[jpp*p+kp, r] * W[k*n+i, r]
                            b[idx] = 1.0 if (ip == i and jp == jpp and kp == k) else 0.0
                            idx += 1
            
            # Solve least squares
            try:
                sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                U_new[ip*m+jp, :] = sol
            except:
                U_new[ip*m+jp, :] = U[ip*m+jp, :]
    
    return U_new

def _solve_for_V(U, V, W, n, m, p, R):
    """Solve for V given U and W using least squares."""
    V_new = np.zeros((m*p, R))
    for jp in range(m):
        for kp in range(p):
            num_eq = n * p * n * m
            A = np.zeros((num_eq, R))
            b = np.zeros(num_eq)
            
            idx = 0
            for ip in range(n):
                for i in range(n):
                    for jpp in range(m):
                        for k in range(p):
                            for r in range(R):
                                A[idx, r] = U[ip*m+jp, r] * W[k*n+i, r]
                            b[idx] = 1.0 if (ip == i and jp == jpp and kp == k) else 0.0
                            idx += 1
            
            try:
                sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                V_new[jp*p+kp, :] = sol
            except:
                V_new[jp*p+kp, :] = V[jp*p+kp, :]
    
    return V_new

def _solve_for_W(U, V, W, n, m, p, R):
    """Solve for W given U and V using least squares."""
    W_new = np.zeros((p*n, R))
    for k in range(p):
        for i in range(n):
            num_eq = n * p * m * p
            A = np.zeros((num_eq, R))
            b = np.zeros(num_eq)
            
            idx = 0
            for ip in range(n):
                for jp in range(m):
                    for jpp in range(m):
                        for kp in range(p):
                            for r in range(R):
                                A[idx, r] = U[ip*m+jp, r] * V[jpp*p+kp, r]
                            b[idx] = 1.0 if (ip == i and jp == jpp and kp == k) else 0.0
                            idx += 1
            
            try:
                sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                W_new[k*n+i, :] = sol
            except:
                W_new[k*n+i, :] = W[k*n+i, :]
    
    return W_new

def _compute_residual(U, V, W, n, m, p):
    """Compute the residual of the tensor decomposition."""
    R = U.shape[1]
    residual = 0.0
    for ip in range(n):
        for jp in range(m):
            for jpp in range(m):
                for kp in range(p):
                    for i in range(n):
                        for k in range(p):
                            val = np.dot(U[ip*m+jp, :], V[jpp*p+kp, :] * W[k*n+i, :])
                            target = 1.0 if (ip == i and jp == jpp and kp == k) else 0.0
                            residual += (val - target) ** 2
    return residual

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition is valid."""
    R = U.shape[1]
    for ip in range(n):
        for jp in range(m):
            for jpp in range(m):
                for kp in range(p):
                    for i in range(n):
                        for k in range(p):
                            val = np.dot(U[ip*m+jp, :], V[jpp*p+kp, :] * W[k*n+i, :])
                            target = 1.0 if (ip == i and jp == jpp and kp == k) else 0.0
                            if abs(val - target) > tol:
                                return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition with rank n*m*p."""
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
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    rank_seed = n * m * p  # 96
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Start with schoolbook
    U_best, V_best, W_best = _schoolbook(n, m, p)
    best_rank = rank_seed
    
    # Try to find lower rank decompositions using ALS
    for R in range(rank_seed - 1, max(50, rank_seed - 30), -1):
        if time.time() > deadline - 10.0:
            break
        
        found = False
        for trial in range(5):
            if time.time() > deadline - 5.0:
                break
            
            np.random.seed(42 + R * 100 + trial)
            U = np.random.randn(n*m, R) * 0.5
            V = np.random.randn(m*p, R) * 0.5
            W = np.random.randn(p*n, R) * 0.5
            
            prev_res = np.inf
            for it in range(50):
                if time.time() > deadline - 5.0:
                    break
                
                U = _solve_for_U(U, V, W, n, m, p, R)
                V = _solve_for_V(U, V, W, n, m, p, R)
                W = _solve_for_W(U, V, W, n, m, p, R)
                
                res = _compute_residual(U, V, W, n, m, p)
                if abs(prev_res - res) < 1e-10:
                    break
                prev_res = res
            
            if prev_res < 1e-8 and _verify(U, V, W, n, m, p, tol=1e-8):
                U_best, V_best, W_best = U, V, W
                best_rank = R
                found = True
                break
        
        if found:
            # Found a valid decomposition at rank R, try lower
            continue
    
    # Round to half-integers if possible
    def _round_to_half(x):
        return np.round(x * 2) / 2
    
    U_rounded = _round_to_half(U_best)
    V_rounded = _round_to_half(V_best)
    W_rounded = _round_to_half(W_best)
    
    if _verify(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-6):
        U_best, V_best, W_best = U_rounded, V_rounded, W_rounded
    
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
