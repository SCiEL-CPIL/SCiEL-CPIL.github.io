"""Combined approach: structured initialization + efficient ALS + polish for 2x5x5 matrix multiplication tensor."""

import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    n, m, p = 2, 5, 5
    nm = n * m  # 10
    mp = m * p  # 25
    pn = p * n  # 10
    n_constraints = n * m * p  # 50
    
    # Build schoolbook decomposition (rank 50)
    def build_schoolbook():
        R = n * m * p
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    best_solution = build_schoolbook()
    best_rank = n * m * p
    
    # Vectorized ALS solvers - solve for one factor given the other two
    def solve_U(V, W, R):
        # V: (mp, R) -> (m, p, R)
        # W: (pn, R) -> (p, n, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        
        # We need M[i, j, k, r] = V[j, k, r] * W[k, i, r]
        # V_3d shape (m, p, R) -> broadcast to (1, m, p, R)
        # W_3d shape (p, n, R) -> broadcast to (n, 1, p, R)
        V_b = V_3d.reshape(1, m, p, R)
        W_b = W_3d.reshape(n, 1, p, R)
        
        M = V_b * W_b # Shape (n, m, p, R)
        M_flat = M.reshape(n*m, p, R)
        
        U = np.zeros((n*m, R))
        b = np.ones(p)
        
        for idx in range(n*m):
            A = M_flat[idx] # Shape (p, R)
            # Solve A.T @ u = b
            try:
                U[idx] = np.linalg.lstsq(A.T, b, rcond=None)[0]
            except:
                U[idx] = np.zeros(R)
                
        return U
    
    def solve_V(U, W, R):
        # U: (nm, R) -> (n, m, R)
        # W: (pn, R) -> (p, n, R)
        U_3d = U.reshape(n, m, R)
        W_3d = W.reshape(p, n, R)
        
        # We need M[j, k, i, r] = U[i, j, r] * W[k, i, r]
        # U_3d shape (n, m, R) -> broadcast to (n, 1, m, R) ? No.
        # We want to index by j, k.
        # M[j, k, i, r] = U[i, j, r] * W[k, i, r]
        
        # U[i, j, r] -> U_3d[i, j, r]. We want to move j to first dim?
        # Let's construct M of shape (m, p, n, R)
        # U_3d is (n, m, R). Transpose to (m, n, R)? No, we need r last.
        # U_3d.T is (R, m, n).
        
        # Let's use broadcasting carefully.
        # U_3d[i, j, r]
        # W_3d[k, i, r]
        
        # Reshape U to (1, m, n, R) ? No, U is (n, m, R).
        # U_reshaped = U_3d.T.reshape(1, m, n, R) ?
        # U_3d.T is (R, m, n).
        # We want U[i, j, r] to be accessible as U_b[j, i, r] effectively for the product.
        
        # Let's try:
        # U_b shape (1, m, n, R) containing U[i, j, r] at [0, j, i, r]?
        # U_3d is (n, m, R).
        # U_3d.transpose(1, 0, 2) is (m, n, R).
        # Reshape to (1, m, n, R).
        
        U_b = U_3d.transpose(1, 0, 2).reshape(1, m, n, R)
        
        # W_3d is (p, n, R).
        # We want W[k, i, r] at [k, i, r].
        # Reshape to (p, 1, n, R).
        W_b = W_3d.reshape(p, 1, n, R)
        
        M = U_b * W_b # Shape (p, m, n, R)
        # We need M[j, k, i, r]. Current M is [k, j, i, r].
        M_trans = M.transpose(1, 0, 2, 3) # Shape (m, p, n, R)
        M_flat = M_trans.reshape(m*p, n, R)
        
        V = np.zeros((m*p, R))
        b = np.ones(n)
        
        for idx in range(m*p):
            A = M_flat[idx] # Shape (n, R)
            # Solve A.T @ v = b
            try:
                V[idx] = np.linalg.lstsq(A.T, b, rcond=None)[0]
            except:
                V[idx] = np.zeros(R)
                
        return V
    
    def solve_W(U, V, R):
        # U: (nm, R) -> (n, m, R)
        # V: (mp, R) -> (m, p, R)
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        
        # We need M[k, i, j, r] = U[i, j, r] * V[j, k, r]
        
        # U_3d is (n, m, R).
        # We want U[i, j, r] at [i, j, r].
        # Reshape to (n, m, 1, R) ?
        U_b = U_3d.reshape(n, m, 1, R)
        
        # V_3d is (m, p, R).
        # We want V[j, k, r] at [j, k, r].
        # Reshape to (1, m, p, R) ?
        V_b = V_3d.reshape(1, m, p, R)
        
        M = U_b * V_b # Shape (n, m, p, R)
        # We need M[k, i, j, r]. Current M is [i, j, k, r].
        # Transpose to (p, n, m, R) ?
        # M.transpose(2, 0, 1, 3) -> [k, i, j, r].
        M_trans = M.transpose(2, 0, 1, 3) # Shape (p, n, m, R)
        M_flat = M_trans.reshape(p*n, m, R)
        
        W = np.zeros((p*n, R))
        b = np.ones(m)
        
        for idx in range(p*n):
            A = M_flat[idx] # Shape (m, R)
            # Solve A.T @ w = b
            try:
                W[idx] = np.linalg.lstsq(A.T, b, rcond=None)[0]
            except:
                W[idx] = np.zeros(R)
                
        return W
    
    def check_error(U, V, W):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err = abs(total - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def polish(U, V, W, R, max_iter=100, tol=1e-12):
        """High-precision polish pass with tighter tolerance."""
        prev_err = np.inf
        for it in range(max_iter):
            if time.time() > deadline - 3:
                break
            
            U = solve_U(V, W, R)
            V = solve_V(U, W, R)
            W = solve_W(U, V, R)
            
            err = check_error(U, V, W)
            
            if err < tol:
                return U, V, W, err
            
            if abs(prev_err - err) < 1e-14:
                break
            prev_err = err
        
        return U, V, W, err
    
    def try_rank(R, n_restarts=10, max_iter=300, tol=1e-9):
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                return None
            
            # Initialization strategies
            if restart == 0:
                # Near-schoolbook: start with schoolbook and randomly combine columns
                U0, V0, W0 = build_schoolbook()
                np.random.seed(42 + restart)
                if R < n * m * p:
                    # Use random linear combinations of schoolbook columns
                    P = np.random.randn(n*m*p, R)
                    U = U0 @ P
                    V = V0 @ P
                    W = W0 @ P
                else:
                    U = U0.copy()
                    V = V0.copy()
                    W = W0.copy()
            elif restart == 1:
                # Random small initialization
                np.random.seed(123 + restart)
                U = np.random.randn(nm, R) * 0.1
                V = np.random.randn(mp, R) * 0.1
                W = np.random.randn(pn, R) * 0.1
            elif restart == 2:
                # Random with some structure (sparse)
                np.random.seed(456 + restart)
                U = np.random.randn(nm, R) * 0.5
                V = np.random.randn(mp, R) * 0.5
                W = np.random.randn(pn, R) * 0.5
                U[np.random.rand(nm, R) < 0.3] = 0
                V[np.random.rand(mp, R) < 0.3] = 0
                W[np.random.rand(pn, R) < 0.3] = 0
            elif restart == 3:
                # Structured: try to exploit n=2 symmetry
                np.random.seed(789 + restart)
                U = np.random.randn(nm, R) * 0.3
                V = np.random.randn(mp, R) * 0.3
                W = np.random.randn(pn, R) * 0.3
                # Make W symmetric in some sense for the two i values
                for k in range(p):
                    W[k*n+0, :] = W[k*n+0, :] + 0.1 * W[k*n+1, :]
                    W[k*n+1, :] = W[k*n+1, :] + 0.1 * W[k*n+0, :]
            else:
                # Pure random
                np.random.seed(1000 + restart)
                U = np.random.randn(nm, R) * 0.2
                V = np.random.randn(mp, R) * 0.2
                W = np.random.randn(pn, R) * 0.2
            
            # Alternating least squares
            prev_err = np.inf
            converged = False
            for it in range(max_iter):
                if time.time() > deadline - 10:
                    break
                
                U = solve_U(V, W, R)
                V = solve_V(U, W, R)
                W = solve_W(U, V, R)
                
                err = check_error(U, V, W)
                
                if err < tol:
                    converged = True
                    break
                
                if abs(prev_err - err) < 1e-12:
                    break
                prev_err = err
            
            # If we got close or converged, do a high-precision polish
            if converged or (err < 1e-6 and time.time() < deadline - 15):
                U, V, W, final_err = polish(U, V, W, R, max_iter=200, tol=1e-12)
                if final_err < 1e-9:
                    return U, V, W
        
        return None
    
    # Try ranks from just below 50 downward
    # Focus on a tighter range first, then expand if time permits
    # Try ranks from just below 50 downward
    # With faster ALS, we can afford more restarts and potentially lower ranks
    ranks_to_try = list(range(49, 30, -1))
    
    for R in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        result = try_rank(R, n_restarts=10, max_iter=300, tol=1e-9)
        if result is not None:
            best_rank = R
            best_solution = result
            # Continue trying lower ranks
            continue
        else:
            # If we can't find rank R, try a few more lower ranks with less effort
            for R2 in range(R-1, max(R-3, 25), -1):
                if time.time() > deadline - 10:
                    break
                result2 = try_rank(R2, n_restarts=5, max_iter=150, tol=1e-9)
                if result2 is not None:
                    best_rank = R2
                    best_solution = result2
                    break
            break
    
    # Convert to lists for return
    U, V, W = best_solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
