import numpy as np
import time

def construct():
    n, m, p = 2, 3, 6
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s limit
    
    # Build the tensor equation system
    # For each (i,j,k), we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Total equations: n*m*p = 36
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact."""
        rank = U.shape[1]
        # Check all (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to find lower rank decompositions
    # Use a greedy approach: start with schoolbook, try to eliminate columns
    
    # Alternative: use a random search with fixed V and W, solve for U, then refine
    # Or use alternating least squares
    
    # Let's try: fix rank r, initialize randomly, then use ALS to converge
    def try_rank(r, max_iter=200, tol=1e-10):
        if r >= best_rank:
            return None
        
        # Try multiple random initializations
        for init_seed in range(5):
            if time.time() > deadline:
                return None
            
            np.random.seed(42 + r + init_seed * 100)
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Build the linear system for each variable
            # For fixed V, W: U[i*m+j, r] appears in equation (i,j,k) with coefficient V[j*p+k, r] * W[k*n+i, r]
            # This is nonlinear, so use ALS
            
            for it in range(max_iter):
                if time.time() > deadline:
                    return None
                
                # Update U: for each (i,j), solve for U[i*m+j, :]
                for i in range(n):
                    for j in range(m):
                        # Equation: sum_k sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 (for each k... wait)
                        # Actually, the equation is per (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                        # For fixed (i,j), we have p equations (one for each k) in r unknowns (U[i*m+j, 0..r-1])
                        # Let's build the system: for each k, sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
                        A_mat = np.zeros((p, r))
                        for k in range(p):
                            for rr in range(r):
                                A_mat[k, rr] = V[j * p + k, rr] * W[k * n + i, rr]
                        b_vec = np.ones(p)
                        # Solve A_mat @ U[i*m+j, :] = b_vec
                        try:
                            U[i * m + j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                        except:
                            break
                    else:
                        continue
                    break
                else:
                    # Update V: for each (j,k), solve for V[j*p+k, :]
                    for j in range(m):
                        for k in range(p):
                            # Equation: sum_i sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 (for each i)
                            # For fixed (j,k), we have n equations in r unknowns
                            A_mat = np.zeros((n, r))
                            for i in range(n):
                                for rr in range(r):
                                    A_mat[i, rr] = U[i * m + j, rr] * W[k * n + i, rr]
                            b_vec = np.ones(n)
                            try:
                                V[j * p + k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                            except:
                                break
                        else:
                            continue
                        break
                    else:
                        # Update W: for each (k,i), solve for W[k*n+i, :]
                        for k in range(p):
                            for i in range(n):
                                # Equation: sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 (for each j)
                                # For fixed (k,i), we have m equations in r unknowns
                                A_mat = np.zeros((m, r))
                                for j in range(m):
                                    for rr in range(r):
                                        A_mat[j, rr] = U[i * m + j, rr] * V[j * p + k, rr]
                                b_vec = np.ones(m)
                                try:
                                    W[k * n + i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                                except:
                                    break
                            else:
                                continue
                            break
                        else:
                            # Check convergence
                            if verify(U, V, W, tol):
                                # Try to round to integers or half-integers
                                U_rounded = np.round(U * 2) / 2
                                V_rounded = np.round(V * 2) / 2
                                W_rounded = np.round(W * 2) / 2
                                if verify(U_rounded, V_rounded, W_rounded, 1e-9):
                                    return U_rounded, V_rounded, W_rounded
                                # Try integer rounding
                                U_rounded = np.round(U)
                                V_rounded = np.round(V)
                                W_rounded = np.round(W)
                                if verify(U_rounded, V_rounded, W_rounded, 1e-9):
                                    return U_rounded, V_rounded, W_rounded
                                # Return as-is if it's close enough (the problem allows half-integers)
                                # But we need exact, so let's try harder
                                # For now, accept if verify passes with the rounded version
                                # If not, continue iterating
                                pass
                # End of iteration
                
            # Final check
            if verify(U, V, W, tol):
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                if verify(U_rounded, V_rounded, W_rounded, 1e-9):
                    return U_rounded, V_rounded, W_rounded
                U_rounded = np.round(U)
                V_rounded = np.round(V)
                W_rounded = np.round(W)
                if verify(U_rounded, V_rounded, W_rounded, 1e-9):
                    return U_rounded, V_rounded, W_rounded
        
        return None
    
    # Try ranks from best_rank-1 down to some minimum
    # Focus on promising ranks: 35, 34, 33, 32, 31, 30, 28, 24, 20, 18, 16, 12
    candidate_ranks = [35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 22, 20, 18, 16, 14, 12]
    
    for r in candidate_ranks:
        if r >= best_rank:
            continue
        if time.time() > deadline:
            break
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_U, best_V, best_W = U, V, W
            break  # Found the lowest rank we tried
    
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
