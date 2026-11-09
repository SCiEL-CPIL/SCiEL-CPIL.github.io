import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(len(U[i*m+j])))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _check_exact(U, V, W, n, m, p):
    """Check exact decomposition using Fractions."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for c in range(len(U[i*m+j])):
                    val += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if val != Fraction(1):
                    return False
    return True

def _try_rank_reduction(U, V, W, n, m, p, deadline):
    """Try to reduce rank by finding linear dependencies among columns."""
    r = len(U[0])
    # Try to find a column that is a linear combination of others
    # For each pair of columns, try to eliminate one
    for target_col in range(r):
        if time.time() > deadline:
            return None, None, None, r
        
        # Try to express column target_col as linear combination of other columns
        # Build system: for each (i,j,k), sum_c alpha_c * (U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]) = U[i*m+j,target_col]*V[j*p+k,target_col]*W[k*n+i,target_col]
        
        # Collect all (i,j,k) equations
        equations = []
        rhs = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = []
                    for c in range(r):
                        if c != target_col:
                            row.append(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c])
                    rhs.append(U[i*m+j][target_col] * V[j*p+k][target_col] * W[k*n+i][target_col])
                    equations.append(row)
        
        if not equations:
            continue
            
        # Solve least squares
        A = np.array(equations, dtype=float)
        b = np.array(rhs, dtype=float)
        
        try:
            alpha, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
        
        # Check if the solution is good
        pred = A @ alpha
        if np.max(np.abs(pred - b)) < 1e-6:
            # Try to round alpha to half-integers
            alpha_r = np.round(alpha * 2) / 2
            
            # Verify
            pred_r = np.array([sum(alpha_r[c] * equations[i][c] for c in range(len(alpha_r))) for i in range(len(equations))])
            if np.max(np.abs(pred_r - b)) < 1e-9:
                # Create new matrices without column target_col
                new_r = r - 1
                keep = [c for c in range(r) if c != target_col]
                
                U_new = np.zeros((n*m, new_r))
                V_new = np.zeros((m*p, new_r))
                W_new = np.zeros((p*n, new_r))
                
                for new_c, old_c in enumerate(keep):
                    U_new[:, new_c] = U[:, old_c]
                    V_new[:, new_c] = V[:, old_c]
                    W_new[:, new_c] = W[:, old_c]
                
                # Add the linear combination
                for new_c, old_c in enumerate(keep):
                    U_new[:, new_c] += alpha_r[new_c] * U[:, target_col]
                    V_new[:, new_c] += alpha_r[new_c] * V[:, target_col]
                    W_new[:, new_c] += alpha_r[new_c] * W[:, target_col]
                
                # Round to half-integers
                U_r = np.round(U_new * 2) / 2
                V_r = np.round(V_new * 2) / 2
                W_r = np.round(W_new * 2) / 2
                
                if _check_decomposition(U_r, V_r, W_r, n, m, p, 1e-9):
                    return U_r, V_r, W_r, new_r
    
    return None, None, None, r

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Initialize with schoolbook
    best_rank = n * m * p
    U_init = np.zeros((n*m, best_rank))
    V_init = np.zeros((m*p, best_rank))
    W_init = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_init[i*m+j, col] = 1
                V_init[j*p+k, col] = 1
                W_init[k*n+i, col] = 1
                col += 1
    
    best_result = {"U": U_init.tolist(), "V": V_init.tolist(), "W": W_init.tolist()}
    
    # Try rank reduction from schoolbook
    U_cur = U_init.copy()
    V_cur = V_init.copy()
    W_cur = W_init.copy()
    r_cur = best_rank
    
    for iteration in range(20):
        if time.time() > deadline:
            break
        
        U_new, V_new, W_new, r_new = _try_rank_reduction(U_cur, V_cur, W_cur, n, m, p, deadline)
        
        if U_new is not None:
            U_cur = U_new
            V_cur = V_new
            W_cur = W_new
            r_cur = r_new
            best_rank = r_new
            best_result = {"U": U_cur.tolist(), "V": V_cur.tolist(), "W": W_cur.tolist()}
        else:
            break
    
    # Try random search for lower ranks
    for r in range(max(10, best_rank - 5), best_rank):
        if time.time() > deadline:
            break
        
        for restart in range(10):
            if time.time() > deadline:
                break
            
            U = np.random.randint(-2, 3, size=(n*m, r))
            V = np.random.randint(-2, 3, size=(m*p, r))
            W = np.random.randint(-2, 3, size=(p*n, r))
            
            converged = False
            for it in range(30):
                if time.time() > deadline:
                    break
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, r))
                        for j in range(m):
                            A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        b = np.ones(m)
                        try:
                            AA_T = A @ A.T
                            AA_T += 1e-8 * np.eye(m)
                            w = A.T @ np.linalg.solve(AA_T, b)
                            W[k*n+i, :] = w
                        except np.linalg.LinAlgError:
                            w = np.linalg.pinv(A) @ b
                            W[k*n+i, :] = w
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, r))
                        for k in range(p):
                            A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        b = np.ones(p)
                        try:
                            AA_T = A @ A.T
                            AA_T += 1e-8 * np.eye(p)
                            u = A.T @ np.linalg.solve(AA_T, b)
                            U[i*m+j, :] = u
                        except np.linalg.LinAlgError:
                            u = np.linalg.pinv(A) @ b
                            U[i*m+j, :] = u
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, r))
                        for i in range(n):
                            A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        b = np.ones(n)
                        try:
                            AA_T = A @ A.T
                            AA_T += 1e-8 * np.eye(n)
                            v = A.T @ np.linalg.solve(AA_T, b)
                            V[j*p+k, :] = v
                        except np.linalg.LinAlgError:
                            v = np.linalg.pinv(A) @ b
                            V[j*p+k, :] = v
                
                # Check residual
                res = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            res += (val - 1.0)**2
                
                if res < 1e-6:
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    if _check_decomposition(U_r, V_r, W_r, n, m, p, 1e-9):
                        best_rank = r
                        best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        converged = True
                        break
                
                if time.time() > deadline:
                    break
            
            if converged:
                break
    
    return best_result
