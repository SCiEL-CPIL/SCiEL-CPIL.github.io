import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin within 120s
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 50
    
    # Initialize with schoolbook decomposition
    U = np.zeros((n * m, schoolbook_rank), dtype=float)
    V = np.zeros((m * p, schoolbook_rank), dtype=float)
    W = np.zeros((p * n, schoolbook_rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = schoolbook_rank
    
    # Define the tensor T[i,j,k] = 1 if (i,j,k) is a valid multiplication term
    # T is implicitly: sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
    
    def verify(U_mat, V_mat, W_mat, tol=1e-6):
        """Check if the decomposition is exact within tolerance."""
        nm = n * m
        mp = m * p
        pn = p * n
        rank = U_mat.shape[1]
        # Reshape for efficient computation
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    val = np.dot(U_mat[a_idx, :], V_mat[b_idx, :] * W_mat[c_idx, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to reduce rank using a greedy column elimination approach
    # For each column, check if it can be expressed as a linear combination of others
    
    # Alternative approach: use random initialization + optimization
    # Since we have time, let's try a few random restarts with lower rank
    
    import random
    rng = random.Random(42)
    
    # Try ranks below schoolbook
    for target_rank in range(schoolbook_rank - 1, max(30, schoolbook_rank - 15), -1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            U_init = np.random.randn(n * m, target_rank)
            V_init = np.random.randn(m * p, target_rank)
            W_init = np.random.randn(p * n, target_rank)
            
            # Scale to have reasonable magnitudes
            scale = 1.0 / (target_rank ** 0.25)
            U_init *= scale
            V_init *= scale
            W_init *= scale
            
            # Simple iterative refinement
            # Objective: minimize sum over (i,j,k) of (sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1)^2
            # Use alternating minimization
            
            U_cur = U_init.copy()
            V_cur = V_init.copy()
            W_cur = W_init.copy()
            
            max_iter = 500
            converged = False
            
            for it in range(max_iter):
                if time.time() > deadline - 10:
                    break
                
                # Update W: for each (k,i), W[k*n+i, r] should satisfy
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
                # For fixed i,k, we need: sum_j (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]) = ... 
                # Actually, for each (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
                # Fix i,k, vary j: this is a system in W[k*n+i, :]
                # Let A_jk = U[i*m+j, :] * V[j*p+k, :] (elementwise product, shape target_rank)
                # Then sum_r A_jk[r] * W[k*n+i, r] = 1 for each j
                # This is: (A_jk @ W[k*n+i,:]) = 1 for each j
                # So we solve the system
                
                for i in range(n):
                    for k in range(p):
                        # Collect the vectors A_jk for each j
                        A_list = []
                        for j in range(m):
                            a_vec = U_cur[i * m + j, :]  # shape (target_rank,)
                            v_vec = V_cur[j * p + k, :]  # shape (target_rank,)
                            A_list.append(a_vec * v_vec)  # elementwise, shape (target_rank,)
                        
                        A_mat = np.array(A_list)  # shape (m, target_rank)
                        # Solve A_mat @ W[k*n+i, :] = ones(m)
                        # Least squares: min ||A_mat @ w - 1||^2
                        try:
                            w_sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                            W_cur[k * n + i, :] = w_sol
                        except:
                            pass
                
                # Update V: for each (j,k), similar approach
                for j in range(m):
                    for k in range(p):
                        A_list = []
                        for i in range(n):
                            u_vec = U_cur[i * m + j, :]
                            w_vec = W_cur[k * n + i, :]
                            A_list.append(u_vec * w_vec)
                        
                        A_mat = np.array(A_list)  # shape (n, target_rank)
                        try:
                            v_sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)
                            V_cur[j * p + k, :] = v_sol
                        except:
                            pass
                
                # Update U: for each (i,j)
                for i in range(n):
                    for j in range(m):
                        A_list = []
                        for k in range(p):
                            v_vec = V_cur[j * p + k, :]
                            w_vec = W_cur[k * n + i, :]
                            A_list.append(v_vec * w_vec)
                        
                        A_mat = np.array(A_list)  # shape (p, target_rank)
                        try:
                            u_sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)
                            U_cur[i * m + j, :] = u_sol
                        except:
                            pass
                
                # Check convergence
                if it % 50 == 0 and time.time() < deadline - 5:
                    if verify(U_cur, V_cur, W_cur, tol=1e-4):
                        converged = True
                        break
            
            if converged:
                if verify(U_cur, V_cur, W_cur, tol=1e-6):
                    best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
                    best_rank = target_rank
                    break  # Move to next lower rank
    
    # Try to snap to half-integers
    def snap_to_half_integer(arr):
        """Snap values to nearest half-integer if close enough."""
        snapped = np.round(arr * 2.0) / 2.0
        if np.allclose(arr, snapped, atol=1e-4):
            return snapped
        return arr
    
    # If we have a solution, try to snap it
    if best_rank < schoolbook_rank:
        U_snap = snap_to_half_integer(best_U)
        V_snap = snap_to_half_integer(best_V)
        W_snap = snap_to_half_integer(best_W)
        if verify(U_snap, V_snap, W_snap, tol=1e-5):
            best_U, best_V, best_W = U_snap, V_snap, W_snap
    
    # If we're at deadline, return best found
    if time.time() > deadline - 5:
        # Ensure we have a valid solution
        if not verify(best_U, best_V, best_W, tol=1e-5):
            # Fall back to schoolbook
            best_U = np.zeros((n * m, schoolbook_rank), dtype=float)
            best_V = np.zeros((m * p, schoolbook_rank), dtype=float)
            best_W = np.zeros((p * n, schoolbook_rank), dtype=float)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        best_U[i * m + j, col] = 1.0
                        best_V[j * p + k, col] = 1.0
                        best_W[k * n + i, col] = 1.0
                        col += 1
            best_rank = schoolbook_rank
    
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

