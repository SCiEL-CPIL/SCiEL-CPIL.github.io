# EVOLVE-BLOCK-START
import time
import sys
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # 130s safety margin
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Initialize with schoolbook
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school

    # Try to find a lower rank solution using ALS with integer snapping
    # Lower bound is max(n*m, m*p, p*n) = 30
    np.random.seed(42)
    
    for trial_rank in range(30, rank_schoolbook):
        if time.time() > deadline - 10:
            break
            
        # Try a few random initializations
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
                
            # Random integer initialization in {-1, 0, 1}
            V_try = np.random.choice([-1, 0, 1], size=(m * p, trial_rank))
            W_try = np.random.choice([-1, 0, 1], size=(p * n, trial_rank))
            
            # ALS loop
            U_try = np.zeros((n * m, trial_rank))
            success = False
            
            for iteration in range(50):
                if time.time() > deadline - 10:
                    break
                
                # Update U
                # For each (i, j), solve for U[i*m+j, :]
                # Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                # Let M_kr = V[j*p+k, r] * W[k*n+i, r]
                # We want U_row @ M.T = ones(p)
                # M is p x R. U_row is 1 x R.
                # M.T @ U_row.T = ones(p)
                for i in range(n):
                    for j in range(m):
                        # Construct M (p x R)
                        M = np.zeros((p, trial_rank))
                        for k in range(p):
                            for r in range(trial_rank):
                                M[k, r] = V_try[j * p + k, r] * W_try[k * n + i, r]
                        
                        # Solve M.T @ u = ones(p)
                        # If R >= p, use least squares
                        try:
                            u_sol, residuals, rank, sv = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                            U_try[i * m + j, :] = u_sol
                        except:
                            pass
                
                # Update V
                # For each (j, k), solve for V[j*p+k, :]
                # Equation: sum_r V[j*p+k, r] * U[i*m+j, r] * W[k*n+i, r] = 1 for all i
                # Let N_ir = U[i*m+j, r] * W[k*n+i, r]
                # We want V_row @ N.T = ones(n)
                for j in range(m):
                    for k in range(p):
                        N = np.zeros((n, trial_rank))
                        for i in range(n):
                            for r in range(trial_rank):
                                N[i, r] = U_try[i * m + j, r] * W_try[k * n + i, r]
                        
                        try:
                            v_sol, residuals, rank, sv = np.linalg.lstsq(N.T, np.ones(n), rcond=None)
                            V_try[j * p + k, :] = v_sol
                        except:
                            pass
                
                # Update W
                # For each (k, i), solve for W[k*n+i, :]
                # Equation: sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1 for all j
                # Let O_jr = U[i*m+j, r] * V[j*p+k, r]
                # We want W_row @ O.T = ones(m)
                for k in range(p):
                    for i in range(n):
                        O = np.zeros((m, trial_rank))
                        for j in range(m):
                            for r in range(trial_rank):
                                O[j, r] = U_try[i * m + j, r] * V_try[j * p + k, r]
                        
                        try:
                            w_sol, residuals, rank, sv = np.linalg.lstsq(O.T, np.ones(m), rcond=None)
                            W_try[k * n + i, :] = w_sol
                        except:
                            pass
                
                # Check convergence and validity
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_try[i * m + j] * V_try[j * p + k] * W_try[k * n + i])
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-9:
                    # Check if entries are close to integers or half-integers
                    is_valid = True
                    for mat in [U_try, V_try, W_try]:
                        for val in mat.flatten():
                            # Check if close to integer
                            if abs(val - round(val)) > 1e-6:
                                # Check if close to half-integer
                                if abs(val - (round(val * 2) / 2.0)) > 1e-6:
                                    is_valid = False
                                    break
                        if not is_valid:
                            break
                    
                    if is_valid:
                        # Snap to integers/half-integers
                        U_snap = np.zeros_like(U_try)
                        V_snap = np.zeros_like(V_try)
                        W_snap = np.zeros_like(W_try)
                        
                        for idx in range(U_try.size):
                            v = U_try.flat[idx]
                            if abs(v - round(v)) < 1e-6:
                                U_snap.flat[idx] = round(v)
                            else:
                                U_snap.flat[idx] = round(v * 2) / 2.0
                                
                        for idx in range(V_try.size):
                            v = V_try.flat[idx]
                            if abs(v - round(v)) < 1e-6:
                                V_snap.flat[idx] = round(v)
                            else:
                                V_snap.flat[idx] = round(v * 2) / 2.0
                                
                        for idx in range(W_try.size):
                            v = W_try.flat[idx]
                            if abs(v - round(v)) < 1e-6:
                                W_snap.flat[idx] = round(v)
                            else:
                                W_snap.flat[idx] = round(v * 2) / 2.0
                                
                        # Final verification of snapped solution
                        final_valid = True
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    val = np.sum(U_snap[i * m + j] * V_snap[j * p + k] * W_snap[k * n + i])
                                    if abs(val - 1.0) > 1e-9:
                                        final_valid = False
                                        break
                            if not final_valid:
                                break
                        
                        if final_valid:
                            best_rank = trial_rank
                            best_U = U_snap
                            best_V = V_snap
                            best_W = W_snap
                            success = True
                            break
                if success:
                    break
            if success:
                break

    # Fallback to schoolbook if no better solution found
    if best_rank == rank_schoolbook:
        best_U = U_school
        best_V = V_school
        best_W = W_school

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
