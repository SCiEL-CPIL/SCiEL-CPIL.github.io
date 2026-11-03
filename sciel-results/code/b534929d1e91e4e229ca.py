import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 20, 25, 20
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Build schoolbook decomposition (rank = 100)
    schoolbook_rank = n * m * p  # 100
    U_school = np.zeros((nm, schoolbook_rank), dtype=float)
    V_school = np.zeros((mp, schoolbook_rank), dtype=float)
    W_school = np.zeros((pn, schoolbook_rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    def verify(U, V, W, tol=1e-9):
        """Verify exact decomposition."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0.0
                    for r in range(rank):
                        s += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Verify schoolbook
    if not verify(U_school, V_school, W_school):
        # Should never happen, but guard
        pass
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # Strategy: For each rank r from 99 down, try to find a valid decomposition
    # Use a structured approach: try to eliminate terms one at a time
    
    def try_reduce_rank(target_rank):
        """Try to find a decomposition of the given target rank."""
        if time.time() > deadline:
            return None
        
        # Try different strategies to find a rank-target_rank decomposition
        # Strategy 1: Start with schoolbook, remove (100 - target_rank) terms, 
        # then solve for the remaining terms
        
        num_remove = schoolbook_rank - target_rank
        if num_remove <= 0 or num_remove > 10:  # Only try small reductions
            return None
        
        # Try removing terms from the end (or systematically)
        # For each combination of terms to remove, try to solve for the rest
        
        # This is expensive, so limit to small num_remove
        if num_remove == 1:
            # Try removing each single term
            for remove_idx in range(schoolbook_rank):
                if time.time() > deadline:
                    return None
                
                # Remove term remove_idx
                keep = [i for i in range(schoolbook_rank) if i != remove_idx]
                U_try = U_school[:, keep]
                V_try = V_school[:, keep]
                W_try = W_school[:, keep]
                
                # Now try to adjust W_try to compensate
                # The constraint is: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                # For fixed U and V, this is linear in W
                # But it's not a standard linear system because each equation involves
                # different entries of W
                
                # Actually, for each (i,j,k), the equation is:
                # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                # Let c_r(i,j,k) = U[i*m+j,r] * V[j*p+k,r]
                # Then: sum_r c_r(i,j,k) * W[k*n+i,r] = 1
                # This is linear in W, but the coefficient c_r depends on (i,j,k)
                
                # This is a system of 100 equations in 20*target_rank unknowns
                # But each equation only involves target_rank variables (for a fixed k*n+i)
                
                # Actually, for a fixed output index (k*n+i), the equations for all j are:
                # For each j: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                # This is a system of m equations in target_rank unknowns (W[k*n+i, :])
                # with coefficients A_j,r = U[i*m+j,r] * V[j*p+k,r]
                
                # So for each (i,k), we have a system:
                # A(i,k) * W[k*n+i, :] = ones(m)
                # where A(i,k) is m x target_rank
                
                # If this system is consistent for all (i,k), we have a solution
                
                W_solution = np.zeros((pn, target_rank), dtype=float)
                valid = True
                for i in range(n):
                    for k in range(p):
                        if time.time() > deadline:
                            return None
                        
                        # Build the system for this (i,k)
                        A = np.zeros((m, target_rank), dtype=float)
                        for j in range(m):
                            for r_idx, r in enumerate(keep):
                                A[j, r_idx] = U_school[i * m + j, r] * V_school[j * p + k, r]
                        
                        b = np.ones(m, dtype=float)
                        
                        # Solve A * x = b
                        try:
                            # Use least squares
                            x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                            # Check if the solution is exact
                            if np.max(np.abs(A @ x - b)) > 1e-9:
                                valid = False
                                break
                            W_solution[k * n + i, :] = x
                        except:
                            valid = False
                            break
                    
                    if not valid:
                        break
                
                if valid:
                    # Verify the solution
                    if verify(U_try, V_try, W_solution):
                        return {"U": U_try.tolist(), "V": V_try.tolist(), "W": W_solution.tolist()}
        
        return None
    
    # Try ranks from 99 down to 90
    for target_rank in range(schoolbook_rank - 1, max(schoolbook_rank - 10, 0), -1):
        if time.time() > deadline:
            break
        
        result = try_reduce_rank(target_rank)
        if result is not None:
            best_rank = target_rank
            best_result = result
            break
    
    # If no improvement found, return schoolbook
    return best_result
