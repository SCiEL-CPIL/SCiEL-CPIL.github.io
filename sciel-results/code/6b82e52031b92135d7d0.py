"""Matrix multiplication 3,5,7 tensor decomposition - rank minimization."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition (Rank 105) as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Strategy: Try to find lower rank decompositions
    # The tensor T[i,j,k] = 1 for all (i,j,k) with i<n, j<m, k<p
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    
    # Approach: Fix U and V, solve linear system for W
    # For each (i,j,k): sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r]) = 1
    # This is a linear system in W entries
    
    # Try various ranks from a lower bound up to schoolbook
    # Lower bound: at least max(n*m, m*p, n*p) = max(15, 35, 21) = 35
    # But realistically, we need higher
    
    # Use random initialization with {-1, 0, 1} entries
    # Solve for W using least squares, check if solution is integer/half-integer
    
    np.random.seed(42)
    
    # Try ranks from 40 to 104 (below schoolbook)
    # For each rank, try multiple random initializations
    for R in range(40, min(rank_sb, 105)):
        if time.time() > deadline - 5.0:
            break
        
        # Number of trials: more for smaller ranks, fewer for larger
        num_trials = 200 if R < 60 else 50
        
        for trial in range(num_trials):
            if time.time() > deadline - 3.0:
                break
            
            # Random U and V with entries in {-1, 0, 1}
            U_rand = np.random.choice([-1, 0, 1], size=(n * m, R))
            V_rand = np.random.choice([-1, 0, 1], size=(m * p, R))
            
            # Build linear system for W
            # Unknowns: W[k*n+i, r] for k in [0,p), i in [0,n), r in [0,R)
            # Total unknowns: p * n * R
            # Equations: n * m * p (one for each (i,j,k))
            
            num_unknowns = p * n * R
            num_equations = n * m * p
            
            A = np.zeros((num_equations, num_unknowns))
            b = np.ones(num_equations)
            
            # Fill A
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * (m * p) + j * p + k
                        for r in range(R):
                            coeff = U_rand[i * m + j, r] * V_rand[j * p + k, r]
                            if coeff != 0:
                                # W index: k*n + i, column r
                                w_row = k * n + i
                                unk_idx = w_row * R + r
                                A[eq_idx, unk_idx] = coeff
            
            # Solve A @ w = b
            try:
                # Use least squares
                w_sol, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                
                # Check if solution is valid
                if len(residuals) > 0:
                    if residuals[0] > 1e-8:
                        continue
                else:
                    # Underdetermined: check A @ w_sol == b
                    if not np.allclose(A @ w_sol, b, atol=1e-8):
                        continue
                
                # Check if w_sol entries are integers or half-integers
                w_rounded = np.round(w_sol * 2) / 2
                if not np.allclose(w_sol, w_rounded, atol=1e-8):
                    continue
                
                # Valid solution found!
                # Construct W matrix
                W_sol = np.zeros((p * n, R))
                for idx in range(len(w_sol)):
                    w_row = idx // R
                    r_col = idx % R
                    W_sol[w_row, r_col] = w_rounded[idx]
                
                # Convert to lists with proper types
                U_list = U_rand.tolist()
                V_list = V_rand.tolist()
                W_list = W_sol.tolist()
                
                # Round to ensure clean integers/half-integers
                for row in U_list:
                    for idx in range(len(row)):
                        row[idx] = round(row[idx])
                for row in V_list:
                    for idx in range(len(row)):
                        row[idx] = round(row[idx])
                for row in W_list:
                    for idx in range(len(row)):
                        val = round(row[idx] * 2) / 2
                        row[idx] = val
                
                # Verify the solution
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(R):
                                val += U_list[i * m + j][r] * V_list[j * p + k][r] * W_list[k * n + i][r]
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best_solution = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = R
                    # Found a solution, try to go lower
                    break
            
            except Exception:
                continue
        
        # If we found a solution at this rank, we can stop (we're going from low to high)
        # But we want the minimum, so if we found one at R, we should try lower ranks
        # However, we're iterating from low to high, so the first valid one is the minimum
        if best_rank < rank_sb:
            break
    
    return best_solution
