import numpy as np
import time
import random

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe
    
    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # We search for ranks from 55 down to 40
    for r in range(55, 39, -1):
        if time.time() > deadline - 10:
            break
            
        # Number of attempts per rank
        attempts = 50 if r > 50 else 100
        
        for _ in range(attempts):
            if time.time() > deadline - 5:
                break
            
            # Generate random U and V with small integer entries
            # Using -1, 0, 1
            U = np.random.randint(-1, 2, size=(n * m, r))
            V = np.random.randint(-1, 2, size=(m * p, r))
            
            # Construct the linear system for W
            # Equations: for each (i, j, k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # Unknowns: W[k*n+i, c]
            
            num_eqs = n * m * p
            num_unknowns = p * n * r
            
            A_mat = np.zeros((num_eqs, num_unknowns))
            b_vec = np.ones(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Equation for (i, j, k)
                        # W index for this eq is k*n + i
                        w_row = k * n + i
                        for c in range(r):
                            coeff = U[i * m + j, c] * V[j * p + k, c]
                            if coeff != 0:
                                A_mat[eq_idx, w_row * r + c] = coeff
                        eq_idx += 1
            
            # Solve A_mat * W_flat = b_vec
            # Since underdetermined, use least squares or pseudoinverse
            # We want a solution with small entries
            try:
                # Using np.linalg.lstsq
                W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except:
                continue
                
            W = W_flat.reshape(p * n, r)
            
            # Check if W is close to half-integers
            # Multiply by 2 to check integers
            W_half = W * 2
            W_rounded = np.round(W_half)
            
            # Check error
            error = np.max(np.abs(W_half - W_rounded))
            if error < 1e-6:
                # Potential candidate
                W_candidate = (W_rounded / 2.0)
                
                # Verify exactly
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(r):
                                val += U[i * m + j, c] * V[j * p + k, c] * W_candidate[k * n + i, c]
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    # Convert to lists
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W_candidate.tolist()
                    
                    # Ensure types are int or float (half-int)
                    # The task says integer/half-integer.
                    # We can return floats if they are .0 or .5
                    return {"U": U_list, "V": V_list, "W": W_list}

    return best_result
# EVOLVE-BLOCK-END
