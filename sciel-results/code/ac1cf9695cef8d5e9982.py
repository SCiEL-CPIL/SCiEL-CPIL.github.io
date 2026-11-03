import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    start_time = time.time()
    deadline = start_time + 110
    
    def make_schoolbook(rank):
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find lower rank decompositions
    # The bilinear system: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # Strategy: fix U and V with small integer entries, solve linear system for W
    
    best_result = make_schoolbook(rank_schoolbook)
    
    # Try ranks from 44 down to 40
    for target_rank in range(rank_schoolbook - 1, 39, -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Generate sparse random U and V with entries in {-1, 0, 1}
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            
            # Make each column have a few non-zero entries
            for r in range(target_rank):
                # Pick random rows for U
                u_rows = random.sample(range(n * m), min(3, n * m))
                for row in u_rows:
                    U[row, r] = random.choice([-1, 1])
                
                v_rows = random.sample(range(m * p), min(3, m * p))
                for row in v_rows:
                    V[row, r] = random.choice([-1, 1])
            
            # For each output (i,k), we need: sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is nonlinear. Instead, let's fix U and V and solve for W using the constraint
            # that for each (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            
            # This is still nonlinear in W. Let's try a different approach:
            # Use the fact that if we fix two of the three matrices, the third is determined
            # by a linear system (if we treat it as a tensor unfolding).
            
            # Alternative: use the "slice" approach. For each r, the contribution is
            # U[:,r] * V[:,r]^T * W[:,r] (outer products)
            # We need sum_r U[:,r] outer V[:,r] outer W[:,r] = T (the multiplication tensor)
            
            # Let's try: fix U and V, then for each (i,k), the equation is:
            # sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # This is linear in W! For fixed (i,k), we have:
            # sum_r (sum_j U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r] = 1
            # Let A_ik[r] = sum_j U[i*m+j,r] * V[j*p+k,r]
            # Then sum_r A_ik[r] * W[k*n+i,r] = 1 for all (i,k)
            
            # This gives n*p = 15 equations in target_rank unknowns (W entries for that (i,k))
            # But W[k*n+i,r] is shared across different j... wait no, W is indexed by (k*n+i, r)
            # So for each (i,k), we have one equation: sum_r A_ik[r] * W[k*n+i, r] = 1
            # That's 15 equations in 15*target_rank unknowns. Underdetermined!
            
            # We can add constraints: W entries should be small. Let's just find any solution.
            # For each (i,k), solve: A_ik @ W_ik = 1 where W_ik is a vector of length target_rank
            # This is underdetermined, so we can pick the minimum-norm solution or a sparse one.
            
            W = np.zeros((p * n, target_rank))
            valid = True
            for i in range(n):
                for k in range(p):
                    # Compute A_ik[r] = sum_j U[i*m+j,r] * V[j*p+k,r]
                    A_ik = np.zeros(target_rank)
                    for j in range(m):
                        A_ik += U[i * m + j] * V[j * p + k]
                    
                    if np.allclose(A_ik, 0):
                        valid = False
                        break
                    
                    # Find w such that A_ik @ w = 1, minimize ||w||
                    # Minimum norm solution: w = A_ik^T / (A_ik @ A_ik^T)
                    denom = np.dot(A_ik, A_ik)
                    if abs(denom) < 1e-10:
                        valid = False
                        break
                    w = A_ik / denom
                    W[k * n + i] = w
            
            if not valid:
                continue
            
            # Verify the solution
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(target_rank):
                            val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        err = abs(val - 1)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-8:
                # Found a valid decomposition!
                # Convert to lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                # Check if entries are close to integers or half-integers
                all_good = True
                for mat in [U_list, V_list, W_list]:
                    for row in mat:
                        for val in row:
                            rounded = round(val * 2) / 2
                            if abs(val - rounded) > 1e-6:
                                all_good = False
                                break
                        if not all_good:
                            break
                    if not all_good:
                        break
                
                if all_good:
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    found = True
                    break
        
        if found:
            break
    
    return best_result
