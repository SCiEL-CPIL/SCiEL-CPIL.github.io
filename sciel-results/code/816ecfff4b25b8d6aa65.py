import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Schoolbook decomposition as guaranteed fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank decompositions using structured search
    # Key insight: use the structure that W[k*n+i, c] selects (k,i) pairs
    # For each (i,k), we need sum_j U[i,j,c]*V[j,k,c]*W[k*n+i,c] = 1
    
    # Try rank values from 12 (theoretical minimum n*p) up to schoolbook
    # Use alternating minimization with structured initialization
    
    T_target = np.ones((n, m, p))
    
    # Try ranks from 12 to schoolbook-1
    for r in range(max(12, n * p), rank_school):
        if time.time() > deadline - 20:
            break
        
        found = False
        # Multiple restarts with different initializations
        for restart in range(8):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(r * 100 + restart * 13 + 7)
            
            # Initialize with structure: W selects (k,i) pairs
            # For each (k,i), assign r/(n*p) terms
            terms_per_ki = r // (n * p)
            if terms_per_ki < 1:
                continue
                
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            
            col = 0
            for k in range(p):
                for i in range(n):
                    # Assign terms_per_ki terms to this (k,i) pair
                    for t in range(terms_per_ki):
                        if col >= r:
                            break
                        W[k * n + i, col] = 1.0
                        # For the j-sum to equal 1, use random split
                        for j in range(m):
                            U[i * m + j, col] = 1.0 / m
                            V[j * p + k, col] = 1.0
                        col += 1
            
            # Alternating minimization to refine
            max_iters = 200
            prev_error = float('inf')
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, r))
                        for k in range(p):
                            M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        try:
                            U[i * m + j, :] = np.linalg.lstsq(M.T, T_target[i, j, :], rcond=None)[0]
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, r))
                        for i in range(n):
                            M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            V[j * p + k, :] = np.linalg.lstsq(M.T, T_target[:, j, k], rcond=None)[0]
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((m, r))
                        for j in range(m):
                            M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            W[k * n + i, :] = np.linalg.lstsq(M.T, T_target[i, :, k], rcond=None)[0]
                        except:
                            pass
                
                # Check convergence
                if it % 10 == 9:
                    error = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                                error += (val - 1.0) ** 2
                    if error < 1e-10:
                        # Verify and check half-integer constraint
                        U_list = U.tolist()
                        V_list = V.tolist()
                        W_list = W.tolist()
                        
                        # Check if entries are close to half-integers
                        def is_half_int(x):
                            return abs(2.0 * x - round(2.0 * x)) < 1e-6
                        
                        if all(is_half_int(x) for row in U_list for x in row) and \
                           all(is_half_int(x) for row in V_list for x in row) and \
                           all(is_half_int(x) for row in W_list for x in row):
                            # Snap to exact half-integers
                            U_snapped = [[round(2.0 * x) / 2.0 for x in row] for row in U_list]
                            V_snapped = [[round(2.0 * x) / 2.0 for x in row] for row in V_list]
                            W_snapped = [[round(2.0 * x) / 2.0 for x in row] for row in W_list]
                            
                            # Verify snapped version
                            Ua = np.array(U_snapped)
                            Va = np.array(V_snapped)
                            Wa = np.array(W_snapped)
                            valid = True
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        val = np.dot(Ua[i * m + j, :], Va[j * p + k, :] * Wa[k * n + i, :])
                                        if abs(val - 1.0) > 1e-9:
                                            valid = False
                                            break
                                    if not valid:
                                        break
                                if not valid:
                                    break
                            if valid:
                                best_rank = r
                                best_U, best_V, best_W = U_snapped, V_snapped, W_snapped
                                found = True
                                break
                    if abs(error - prev_error) < 1e-13:
                        break
                    prev_error = error
            
            if found:
                break
        
        if found:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
