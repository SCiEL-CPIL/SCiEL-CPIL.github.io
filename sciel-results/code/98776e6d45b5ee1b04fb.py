import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    num_a = n * m  # 16
    num_b = m * p  # 32
    num_c = p * n  # 32
    
    start_time = time.time()
    timeout = 115  # seconds
    
    # The tensor T has non-zero entries at (a, b, c) where:
    # a = i*m + j, b = j*p + k, c = k*n + i
    # for i in range(n), j in range(m), k in range(p)
    
    # Non-zero indices
    nonzero = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero.append((a, b, c))
    
    # Total number of constraints:
    # 1. For each (a,b,c) in nonzero: sum_t U[a,t]*V[b,t]*W[c,t] = 1
    # 2. For each (a,b,c) NOT in nonzero: sum_t U[a,t]*V[b,t]*W[c,t] = 0
    
    # We use the strategy from the directive: formulate as a sparse linear system
    # and find a kernel basis using Gaussian elimination over rationals.
    
    # However, the system is nonlinear (trilinear). Instead, we fix U and V randomly 
    # with small integer entries, then solve for W using the linear constraints.
    # This is similar to the original approach but more systematic.
    
    # Actually, let's try a different approach: 
    # Fix U to be a random matrix, then for each c, the constraint is:
    # sum_t U[a,t] * V[b,t] * W[c,t] = T[a,b,c] for all (a,b)
    # This is linear in V and W jointly, but not separately.
    
    # Let's stick with the original approach but improve the search:
    # 1. Try smaller ranks first (start from theoretical lower bound)
    # 2. Use better random initialization
    # 3. Add more trials for promising ranks
    
    # Theoretical lower bound: max(num_a, num_b, num_c) = 32
    # But we need at least n*m*p / something... Actually for matrix multiplication,
    # the rank is at least max(n*m, m*p, p*n) = 32 for 4x4x8.
    # Known results suggest rank 48 might be achievable or even lower.
    
    # Let's try ranks from 40 to 128 with increasing effort
    for R in range(40, 129):
        if time.time() - start_time > timeout:
            break
        
        # Number of trials increases with rank difficulty
        if R < 48:
            num_trials = 500
        elif R < 56:
            num_trials = 300
        else:
            num_trials = 150
            
        for trial in range(num_trials):
            if time.time() - start_time > timeout:
                break
                
            # Random U, V with entries in {-1, 0, 1} for sparsity and simplicity
            U = np.random.randint(-1, 2, size=(num_a, R))
            V = np.random.randint(-1, 2, size=(num_b, R))
            
            # For each c, solve the linear system M_c * W_c = rhs
            W = np.zeros((num_c, R))
            valid = True
            for c in range(num_c):
                i_out = c % n
                k_out = c // n
                
                # M_c is m x R, where M_c[j, t] = U[i_out * m + j, t] * V[j * p + k_out, t]
                M_c = np.zeros((m, R))
                for j in range(m):
                    a_idx = i_out * m + j
                    b_idx = j * p + k_out
                    M_c[j, :] = U[a_idx, :] * V[b_idx, :]
                
                rhs = np.ones(m)
                
                # Solve M_c * W_c = rhs using least squares
                try:
                    W_c, residuals, rank_m, s = np.linalg.lstsq(M_c, rhs, rcond=None)
                    residual = np.linalg.norm(M_c @ W_c - rhs)
                    if residual > 1e-9:
                        valid = False
                        break
                    W[c, :] = W_c
                except:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Verify the full decomposition
            ok = True
            # Check non-zero entries
            for a, b, c in nonzero:
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-9:
                    ok = False
                    break
            if not ok:
                continue
                
            # Check zero entries (sample to save time)
            nonzero_set = set(nonzero)
            import random as rnd
            rnd.seed(trial + R)
            for _ in range(1000):
                a_idx = rnd.randint(0, num_a - 1)
                b_idx = rnd.randint(0, num_b - 1)
                c_idx = rnd.randint(0, num_c - 1)
                if (a_idx, b_idx, c_idx) in nonzero_set:
                    continue
                val = np.dot(U[a_idx, :], V[b_idx, :] * W[c_idx, :])
                if abs(val) > 1e-9:
                    ok = False
                    break
            if not ok:
                continue
                
            # Success! Return the decomposition
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
