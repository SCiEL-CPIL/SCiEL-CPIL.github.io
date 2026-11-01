import numpy as np
import time
import itertools
import random

DIMENSIONS = [3, 3, 3]
TIME_LIMIT = 150.0

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    timeout = TIME_LIMIT - 5.0  # Safety margin
    
    # Target: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # Shapes: U (n*m, R), V (m*p, R), W (p*n, R)
    
    def verify(U, V, W, R):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_r[i, j, :] * V_r[j, k, :] * W_r[k, i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def error(U, V, W, R):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_r[i, j, :] * V_r[j, k, :] * W_r[k, i, :])
                    err += (s - 1.0) ** 2
        return err

    # Strategy:
    # 1. Try known low-rank decompositions (e.g., Strassen rank 7 for 2x2, 
    #    but for 3x3, best known is 23).
    #    We don't have a hardcoded 23-rank solution easily available without 
    #    complex symbolic derivation in this short time.
    # 2. Instead, we perform a "discrete local search" on the space of 
    #    half-integers.
    # 3. We initialize with random half-integers and perform coordinate ascent/descent 
    #    or random perturbations to minimize the error.
    # 4. We try ranks from 23 upwards.
    
    best_result = None
    best_rank = n * m * p  # 27
    
    # Fallback: Schoolbook rank 27
    U_fb = np.zeros((n * m, best_rank))
    V_fb = np.zeros((m * p, best_rank))
    W_fb = np.zeros((p * n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1.0
                V_fb[j * p + k, col] = 1.0
                W_fb[k * n + i, col] = 1.0
                col += 1
    
    if verify(U_fb, V_fb, W_fb, best_rank):
        best_result = (U_fb, V_fb, W_fb, best_rank)

    # Search for lower ranks
    for R in range(23, best_rank):
        if time.time() - start_time > timeout:
            break
            
        found = False
        # Multiple restarts for this rank
        num_restarts = 200
        for restart in range(num_restarts):
            if time.time() - start_time > timeout:
                break
                
            # Initialize with random half-integers in a small range
            # Scale is important. If entries are too large, error is huge.
            # If too small, error is huge (sum ~ 0).
            # For R=23, average magnitude might be around 1/sqrt(R) * constant.
            # Let's try entries from {-1, -0.5, 0, 0.5, 1}
            
            U = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(n * m, R))
            V = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(m * p, R))
            W = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(p * n, R))
            
            curr_err = error(U, V, W, R)
            
            # Local search: Randomly perturb one entry in one matrix
            # and accept if error decreases (Simulated Annealing style or simple hill climbing)
            # Given the time limit, simple hill climbing with random restarts is safer.
            
            for step in range(2000):
                if time.time() - start_time > timeout:
                    break
                    
                # Pick a random matrix and a random entry
                mat_idx = np.random.randint(0, 3)
                if mat_idx == 0:
                    mat = U
                    rows, cols = n * m, R
                elif mat_idx == 1:
                    mat = V
                    rows, cols = m * p, R
                else:
                    mat = W
                    rows, cols = p * n, R
                    
                r_idx = np.random.randint(0, rows)
                c_idx = np.random.randint(0, cols)
                
                old_val = mat[r_idx, c_idx]
                
                # Try a small change: +/- 0.5 or +/- 1.0
                # Or just pick a new random value from the set
                # To ensure progress, let's try neighbors: old +/- 0.5, old +/- 1.0
                # And clamp to reasonable bounds [-2, 2]
                
                candidates = [old_val + 0.5, old_val - 0.5, old_val + 1.0, old_val - 1.0]
                # Add some random exploration
                candidates.append(np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -1.5]))
                
                best_candidate = None
                best_candidate_err = curr_err
                
                for cand in candidates:
                    if cand < -2.0 or cand > 2.0:
                        continue
                    mat[r_idx, c_idx] = cand
                    new_err = error(U, V, W, R)
                    if new_err < best_candidate_err:
                        best_candidate_err = new_err
                        best_candidate = cand
                    mat[r_idx, c_idx] = old_val
                
                if best_candidate is not None:
                    mat[r_idx, c_idx] = best_candidate
                    curr_err = best_candidate_err
                    
                    if curr_err < 1e-9:
                        # Check exactness
                        if verify(U, V, W, R):
                            found = True
                            break
                else:
                    # Stuck in local minimum, break to restart
                    break
            
            if found:
                best_result = (U, V, W, R)
                break
        
        if found:
            break

    if best_result is not None:
        U, V, W, R = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Return fallback
        U, V, W, R = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# Note: The above logic has a bug in the fallback return if best_result is None initially, 
# but best_result is initialized with the schoolbook solution.
# However, the discrete search is very slow for R=23. 
# A better strategy for 3x3 might be to use the known Strassen-based extension or 
# just return the schoolbook if no improvement is found quickly.
# Given the 150s limit, and the difficulty of finding rank 23 via random local search 
# on half-integers (which is a huge space), the program will likely time out 
# and return the schoolbook rank 27.
# To potentially improve, one could hardcode a known rank 23 decomposition if available.
# Since I cannot derive it instantly, I will stick to the search but optimize the 
# verification step to be faster using numpy vectorization where possible, 
# though for R=23 and 27 points, the loop is small.

# Let's refine the search to be faster.
# Instead of full error calculation for every candidate, we can use 
# gradient information? No, discrete.
# We can reduce the number of steps if no improvement is made.

# Revised plan:
# 1. Check if we can find a solution for R=23 quickly.
# 2. If not, return 27.

# To make the discrete search more effective, we can use a "restart with perturbation" 
# strategy more aggressively.

# However, there is a known rank 23 decomposition. 
# If I could find it online or derive it... 
# Since I am an AI, I should try to generate a valid one if possible.
# But generating a correct rank 23 tensor decomposition from scratch is extremely hard.
# The prompt asks for a "different strategy". 
# The current strategy is gradient descent. 
# My new strategy is discrete local search.
# It is structurally different.
# It might fail to find R=23 and return R=27.
# This is acceptable as a "first try in a new direction".

# Let's ensure the code is robust and doesn't crash.

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    timeout = 145.0
    
    def verify(U, V, W, R):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_r[i, j, :] * V_r[j, k, :] * W_r[k, i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def error(U, V, W, R):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_r[i, j, :] * V_r[j, k, :] * W_r[k, i, :])
                    err += (s - 1.0) ** 2
        return err

    # Fallback: Schoolbook rank 27
    R_fb = n * m * p
    U_fb = np.zeros((n * m, R_fb))
    V_fb = np.zeros((m * p, R_fb))
    W_fb = np.zeros((p * n, R_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1.0
                V_fb[j * p + k, col] = 1.0
                W_fb[k * n + i, col] = 1.0
                col += 1
    best_result = (U_fb, V_fb, W_fb, R_fb)

    # Try to find lower rank
    # We'll try R=23.
    # We use a randomized local search on the discrete set of half-integers.
    # To increase the chance of success, we can try to "fix" the matrices 
    # by solving linear systems? No, it's trilinear.
    
    # Let's try a few restarts.
    for R in [23]:
        if time.time() - start_time > timeout:
            break
        
        for restart in range(100):
            if time.time() - start_time > timeout:
                break
            
            # Initialize
            U = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(n * m, R))
            V = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(m * p, R))
            W = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(p * n, R))
            
            curr_err = error(U, V, W, R)
            
            # Local search
            for step in range(1000):
                if time.time() - start_time > timeout:
                    break
                
                # Pick random entry
                mat_idx = np.random.randint(0, 3)
                if mat_idx == 0:
                    mat = U
                    rows = n * m
                elif mat_idx == 1:
                    mat = V
                    rows = m * p
                else:
                    mat = W
                    rows = p * n
                
                r_idx = np.random.randint(0, rows)
                c_idx = np.random.randint(0, R)
                
                old_val = mat[r_idx, c_idx]
                
                # Try neighbors
                # We can also try to set to 0 if it's not helping
                candidates = [old_val + 0.5, old_val - 0.5, 0.0]
                
                best_cand = None
                best_err = curr_err
                
                for cand in candidates:
                    if cand < -2.0 or cand > 2.0:
                        continue
                    mat[r_idx, c_idx] = cand
                    new_err = error(U, V, W, R)
                    if new_err < best_err:
                        best_err = new_err
                        best_cand = cand
                    mat[r_idx, c_idx] = old_val
                
                if best_cand is not None:
                    mat[r_idx, c_idx] = best_cand
                    curr_err = best_err
                    if curr_err < 1e-9:
                        if verify(U, V, W, R):
                            best_result = (U, V, W, R)
                            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                else:
                    break # Stuck
            
    return {"U": best_result[0].tolist(), "V": best_result[1].tolist(), "W": best_result[2].tolist()}
