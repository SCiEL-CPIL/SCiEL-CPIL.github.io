import numpy as np
import time
import random

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    # Try to find a lower rank solution
    # Target rank 28
    target_rank = 28
    
    # Deadline guard
    start_time = time.time()
    time_limit = 100  # seconds
    
    # We will try to find a solution over integers or rationals.
    # A common strategy is to use a finite field or just random integers.
    # Given the small size, we can try to solve the system of equations.
    # However, solving nonlinear systems is hard.
    
    # Instead, we can try to construct a solution based on Strassen's method for 2x2 and extend.
    # But 2x3x5 is specific.
    
    # Let's try a random search with local improvement.
    # Initialize U, V, W with random integers in [-1, 1] or similar.
    # Then minimize the error.
    
    best_solution = None
    best_error = float('inf')
    
    # To speed up, we can try to fix one matrix and solve for others?
    # Or just random restarts.
    
    for _ in range(1000): # Max iterations
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization
        # Use small integers
        U = np.random.randint(-2, 3, size=(n * m, target_rank))
        V = np.random.randint(-2, 3, size=(m * p, target_rank))
        W = np.random.randint(-2, 3, size=(p * n, target_rank))
        
        # Verify
        # We need sum_r U[i,j,r] V[j',k,r] W[i',k',r] = delta(j,j') delta(i,i') delta(k,k')
        
        # Efficient check:
        # For each r, we add outer products? No, it's a 3-tensor contraction.
        # We can check specific entries.
        
        # There are n*m * m*p * p*n = 30 * 15 * 10 = 4500 constraints?
        # No, the non-zero constraints are only n*m*p = 30.
        # The zero constraints are the rest.
        
        # Let's check the 30 non-zero constraints first.
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Sum over r
                    s = 0
                    for r in range(target_rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if s != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            continue
            
        # Check zero constraints (sample a few to save time)
        # Total zeros: (n*m * m*p * p*n) - (n*m*p)
        # We can check a random subset of zeros.
        for _ in range(100):
            i = random.randint(0, n - 1)
            j = random.randint(0, m - 1)
            jp = random.randint(0, m - 1) # j'
            k = random.randint(0, p - 1)
            ip = random.randint(0, n - 1) # i'
            kp = random.randint(0, p - 1) # k'
            
            if i == ip and k == kp and j == jp:
                continue # This is a non-zero constraint, already checked or skip
                
            s = 0
            for r in range(target_rank):
                s += U[i * m + j, r] * V[jp * p + k, r] * W[kp * n + ip, r]
            if s != 0:
                valid = False
                break
        
        if valid:
            # Found a solution!
            best_solution = (U, V, W)
            break
            
    if best_solution:
        U, V, W = best_solution
        # Convert to lists of ints
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Return schoolbook
        U_list = U_school.tolist()
        V_list = V_school.tolist()
        W_list = W_school.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
