"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication.
    Returns {'U': list of lists, 'V': list of lists, 'W': list of lists}
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The decomposition satisfies: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Precompute index mappings for speed
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # We search for integer solutions to:
    # sum_r U_int[i*m+j][r] * V_int[j*p+k][r] * W_int[k*n+i][r] = 8
    # Then scale by 1/2 to get half-integers.
    
    # We try ranks from 14 upwards.
    # For each rank, we try to find a solution using a hybrid approach:
    # 1. Random initialization
    # 2. Local search (hill climbing) to minimize max error
    # 3. If stuck, try to fix specific equations
    
    best_result = None
    
    for R in range(14, 25):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        found = False
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            # Initialize with random small integers
            # Using a range of [-4, 4] to allow some flexibility
            U_int = [[random.randint(-4, 4) for _ in range(R)] for _ in range(n * m)]
            V_int = [[random.randint(-4, 4) for _ in range(R)] for _ in range(m * p)]
            W_int = [[random.randint(-4, 4) for _ in range(R)] for _ in range(p * n)]
            
            # Check if this random guess works
            if _check_int_decomposition(U_int, V_int, W_int, n, m, p, 8):
                best_result = (R, U_int, V_int, W_int)
                found = True
                break
            
            # Local search
            # We want to minimize the max absolute error
            # Error is sum_r U_int[i*m+j][r] * V_int[j*p+k][r] * W_int[k*n+i][r] - 8
            
            best_err = float('inf')
            best_state = None
            
            for step in range(2000): # Limit steps per restart
                if time.time() > deadline:
                    break
                
                # Compute errors
                errors = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            ui = U_int[i*m+j]
                            vj = V_int[j*p+k]
                            wk = W_int[k*n+i]
                            for r in range(R):
                                s += ui[r] * vj[r] * wk[r]
                            errors.append(s - 8)
                
                max_err = max(abs(e) for e in errors)
                if max_err == 0:
                    # Found a solution
                    U_out = [[x/2 for x in row] for row in U_int]
                    V_out = [[x/2 for x in row] for row in V_int]
                    W_out = [[x/2 for x in row] for row in W_int]
                    # Verify
                    if _check_decomposition(U_out, V_out, W_out, n, m, p):
                        best_result = (R, U_int, V_int, W_int)
                        found = True
                        break
                
                if max_err < best_err:
                    best_err = max_err
                    best_state = (U_int, V_int, W_int)
                
                if max_err == 0:
                    break
                    
                # Random move
                # Pick a random variable to change
                mat = random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    idx = random.randint(0, n*m-1)
                    r = random.randint(0, R-1)
                    # Try small changes
                    delta = random.choice([-2, -1, 1, 2])
                    old_val = U_int[idx][r]
                    U_int[idx][r] = old_val + delta
                elif mat == 'V':
                    idx = random.randint(0, m*p-1)
                    r = random.randint(0, R-1)
                    delta = random.choice([-2, -1, 1, 2])
                    old_val = V_int[idx][r]
                    V_int[idx][r] = old_val + delta
                else:
                    idx = random.randint(0, p*n-1)
                    r = random.randint(0, R-1)
                    delta = random.choice([-2, -1, 1, 2])
                    old_val = W_int[idx][r]
                    W_int[idx][r] = old_val + delta
                
                # Check if error decreased (optional, but helps)
                # For simplicity, we just accept and let the loop handle it, 
                # or we can do a proper hill climb.
                # Let's do a simple hill climb: if error increased, revert.
                
                # Recompute max_err quickly? 
                # It's O(R*n*m*p) which is small.
                errors_new = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            ui = U_int[i*m+j]
                            vj = V_int[j*p+k]
                            wk = W_int[k*n+i]
                            for r in range(R):
                                s += ui[r] * vj[r] * wk[r]
                            errors_new.append(s - 8)
                
                new_max_err = max(abs(e) for e in errors_new)
                
                if new_max_err > max_err:
                    # Revert
                    if mat == 'U':
                        U_int[idx][r] = old_val
                    elif mat == 'V':
                        V_int[idx][r] = old_val
                    else:
                        W_int[idx][r] = old_val
                else:
                    max_err = new_max_err
                    if new_max_err < best_err:
                        best_err = new_max_err
                        best_state = (U_int, V_int, W_int)
                        
                    if new_max_err == 0:
                        break
            
            if found:
                break
        
        if found:
            break
    
    # If no low-rank solution found, fall back to schoolbook
    if best_result is None:
        R = n * m * p
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    R, U_int, V_int, W_int = best_result
    U_out = [[x/2 for x in row] for row in U_int]
    V_out = [[x/2 for x in row] for row in V_int]
    W_out = [[x/2 for x in row] for row in W_int]
    
    return {"U": U_out, "V": V_out, "W": W_out}

def _check_decomposition(U, V, W, n, m, p):
    """Check if U, V, W form a valid decomposition."""
    R = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(R):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _check_int_decomposition(U_int, V_int, W_int, n, m, p, target):
    """Check if U_int, V_int, W_int form a valid integer decomposition."""
    R = len(U_int[0]) if U_int else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(R):
                    s += U_int[i * m + j][r] * V_int[j * p + k][r] * W_int[k * n + i][r]
                if s != target:
                    return False
    return True
# EVOLVE-BLOCK-END
