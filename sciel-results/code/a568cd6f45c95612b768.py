"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication.
    Returns {'U': list of lists, 'V': list of lists, 'W': list of lists}
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The decomposition satisfies: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    We search for half-integer solutions by scaling by 2.
    Let U = U_int / 2, V = V_int / 2, W = W_int / 2.
    Then sum (U_int * V_int * W_int) / 8 = 1 => sum (U_int * V_int * W_int) = 8.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Precompute indices for faster access
    # U indices: i*m+j
    # V indices: j*p+k
    # W indices: k*n+i
    
    # We will search for R from 14 upwards
    for R in range(14, 24):
        if time.time() > deadline:
            break
            
        # Try multiple random restarts for this R
        for restart in range(200):
            if time.time() > deadline:
                break
                
            # Initialize with small random integers
            # Using a range of [-4, 4] to allow some flexibility
            U_int = [[random.randint(-4, 4) for _ in range(R)] for _ in range(n * m)]
            V_int = [[random.randint(-4, 4) for _ in range(R)] for _ in range(m * p)]
            W_int = [[random.randint(-4, 4) for _ in range(R)] for _ in range(p * n)]
            
            # Local search
            # We want sum_r U_int[i*m+j][r] * V_int[j*p+k][r] * W_int[k*n+i][r] == 8 for all i,j,k
            
            # Calculate initial error
            # Error is sum of squared differences from 8
            # Or just check if all are 8.
            
            # To speed up, we can compute the current sums S[i,j,k]
            # S[i,j,k] = sum_r U_int[i*m+j][r] * V_int[j*p+k][r] * W_int[k*n+i][r]
            
            # Let's use a simple hill climbing on the sum of squared errors
            # But checking exactness is the goal.
            
            # We can try to satisfy one equation at a time?
            # Or just do random walks and check if all are 8.
            
            # A better heuristic:
            # Pick a random (i,j,k).
            # Calculate current sum S.
            # We want S=8.
            # Change a random term in the sum to reduce |S-8|.
            
            # However, changing one term affects multiple (i,j,k) equations.
            # U[i*m+j][r] appears in all k for fixed i,j.
            # V[j*p+k][r] appears in all i for fixed j,k.
            # W[k*n+i][r] appears in all j for fixed k,i.
            
            # Let's try a randomized local search minimizing the max absolute error.
            
            best_err = float('inf')
            best_state = None
            
            for step in range(5000): # Limit steps per restart
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
                    if _verify(U_out, V_out, W_out, n, m, p):
                        return {"U": U_out, "V": V_out, "W": W_out}
                
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
                        U_out = [[x/2 for x in row] for row in U_int]
                        V_out = [[x/2 for x in row] for row in V_int]
                        W_out = [[x/2 for x in row] for row in W_int]
                        if _verify(U_out, V_out, W_out, n, m, p):
                            return {"U": U_out, "V": V_out, "W": W_out}
                        break

    # Fallback to schoolbook
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

def _verify(U, V, W, n, m, p):
    """Verify the decomposition."""
    R = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
