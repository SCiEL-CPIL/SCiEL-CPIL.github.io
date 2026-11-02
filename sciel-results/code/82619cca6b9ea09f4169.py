import numpy as np
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank
    target_rank = 24
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s buffer
    
    # We want to find U, V, W such that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    
    # Initialize with random integers
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Fallback: Schoolbook
    U_school = [[0] * best_rank for _ in range(n * m)]
    V_school = [[0] * best_rank for _ in range(m * p)]
    W_school = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
                
    # Try to find lower rank
    # We will use a simple randomized local search / simulated annealing
    # Objective: Minimize error, then minimize rank
    
    def check_solution(U, V, W, rank):
        # Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True

    def get_error(U, V, W, rank):
        err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    err += (s - 1) ** 2
        return err

    # Try ranks from 24 to 35
    for rank in range(24, 36):
        if time.time() > deadline:
            break
            
        # Random initialization
        for attempt in range(500):
            if time.time() > deadline:
                break
                
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            # Initialize with small random integers
            for i in range(n * m):
                for r in range(rank):
                    U[i][r] = np.random.randint(-2, 3)
            for j in range(m * p):
                for r in range(rank):
                    V[j][r] = np.random.randint(-2, 3)
            for k in range(p * n):
                for r in range(rank):
                    W[k][r] = np.random.randint(-2, 3)
                    
            # Local search
            for step in range(200):
                if time.time() > deadline:
                    break
                    
                # Pick a random element to change
                mat = np.random.choice(['U', 'V', 'W'])
                idx = np.random.randint(0, {'U': n*m, 'V': m*p, 'W': p*n}[mat])
                r = np.random.randint(0, rank)
                
                old_val = {'U': U, 'V': V, 'W': W}[mat][idx][r]
                new_val = old_val + np.random.randint(-1, 2)
                
                # Apply change
                {'U': U, 'V': V, 'W': W}[mat][idx][r] = new_val
                
                err_new = get_error(U, V, W, rank)
                
                # If error is 0, we found a solution
                if err_new == 0:
                    if check_solution(U, V, W, rank):
                        return {"U": U, "V": V, "W": W}
                        
                # Revert if worse (simple hill climbing)
                # Note: We don't track old error efficiently here, so this is slow.
                # Let's just keep the change if it's an integer solution candidate?
                # Actually, let's just do a few iterations and check.
                
                # To speed up, we'll just check every 10 steps
                if step % 10 == 0:
                     if err_new == 0:
                         if check_solution(U, V, W, rank):
                             return {"U": U, "V": V, "W": W}

    # If no better solution found, return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
