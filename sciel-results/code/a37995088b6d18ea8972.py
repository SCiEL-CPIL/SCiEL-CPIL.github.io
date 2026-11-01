import time
import random
import itertools

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 42
    
    # Trivial decomposition as baseline
    U = [[0] * trivial_rank for _ in range(n * m)]
    V = [[0] * trivial_rank for _ in range(m * p)]
    W = [[0] * trivial_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    best_rank = trivial_rank
    best_result = {"U": U, "V": V, "W": W}
    
    def check_decomposition(U_r, V_r, W_r, rank):
        """Check if decomposition is valid: sum_t U[i*m+j][t]*V[j*p+k][t]*W[k*n+i][t] == 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for t in range(rank):
                        s += U_r[i * m + j][t] * V_r[j * p + k][t] * W_r[k * n + i][t]
                    if s != 1:
                        return False
        return True
    
    def try_rank(target_rank, max_attempts=1000):
        """Try to find a valid decomposition of given rank"""
        if time.time() > deadline - 5:
            return None
        
        # Strategy 1: Start from trivial and try to merge terms
        # Strategy 2: Random search with structured initialization
        # Strategy 3: Pattern-based search
        
        # For small reductions, try systematic approaches
        reduction = trivial_rank - target_rank
        
        if reduction <= 3:
            # Try removing specific terms from trivial decomposition
            # and see if we can compensate
            for attempt in range(min(max_attempts, 500)):
                if time.time() > deadline - 5:
                    break
                
                # Randomly select 'reduction' terms to remove
                indices_to_remove = random.sample(range(trivial_rank), reduction)
                
                U_r = [[0] * target_rank for _ in range(n * m)]
                V_r = [[0] * target_rank for _ in range(m * p)]
                W_r = [[0] * target_rank for _ in range(p * n)]
                
                # Copy remaining terms
                kept_indices = [i for i in range(trivial_rank) if i not in indices_to_remove]
                for new_col, old_col in enumerate(kept_indices):
                    for i in range(n):
                        for j in range(m):
                            U_r[i * m + j][new_col] = U[i * m + j][old_col]
                    for j in range(m):
                        for k in range(p):
                            V_r[j * p + k][new_col] = V[j * p + k][old_col]
                    for k in range(p):
                        for i in range(n):
                            W_r[k * n + i][new_col] = W[k * n + i][old_col]
                
                # Now try to adjust entries to make it work
                # This is a constrained optimization - try random perturbations
                for pert_attempt in range(20):
                    if time.time() > deadline - 5:
                        break
                    
                    U_p = [row[:] for row in U_r]
                    V_p = [row[:] for row in V_r]
                    W_p = [row[:] for row in W_r]
                    
                    # Randomly perturb some entries
                    for _ in range(reduction * 2):
                        mat_choice = random.choice(['U', 'V', 'W'])
                        if mat_choice == 'U':
                            row = random.randint(0, n*m-1)
                            col = random.randint(0, target_rank-1)
                            U_p[row][col] += random.choice([-1, 1])
                        elif mat_choice == 'V':
                            row = random.randint(0, m*p-1)
                            col = random.randint(0, target_rank-1)
                            V_p[row][col] += random.choice([-1, 1])
                        else:
                            row = random.randint(0, p*n-1)
                            col = random.randint(0, target_rank-1)
                            W_p[row][col] += random.choice([-1, 1])
                    
                    if check_decomposition(U_p, V_p, W_p, target_rank):
                        return {"U": U_p, "V": V_p, "W": W_p}
        
        # Strategy 2: Pure random search with small integers
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Use small range for better chance of hitting exact 1
            val_range = [-2, -1, 0, 1, 2]
            
            U_r = [[random.choice(val_range) for _ in range(target_rank)] for _ in range(n * m)]
            V_r = [[random.choice(val_range) for _ in range(target_rank)] for _ in range(m * p)]
            W_r = [[random.choice(val_range) for _ in range(target_rank)] for _ in range(p * n)]
            
            if check_decomposition(U_r, V_r, W_r, target_rank):
                return {"U": U_r, "V": V_r, "W": W_r}
        
        # Strategy 3: Structured Winograd-like patterns
        # For 2x3x7, try patterns based on small matrix multiplication decompositions
        if target_rank <= 35:
            for attempt in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Try a pattern where some terms share structure
                U_r = [[0] * target_rank for _ in range(n * m)]
                V_r = [[0] * target_rank for _ in range(m * p)]
                W_r = [[0] * target_rank for _ in range(p * n)]
                
                # Initialize with random sparse patterns
                for t in range(target_rank):
                    # Each term has a certain density
                    density = random.randint(1, 3)
                    
                    # Choose which U entries are non-zero
                    u_rows = random.sample(range(n * m), min(density, n * m))
                    v_rows = random.sample(range(m * p), min(density, m * p))
                    w_rows = random.sample(range(p * n), min(density, p * n))
                    
                    for r in u_rows:
                        U_r[r][t] = random.choice([-1, 1])
                    for r in v_rows:
                        V_r[r][t] = random.choice([-1, 1])
                    for r in w_rows:
                        W_r[r][t] = random.choice([-1, 1])
                
                if check_decomposition(U_r, V_r, W_r, target_rank):
                    return {"U": U_r, "V": V_r, "W": W_r}
        
        return None
    
    # Search for better ranks
    # Start from just below trivial and go down
    for target_rank in range(trivial_rank - 1, max(20, trivial_rank - 15), -1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(target_rank, max_attempts=500)
        if result is not None:
            best_rank = target_rank
            best_result = result
            # Found a better rank, continue searching lower
            continue
    
    return best_result
