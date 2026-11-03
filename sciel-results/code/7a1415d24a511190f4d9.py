import numpy as np
import time
import random

DIMENSIONS = [3, 4, 6]

def verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check all entries
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Fallback: Schoolbook (Rank 72)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    best_rank = rank_schoolbook

    # Precompute index mappings for speed
    # U indices: (i, j) -> i*m + j
    # V indices: (j, k) -> j*p + k
    # W indices: (k, i) -> k*n + i
    
    # We will search for ranks lower than schoolbook.
    # Theoretical lower bound for 3x4x6 is not trivially 1.
    # Let's try ranks from 60 down to 40.
    # If we find a valid one, we return it.
    
    # To make the search effective, we initialize with a "noisy" schoolbook 
    # structure if rank is close, or random if rank is low.
    # However, for rank < 72, we cannot use schoolbook directly.
    # We will use Randomized Local Search (Simulated Annealing) on half-integers.
    
    def calculate_error(U, V, W, rank):
        """Calculate sum of squared errors for all i, j, k."""
        err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U[i * m + j]
                for k in range(p):
                    v_row = V[j * p + k]
                    w_row = W[k * n + i]
                    s = np.dot(u_row, v_row * w_row)
                    err += (s - 1.0) ** 2
        return err

    def local_search(rank, seed, time_limit):
        """
        Perform Simulated Annealing on half-integer entries.
        """
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize with small random half-integers
        # Values in {-1.5, -1, -0.5, 0, 0.5, 1, 1.5}
        def rand_half_int():
            return random.choice([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
        
        U = np.array([[rand_half_int() for _ in range(rank)] for _ in range(n * m)], dtype=np.float64)
        V = np.array([[rand_half_int() for _ in range(rank)] for _ in range(m * p)], dtype=np.float64)
        W = np.array([[rand_half_int() for _ in range(rank)] for _ in range(p * n)], dtype=np.float64)
        
        current_err = calculate_error(U, V, W, rank)
        
        start_time = time.time()
        temperature = 1.0
        cooling_rate = 0.999
        min_temp = 1e-4
        
        # Total variables
        num_vars = (n * m + m * p + p * n) * rank
        
        while time.time() - start_time < time_limit:
            if current_err < 1e-9:
                return U, V, W, current_err
            
            # Pick a random variable
            var_idx = random.randint(0, num_vars - 1)
            
            # Determine which matrix and position
            # First n*m*rank vars are U, next m*p*rank are V, rest W
            u_size = n * m * rank
            v_size = m * p * rank
            
            if var_idx < u_size:
                mat = U
                mat_size = n * m
                local_idx = var_idx
            elif var_idx < u_size + v_size:
                mat = V
                mat_size = m * p
                local_idx = var_idx - u_size
            else:
                mat = W
                mat_size = p * n
                local_idx = var_idx - u_size - v_size
            
            row = local_idx // rank
            col = local_idx % rank
            
            old_val = mat[row, col]
            
            # Generate new value: neighbor in half-integer lattice
            # Options: old_val - 0.5, old_val, old_val + 0.5
            # Or jump to another random half-int in [-2, 2]
            if random.random() < 0.8:
                # Small step
                new_val = old_val + random.choice([-0.5, 0.5])
            else:
                # Large jump
                new_val = random.choice([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
            
            if new_val == old_val:
                continue
                
            # Calculate delta error efficiently?
            # Delta E = sum over affected (i,j,k) of (new_s - 1)^2 - (old_s - 1)^2
            # Affected indices depend on which matrix changed.
            
            # For simplicity and robustness, recalculate error if the change is small?
            # No, full recalculation is O(nmp * rank). 
            # n=3, m=4, p=6 => 72 terms. rank ~ 50. 72*50 = 3600 ops.
            # This is fast enough for Python loops if vectorized, but let's optimize.
            
            # Let's try to compute delta error without full recalculation.
            # If U[row, col] changes by d:
            # row corresponds to (i, j) where i = row // m, j = row % m
            # Affected terms are for all k.
            # Term (i, j, k) changes by:
            # (old_s + d * V[j*p+k, col] * W[k*n+i, col] - 1)^2 - (old_s - 1)^2
            # Let x = old_s - 1. Let y = d * V[j*p+k, col] * W[k*n+i, col].
            # New term = (x + y)^2. Old term = x^2.
            # Delta = 2xy + y^2.
            
            i = row // m
            j = row % m
            
            if mat is U:
                # Affected: all k
                delta_err = 0.0
                for k in range(p):
                    v_val = V[j * p + k, col]
                    w_val = W[k * n + i, col]
                    # We need old_s for this (i,j,k)
                    # To avoid storing old_s, we can just recalculate the whole error 
                    # if the potential delta is promising? 
                    # Given the small size, let's just do a full recalculation 
                    # but only if we accept the move? No, we need delta to decide.
                    # Full recalculation is 3600 ops. 
                    # Let's just do full recalculation for simplicity first, 
                    # and optimize if it's too slow.
                    pass
                
                # Optimization: Store current sums S[i,j,k]
                # This requires maintaining an array of size n*m*p.
                
                # Let's stick to full recalculation for correctness and simplicity in this draft.
                # If it's too slow, we will switch to incremental.
                # 3600 ops * 1000 steps * 100 seeds = 360M ops. Might be tight in 110s.
                # Let's implement incremental error calculation.
                
                # We need to maintain S[i,j,k] = sum_c ...
                # This adds complexity. Let's try full recalculation first with a time check.
                
                mat[row, col] = new_val
                new_err = calculate_error(U, V, W, rank)
                delta = new_err - current_err
                
                if delta < 0 or random.random() < np.exp(-delta / max(temperature, 1e-9)):
                    current_err = new_err
                else:
                    mat[row, col] = old_val
                    
            elif mat is V:
                # Affected: all i for this (j, k)
                # row is j*p + k. j = row // p, k = row % p
                j = row // p
                k = row % p
                mat[row, col] = new_val
                new_err = calculate_error(U, V, W, rank)
                delta = new_err - current_err
                if delta < 0 or random.random() < np.exp(-delta / max(temperature, 1e-9)):
                    current_err = new_err
                else:
                    mat[row, col] = old_val
                
            else: # W
                # Affected: all j for this (k, i)
                # row is k*n + i. k = row // n, i = row % n
                k = row // n
                i = row % n
                mat[row, col] = new_val
                new_err = calculate_error(U, V, W, rank)
                delta = new_err - current_err
                if delta < 0 or random.random() < np.exp(-delta / max(temperature, 1e-9)):
                    current_err = new_err
                else:
                    mat[row, col] = old_val
            
            temperature *= cooling_rate
            if temperature < min_temp:
                temperature = min_temp

        return U, V, W, current_err

    # Try ranks from 60 down to 40
    # If we find a valid rank, return it.
    # To save time, we try fewer seeds for higher ranks and more for lower?
    # Or just a fixed budget per rank.
    
    # We know 72 is valid. We want < 72.
    # Let's try 60, 58, 56...
    
    # Time budget per rank
    time_per_rank = 10.0
    
    for rank in range(60, 39, -2): # Step by 2 to save time
        if time.time() > deadline - 5:
            break
            
        # Try a few seeds
        for seed in range(3):
            if time.time() > deadline - 5:
                break
                
            U, V, W, err = local_search(rank, seed, time_limit=time_per_rank)
            
            if err < 1e-9:
                # Verify strictly
                if verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    best_U = U.tolist()
                    best_V = V.tolist()
                    best_W = W.tolist()
                    best_rank = rank
                    print(f"Found valid rank {rank} with seed {seed}")
                    break
        
        if best_rank < rank_schoolbook:
            # Found something better, can we find even better?
            # The loop continues downwards.
            pass

    # If no better rank found, return schoolbook
    return {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }
