import time
import itertools

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np

def _verify(U, V, W, n=3, m=3, p=3):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    rank = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i])
                if s != 1:
                    return False
    return True


def _try_rank(r, n=3, m=3, p=3, time_budget=10.0, seed=0):
    """
    Attempt to find an exact rank-r decomposition using structured initialization
    and targeted local search with NumPy-accelerated error computation.
    Returns (U, V, W) or None.
    """
    start = time.time()
    import random
    rng = random.Random(seed)
    
    U = np.zeros((n * m, r), dtype=np.int64)
    V = np.zeros((m * p, r), dtype=np.int64)
    W = np.zeros((p * n, r), dtype=np.int64)
    
    def compute_errors():
        """Compute error tensor E[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - 1"""
        errors = np.zeros((n, m, p), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                    errors[i, j, k] = s - 1
        return errors
    
    def total_error(errors):
        return np.sum(errors * errors)
    
    # Strategy 1: Try structured initialization based on patterns
    # For small ranks, we can try to build from known constructions
    
    # Initialize with a pattern that partially satisfies the constraints
    # Each column c should contribute to a subset of (i,j,k) triples
    
    # Try multiple initialization strategies
    best_state = None
    best_err = float('inf')
    
    for init_strategy in range(5):
        if time.time() - start > time_budget:
            break
        
        U[:] = 0
        V[:] = 0
        W[:] = 0
        
        if init_strategy == 0:
            # Random sparse initialization
            for idx in range(n * m):
                for c in range(r):
                    if rng.random() < 0.25:
                        U[idx, c] = rng.choice([1, -1])
            for idx in range(m * p):
                for c in range(r):
                    if rng.random() < 0.25:
                        V[idx, c] = rng.choice([1, -1])
            for idx in range(p * n):
                for c in range(r):
                    if rng.random() < 0.25:
                        W[idx, c] = rng.choice([1, -1])
        
        elif init_strategy == 1:
            # Block-diagonal-like initialization
            # Assign columns to groups of (i,j,k)
            triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
            rng.shuffle(triples)
            for c, (i, j, k) in enumerate(triples[:r]):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
        
        elif init_strategy == 2:
            # Strassen-like pattern for 3x3
            # Use the 7-term formula adapted to 3x3
            # This is heuristic
            for c in range(min(r, 7)):
                # Define patterns for each term
                if c == 0:
                    U[0, c] = 1; V[0, c] = 1; W[0, c] = 1
                elif c == 1:
                    U[0, c] = 1; V[3, c] = 1; W[3, c] = 1
                elif c == 2:
                    U[1, c] = 1; V[0, c] = 1; W[1, c] = 1
                elif c == 3:
                    U[1, c] = 1; V[3, c] = 1; W[4, c] = 1
                elif c == 4:
                    U[3, c] = 1; V[0, c] = 1; W[0, c] = 1
                elif c == 5:
                    U[3, c] = 1; V[3, c] = 1; W[3, c] = 1
                elif c == 6:
                    U[4, c] = 1; V[0, c] = 1; W[1, c] = 1
        
        elif init_strategy == 3:
            # Random with bias toward covering all indices
            for c in range(r):
                # Pick random i, j, k for this column
                i = rng.randrange(n)
                j = rng.randrange(m)
                k = rng.randrange(p)
                U[i * m + j, c] = rng.choice([1, -1])
                V[j * p + k, c] = rng.choice([1, -1])
                W[k * n + i, c] = rng.choice([1, -1])
                # Add some extra terms
                if rng.random() < 0.3:
                    i2 = rng.randrange(n)
                    j2 = rng.randrange(m)
                    k2 = rng.randrange(p)
                    U[i2 * m + j2, c] += rng.choice([1, -1])
                    V[j2 * p + k2, c] += rng.choice([1, -1])
                    W[k2 * n + i2, c] += rng.choice([1, -1])
        
        elif init_strategy == 4:
            # Gradient-inspired: start with small random values and amplify
            for idx in range(n * m):
                for c in range(r):
                    U[idx, c] = rng.choice([0, 1, -1])
            for idx in range(m * p):
                for c in range(r):
                    V[idx, c] = rng.choice([0, 1, -1])
            for idx in range(p * n):
                for c in range(r):
                    W[idx, c] = rng.choice([0, 1, -1])
        
        errors = compute_errors()
        err = total_error(errors)
        
        if err < best_err:
            best_err = err
            best_state = (U.copy(), V.copy(), W.copy())
        
        if err == 0:
            break
    
    if best_state is None:
        return None
    
    U, V, W = best_state
    
    # Local search phase
    current_err = best_err
    no_improve_count = 0
    max_steps = 20000
    
    while no_improve_count < 500 and max_steps > 0:
        if time.time() - start > time_budget:
            break
        max_steps -= 1
        
        # Pick a random factor, row, col to mutate
        factor = rng.choice(['U', 'V', 'W'])
        if factor == 'U':
            row = rng.randrange(n * m)
        elif factor == 'V':
            row = rng.randrange(m * p)
        else:
            row = rng.randrange(p * n)
        col = rng.randrange(r)
        
        M = {'U': U, 'V': V, 'W': W}[factor]
        old_val = M[row, col]
        new_val = rng.choice([0, 1, -1])
        if new_val == old_val:
            continue
        
        M[row, col] = new_val
        
        errors = compute_errors()
        new_err = total_error(errors)
        
        if new_err < current_err:
            current_err = new_err
            no_improve_count = 0
        else:
            # Revert with high probability
            if rng.random() > 0.05:
                M[row, col] = old_val
            no_improve_count += 1
        
        if current_err == 0:
            break
    
    # Convert back to list format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    if _verify(U_list, V_list, W_list, n, m, p):
        return U_list, V_list, W_list
    
    return None


def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # First, build the guaranteed schoolbook rank-27 decomposition
    rank_schoolbook = n * m * p  # 27
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try to find lower-rank decompositions
    # Best known upper bound for 3x3 is 23 (Coppersmith-Winograd family)
    # Lower bound is 19 (Landsberg)
    # We'll try ranks from 23 down to 19
    
    # Allocate time budget across rank attempts
    # Give more time to higher ranks (more likely to succeed)
    rank_time_budgets = {
        23: 20.0,
        22: 15.0,
        21: 12.0,
        20: 10.0,
        19: 8.0
    }
    
    for target_rank in range(23, 18, -1):
        if time.time() - start_time > timeout - 3:
            break
        
        budget = rank_time_budgets.get(target_rank, 8.0)
        
        # Try multiple seeds
        for seed in range(4):
            if time.time() - start_time > timeout - 3:
                break
            
            result = _try_rank(target_rank, n, m, p, time_budget=budget / 4.0, seed=seed)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break  # Found a valid decomposition for this rank, move to lower
    
    return best_result
# EVOLVE-BLOCK-END
