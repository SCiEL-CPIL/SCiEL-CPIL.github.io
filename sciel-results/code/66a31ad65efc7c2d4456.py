import time
import itertools

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=3, m=3, p=3):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True


def _try_rank(r, n=3, m=3, p=3, time_budget=10.0, seed=0):
    """
    Attempt to find an exact rank-r decomposition by enumerating sparse
    {0,1,-1} factor matrices and checking the tensor identity.
    Returns (U, V, W) or None.
    """
    start = time.time()
    total_cells = n * m * r + m * p * r + p * n * r  # total non-zero budget to consider
    
    # Strategy: for small r, the factors must be very sparse.
    # Each factor matrix has (n*m)*r, (m*p)*r, (p*n)*r entries.
    # For the identity to hold, specific patterns are needed.
    # 
    # We use a randomized local search: start from a random sparse configuration,
    # then mutate and check.
    
    import random
    rng = random.Random(seed)
    
    # For r <= 20, we can try to construct via known identities or search.
    # Let's try a simple hill-climbing:
    # 1. Initialize U, V, W with random sparse {0, 1, -1} entries
    # 2. Compute the "error tensor" E[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - 1
    # 3. Mutate a single entry to reduce error
    
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    # Initialize with a random pattern
    for row in range(n * m):
        for col in range(r):
            if rng.random() < 0.3:
                U[row][col] = rng.choice([1, -1])
    for row in range(m * p):
        for col in range(r):
            if rng.random() < 0.3:
                V[row][col] = rng.choice([1, -1])
    for row in range(p * n):
        for col in range(r):
            if rng.random() < 0.3:
                W[row][col] = rng.choice([1, -1])
    
    def compute_error():
        """Compute sum of squared errors over all (i,j,k)."""
        err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    err += (s - 1) ** 2
        return err
    
    current_err = compute_error()
    best_err = current_err
    best_state = (
        [row[:] for row in U],
        [row[:] for row in V],
        [row[:] for row in W]
    )
    
    no_improve_count = 0
    max_steps = 5000
    
    while no_improve_count < 200 and max_steps > 0:
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
        old_val = M[row][col]
        new_val = rng.choice([0, 1, -1])
        if new_val == old_val:
            continue
        M[row][col] = new_val
        
        new_err = compute_error()
        if new_err < current_err:
            current_err = new_err
            if new_err < best_err:
                best_err = new_err
                best_state = (
                    [row[:] for row in U],
                    [row[:] for row in V],
                    [row[:] for row in W]
                )
            no_improve_count = 0
        else:
            # Revert with some probability (simulated annealing-like)
            if rng.random() > 0.1:
                M[row][col] = old_val
            no_improve_count += 1
    
    # Check if we found an exact solution
    U_best, V_best, W_best = best_state
    if _verify(U_best, V_best, W_best, n, m, p):
        return U_best, V_best, W_best
    
    return None


def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s limit
    
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
    # Known lower bounds: for 3x3 matrix multiplication, rank is at least 19
    # (Landsberg), and Strassen gives 7 for 2x2. For 3x3, best known is around 21-23.
    # We'll try ranks from 22 down to 19.
    
    for target_rank in range(22, 18, -1):
        if time.time() - start_time > timeout - 5:
            break
        
        # Try a few random seeds for each rank
        for seed in range(3):
            if time.time() - start_time > timeout - 5:
                break
            
            result = _try_rank(target_rank, n, m, p, time_budget=8.0, seed=seed)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break  # Found a valid decomposition for this rank, move to lower
    
    return best_result
# EVOLVE-BLOCK-END
