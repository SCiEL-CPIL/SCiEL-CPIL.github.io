"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    
    Strategy:
    1. Start with schoolbook (rank 36) as fallback.
    2. Use structured random search with better initialization.
    3. Exploit the fact that we only need to verify 36 active triples + ensure
       all other combinations sum to 0.
    4. Use fast incremental error computation for local search.
    5. Use vectorized numpy operations for faster evaluation.
    6. Add local search to improve near-miss candidates.
    """
    start_time = time.time()
    deadline = start_time + 145  # 145s budget, leaving 5s margin for 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    nm, mp, pn = n * m, m * p, p * n  # 9, 12, 12
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Active triples: (a, b, c) where a=i*m+j, b=j*p+k, c=k*n+i
    active_triples = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                active_triples.add((a, b, c))
    
    # Precompute all triples that need to be checked
    all_triples = list(itertools.product(range(nm), range(mp), range(pn)))
    active_set = set(active_triples)
    
    # Create index arrays for vectorized computation
    A_idx = np.array([a for a, b, c in all_triples])
    B_idx = np.array([b for a, b, c in all_triples])
    C_idx = np.array([c for a, b, c in all_triples])
    
    # Expected values: 1 for active triples, 0 for inactive
    expected_vals = np.array([1 if (a, b, c) in active_set else 0 for a, b, c in all_triples])
    
    # Precompute for fast error computation
    # For each triple index, store (a, b, c)
    triple_a = np.array([a for a, b, c in all_triples])
    triple_b = np.array([b for a, b, c in all_triples])
    triple_c = np.array([c for a, b, c in all_triples])
    
    def compute_prod(U_arr, V_arr, W_arr, r):
        """Compute the tensor product sum for all triples."""
        prod = np.zeros(len(all_triples), dtype=np.int32)
        for ci in range(r):
            prod += U_arr[A_idx, ci] * V_arr[B_idx, ci] * W_arr[C_idx, ci]
        return prod
    
    def verify_numpy(U, V, W, r):
        """Check if U, V, W with rank r satisfy all constraints using numpy."""
        U_arr = np.array(U, dtype=np.int32)
        V_arr = np.array(V, dtype=np.int32)
        W_arr = np.array(W, dtype=np.int32)
        prod = compute_prod(U_arr, V_arr, W_arr, r)
        return np.array_equal(prod, expected_vals)
    
    def error_count_numpy(U, V, W, r):
        """Count how many triples are violated using numpy."""
        U_arr = np.array(U, dtype=np.int32)
        V_arr = np.array(V, dtype=np.int32)
        W_arr = np.array(W, dtype=np.int32)
        prod = compute_prod(U_arr, V_arr, W_arr, r)
        return int(np.sum(prod != expected_vals))
    
    def fast_local_search(U, V, W, r, max_iters=10000):
        """
        Fast local search using incremental error updates.
        When we change U[row][col] from old to new, the change in prod for triple t is:
        (new - old) * V[b_t][col] * W[c_t][col]
        """
        U_arr = np.array(U, dtype=np.int32)
        V_arr = np.array(V, dtype=np.int32)
        W_arr = np.array(W, dtype=np.int32)
        
        prod = compute_prod(U_arr, V_arr, W_arr, r)
        current_errors = int(np.sum(prod != expected_vals))
        
        # Precompute V and W columns for fast lookup
        # V_col[b][col] = V_arr[b, col]
        # W_col[c][col] = W_arr[c, col]
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if current_errors == 0:
                return U_arr.tolist(), V_arr.tolist(), W_arr.tolist()
            
            # Randomly pick a matrix, row, column
            mat_choice = random.random()
            if mat_choice < 0.33:
                mat = 'U'
                row = random.randint(0, nm - 1)
            elif mat_choice < 0.66:
                mat = 'V'
                row = random.randint(0, mp - 1)
            else:
                mat = 'W'
                row = random.randint(0, pn - 1)
            
            col = random.randint(0, r - 1)
            
            if mat == 'U':
                old_val = U_arr[row, col]
                new_val = random.choice([-2, -1, 0, 1, 2])
                if new_val == old_val:
                    continue
                delta = new_val - old_val
                # Change in prod for each triple t: delta * V[b_t][col] * W[c_t][col]
                delta_prod = delta * V_arr[triple_b, col] * W_arr[triple_c, col]
                new_prod = prod + delta_prod
                new_errors = int(np.sum(new_prod != expected_vals))
                if new_errors < current_errors:
                    U_arr[row, col] = new_val
                    prod = new_prod
                    current_errors = new_errors
                # else: no change
            elif mat == 'V':
                old_val = V_arr[row, col]
                new_val = random.choice([-2, -1, 0, 1, 2])
                if new_val == old_val:
                    continue
                delta = new_val - old_val
                # Change in prod for each triple t: delta * U[a_t][col] * W[c_t][col]
                delta_prod = delta * U_arr[triple_a, col] * W_arr[triple_c, col]
                new_prod = prod + delta_prod
                new_errors = int(np.sum(new_prod != expected_vals))
                if new_errors < current_errors:
                    V_arr[row, col] = new_val
                    prod = new_prod
                    current_errors = new_errors
            else:  # W
                old_val = W_arr[row, col]
                new_val = random.choice([-2, -1, 0, 1, 2])
                if new_val == old_val:
                    continue
                delta = new_val - old_val
                # Change in prod for each triple t: delta * U[a_t][col] * V[b_t][col]
                delta_prod = delta * U_arr[triple_a, col] * V_arr[triple_b, col]
                new_prod = prod + delta_prod
                new_errors = int(np.sum(new_prod != expected_vals))
                if new_errors < current_errors:
                    W_arr[row, col] = new_val
                    prod = new_prod
                    current_errors = new_errors
        
        return U_arr.tolist(), V_arr.tolist(), W_arr.tolist()
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def try_rank_structured(r, max_attempts=200):
        """Try to find a rank-r decomposition using structured random search."""
        if time.time() > deadline:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Initialize with structured patterns
            strategy = attempt % 5
            
            if strategy == 0:
                # Sparse random: mostly zeros with some ±1
                U = [[0] * r for _ in range(nm)]
                V = [[0] * r for _ in range(mp)]
                W = [[0] * r for _ in range(pn)]
                for a in range(nm):
                    num_nonzero = random.randint(1, min(3, r))
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        U[a][ci] = random.choice([-1, 1])
                for b in range(mp):
                    num_nonzero = random.randint(1, min(3, r))
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        V[b][ci] = random.choice([-1, 1])
                for c in range(pn):
                    num_nonzero = random.randint(1, min(3, r))
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        W[c][ci] = random.choice([-1, 1])
                        
            elif strategy == 1:
                # Dense random with small values
                U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(nm)]
                V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(mp)]
                W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(pn)]
                
            elif strategy == 2:
                # Strassen-inspired: use block structure with correlated entries
                U = [[0] * r for _ in range(nm)]
                V = [[0] * r for _ in range(mp)]
                W = [[0] * r for _ in range(pn)]
                
                for ci in range(r):
                    # Use structured patterns based on ci
                    if ci % 3 == 0:
                        for i in range(n):
                            for j in range(m):
                                U[i * m + j][ci] = random.choice([-1, 1])
                        for j in range(m):
                            for k in range(p):
                                V[j * p + k][ci] = random.choice([-1, 1])
                        for k in range(p):
                            for i in range(n):
                                W[k * n + i][ci] = random.choice([-1, 1])
                    elif ci % 3 == 1:
                        for i in range(n):
                            for j in range(m):
                                U[i * m + j][ci] = random.choice([-1, 0, 1])
                        for j in range(m):
                            for k in range(p):
                                V[j * p + k][ci] = random.choice([-1, 0, 1])
                        for k in range(p):
                            for i in range(n):
                                W[k * n + i][ci] = random.choice([-1, 0, 1])
                    else:
                        for i in range(n):
                            for j in range(m):
                                U[i * m + j][ci] = random.choice([-1, 0, 1])
                        for j in range(m):
                            for k in range(p):
                                V[j * p + k][ci] = random.choice([-1, 0, 1])
                        for k in range(p):
                            for i in range(n):
                                W[k * n + i][ci] = random.choice([-1, 0, 1])
                        
            elif strategy == 3:
                # Very sparse: each row has exactly 1-2 nonzeros
                U = [[0] * r for _ in range(nm)]
                V = [[0] * r for _ in range(mp)]
                W = [[0] * r for _ in range(pn)]
                for a in range(nm):
                    num_nz = random.randint(1, 2)
                    cols = random.sample(range(r), min(num_nz, r))
                    for ci in cols:
                        U[a][ci] = random.choice([-1, 1])
                for b in range(mp):
                    num_nz = random.randint(1, 2)
                    cols = random.sample(range(r), min(num_nz, r))
                    for ci in cols:
                        V[b][ci] = random.choice([-1, 1])
                for c in range(pn):
                    num_nz = random.randint(1, 2)
                    cols = random.sample(range(r), min(num_nz, r))
                    for ci in cols:
                        W[c][ci] = random.choice([-1, 1])
            
            else:
                # Mixed strategy: start from a random base and perturb
                U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(nm)]
                V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(mp)]
                W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(pn)]
                # Zero out some entries to make it sparser
                for a in range(nm):
                    for ci in range(r):
                        if random.random() < 0.4:
                            U[a][ci] = 0
                for b in range(mp):
                    for ci in range(r):
                        if random.random() < 0.4:
                            V[b][ci] = 0
                for c in range(pn):
                    for ci in range(r):
                        if random.random() < 0.4:
                            W[c][ci] = 0
            
            # Quick check with numpy
            if verify_numpy(U, V, W, r):
                return {"U": U, "V": V, "W": W}
            
            # If close, try fast local search
            errs = error_count_numpy(U, V, W, r)
            if errs <= 15 and time.time() < deadline - 5:
                U_ls, V_ls, W_ls = fast_local_search(U, V, W, r, max_iters=5000)
                if error_count_numpy(U_ls, V_ls, W_ls, r) == 0:
                    return {"U": U_ls, "V": V_ls, "W": W_ls}
        
        return None
    
    # Try to find lower rank decompositions, starting from a more realistic rank
    # Known lower bounds for 3x3x4 are around 21-22
    for r in range(21, best_rank):
        if time.time() > deadline:
            break
        
        time_remaining = deadline - time.time()
        if time_remaining < 3:
            break
        
        # Spend more time on promising ranks
        if r <= 23:
            max_attempts = min(600, int(time_remaining * 20))
        elif r <= 25:
            max_attempts = min(400, int(time_remaining * 12))
        else:
            max_attempts = min(200, int(time_remaining * 6))
        
        result = try_rank_structured(r, max_attempts)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    return best_result
# EVOLVE-BLOCK-END
