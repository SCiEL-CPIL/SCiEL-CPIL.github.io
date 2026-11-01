"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from fractions import Fraction

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
    4. Use local search / hill climbing to refine candidates.
    5. Use vectorized numpy operations for faster evaluation.
    6. Add local search to improve near-miss candidates.
    """
    start_time = time.time()
    deadline = start_time + 140  # 140s budget, leaving 10s margin for 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
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
    all_triples = list(itertools.product(range(n * m), range(m * p), range(p * n)))
    active_set = set(active_triples)
    
    # Precompute numpy arrays for fast evaluation
    import numpy as np
    
    # Create index arrays for vectorized computation
    A_idx = np.array([a for a, b, c in all_triples])
    B_idx = np.array([b for a, b, c in all_triples])
    C_idx = np.array([c for a, b, c in all_triples])
    
    # Expected values: 1 for active triples, 0 for inactive
    expected_vals = np.array([1 if (a, b, c) in active_set else 0 for a, b, c in all_triples])
    
    def verify_numpy(U, V, W, r):
        """Check if U, V, W with rank r satisfy all constraints using numpy."""
        U_arr = np.array(U, dtype=int)
        V_arr = np.array(V, dtype=int)
        W_arr = np.array(W, dtype=int)
        
        # For each rank component ci, compute U[a,ci] * V[b,ci] * W[c,ci]
        # Sum over ci
        # Shape: (len(all_triples),)
        prod = np.zeros(len(all_triples), dtype=int)
        for ci in range(r):
            prod += U_arr[A_idx, ci] * V_arr[B_idx, ci] * W_arr[C_idx, ci]
        
        return np.array_equal(prod, expected_vals)
    
    def error_count_numpy(U, V, W, r):
        """Count how many triples are violated using numpy."""
        U_arr = np.array(U, dtype=int)
        V_arr = np.array(V, dtype=int)
        W_arr = np.array(W, dtype=int)
        
        prod = np.zeros(len(all_triples), dtype=int)
        for ci in range(r):
            prod += U_arr[A_idx, ci] * V_arr[B_idx, ci] * W_arr[C_idx, ci]
        
        return int(np.sum(prod != expected_vals))
    
    def local_search(U, V, W, r, max_iters=5000):
        """Local search to reduce error count."""
        U_arr = [row[:] for row in U]
        V_arr = [row[:] for row in V]
        W_arr = [row[:] for row in W]
        
        current_errors = error_count_numpy(U_arr, V_arr, W_arr, r)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if current_errors == 0:
                return U_arr, V_arr, W_arr
            
            # Randomly pick a matrix, row, column, and value to change
            mat_choice = random.choice(['U', 'V', 'W'])
            if mat_choice == 'U':
                row = random.randint(0, n * m - 1)
                arr = U_arr
            elif mat_choice == 'V':
                row = random.randint(0, m * p - 1)
                arr = V_arr
            else:
                row = random.randint(0, p * n - 1)
                arr = W_arr
            
            col = random.randint(0, r - 1)
            old_val = arr[row][col]
            
            # Try changing to a new value
            new_val = random.choice([-2, -1, 0, 1, 2])
            if new_val == old_val:
                new_val = (old_val + 1) % 3 - 1  # cycle through -1, 0, 1
            
            arr[row][col] = new_val
            
            new_errors = error_count_numpy(U_arr, V_arr, W_arr, r)
            
            if new_errors >= current_errors:
                # Revert
                arr[row][col] = old_val
            else:
                current_errors = new_errors
        
        return U_arr, V_arr, W_arr
    
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
            strategy = attempt % 4
            
            if strategy == 0:
                # Sparse random: mostly zeros with some ±1
                U = [[0] * r for _ in range(n * m)]
                V = [[0] * r for _ in range(m * p)]
                W = [[0] * r for _ in range(p * n)]
                for a in range(n * m):
                    num_nonzero = random.randint(1, min(4, r))
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        U[a][ci] = random.choice([-1, 1])
                for b in range(m * p):
                    num_nonzero = random.randint(1, min(4, r))
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        V[b][ci] = random.choice([-1, 1])
                for c in range(p * n):
                    num_nonzero = random.randint(1, min(4, r))
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        W[c][ci] = random.choice([-1, 1])
                        
            elif strategy == 1:
                # Dense random with small values
                scale = 2
                U = [[random.randint(-scale, scale) for _ in range(r)] for _ in range(n * m)]
                V = [[random.randint(-scale, scale) for _ in range(r)] for _ in range(m * p)]
                W = [[random.randint(-scale, scale) for _ in range(r)] for _ in range(p * n)]
                
            elif strategy == 2:
                # Strassen-inspired: use block structure
                U = [[0] * r for _ in range(n * m)]
                V = [[0] * r for _ in range(m * p)]
                W = [[0] * r for _ in range(p * n)]
                
                for ci in range(r):
                    u_vals = [random.choice([-1, 0, 1]) for _ in range(n * m)]
                    v_vals = [random.choice([-1, 0, 1]) for _ in range(m * p)]
                    w_vals = [random.choice([-1, 0, 1]) for _ in range(p * n)]
                    for a in range(n * m):
                        U[a][ci] = u_vals[a]
                    for b in range(m * p):
                        V[b][ci] = v_vals[b]
                    for c in range(p * n):
                        W[c][ci] = w_vals[c]
                        
            else:
                # Very sparse: each row has exactly 1 nonzero
                U = [[0] * r for _ in range(n * m)]
                V = [[0] * r for _ in range(m * p)]
                W = [[0] * r for _ in range(p * n)]
                for a in range(n * m):
                    ci = random.randint(0, r - 1)
                    U[a][ci] = random.choice([-1, 1])
                for b in range(m * p):
                    ci = random.randint(0, r - 1)
                    V[b][ci] = random.choice([-1, 1])
                for c in range(p * n):
                    ci = random.randint(0, r - 1)
                    W[c][ci] = random.choice([-1, 1])
            
            # Quick check with numpy
            if verify_numpy(U, V, W, r):
                return {"U": U, "V": V, "W": W}
            
            # If close, try local search
            errs = error_count_numpy(U, V, W, r)
            if errs <= 5 and time.time() < deadline - 10:
                U_ls, V_ls, W_ls = local_search(U, V, W, r, max_iters=2000)
                if error_count_numpy(U_ls, V_ls, W_ls, r) == 0:
                    return {"U": U_ls, "V": V_ls, "W": W_ls}
        
        return None
    
    # Try to find lower rank decompositions
    for r in range(18, best_rank):
        if time.time() > deadline:
            break
        
        time_remaining = deadline - time.time()
        if time_remaining < 5:
            break
        
        # Spend more time on promising ranks
        if r <= 22:
            max_attempts = min(800, int(time_remaining * 15))
        elif r <= 26:
            max_attempts = min(400, int(time_remaining * 8))
        else:
            max_attempts = min(200, int(time_remaining * 4))
        
        result = try_rank_structured(r, max_attempts)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    return best_result
# EVOLVE-BLOCK-END
