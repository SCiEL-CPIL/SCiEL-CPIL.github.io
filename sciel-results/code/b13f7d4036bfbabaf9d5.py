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
    7. Use incremental error computation for faster local search.
    8. Try lower ranks first with more aggressive search.
    """
    start_time = time.time()
    deadline = start_time + 145  # 145s budget, leaving 5s margin
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    Nm, Mp, Pn = n * m, m * p, p * n  # 9, 12, 12
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(Nm)]
        V = [[0] * rank for _ in range(Mp)]
        W = [[0] * rank for _ in range(Pn)]
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
    all_triples = list(itertools.product(range(Nm), range(Mp), range(Pn)))
    active_set = set(active_triples)
    num_triples = len(all_triples)
    
    # Precompute numpy arrays for fast evaluation
    import numpy as np
    
    # Create index arrays for vectorized computation
    A_idx = np.array([a for a, b, c in all_triples], dtype=np.int32)
    B_idx = np.array([b for a, b, c in all_triples], dtype=np.int32)
    C_idx = np.array([c for a, b, c in all_triples], dtype=np.int32)
    
    # Expected values: 1 for active triples, 0 for inactive
    expected_vals = np.array([1 if (a, b, c) in active_set else 0 for a, b, c in all_triples], dtype=np.int32)
    
    def compute_product(U_arr, V_arr, W_arr, r):
        """Compute the full product tensor using numpy."""
        prod = np.zeros(num_triples, dtype=np.int32)
        for ci in range(r):
            prod += U_arr[A_idx, ci] * V_arr[B_idx, ci] * W_arr[C_idx, ci]
        return prod
    
    def verify_numpy(U, V, W, r):
        """Check if U, V, W with rank r satisfy all constraints using numpy."""
        U_arr = np.array(U, dtype=np.int32)
        V_arr = np.array(V, dtype=np.int32)
        W_arr = np.array(W, dtype=np.int32)
        prod = compute_product(U_arr, V_arr, W_arr, r)
        return np.array_equal(prod, expected_vals)
    
    def error_count_numpy(U, V, W, r):
        """Count how many triples are violated using numpy."""
        U_arr = np.array(U, dtype=np.int32)
        V_arr = np.array(V, dtype=np.int32)
        W_arr = np.array(W, dtype=np.int32)
        prod = compute_product(U_arr, V_arr, W_arr, r)
        return int(np.sum(prod != expected_vals))
    
    def fast_local_search(U, V, W, r, max_iters=3000):
        """
        Fast local search using incremental error updates.
        Precomputes which triples are affected by each (matrix, row, col) change.
        """
        U_arr = [row[:] for row in U]
        V_arr = [row[:] for row in V]
        W_arr = [row[:] for row in W]
        
        U_np = np.array(U_arr, dtype=np.int32)
        V_np = np.array(V_arr, dtype=np.int32)
        W_np = np.array(W_arr, dtype=np.int32)
        
        current_prod = compute_product(U_np, V_np, W_np, r)
        current_errors = int(np.sum(current_prod != expected_vals))
        
        # Precompute for each matrix/row/col which triples are affected
        # For U[a, ci]: affected triples are those with A_idx == a
        # For V[b, ci]: affected triples are those with B_idx == b
        # For W[c, ci]: affected triples are those with C_idx == c
        
        # Build lookup: for each (mat_type, row, col), get list of triple indices
        # We'll compute affected indices on the fly using precomputed masks
        
        # Precompute masks for each row in each matrix
        U_masks = {}  # U_masks[a][ci] = boolean mask of triples with A_idx == a
        V_masks = {}  # V_masks[b][ci] = boolean mask of triples with B_idx == b
        W_masks = {}  # W_masks[c][ci] = boolean mask of triples with C_idx == c
        
        # Actually, for a given (matrix, row, col), the affected triples are those
        # where the corresponding index matches the row. The col doesn't matter for
        # which triples are affected, only for the delta value.
        
        # For U[a, ci]: triples where A_idx == a
        for a in range(Nm):
            U_masks[a] = (A_idx == a)
        for b in range(Mp):
            V_masks[b] = (B_idx == b)
        for c in range(Pn):
            W_masks[c] = (C_idx == c)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if current_errors == 0:
                return U_arr, V_arr, W_arr
            
            # Randomly pick a matrix, row, column, and value to change
            mat_choice = random.choice(['U', 'V', 'W'])
            if mat_choice == 'U':
                row = random.randint(0, Nm - 1)
                arr = U_arr
                arr_np = U_np
                mask = U_masks[row]
            elif mat_choice == 'V':
                row = random.randint(0, Mp - 1)
                arr = V_arr
                arr_np = V_np
                mask = V_masks[row]
            else:
                row = random.randint(0, Pn - 1)
                arr = W_arr
                arr_np = W_np
                mask = W_masks[row]
            
            col = random.randint(0, r - 1)
            old_val = arr[row][col]
            
            # Try changing to a new value
            new_val = random.choice([-2, -1, 0, 1, 2])
            if new_val == old_val:
                new_val = (old_val + 1) % 3 - 1
            
            # Compute delta: how much each affected triple's product changes
            # For U: delta = (new_val - old_val) * V[b, ci] * W[c, ci]
            # For V: delta = U[a, ci] * (new_val - old_val) * W[c, ci]
            # For W: delta = U[a, ci] * V[b, ci] * (new_val - old_val)
            
            delta_val = new_val - old_val
            
            # Get the affected triple indices
            affected_indices = np.where(mask)[0]
            
            if len(affected_indices) == 0:
                arr[row][col] = new_val
                continue
            
            # Compute the delta for each affected triple
            if mat_choice == 'U':
                # delta[i] = delta_val * V[B_idx[i], ci] * W[C_idx[i], ci]
                deltas = delta_val * V_np[B_idx[affected_indices], ci] * W_np[C_idx[affected_indices], ci]
            elif mat_choice == 'V':
                deltas = U_np[A_idx[affected_indices], ci] * delta_val * W_np[C_idx[affected_indices], ci]
            else:
                deltas = U_np[A_idx[affected_indices], ci] * V_np[B_idx[affected_indices], ci] * delta_val
            
            # Apply delta to current_prod for affected indices
            old_prod_vals = current_prod[affected_indices].copy()
            new_prod_vals = old_prod_vals + deltas
            
            # Count how many errors change
            old_errors_affected = int(np.sum(old_prod_vals != expected_vals[affected_indices]))
            new_errors_affected = int(np.sum(new_prod_vals != expected_vals[affected_indices]))
            
            delta_errors = new_errors_affected - old_errors_affected
            
            if delta_errors >= 0:
                # Revert or try different value
                continue
            else:
                # Accept the change
                arr[row][col] = new_val
                arr_np[row, col] = new_val
                current_prod[affected_indices] = new_prod_vals
                current_errors += delta_errors
        
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
            strategy = attempt % 5
            
            if strategy == 0:
                # Sparse random: mostly zeros with some ±1
                U = [[0] * r for _ in range(Nm)]
                V = [[0] * r for _ in range(Mp)]
                W = [[0] * r for _ in range(Pn)]
                for a in range(Nm):
                    num_nonzero = random.randint(1, min(3, r))
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        U[a][ci] = random.choice([-1, 1])
                for b in range(Mp):
                    num_nonzero = random.randint(1, min(3, r))
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        V[b][ci] = random.choice([-1, 1])
                for c in range(Pn):
                    num_nonzero = random.randint(1, min(3, r))
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        W[c][ci] = random.choice([-1, 1])
                        
            elif strategy == 1:
                # Dense random with small values
                scale = 2
                U = [[random.randint(-scale, scale) for _ in range(r)] for _ in range(Nm)]
                V = [[random.randint(-scale, scale) for _ in range(r)] for _ in range(Mp)]
                W = [[random.randint(-scale, scale) for _ in range(r)] for _ in range(Pn)]
                
            elif strategy == 2:
                # Strassen-inspired: use block structure
                U = [[0] * r for _ in range(Nm)]
                V = [[0] * r for _ in range(Mp)]
                W = [[0] * r for _ in range(Pn)]
                
                for ci in range(r):
                    u_vals = [random.choice([-1, 0, 1]) for _ in range(Nm)]
                    v_vals = [random.choice([-1, 0, 1]) for _ in range(Mp)]
                    w_vals = [random.choice([-1, 0, 1]) for _ in range(Pn)]
                    for a in range(Nm):
                        U[a][ci] = u_vals[a]
                    for b in range(Mp):
                        V[b][ci] = v_vals[b]
                    for c in range(Pn):
                        W[c][ci] = w_vals[c]
                        
            elif strategy == 3:
                # Very sparse: each row has exactly 1 nonzero
                U = [[0] * r for _ in range(Nm)]
                V = [[0] * r for _ in range(Mp)]
                W = [[0] * r for _ in range(Pn)]
                for a in range(Nm):
                    ci = random.randint(0, r - 1)
                    U[a][ci] = random.choice([-1, 1])
                for b in range(Mp):
                    ci = random.randint(0, r - 1)
                    V[b][ci] = random.choice([-1, 1])
                for c in range(Pn):
                    ci = random.randint(0, r - 1)
                    W[c][ci] = random.choice([-1, 1])
            else:
                # Half-integer friendly: use values from {-2,-1,0,1,2} with bias toward small
                U = [[random.choice([-1, 0, 0, 1]) for _ in range(r)] for _ in range(Nm)]
                V = [[random.choice([-1, 0, 0, 1]) for _ in range(r)] for _ in range(Mp)]
                W = [[random.choice([-1, 0, 0, 1]) for _ in range(r)] for _ in range(Pn)]
            
            # Quick check with numpy
            if verify_numpy(U, V, W, r):
                return {"U": U, "V": V, "W": W}
            
            # If close, try fast local search
            errs = error_count_numpy(U, V, W, r)
            if errs <= 10 and time.time() < deadline - 5:
                U_ls, V_ls, W_ls = fast_local_search(U, V, W, r, max_iters=5000)
                if error_count_numpy(U_ls, V_ls, W_ls, r) == 0:
                    return {"U": U_ls, "V": V_ls, "W": W_ls}
        
        return None
    
    # Try to find lower rank decompositions, starting from lower ranks
    # Known: 3x3x3 has rank 23 (Strassen-like), 3x3x4 likely needs ~25-30
    for r in range(15, best_rank):
        if time.time() > deadline:
            break
        
        time_remaining = deadline - time.time()
        if time_remaining < 3:
            break
        
        # Spend more time on promising ranks
        if r <= 20:
            max_attempts = min(500, int(time_remaining * 10))
        elif r <= 25:
            max_attempts = min(800, int(time_remaining * 15))
        elif r <= 30:
            max_attempts = min(600, int(time_remaining * 10))
        else:
            max_attempts = min(300, int(time_remaining * 5))
        
        result = try_rank_structured(r, max_attempts)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    return best_result
# EVOLVE-BLOCK-END
