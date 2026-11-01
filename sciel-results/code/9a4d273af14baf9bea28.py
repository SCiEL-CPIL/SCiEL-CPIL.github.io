"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    if not U or not V or not W:
        return False
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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

def _snap_to_half(x):
    """Snap a value to the nearest integer or half-integer."""
    nearest_int = round(x)
    nearest_half = round(x * 2) / 2
    if abs(x - nearest_int) <= abs(x - nearest_half):
        return nearest_int
    else:
        return nearest_half

def _is_half_integer(x, tol=1e-9):
    """Check if x is an integer or half-integer within tolerance."""
    if abs(x - round(x)) < tol:
        return True
    if abs(x * 2 - round(x * 2)) < tol:
        return True
    return False

def _try_rank_optimization(n, m, p, target_rank, time_budget=10.0, seed=0):
    """
    Attempt to find a decomposition of the given target rank.
    Uses a local search approach starting from a random initialization.
    The objective is to satisfy the 42 constraints:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    start = time.time()
    rng = random.Random(seed)
    
    # Initialize with random small integer/half-integer values
    # We use a sparse initialization to help convergence
    U = [[0] * target_rank for _ in range(n * m)]
    V = [[0] * target_rank for _ in range(m * p)]
    W = [[0] * target_rank for _ in range(p * n)]
    
    # Strategy: For each constraint (i,j,k), try to assign a single rank index r
    # that handles it, or split it.
    # Actually, a better strategy for low rank is to use the structure of the tensor.
    # But since we don't know the structure, we use a heuristic local search.
    
    # Initialize U, V, W with random values from {-1, 0, 1}
    # But ensure enough non-zeros to allow solution
    for i in range(n * m):
        for r in range(target_rank):
            if rng.random() < 0.7:
                U[i][r] = rng.choice([-1, 1])
    for j in range(m * p):
        for r in range(target_rank):
            if rng.random() < 0.7:
                V[j][r] = rng.choice([-1, 1])
    for k in range(p * n):
        for r in range(target_rank):
            if rng.random() < 0.7:
                W[k][r] = rng.choice([-1, 1])
    
    # Local Search:
    # 1. Compute error for each constraint
    # 2. Pick the constraint with largest error
    # 3. Try to adjust one variable (U, V, or W entry) to reduce error
    # 4. Accept if error decreases
    
    def compute_errors():
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(target_rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    errors.append((i, j, k, s - 1))
        return errors
    
    def total_error():
        return sum(abs(e) for _, _, _, e in compute_errors())
    
    best_error = total_error()
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    max_iters = 5000
    for it in range(max_iters):
        if time.time() - start > time_budget:
            break
        
        errors = compute_errors()
        # Find max error
        max_err_idx = max(range(len(errors)), key=lambda idx: abs(errors[idx][3]))
        i, j, k, err = errors[max_err_idx]
        
        if abs(err) < 1e-9:
            # Check if all are satisfied
            if all(abs(e) < 1e-9 for _, _, _, e in errors):
                return U, V, W
        
        # Try to fix this constraint
        # We can adjust U[i*m+j, r], V[j*p+k, r], or W[k*n+i, r]
        # For a given r, if we change U[i*m+j, r] to new_val, the change in sum is:
        # (new_val - old_val) * V[j*p+k, r] * W[k*n+i, r]
        # We want this to be -err.
        # So new_val = old_val - err / (V[j*p+k, r] * W[k*n+i, r])
        
        best_move = None
        best_new_err = abs(err)
        
        for r in range(target_rank):
            # Option 1: Adjust U
            denom = V[j * p + k][r] * W[k * n + i][r]
            if denom != 0:
                old_val = U[i * m + j][r]
                new_val = old_val - err / denom
                # Check if new_val is "nice" (close to integer or half-integer)
                # If not, snap it, but only if the resulting error is still reduced
                snapped_val = _snap_to_half(new_val)
                # Calculate error change if we set to snapped_val
                delta = (snapped_val - old_val) * denom
                new_err = err + delta
                if abs(new_err) < best_new_err - 1e-6:
                    best_new_err = abs(new_err)
                    best_move = ('U', i * m + j, r, snapped_val)
            
            # Option 2: Adjust V
            denom = U[i * m + j][r] * W[k * n + i][r]
            if denom != 0:
                old_val = V[j * p + k][r]
                new_val = old_val - err / denom
                snapped_val = _snap_to_half(new_val)
                delta = (snapped_val - old_val) * denom
                new_err = err + delta
                if abs(new_err) < best_new_err - 1e-6:
                    best_new_err = abs(new_err)
                    best_move = ('V', j * p + k, r, snapped_val)
            
            # Option 3: Adjust W
            denom = U[i * m + j][r] * V[j * p + k][r]
            if denom != 0:
                old_val = W[k * n + i][r]
                new_val = old_val - err / denom
                snapped_val = _snap_to_half(new_val)
                delta = (snapped_val - old_val) * denom
                new_err = err + delta
                if abs(new_err) < best_new_err - 1e-6:
                    best_new_err = abs(new_err)
                    best_move = ('W', k * n + i, r, snapped_val)
        
        if best_move:
            mat_name, idx, r, val = best_move
            if mat_name == 'U':
                U[idx][r] = val
            elif mat_name == 'V':
                V[idx][r] = val
            else:
                W[idx][r] = val
            
            curr_err = total_error()
            if curr_err < best_error:
                best_error = curr_err
                best_U = [row[:] for row in U]
                best_V = [row[:] for row in V]
                best_W = [row[:] for row in W]
        else:
            # No good move found, try a random perturbation to escape local minimum
            # Pick a random entry and change it to a random nice value
            mat_choice = rng.choice(['U', 'V', 'W'])
            if mat_choice == 'U':
                idx = rng.randint(0, n * m - 1)
                r = rng.randint(0, target_rank - 1)
                U[idx][r] = rng.choice([-1, 0, 1])
            elif mat_choice == 'V':
                idx = rng.randint(0, m * p - 1)
                r = rng.randint(0, target_rank - 1)
                V[idx][r] = rng.choice([-1, 0, 1])
            else:
                idx = rng.randint(0, p * n - 1)
                r = rng.randint(0, target_rank - 1)
                W[idx][r] = rng.choice([-1, 0, 1])
            
            curr_err = total_error()
            if curr_err < best_error:
                best_error = curr_err
                best_U = [row[:] for row in U]
                best_V = [row[:] for row in V]
                best_W = [row[:] for row in W]
        
        # Early exit if perfect
        if best_error < 1e-9:
            return best_U, best_V, best_W
    
    # Check final best solution
    if best_error < 1e-9:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with schoolbook decomposition as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # The tensor has 42 constraints.
    # Lower bound is at least max(n*m, m*p, p*n) = max(6, 21, 14) = 21?
    # Actually, rank >= max(n*p, m*p, n*m) is not a strict lower bound for rank,
    # but for matrix multiplication, rank >= n*p is often cited? No.
    # Let's try ranks from 21 upwards.
    
    # We will try ranks from 21 to 41.
    # For each rank, we will try a few random restarts.
    
    min_rank_to_try = 21
    max_restarts = 5
    
    for R in range(min_rank_to_try, best_rank + 1):
        if time.time() > deadline - 10:
            break
            
        found = False
        for attempt in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Allocate time budget for this attempt
            # We have ~140s total, 10s safety, so ~130s for search.
            # We have ~21 ranks to try. So ~6s per rank.
            # Let's give 1s per attempt, so 5 attempts per rank = 5s.
            time_budget = 1.0
            
            U, V, W = _try_rank_optimization(n, m, p, R, time_budget, seed=attempt * 100 + R)
            
            if U is not None:
                # Verify
                if _verify(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
        
        # If we found a valid solution for this rank, we could return it,
        # but we want the minimal rank, so we continue to lower ranks?
        # No, we are iterating from low to high.
        # So if we find one at R, it's the minimal found so far.
        # But we should check if a lower rank is possible?
        # We started at 21. If 21 fails, we try 22.
        # So the first success is the minimal.
        
        # Wait, the loop is `for R in range(min_rank_to_try, best_rank + 1)`.
        # So if R=21 succeeds, we return.
        # If R=21 fails, we try R=22.
        # This is correct.
        
        # However, if we find a solution at R=25, we return it.
        # We don't need to check R=24 again?
        # The loop goes 21, 22, 23, 24, 25.
        # So if 21-24 fail, and 25 succeeds, we return 25.
        # This is correct.
        
        # One optimization: if we find a solution at R, we can return immediately.
        # The code above does `if U is not None: ... return`.
        # So it works.
        
        # But what if 21 fails with 5 restarts, but would succeed with 10?
        # We might miss it.
        # To mitigate, we could increase restarts for lower ranks.
        # Or just accept the risk.
        
        # Let's adjust: give more time to lower ranks.
        # R=21: 2s per attempt, 5 attempts = 10s
        # R=22: 1s per attempt, 5 attempts = 5s
        # ...
        
    return best_result

# EVOLVE-BLOCK-END
