"""Improved program: exploit LCM periodicity + optimized backtracking for span minimization."""
import time
import math
import random

# EVOLVE-BLOCK-START
DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAXV = 1000000
N = len(DIVISORS)

def _compute_lcm():
    """Compute LCM of all divisors."""
    result = 1
    for d in DIVISORS:
        result = result * d // math.gcd(result, d)
    return result

LCM = _compute_lcm()  # 32730

_START_TIME = time.time()
DEADLINE = _START_TIME + 85  # safety margin vs 90s wall-clock


def _backtrack_assign(candidates, index=0, assigned=None, used=None, remaining=None):
    """
    Backtrack with MRV heuristic and forward checking.
    candidates: list of lists, candidates[i] = list of possible values for divisor i
    Returns the assignment list or None.
    """
    if assigned is None:
        assigned = [None] * N
        used = set()
        remaining = list(range(N))
    
    if not remaining:
        return assigned
    
    # MRV: pick unassigned variable with smallest candidate list
    min_idx = -1
    min_size = float('inf')
    for idx in remaining:
        avail = len(candidates[idx]) - sum(1 for v in candidates[idx] if v in used)
        if avail < min_size:
            min_size = avail
            min_idx = idx
        if min_size <= 0:
            break
    
    if min_size <= 0:
        return None
    
    new_remaining = [i for i in remaining if i != min_idx]
    
    for val in candidates[min_idx]:
        if val not in used:
            used.add(val)
            assigned[min_idx] = val
            result = _backtrack_assign(candidates, index, assigned, used, new_remaining)
            if result is not None:
                return result
            used.remove(val)
            assigned[min_idx] = None
    
    return None


def _check_span_feasible(S, max_L=None):
    """
    Check if there exists a window [L, L+S] where we can assign distinct multiples.
    Returns (L, reps) if found, else None.
    Uses LCM periodicity: only need to check L in [1, LCM].
    """
    if max_L is None:
        max_L = LCM  # periodicity
    
    upper_L = min(max_L, MAXV - S)
    if upper_L < 1:
        return None
    
    for L in range(1, upper_L + 1):
        if time.time() > DEADLINE:
            return None
        
        # For each divisor, find all multiples in [L, L+S]
        candidates = []
        feasible = True
        for i in range(N):
            d = DIVISORS[i]
            # First multiple of d >= L
            first_k = (L + d - 1) // d
            first_val = first_k * d
            # Last multiple of d <= L+S
            last_val = L + S
            
            if first_val > last_val:
                feasible = False
                break
            
            # Generate all multiples in [L, L+S]
            vals = []
            v = first_val
            while v <= last_val:
                vals.append(v)
                v += d
            candidates.append(vals)
        
        if not feasible:
            continue
        
        # Use backtracking with MRV + forward checking
        result = _backtrack_assign(candidates)
        if result is not None:
            return (L, result)
    
    return None


def _local_search(reps, max_iterations=5000):
    """Local search: try perturbing each representative by +/- k*d for small k."""
    best_span = max(reps) - min(reps)
    best_reps = reps[:]
    current_reps = reps[:]
    random.seed(42)
    
    for iteration in range(max_iterations):
        if time.time() > DEADLINE:
            break
        improved = False
        
        # Try all single-variable moves with wider range
        for i in range(N):
            d = DIVISORS[i]
            for k in range(-20, 21):
                if k == 0:
                    continue
                new_val = current_reps[i] + k * d
                if new_val <= 0 or new_val > MAXV:
                    continue
                new_reps = current_reps[:]
                new_reps[i] = new_val
                if len(set(new_reps)) != len(new_reps):
                    continue
                span = max(new_reps) - min(new_reps)
                if span < best_span:
                    best_span = span
                    best_reps = new_reps[:]
                    current_reps = new_reps[:]
                    improved = True
                    break
            if improved:
                break
        
        # Try 2-variable moves
        if not improved:
            for i in range(N):
                for j in range(i + 1, N):
                    d_i = DIVISORS[i]
                    d_j = DIVISORS[j]
                    for k_i in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
                        for k_j in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
                            new_val_i = current_reps[i] + k_i * d_i
                            new_val_j = current_reps[j] + k_j * d_j
                            if new_val_i <= 0 or new_val_i > MAXV:
                                continue
                            if new_val_j <= 0 or new_val_j > MAXV:
                                continue
                            new_reps = current_reps[:]
                            new_reps[i] = new_val_i
                            new_reps[j] = new_val_j
                            if len(set(new_reps)) != len(new_reps):
                                continue
                            span = max(new_reps) - min(new_reps)
                            if span < best_span:
                                best_span = span
                                best_reps = new_reps[:]
                                current_reps = new_reps[:]
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        # Try 3-variable moves
        if not improved:
            for i in range(N):
                for j in range(i + 1, N):
                    for k in range(j + 1, N):
                        d_i = DIVISORS[i]
                        d_j = DIVISORS[j]
                        d_k = DIVISORS[k]
                        for ki in [-2, -1, 1, 2]:
                            for kj in [-2, -1, 1, 2]:
                                for kk in [-2, -1, 1, 2]:
                                    new_val_i = current_reps[i] + ki * d_i
                                    new_val_j = current_reps[j] + kj * d_j
                                    new_val_k = current_reps[k] + kk * d_k
                                    if new_val_i <= 0 or new_val_i > MAXV:
                                        continue
                                    if new_val_j <= 0 or new_val_j > MAXV:
                                        continue
                                    if new_val_k <= 0 or new_val_k > MAXV:
                                        continue
                                    new_reps = current_reps[:]
                                    new_reps[i] = new_val_i
                                    new_reps[j] = new_val_j
                                    new_reps[k] = new_val_k
                                    if len(set(new_reps)) != len(new_reps):
                                        continue
                                    span = max(new_reps) - min(new_reps)
                                    if span < best_span:
                                        best_span = span
                                        best_reps = new_reps[:]
                                        current_reps = new_reps[:]
                                        improved = True
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        # Random perturbation to escape local minima
        if not improved:
            i = random.randint(0, N - 1)
            d = DIVISORS[i]
            k = random.randint(-25, 25)
            if k != 0:
                new_val = current_reps[i] + k * d
                if 0 < new_val <= MAXV:
                    new_reps = current_reps[:]
                    new_reps[i] = new_val
                    if len(set(new_reps)) == len(new_reps):
                        current_reps = new_reps[:]
        
        if not improved and iteration > 2000:
            break
    
    return (best_span, best_reps)


def construct():
    best_span = float('inf')
    best_reps = None
    
    # Strategy: Search for minimum feasible span S
    # Start from lower bound N-1=9 and increase
    
    for S in range(9, 300):
        if time.time() > DEADLINE - 5:
            break
        
        result = _check_span_feasible(S)
        if result is not None:
            L, reps = result
            # Verify all constraints
            valid = True
            for i in range(N):
                if reps[i] <= 0 or reps[i] > MAXV:
                    valid = False
                    break
                if reps[i] % DIVISORS[i] != 0:
                    valid = False
                    break
            if not valid:
                continue
            if len(set(reps)) != len(reps):
                continue
            
            span = max(reps) - min(reps)
            if span <= S:
                best_span = span
                best_reps = reps[:]
                break  # Found minimum feasible span
    
    # If we found a solution, try to improve with local search
    if best_reps is not None:
        span, reps = _local_search(best_reps, max_iterations=20000)
        if span < best_span:
            best_span = span
            best_reps = reps
    
    # Fallback: if no solution found via span search, do a broader scan
    if best_reps is None:
        def _smallest_multiple_at_least(d, L):
            return ((L + d - 1) // d) * d
        
        for L in range(1, 10000):
            if time.time() > DEADLINE - 5:
                break
            reps = []
            valid = True
            for d in DIVISORS:
                m = _smallest_multiple_at_least(d, L)
                if m > MAXV:
                    valid = False
                    break
                reps.append(m)
            if not valid:
                continue
            if len(set(reps)) != len(reps):
                continue
            span = max(reps) - min(reps)
            if span < best_span:
                best_span = span
                best_reps = reps[:]
        
        if best_reps is not None:
            span, reps = _local_search(best_reps, max_iterations=10000)
            if span < best_span:
                best_span = span
                best_reps = reps
    
    if best_reps is None:
        # Last resort fallback
        return DIVISORS[:]
    
    # Ensure we return Python ints
    return [int(x) for x in best_reps]
# EVOLVE-BLOCK-END
