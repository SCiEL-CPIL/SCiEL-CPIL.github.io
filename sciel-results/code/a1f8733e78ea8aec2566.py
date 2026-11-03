import time
import random
import math
import numpy as np

_DEADLINE = 0.0
_BEST = None

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _scope(rows):
    return max(a for r in rows for a in r[1:])

def _fast_greedy(k, n_rows, rng, spread=False):
    used = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = 0
            for a in row:
                new |= (1 << (x - a))
            if (used & new) == 0:
                row.append(x)
                used |= new
                if spread:
                    x += max(1, (k - len(row) + 1) * 2 - 2)
        rows.append(row)
    return rows

def _compute_stats(rows, k, max_scope_hint=200):
    """
    Compute scope and collision count for a set of rows.
    Returns (scope, collisions).
    collisions is the number of differences that appear more than once.
    """
    scope = 0
    counts = np.zeros(max_scope_hint + 1, dtype=np.int32)
    
    for r in rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d > scope:
                    scope = d
                if d <= max_scope_hint:
                    counts[d] += 1
                else:
                    # If scope exceeds hint, we need to expand or handle it
                    # For SA, we usually bound the search space.
                    # If d > max_scope_hint, it's a "large" difference.
                    # We can still count it if we dynamically resize, but for speed,
                    # let's assume max_scope_hint is large enough for valid solutions.
                    # If not, we just ignore collisions for d > max_scope_hint in the penalty,
                    # but that's bad. Better to just make max_scope_hint large (e.g. 300).
                    pass
    
    # Number of differences that are not unique
    # If a difference appears c times, it contributes c-1 to collisions? 
    # Or just count how many pairs of rows share a diff?
    # Standard penalty: sum over d of max(0, counts[d] - 1)
    collisions = np.sum(np.maximum(0, counts - 1))
    
    return scope, collisions

def _sa_optimize(initial_rows, n_rows, k, deadline, rng):
    """
    Simulated Annealing to minimize scope while resolving collisions.
    """
    rows = [list(r) for r in initial_rows]
    
    # Determine a reasonable max_scope_hint for the SA loop
    # Start with current scope + some margin
    current_scope = _scope(rows)
    max_scope_hint = max(200, current_scope + 50)
    
    # Ensure arrays are large enough
    # We'll use a fixed size for speed, e.g. 300
    MAX_S = 300
    
    # Initial state
    scope, collisions = _compute_stats(rows, k, MAX_S)
    current_cost = scope + 10 * collisions  # Weight collisions heavily
    
    best_rows = [list(r) for r in rows]
    best_scope = scope
    best_cost = current_cost
    
    T = 50.0
    cooling = 0.9995
    min_T = 0.01
    
    start_time = time.time()
    
    while time.time() < deadline and T > min_T:
        # Pick a random row and a random element to move
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)  # Don't move the 0
        
        old_val = rows[i][j]
        
        # Determine move range
        # We want to try moving the value to a nearby position
        # Or to a random position within a certain range
        # Let's try a random value in [prev+1, next-1] or a random shift
        
        prev_val = rows[i][j-1]
        next_val = rows[i][j+1] if j < k else rows[i][-1] + 10
        
        # Generate a candidate value
        # 50% chance: small shift
        # 50% chance: random within bounds
        if rng.random() < 0.5:
            shift = rng.randint(-5, 5)
            new_val = old_val + shift
        else:
            # Random within valid range
            lo = prev_val + 1
            hi = min(next_val - 1, MAX_S)
            if lo >= hi:
                continue
            new_val = rng.randint(lo, hi)
        
        if new_val == old_val or new_val <= prev_val or new_val >= next_val:
            continue
            
        # Apply move
        rows[i][j] = new_val
        
        # Calculate new cost
        # Efficient delta calculation would be better, but for simplicity and given small N,
        # full recalculation is O(N * K^2). N=7, K=5. 7 * 25 = 175 ops. Very fast.
        new_scope, new_collisions = _compute_stats(rows, k, MAX_S)
        new_cost = new_scope + 10 * new_collisions
        
        delta = new_cost - current_cost
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            # Accept
            current_cost = new_cost
            scope = new_scope
            collisions = new_collisions
            
            if scope < best_scope and collisions == 0:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                best_cost = current_cost
        else:
            # Reject
            rows[i][j] = old_val
            
        T *= cooling
        
    return best_rows, best_scope

def construct(n_rows, k):
    global _BEST, _DEADLINE
    _BEST = None
    _DEADLINE = time.time() + 145.0  # Safety margin for 150s limit

    def consider(rows):
        global _BEST
        if not _valid(rows, k):
            return
        s = _scope(rows)
        if _BEST is None or s < _BEST[0]:
            _BEST = (s, [list(r) for r in rows])

    rng = random.Random(42)

    # 1. Initial greedy
    greedy_rows = _fast_greedy(k, n_rows, rng, spread=False)
    consider(greedy_rows)
    
    # Also try spread greedy
    greedy_rows2 = _fast_greedy(k, n_rows, rng, spread=True)
    consider(greedy_rows2)
    
    # 2. SA Optimization
    # Run SA from the best initial solution found so far
    if _BEST is not None:
        start_rows = _BEST[1]
        if time.time() < _DEADLINE - 5:
            sa_rows, sa_scope = _sa_optimize(start_rows, n_rows, k, _DEADLINE - 2, rng)
            consider(sa_rows)
            
    # 3. If SA didn't find a valid solution or improved, try a few random restarts of SA
    # from different initializations if time permits
    if time.time() < _DEADLINE - 10:
        for _ in range(3):
            # Generate a new random-ish initialization
            # Perturb the greedy solution
            init_rows = [list(r) for r in greedy_rows]
            for r in init_rows:
                for j in range(1, k+1):
                    if rng.random() < 0.3:
                        # Small random shift
                        shift = rng.randint(-2, 2)
                        r[j] += shift
                        # Ensure validity locally (increasing)
                        if r[j] <= r[j-1] or (j < k and r[j] >= r[j+1]):
                            r[j] = r[j-1] + 1 # crude fix
                            
            # Ensure strictly increasing
            for r in init_rows:
                for j in range(1, k+1):
                    if r[j] <= r[j-1]:
                        r[j] = r[j-1] + 1
                        
            if time.time() < _DEADLINE - 5:
                sa_rows, sa_scope = _sa_optimize(init_rows, n_rows, k, _DEADLINE - 2, rng)
                consider(sa_rows)
                
    if _BEST is not None:
        return _BEST[1]
    return greedy_rows
