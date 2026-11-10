import time
import random
import numpy as np

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i in range(n + 1) if sieve[i]]

_PRIMES = _primes_up_to(1000)
_NPRIMES = len(_PRIMES)
# We want to maximize L. The problem states L >= 1000 is the target.
# Let's set a high limit for search, e.g., 5000 or 10000.
# The score is the largest L.
# If we find L=1500, score is 1500/seed.
# Let's aim for as high as possible within 120s.
_L_LIMIT = 5000

def _build_count(residues, limit):
    count = np.zeros(limit + 1, dtype=np.int32)
    for i, p in enumerate(_PRIMES):
        r = residues[i]
        start = p if r == 0 else r
        # Only update up to limit
        if start <= limit:
            count[start::p] += 1
    return count

def _find_L(count, limit):
    # Find the first index where count < 2
    # count[0] is unused, we care about 1..limit
    below = np.where(count[1:limit+1] < 2)[0]
    if len(below) == 0:
        return limit
    return int(below[0])

def _greedy_repair(residues, deadline):
    """
    Greedy repair: Find the first gap, fix it with the least damage.
    """
    count = _build_count(residues, _L_LIMIT)
    current_L = _find_L(count, _L_LIMIT)
    best_L = current_L
    best_residues = list(residues)
    
    # Precompute positions for each prime for fast access?
    # Or just use slicing.
    
    while time.time() < deadline - 10:
        if current_L >= _L_LIMIT:
            break
            
        gap = current_L + 1
        if gap > _L_LIMIT:
            break
            
        # If count[gap] >= 2, we can just extend L
        if count[gap] >= 2:
            # Check how far we can extend
            while gap <= _L_LIMIT and count[gap] >= 2:
                current_L = gap
                gap += 1
            if current_L > best_L:
                best_L = current_L
                best_residues = list(residues)
            continue
        
        # We have a gap at 'gap'. We need to increase count[gap] to at least 2.
        # This means we need to change a prime's residue to 'gap % p'.
        # We evaluate all primes to find the best move.
        
        best_move = None
        min_violations = float('inf')
        best_prime_idx = -1
        
        for i in range(_NPRIMES):
            p = _PRIMES[i]
            old_r = residues[i]
            new_r = gap % p
            
            if new_r == old_r:
                # Already covering? But count[gap] < 2, so this prime is either
            # not covering it (impossible if new_r == old_r and it was covering)
            # or covering it but count is still < 2?
            # If new_r == old_r, it means p is already assigned to cover gap.
            # But count[gap] < 2. So changing it to the same thing does nothing.
                continue
            
            # Calculate the impact of changing r_i from old_r to new_r
            # We need to check violations in the range [1, _L_LIMIT]
            # Affected positions are those congruent to old_r or new_r mod p.
            
            # To be efficient, we can just simulate the change on a copy of count?
            # No, that's slow.
            # Let's count violations analytically or with small numpy ops.
            
            # Positions where count might drop: x s.t. x % p == old_r
            # Positions where count might rise: x s.t. x % p == new_r
            
            # We are interested in positions where count becomes < 2.
            # This can only happen if count was 2 and we remove 1 (drop to 1),
            # or count was 1 and we remove 1 (drop to 0).
            
            # Let's check the set of positions affected.
            # Since p can be large, the number of affected positions in [1, _L_LIMIT] is _L_LIMIT // p.
            # For small p, this is large. For large p, it's small.
            
            # Optimization: Only check positions in [1, _L_LIMIT]
            
            start_old = p if old_r == 0 else old_r
            start_new = p if new_r == 0 else new_r
            
            # Get indices
            # This might be slow if done for every prime in every iteration.
            # Let's try to optimize.
            
            # We can precompute the "risk" of each prime.
            # But the count array changes.
            
            # Let's just do the numpy subtraction/addition on a temporary view?
            # No, we need to evaluate *before* committing.
            
            # Fast check:
            # How many positions x in [1, _L_LIMIT] have count[x] == 2 and x % p == old_r?
            # If we remove p from these, they drop to 1 (violation).
            # How many positions x in [1, _L_LIMIT] have count[x] == 1 and x % p == old_r?
            # If we remove p from these, they drop to 0 (violation).
            
            # So violations = count of x in [1, _L_LIMIT] such that (x % p == old_r) and (count[x] < 3)
            # Wait, if count[x] == 2, removing 1 makes it 1 (bad).
            # If count[x] == 1, removing 1 makes it 0 (bad).
            # If count[x] >= 3, removing 1 makes it >= 2 (ok).
            
            # So we need to count how many x in [1, _L_LIMIT] satisfy:
            # x % p == old_r AND count[x] <= 2
            
            # We can do this with numpy.
            # Create a mask for x % p == old_r?
            # That's O(L). Doing this for 168 primes is 168 * 5000 = 840k ops per gap fix.
            # If we have 1000 gaps, that's 840M ops. Might be too slow in Python/NumPy if not careful.
            
            # Alternative:
            # We only care about the "damage".
            # Let's iterate through the positions affected by old_r.
            # Number of positions is L // p.
            # Sum over i in range(start_old, L+1, p):
            #   if count[i] <= 2: violations += 1
            
            # This is a Python loop. Might be slow.
            
            # Let's use NumPy vectorization for the check.
            # We can precompute a boolean array for each prime? No, residue changes.
            
            # Let's try a different approach:
            # Instead of evaluating all primes, evaluate a subset?
            # Or just accept the cost if it's not too high.
            
            # Let's optimize the violation count.
            # We can precompute for each prime p, the list of positions it covers for a given residue?
            # No, residue is dynamic.
            
            # Let's just use the loop for now and see if it's fast enough.
            # If L=5000, and p=2, 2500 iterations.
            # 168 primes.
            # If we fix 1000 gaps, 168 * 1000 * (5000/p) operations.
            # For p=2, 168 * 1000 * 2500 = 420M operations. Too slow in Python.
            
            # We need a faster way to estimate damage.
            
            # Idea:
            # Keep a separate array `count` and update it.
            # When considering a move, we can just *apply* it, check the new L, and *revert* if it's bad?
            # But we want the *best* move.
            
            # Maybe we can prioritize small primes?
            # Small primes have more "damage" potential because they cover more positions.
            # But they also provide more "benefit".
            
            # Let's try a heuristic:
            # 1. Try small primes first.
            # 2. For each prime, calculate the number of "critical" positions (count <= 2) that it currently covers.
            #    If this number is 0, the move is "free" (no new gaps created by removal).
            #    If this number is small, it's a good move.
            
            # We can maintain `critical_count[i]` = number of positions x covered by prime i where count[x] <= 2.
            # This can be updated incrementally.
            
            # Let's implement this incremental tracking.
            
            # This is getting complex. Let's stick to the previous SA but with a better initialization and longer run time?
            # No, the directive asks for a structural change.
            
            # Let's go with the "Apply and Revert" strategy but smarter.
            # We know we *must* cover the gap.
            # So we *must* pick a prime to change.
            # We can pick the prime that minimizes the number of *new* gaps.
            # We can approximate this by looking at the "density" of critical points.
            
            # Let's try this:
            # For each prime p, count how many positions x in [1, L] have x % p == old_r and count[x] <= 2.
            # Let this be `risk[p]`.
            # We want to minimize `risk[p]`.
            
            # To speed up `risk[p]` calculation:
            # We can precompute `is_critical[x]` = 1 if count[x] <= 2 else 0.
            # Then `risk[p]` is the sum of `is_critical[x]` for x in [1, L] where x % p == old_r.
            
            # We can compute this sum using numpy if we have a precomputed array of positions?
            # No, positions depend on residue.
            
            # However, we can precompute for each prime p, an array `pos_p` of length L/p?
            # No.
            
            # Let's just use the loop but break early if risk exceeds current best?
            # Or just limit the search to the first 50 primes?
            # Small primes are more likely to be the "bottleneck" for coverage.
            
            # Let's try limiting to the first 100 primes for the repair step.
            
            pass # Placeholder for the loop logic below

    return best_L, best_residues

def _fast_greedy_repair(residues, deadline):
    """
    Optimized greedy repair using incremental critical count tracking.
    """
    count = _build_count(residues, _L_LIMIT)
    current_L = _find_L(count, _L_LIMIT)
    best_L = current_L
    best_residues = list(residues)
    
    # critical[x] = 1 if count[x] <= 2 else 0
    critical = (count <= 2).astype(np.int32)
    
    # risk[i] = number of critical positions covered by prime i
    # We need to update risk when count changes.
    # But count changes affect many positions.
    
    # Let's just recompute risk for candidate primes on the fly.
    # To speed up, we only consider primes p where L // p is small?
    # No, small p are important.
    
    # Let's try a different structural change:
    # **Simulated Annealing with Problem-Specific Moves**
    # Instead of random residue change, propose moves that specifically target gaps.
    
    # Move 1: Pick a random gap x. Pick a random prime p. Set r_p = x % p.
    # Move 2: Pick a random prime p. Change r_p to a random value.
    # Mix these two.
    
    # This is a structural change to the *move generation* in SA.
    
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    random.seed(42)
    
    best_L_overall = 0
    best_residues_overall = None
    
    # Initialization: All zeros is a bad start.
    # Better start: Assign residues to cover 1..100 as much as possible.
    # Or just use random initialization with SA.
    
    # Let's use the "Gap-Targeted SA" approach.
    
    residues = [0] * _NPRIMES
    count = _build_count(residues, _L_LIMIT)
    current_L = _find_L(count, _L_LIMIT)
    best_L = current_L
    best_residues = list(residues)
    
    T = 50.0
    T_min = 0.1
    cooling = 0.999
    
    # Precompute prime indices
    prime_indices = list(range(_NPRIMES))
    
    while time.time() < deadline - 10 and T > T_min:
        # Choose move type
        if random.random() < 0.5:
            # Gap-Targeted Move
            if current_L < _L_LIMIT:
                # Find a gap
                # Use a random gap? Or the first gap?
                # First gap is more deterministic.
                gap = current_L + 1
                if count[gap] < 2:
                    # Pick a random prime to fix it
                    i = random.randint(0, _NPRIMES - 1)
                    p = _PRIMES[i]
                    new_r = gap % p
                else:
                    # No gap? Then we can extend L.
                    # Just do a random move to explore.
                    i = random.randint(0, _NPRIMES - 1)
                    p = _PRIMES[i]
                    new_r = random.randint(0, p - 1)
            else:
                i = random.randint(0, _NPRIMES - 1)
                p = _PRIMES[i]
                new_r = random.randint(0, p - 1)
        else:
            # Random Move
            i = random.randint(0, _NPRIMES - 1)
            p = _PRIMES[i]
            new_r = random.randint(0, p - 1)
            
        old_r = residues[i]
        if new_r == old_r:
            T *= cooling
            continue
        
        start_old = p if old_r == 0 else old_r
        start_new = p if new_r == 0 else new_r
        
        count[start_old::p] -= 1
        count[start_new::p] += 1
        
        new_L = _find_L(count, _L_LIMIT)
        
        if new_L >= current_L:
            residues[i] = new_r
            current_L = new_L
            if current_L > best_L:
                best_L = current_L
                best_residues = list(residues)
        else:
            delta = new_L - current_L
            if random.random() < np.exp(delta / T):
                residues[i] = new_r
                current_L = new_L
            else:
                count[start_new::p] -= 1
                count[start_old::p] += 1
        
        T *= cooling
        
    if best_residues_overall is None or best_L > best_L_overall:
        best_L_overall = best_L
        best_residues_overall = best_residues
    
    return [int(r) for r in best_residues_overall]

# Wait, the above SA is very similar to the original.
# The "Gap-Targeted" move is a slight improvement, but not a huge structural change.
# Let's try the **Greedy Repair with Risk Calculation** properly.

def _risk_calc(count, p, r, limit):
    """Calculate number of critical positions covered by p with residue r."""
    # Critical: count <= 2
    # We want sum of 1 for x in [1, limit] where x % p == r and count[x] <= 2
    # We can use numpy to get indices and sum.
    start = p if r == 0 else r
    if start > limit:
        return 0
    indices = np.arange(start, limit + 1, p)
    # count is a numpy array
    return np.sum(count[indices] <= 2)

def _greedy_repair_v2(residues, deadline):
    count = _build_count(residues, _L_LIMIT)
    current_L = _find_L(count, _L_LIMIT)
    best_L = current_L
    best_residues = list(residues)
    
    while time.time() < deadline - 10:
        if current_L >= _L_LIMIT:
            break
        
        gap = current_L + 1
        if count[gap] >= 2:
            while gap <= _L_LIMIT and count[gap] >= 2:
                current_L = gap
                gap += 1
            if current_L > best_L:
                best_L = current_L
                best_residues = list(residues)
            continue
        
        # Find best prime to fix gap
        best_move = None
        min_risk = float('inf')
        
        # Only check first 100 primes to save time?
        # Small primes are more critical.
        search_range = min(_NPRIMES, 100)
        
        for i in range(search_range):
            p = _PRIMES[i]
            old_r = residues[i]
            new_r = gap % p
            
            if new_r == old_r:
                continue
            
            # Risk is the number of critical positions we will *lose* coverage for.
            # We lose coverage at positions x where x % p == old_r.
            # If count[x] <= 2, losing coverage makes it < 2 (bad).
            risk = _risk_calc(count, p, old_r, _L_LIMIT)
            
            # We also gain coverage at new_r.
            # If count[x] < 2, gaining coverage is good.
            # But we are fixing a gap, so we *must* gain at least 1 at 'gap'.
            
            if risk < min_risk:
                min_risk = risk
                best_move = (i, new_r)
                if risk == 0:
                    break
        
        if best_move:
            i, new_r = best_move
            p = _PRIMES[i]
            old_r = residues[i]
            start_old = p if old_r == 0 else old_r
            start_new = p if new_r == 0 else new_r
            
            count[start_old::p] -= 1
            count[start_new::p] += 1
            residues[i] = new_r
            
            new_L = _find_L(count, _L_LIMIT)
            
            if new_L > current_L:
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = list(residues)
            else:
                # Revert if it didn't help (shouldn't happen if we fixed the gap, but maybe created others)
                count[start_new::p] -= 1
                count[start_old::p] += 1
                residues[i] = old_r
        else:
            # No good move found? Stuck.
            # Try a random move to escape?
            break
            
    return best_L, best_residues

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    random.seed(42)
    
    best_L_overall = 0
    best_residues_overall = None
    
    # Restart strategy
    n_restarts = 3
    time_per_restart = (deadline - start_time - 10) / n_restarts
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # Random initialization
        residues = [random.randint(0, p - 1) for p in _PRIMES]
        
        # Greedy Repair
        sa_deadline = min(time.time() + time_per_restart * 0.7, deadline - 10)
        L_rep, residues_rep = _greedy_repair_v2(residues, sa_deadline)
        
        # SA Polish
        sa_deadline2 = min(time.time() + time_per_restart * 0.3, deadline - 10)
        # Use a simple SA on the result of repair
        count = _build_count(residues_rep, _L_LIMIT)
        current_L = _find_L(count, _L_LIMIT)
        
        T = 10.0
        T_min = 0.1
        cooling = 0.999
        
        while time.time() < sa_deadline2 and T > T_min:
            i = random.randint(0, _NPRIMES - 1)
            p = _PRIMES[i]
            new_r = random.randint(0, p - 1)
            old_r = residues_rep[i]
            
            if new_r == old_r:
                T *= cooling
                continue
                
            start_old = p if old_r == 0 else old_r
            start_new = p if new_r == 0 else new_r
            
            count[start_old::p] -= 1
            count[start_new::p] += 1
            
            new_L = _find_L(count, _L_LIMIT)
            
            if new_L >= current_L:
                residues_rep[i] = new_r
                current_L = new_L
                if current_L > L_rep:
                    L_rep = current_L
            else:
                delta = new_L - current_L
                if random.random() < np.exp(delta / T):
                    residues_rep[i] = new_r
                    current_L = new_L
                else:
                    count[start_new::p] -= 1
                    count[start_old::p] += 1
            
            T *= cooling
        
        if L_rep > best_L_overall:
            best_L_overall = L_rep
            best_residues_overall = residues_rep
            
    if best_residues_overall is None:
        best_residues_overall = [0] * _NPRIMES
        
    return [int(r) for r in best_residues_overall]
