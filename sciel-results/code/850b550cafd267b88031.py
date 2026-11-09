import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    # Generate primes <= 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = get_primes(1000)
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    
    # We will evaluate up to a sufficiently large L to ensure we can find improvements
    # The score is the largest L such that 1..L is doubly covered.
    # We cap the search space for evaluation at MAX_EVAL to keep memory/time low.
    MAX_EVAL = 20000 
    
    # Precompute x % p for all x in 1..MAX_EVAL and all primes?
    # That's 20000 * 168 = 3.36M integers. ~27MB. Feasible.
    # But we need to check if x % p == r_p.
    # Let's precompute the residue of each x for each prime.
    # residues_of_x[x-1, i] = x % primes[i]
    # Then x is covered by prime i if residues_of_x[x-1, i] == current_residues[i]
    
    # Actually, computing this matrix once is fast.
    xs = np.arange(1, MAX_EVAL + 1, dtype=np.int64)
    # diff = xs[None, :] - 0? No, we want x % p.
    # mods = np.remainder(xs[None, :], primes_arr[:, None])
    # This creates a (168, 20000) array.
    # Let's transpose it to (20000, 168) for easier lookup per x.
    # mods_T = mods.T -> (20000, 168)
    
    # Memory: 20000 * 168 * 8 bytes = 26.8 MB. OK.
    mods_T = np.remainder(xs[None, :], primes_arr[:, None]).T
    
    def evaluate(residues):
        """
        Computes the largest L such that every integer 1..L is covered by at least 2 congruences.
        Returns L (capped at MAX_EVAL).
        """
        # residues is (168,)
        # mods_T is (20000, 168)
        # We need to count for each row x, how many columns i satisfy mods_T[x, i] == residues[i]
        
        # Vectorized comparison:
        # diff = mods_T - residues[None, :]
        # hits = (diff == 0)
        # coverage = np.sum(hits, axis=1)
        
        # This creates a (20000, 168) boolean array. 3.36M bytes. Fast.
        diff = mods_T - residues[None, :]
        hits = (diff == 0)
        coverage = np.sum(hits, axis=1)
        
        mask = coverage < 2
        if not np.any(mask):
            return MAX_EVAL
        
        first_fail_idx = np.argmax(mask)
        return int(first_fail_idx)

    def get_coverage(residues):
        diff = mods_T - residues[None, :]
        hits = (diff == 0)
        return np.sum(hits, axis=1)

    # Strategy: Sequential Greedy Construction with Backtracking/Repair
    # Instead of optimizing L directly, we try to extend the valid prefix one by one.
    # State: residues (168,)
    # We maintain a list of "constraints" satisfied so far.
    # When we extend to x = L+1, we check if it's covered by >= 2 primes.
    # If not, we try to change residues of primes that cover x or could cover x.
    # But changing a residue might break coverage of previous x's.
    
    # Let's use a "Conflict-Driven" approach.
    # 1. Initialize residues randomly.
    # 2. Find L.
    # 3. If L < MAX_EVAL, let x = L+1.
    # 4. Check coverage of x. Let S_x be the set of primes covering x.
    # 5. If |S_x| >= 2, then x is covered. Why did evaluate stop at L?
    #    Ah, evaluate stops at the FIRST x with coverage < 2.
    #    So if evaluate returns L, then x=L+1 has coverage < 2.
    #    This means |S_{L+1}| <= 1.
    # 6. We need to increase coverage of x = L+1 to at least 2.
    # 7. We can do this by changing the residue of a prime p NOT in S_x to x % p.
    # 8. However, this might reduce coverage of some y in 1..L.
    # 9. We check if the new state maintains coverage >= 2 for all y in 1..L.
    # 10. If yes, accept. If no, try another prime or revert.
    
    # To make this efficient, we can precompute for each prime i, which x's in 1..MAX_EVAL it covers.
    # But residues change.
    
    # Let's stick to the repair logic but make it smarter.
    # Key insight: Changing a prime's residue affects all x such that x % p == old_r (removes coverage) 
    # and x % p == new_r (adds coverage).
    # For a specific x_target, we want to add coverage.
    # We pick a prime i. new_r = x_target % p_i.
    # This adds coverage to x_target.
    # It removes coverage from any y where y % p_i == old_r.
    # We need to ensure that for all such y, the remaining coverage is still >= 2.
    
    # Let's implement a "Safe Move" generator.
    # 1. Identify x_target = L+1.
    # 2. Identify primes not covering x_target.
    # 3. For each such prime i (sampled), calculate the set of y in 1..L that are currently covered by i.
    #    These are y such that y % p_i == old_r.
    # 4. For each such y, check if coverage[y] > 2. If coverage[y] == 2, removing i's coverage makes it 1 -> FAIL.
    # 5. If no such y exists with coverage[y] == 2, the move is "safe" and will extend L to at least x_target.
    #    Actually, it might extend further if x_target+1 also becomes covered, etc.
    
    # This "Safe Move" check is O(L * (1/p_i)) roughly? 
    # Number of y in 1..L covered by i is ~ L/p_i.
    # For small p_i, this is large. For large p_i, this is small.
    # So we should prefer large primes for safe moves?
    # Large primes have fewer multiples in 1..L, so they are less likely to break coverage.
    
    residues = np.zeros(n_primes, dtype=np.int64)
    for i, p in enumerate(primes):
        residues[i] = random.randint(0, p - 1)
    
    current_score = evaluate(residues)
    best_residues = residues.copy()
    best_score = current_score
    
    current_coverage = get_coverage(residues)
    
    # Precompute indices of primes sorted by size? 
    # We want to try large primes first for safe moves.
    prime_indices_by_size = np.argsort(primes_arr)[::-1] # Largest first
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    
    while time.time() < deadline:
        if current_score >= MAX_EVAL:
            # Try to push higher? No, capped.
            # Just do random moves to maintain diversity? 
            # Or break if we've hit the cap for a long time.
            # Let's just do random moves.
            i = random.randint(0, n_primes - 1)
            new_r = random.randint(0, primes_arr[i] - 1)
            old_r = residues[i]
            residues[i] = new_r
            
            new_score = evaluate(residues)
            delta = new_score - current_score
            current_coverage = get_coverage(residues)
            
            if delta > 0:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_residues = residues.copy()
            else:
                if delta == 0:
                    accept_prob = 0.5
                else:
                    try:
                        accept_prob = np.exp(delta / T)
                    except OverflowError:
                        accept_prob = 0.0
                        
                if random.random() < accept_prob:
                    current_score = new_score
                else:
                    residues[i] = old_r
                    current_coverage = get_coverage(residues)
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
            continue
            
        x_target = current_score + 1
        idx_target = x_target - 1
        c_target = current_coverage[idx_target]
        
        # Find primes that do NOT cover x_target
        # A prime i covers x_target if (x_target - residues[i]) % primes[i] == 0
        # Or simply: mods_T[idx_target, i] == residues[i]
        
        covers_target = (mods_T[idx_target, :] == residues)
        candidate_indices = np.where(~covers_target)[0]
        
        move_made = False
        
        # Try safe moves first
        # Sort candidates by prime size (largest first) to minimize breakage risk
        candidates_sorted = candidate_indices[np.argsort(primes_arr[candidate_indices])[::-1]]
        
        for i in candidates_sorted[:50]: # Try top 50 largest primes that don't cover x_target
            p_i = primes_arr[i]
            old_r = residues[i]
            new_r = x_target % p_i
            
            # Check safety: 
            # Find all y in 1..current_score such that y % p_i == old_r
            # These are the y's that will lose coverage from prime i.
            # If any such y has coverage == 2, the move is unsafe.
            
            # Efficient check:
            # We have current_coverage array.
            # We need to check y in 1..current_score where (y % p_i) == old_r.
            # We can generate these y's: y = old_r + k*p_i.
            # Range: 1 <= y <= current_score.
            
            # k_min = ceil((1 - old_r) / p_i)
            # k_max = floor((current_score - old_r) / p_i)
            
            if old_r == 0:
                # y = k * p_i
                # k >= 1/p_i -> k=1 if p_i <= current_score
                start_k = 1
            else:
                # y = old_r + k*p_i >= 1
                # k*p_i >= 1 - old_r
                # k >= ceil((1 - old_r)/p_i)
                # Since old_r < p_i, 1-old_r is between 1-(p_i-1) and 1-1=0.
                # If old_r >= 1, 1-old_r <= 0, so k >= 0.
                # If old_r = 0, handled above.
                start_k = 0
            
            end_k = int((current_score - old_r) // p_i)
            
            if end_k < start_k:
                # No y in 1..current_score is covered by this prime with old_r?
                # That means changing it doesn't break any existing coverage.
                # This is a perfectly safe move.
                residues[i] = new_r
                new_score = evaluate(residues)
                delta = new_score - current_score
                current_coverage = get_coverage(residues)
                
                if delta > 0:
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_residues = residues.copy()
                    move_made = True
                    break
                else:
                    # Revert
                    residues[i] = old_r
                    current_coverage = get_coverage(residues)
                    continue
            
            # Check coverage of affected y's
            # y_vals = old_r + np.arange(start_k, end_k + 1) * p_i
            # We only care about y_vals <= current_score (guaranteed by end_k)
            # And y_vals >= 1 (guaranteed by start_k logic? 
            # If old_r=0, start_k=1 -> y=p_i >= 2. OK.
            # If old_r>0, start_k=0 -> y=old_r >= 1. OK.)
            
            # However, we only have coverage for 1..MAX_EVAL.
            # y_vals are in 1..current_score <= MAX_EVAL.
            
            y_vals = old_r + np.arange(start_k, end_k + 1, dtype=np.int64) * p_i
            
            # Get coverage of these y's
            # y_vals are 1-indexed values. Array indices are y_vals - 1.
            idxs = y_vals - 1
            coverages_affected = current_coverage[idxs]
            
            # If any coverage is exactly 2, removing this prime makes it 1 -> Bad.
            if np.any(coverages_affected == 2):
                continue
            else:
                # Safe move!
                residues[i] = new_r
                new_score = evaluate(residues)
                delta = new_score - current_score
                current_coverage = get_coverage(residues)
                
                if delta > 0:
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_residues = residues.copy()
                    move_made = True
                    break
                else:
                    # Shouldn't happen if safety check was correct, but just in case
                    residues[i] = old_r
                    current_coverage = get_coverage(residues)
                    continue
        
        if not move_made:
            # No safe move found. Try random move or simulated annealing move.
            # Pick a random prime
            i = random.randint(0, n_primes - 1)
            new_r = random.randint(0, primes_arr[i] - 1)
            old_r = residues[i]
            residues[i] = new_r
            
            new_score = evaluate(residues)
            delta = new_score - current_score
            current_coverage = get_coverage(residues)
            
            if delta > 0:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_residues = residues.copy()
            else:
                if delta == 0:
                    accept_prob = 0.5
                else:
                    try:
                        accept_prob = np.exp(delta / T)
                    except OverflowError:
                        accept_prob = 0.0
                        
                if random.random() < accept_prob:
                    current_score = new_score
                else:
                    residues[i] = old_r
                    current_coverage = get_coverage(residues)

        T *= cooling_rate
        if T < T_min:
            T = T_min

        if time.time() > deadline - 5.0:
            break

    return best_residues.tolist()
