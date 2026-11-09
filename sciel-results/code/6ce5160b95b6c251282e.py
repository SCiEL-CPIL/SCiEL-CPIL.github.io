import time
import random
import numpy as np

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
    
    # We evaluate up to a large limit to ensure we can find improvements
    # The score is the largest L such that 1..L is doubly covered.
    # We cap the search space for evaluation at MAX_EVAL to keep memory/time low.
    # 20000 is sufficient for this task size.
    MAX_EVAL = 20000
    
    xs = np.arange(1, MAX_EVAL + 1, dtype=np.int64)
    
    def evaluate(residues):
        """
        Computes the largest L such that every integer 1..L is covered by at least 2 congruences.
        Returns L (capped at MAX_EVAL).
        """
        # coverage[x] = number of primes p such that x = r_p (mod p)
        # residues is an array of shape (n_primes,)
        # primes_arr is an array of shape (n_primes,)
        
        # Vectorized calculation of coverage
        # diff[i, j] = xs[j] - residues[i]
        # mods[i, j] = diff[i, j] % primes_arr[i]
        # hits[i, j] = (mods[i, j] == 0)
        # coverage[j] = sum(hits[i, j] for i in range(n_primes))
        
        # To save memory, we can do this in chunks if MAX_EVAL * n_primes is too large,
        # but 20000 * 168 ~ 3.3M elements, which is fine for int64.
        
        diff = xs[None, :] - residues[:, None]
        mods = np.remainder(diff, primes_arr[:, None])
        hits = (mods == 0)
        coverage = np.sum(hits, axis=0)
        
        # Find the first index where coverage < 2
        mask = coverage < 2
        if not np.any(mask):
            return MAX_EVAL
        first_fail_idx = np.argmax(mask)
        # xs[0] is 1, so if xs[first_fail_idx] fails, L is xs[first_fail_idx] - 1
        # But wait, the mask is on the array xs.
        # If xs[k] is the first to fail, then 1..xs[k]-1 are good.
        # Since xs is 1..MAX_EVAL, xs[k] is the integer value.
        # So L = xs[first_fail_idx] - 1.
        return int(xs[first_fail_idx] - 1)

    def get_coverage(residues):
        """
        Returns the coverage array for all x in 1..MAX_EVAL.
        """
        diff = xs[None, :] - residues[:, None]
        mods = np.remainder(diff, primes_arr[:, None])
        hits = (mods == 0)
        return np.sum(hits, axis=0)

    # Initial solution: Random residues
    # A random start is often better than a structured one for this repair strategy
    # because the repair process will find the structure needed for the prefix.
    residues = np.array([random.randint(0, p - 1) for p in primes], dtype=np.int64)
    
    current_score = evaluate(residues)
    best_residues = residues.copy()
    best_score = current_score
    
    # Greedy Sequential Repair Loop
    # We try to extend the prefix L by fixing the first integer that fails.
    
    # To avoid getting stuck in a specific local minimum due to random choices,
    # we can restart the process a few times with different random seeds if time permits.
    # However, a single strong greedy run is often very effective.
    
    # Let's run a loop that attempts to repair the current gap.
    # If no repair is found for the current gap, we might be stuck.
    # In that case, we can try a "random walk" to escape, or just restart.
    
    # Strategy:
    # 1. Find current L.
    # 2. Let x_fail = L + 1.
    # 3. Check coverage of x_fail. It must be < 2.
    # 4. Identify primes that currently cover x_fail (coverage 1) and primes that don't (coverage 0).
    # 5. To increase coverage of x_fail to 2, we need to change residues of primes that do NOT cover x_fail
    #    to ones that DO cover x_fail.
    # 6. Changing a prime's residue might break coverage for other numbers in 1..L.
    #    We must verify that the change doesn't reduce coverage of any x in 1..L below 2.
    # 7. We iterate through candidate primes (those not covering x_fail) and try to set their residue
    #    to x_fail % p. We check if this new configuration maintains coverage >= 2 for all 1..L.
    # 8. If yes, accept the change. The new L will be at least L (likely larger).
    # 9. Repeat.
    
    # Optimization: Instead of re-evaluating the whole 1..L for every candidate,
    # we can precompute the "criticality" of each prime for each x in 1..L.
    # But L grows, so this is dynamic.
    # Given MAX_EVAL=20000 and n_primes=168, a full evaluation takes ~1-2ms.
    # 115s / 2ms = ~50,000 evaluations possible.
    # Each repair step might try 10-50 candidates. So ~1000-5000 repair steps.
    # This should be enough to reach L=1000+.
    
    # Let's implement the repair loop.
    
    # To handle the case where we get stuck (no prime can be changed to cover x_fail
    # without breaking 1..L), we will introduce a "perturbation" step.
    # If stuck for K iterations, we randomly change a few residues to escape the local minimum.
    
    stuck_count = 0
    MAX_STUCK = 50
    
    while time.time() < deadline:
        current_score = evaluate(residues)
        if current_score > best_score:
            best_score = current_score
            best_residues = residues.copy()
            stuck_count = 0  # Reset stuck counter on improvement
            
        if current_score >= MAX_EVAL:
            break
            
        # The next integer to cover is L + 1
        target_x = current_score + 1
        
        # Get current coverage for target_x
        # We can compute this quickly without full array
        # coverage_target = sum(1 for i, p in enumerate(primes) if (target_x - residues[i]) % p == 0)
        
        # Let's compute full coverage to identify which primes cover target_x
        # and which don't.
        # Actually, we just need to know which primes are candidates for change.
        # Candidates: primes p where (target_x - residues[i]) % p != 0.
        
        # We want to change one prime's residue to target_x % p.
        # Let's collect candidates.
        candidates = []
        for i, p in enumerate(primes):
            if (target_x - residues[i]) % p != 0:
                candidates.append(i)
                
        if not candidates:
            # All primes already cover target_x? That means coverage >= 168, which is >= 2.
            # This shouldn't happen if current_score is the first failure.
            # Unless current_score was calculated wrong or target_x is not the first failure.
            # If target_x is covered by >= 2, then current_score should have been higher.
            # So this case implies a logic error or numerical issue.
            # We'll break to avoid infinite loop.
            break

        # Shuffle candidates to randomize the order of attempts
        random.shuffle(candidates)
        
        found_repair = False
        # Try to repair by changing one candidate
        for idx in candidates:
            old_res = residues[idx]
            new_res = target_x % primes[idx]
            
            # If the residue doesn't change, skip
            if old_res == new_res:
                continue
                
            # Apply change
            residues[idx] = new_res
            
            # Check if the prefix 1..current_score is still validly covered (>= 2)
            # We only need to check up to current_score, not MAX_EVAL, because
            # we are trying to extend from current_score.
            # However, evaluate() checks the whole range.
            # We can write a faster check for just 1..current_score.
            
            # Let's use a fast check function
            # coverage_check = get_coverage(residues)
            # prefix_coverage = coverage_check[:current_score]
            # if np.all(prefix_coverage >= 2):
            #     found_repair = True
            #     break
            # else:
            #     residues[idx] = old_res # Revert
            
            # To save time, let's just call evaluate. It's fast enough.
            # But evaluate returns the new L. If new L >= current_score, it's a valid step.
            # Actually, if we fix target_x, the new L will be >= target_x - 1 = current_score.
            # So if evaluate() returns >= current_score, we are good.
            # Wait, if we break coverage at some x < target_x, evaluate will return < current_score.
            
            new_score = evaluate(residues)
            
            if new_score >= current_score:
                # This change is valid and potentially improves the score
                found_repair = True
                # We keep the change
                break
            else:
                # Revert
                residues[idx] = old_res
        
        if found_repair:
            stuck_count = 0
        else:
            # No single prime change could fix target_x without breaking the prefix
            stuck_count += 1
            
            if stuck_count > MAX_STUCK:
                # We are stuck. Let's try to escape by randomizing a few residues.
                # This is a "kick" to jump out of the local minimum.
                kick_size = random.randint(3, 10)
                kick_indices = random.sample(range(n_primes), kick_size)
                for i in kick_indices:
                    residues[i] = random.randint(0, primes[i] - 1)
                
                # Reset stuck counter
                stuck_count = 0
                # Note: This might lower the current score significantly.
                # The loop will then try to repair from this new lower score.
                # This is a form of simulated annealing but driven by the repair logic.
                
        # Small sleep to prevent CPU starvation if loop is too fast?
        # No, we want max speed.
        
    return best_residues.tolist()
