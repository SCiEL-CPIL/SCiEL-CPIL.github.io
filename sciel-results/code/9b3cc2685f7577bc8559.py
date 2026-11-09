import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit+1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    num_primes = len(primes)
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
        # Vectorized calculation of coverage
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
        return int(xs[first_fail_idx] - 1)

    def get_coverage(residues):
        """
        Returns the coverage array for all x in 1..MAX_EVAL.
        """
        diff = xs[None, :] - residues[:, None]
        mods = np.remainder(diff, primes_arr[:, None])
        hits = (mods == 0)
        return np.sum(hits, axis=0)

    def repair_step(residues, current_L):
        """
        Attempts to extend current_L by fixing the first integer that fails.
        Returns (new_residues, new_L, improved)
        """
        if current_L >= MAX_EVAL:
            return residues, current_L, False

        target_x = current_L + 1
        # Get current coverage for target_x
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
            # This shouldn't happen if current_L is the first failure.
            # Unless current_L was calculated wrong or target_x is not the first failure.
            # If target_x is covered by >= 2, then current_L should have been higher.
            # So this case implies a logic error or numerical issue.
            # We'll break to avoid infinite loop.
            return residues, current_L, False

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
            
            # Check if the prefix 1..current_L is still validly covered (>= 2)
            # We only need to check the prefix up to current_L, not the whole MAX_EVAL
            # because we are trying to extend from current_L.
            # However, changing a residue might break coverage for some x in 1..current_L.
            # So we must verify coverage for 1..current_L.
            
            # Optimization: Instead of checking the whole array, we can check only up to current_L.
            # But get_coverage computes for the whole array.
            # Let's create a faster check function that only checks up to a limit.
            
            # For now, let's just use get_coverage and check the first current_L elements.
            # This is O(MAX_EVAL * num_primes) which is ~3.3M ops.
            # 115s / 3.3M ops * 1e9 ops/s ~ 34s for 10000 evals.
            # So we can do ~34000 evals.
            # Each repair step tries ~168 candidates. So ~200 repair steps.
            # This might be enough to reach L=1000.
            
            coverage = get_coverage(residues)
            
            # Check if all x in 1..current_L have coverage >= 2
            # xs[0] is 1, so we check xs[0..current_L-1]
            if np.all(coverage[:current_L] >= 2):
                # The change is valid.
                # The new L is at least current_L.
                # It might be larger if target_x is now covered and subsequent numbers are too.
                new_L = evaluate(residues)
                found_repair = True
                return residues, new_L, True
            else:
                # Revert change
                residues[idx] = old_res
        
        return residues, current_L, False

    # Initial solution: Random residues
    residues = np.array([random.randint(0, p - 1) for p in primes], dtype=np.int64)
    
    current_L = evaluate(residues)
    best_residues = residues.copy()
    best_L = current_L
    
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
        current_L = evaluate(residues)
        if current_L > best_L:
            best_L = current_L
            best_residues = residues.copy()
            stuck_count = 0  # Reset stuck counter on improvement
            
        if current_L >= MAX_EVAL:
            break
            
        # The next integer to cover is L + 1
        target_x = current_L + 1
        
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
            # This shouldn't happen if current_L is the first failure.
            # Unless current_L was calculated wrong or target_x is not the first failure.
            # If target_x is covered by >= 2, then current_L should have been higher.
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
            
            # Check if the prefix 1..current_L is still validly covered (>= 2)
            # We only need to check the prefix up to current_L, not the whole MAX_EVAL
            # because we are trying to extend from current_L.
            # However, changing a residue might break coverage for some x in 1..current_L.
            # So we must verify coverage for 1..current_L.
            
            # Optimization: Instead of checking the whole array, we can check only up to current_L.
            # But get_coverage computes for the whole array.
            # Let's create a faster check function that only checks up to a limit.
            
            # For now, let's just use get_coverage and check the first current_L elements.
            # This is O(MAX_EVAL * num_primes) which is ~3.3M ops.
            # 115s / 3.3M ops * 1e9 ops/s ~ 34s for 10000 evals.
            # So we can do ~34000 evals.
            # Each repair step tries ~168 candidates. So ~200 repair steps.
            # This might be enough to reach L=1000.
            
            coverage = get_coverage(residues)
            
            # Check if all x in 1..current_L have coverage >= 2
            # xs[0] is 1, so we check xs[0..current_L-1]
            if np.all(coverage[:current_L] >= 2):
                # The change is valid.
                # The new L is at least current_L.
                # It might be larger if target_x is now covered and subsequent numbers are too.
                new_L = evaluate(residues)
                found_repair = True
                stuck_count = 0
                break
            else:
                # Revert change
                residues[idx] = old_res
        
        if not found_repair:
            stuck_count += 1
            if stuck_count > MAX_STUCK:
                # Perturb: change a few random residues to escape local minimum
                num_perturb = random.randint(3, 10)
                for _ in range(num_perturb):
                    i = random.randint(0, num_primes - 1)
                    p = primes[i]
                    residues[i] = random.randint(0, p - 1)
                stuck_count = 0
                current_L = evaluate(residues)
                if current_L < best_L:
                    # If perturbation made it worse, revert to best
                    residues = best_residues.copy()
                    current_L = best_L

    return [int(r) for r in best_residues]
