import time
import random
from fractions import Fraction
import numpy as np

# EVOLVE-BLOCK-START

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _get_primes(limit):
    return [p for p in range(2, limit + 1) if _is_prime(p)]

def _compute_coverage(primes, residues, max_check=100000):
    """
    Computes the largest L such that all integers 1..L are covered.
    Uses a boolean array for speed.
    """
    if not primes:
        return 0
    
    covered = np.zeros(max_check + 1, dtype=bool)
    
    for p, r in zip(primes, residues):
        if p > max_check:
            if 1 <= r <= max_check:
                covered[r] = True
        else:
            start = r
            if start == 0:
                start = p
            if start > max_check:
                if r == 0 and p <= max_check:
                    start = p
                else:
                    continue
            
            indices = np.arange(start, max_check + 1, p)
            if len(indices) > 0:
                covered[indices] = True
        
    uncovered = np.where(~covered[1:])[0]
    if len(uncovered) == 0:
        return max_check
    else:
        return uncovered[0] + 1

def _compute_coverage_delta(current_covered, p, old_r, new_r, max_check=100000):
    """
    Computes the new coverage length given the current covered array,
    assuming we change the residue of prime p from old_r to new_r.
    Returns the new coverage length L.
    """
    # We don't modify current_covered in place to avoid side effects in the loop
    # but we can create a view or copy. Copying 100k bools is fast (~100us).
    # However, we can be smarter: 
    # 1. Unmark old_r
    # 2. Mark new_r
    # 3. Find first False
    
    # To be safe and simple, let's just update a temporary copy or do it in place and revert.
    # In-place with revert is faster.
    
    # Unmark old residue
    if old_r != new_r:
        if p > max_check:
            if 1 <= old_r <= max_check:
                current_covered[old_r] = False
        else:
            start = old_r
            if start == 0:
                start = p
            if start <= max_check:
                indices = np.arange(start, max_check + 1, p)
                # Only unmark if this prime was the *only* one covering them?
                # No, simple boolean OR logic means we can't just unmark if others cover it.
                # This "delta" approach only works if we track counts or if we accept that 
                # unmarking might leave a False where it should be True (if another prime covers it).
                # If we unmark and it becomes False, but another prime covers it, we've under-estimated coverage.
                # To be exact, we must recompute or use a counter array.
                
                # Counter array approach is more robust for incremental updates.
                # But switching to counters now is a big change.
                
                # Alternative: Just recompute coverage from scratch for the candidate?
                # That was the bottleneck.
                
                # Let's stick to a robust local search that *recomputes* coverage 
                # but is much faster by using Numpy vectorization and early termination.
                
                pass
                
    # Since exact incremental update with boolean arrays is tricky due to OR semantics,
    # we will use a different optimization: 
    # 1. Maintain a list of (p, r).
    # 2. For each step, pick a random prime in the list.
    # 3. Try a few random new residues for it.
    # 4. For each candidate, compute coverage.
    #    Optimization: We can pre-compute the "uncovered" set.
    #    If the new residue doesn't cover any currently uncovered numbers, it's likely worse or neutral.
    #    We can quickly check if the new residue covers the *first* uncovered number.
    
    # Actually, the simplest speedup is to reduce the max_check for intermediate steps?
    # No, we need exact L.
    
    # Let's try a different strategy: 
    # The previous program was too slow because it did 50k steps of full coverage calc.
    # Full coverage calc is ~1ms. 50k * 1ms = 50s. That's actually doable in 120s.
    # Why did it score 1.0? 
    # The seed score is 1.0. The inspiration 1 scored 1.67 with "randomized local search".
    # The current program's SA was likely stuck or the initialization was bad.
    
    # Let's implement a **Greedy + Local Search** hybrid.
    # 1. Start with a set of small primes (high coverage density).
    # 2. Greedily add primes/residues that cover the most new numbers until budget is tight.
    # 3. Local search: Randomly swap residues. If coverage improves, keep. If it stays same, keep with prob 0.5 (to escape plateaus).
    
    return -1 # Placeholder, we will not use this function directly in the loop if we switch strategies.

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for cleanup/return
    
    primes_all = _get_primes(1000)
    
    # Strategy:
    # 1. Initialize with a set of small primes. Small primes cover more numbers per residue.
    #    Primes 2, 3, 5, 7, 11, 13...
    #    Sum of 1/p for small primes grows log-log.
    #    Sum 1/p for p <= 100 is ~ 0.5.
    #    Sum 1/p for p <= 1000 is ~ 0.92.
    #    We have budget 1.
    
    # Let's start with primes up to 100.
    initial_primes = [p for p in primes_all if p <= 100]
    
    # Initial residues: random
    random.seed(42)
    initial_residues = [random.randint(0, p - 1) for p in initial_primes]
    
    current_primes = list(initial_primes)
    current_residues = list(initial_residues)
    
    # Calculate initial coverage
    current_coverage = _compute_coverage(current_primes, current_residues, 100000)
    
    best_classes = [[p, r] for p, r in zip(current_primes, current_residues)]
    best_coverage = current_coverage
    
    # Check budget
    budget = sum(Fraction(1, p) for p in current_primes)
    
    # If budget is too high, remove largest primes (smallest 1/p cost reduction? No, remove largest p to keep small p which cover more).
    # Wait, if budget > 1, we must remove primes. Removing small primes (large 1/p) reduces budget most.
    # But small primes are valuable for coverage.
    # Let's check if initial budget <= 1.
    # Sum 1/p for p <= 100:
    # 1/2+1/3+1/5+1/7+1/11+... 
    # 0.5 + 0.33 + 0.2 + 0.14 + 0.09 + ...
    # It's roughly 0.6-0.7. So we are safe.
    
    # Local Search Loop
    # We want to maximize coverage.
    # Move: Pick a random index i. Pick a new random residue r_new.
    # Evaluate coverage.
    # If coverage > current, accept.
    # If coverage == current, accept with prob 0.5 (random walk).
    # If coverage < current, reject.
    
    # To speed up evaluation, we can't avoid the O(N) coverage check easily without complex bookkeeping.
    # But we can limit the number of primes to keep the array operations fast?
    # No, Numpy is fast. 100k bools is tiny.
    
    # Let's try to be smarter about *which* residue to try.
    # Instead of random, try to cover the *first uncovered* number.
    
    # Find first uncovered number
    def get_first_uncovered(primes, residues):
        covered = np.zeros(100001, dtype=bool)
        for p, r in zip(primes, residues):
            if p > 100000:
                if 1 <= r <= 100000:
                    covered[r] = True
            else:
                start = r if r != 0 else p
                if start <= 100000:
                    indices = np.arange(start, 100001, p)
                    covered[indices] = True
        unc = np.where(~covered[1:])[0]
        if len(unc) == 0: return 100000
        return unc[0] + 1

    # Greedy Phase:
    # While budget allows and we have primes available:
    # 1. Find first uncovered number U.
    # 2. For each available prime p, find residue r such that r == U % p.
    # 3. This guarantees U is covered.
    # 4. Check if adding this (p, r) increases coverage significantly (by checking coverage of U and neighbors).
    # 5. Pick the one that maximizes coverage gain per cost (1/p).
    
    # Actually, a simpler greedy:
    # Sort primes by 1/p descending (small primes first).
    # Add them one by one with residue chosen to cover the current first gap.
    
    current_primes = []
    current_residues = []
    current_budget = Fraction(0, 1)
    
    # Use all primes up to 1000, sorted by value
    candidates = sorted(primes_all, key=lambda x: x)
    
    for p in candidates:
        cost = Fraction(1, p)
        if current_budget + cost > 1:
            continue
            
        # Find first uncovered number in current configuration
        if not current_primes:
            first_uncovered = 1
        else:
            first_uncovered = get_first_uncovered(current_primes, current_residues)
            
        if first_uncovered >= 100000:
            break
            
        # Choose residue to cover first_uncovered
        r = first_uncovered % p
        
        # Tentatively add
        test_primes = current_primes + [p]
        test_residues = current_residues + [r]
        
        # Check new coverage
        new_cov = _compute_coverage(test_primes, test_residues, 100000)
        
        # We only accept if it improves coverage or is equal (to keep budget low? No, we want max coverage)
        # Actually, adding a prime always covers at least one number (first_uncovered).
        # So coverage should not decrease.
        
        current_primes.append(p)
        current_residues.append(r)
        current_budget += cost
        
        # Update best
        if new_cov > best_coverage:
            best_coverage = new_cov
            best_classes = [[pp, rr] for pp, rr in zip(current_primes, current_residues)]
            
        if time.time() > deadline - 20:
            break
            
    # Local Search Phase
    # Now we have a set of primes. We can swap residues.
    # We can also swap primes if we have budget slack.
    
    # For local search, we stick to the current set of primes to maintain budget.
    # We optimize residues.
    
    T = 1.0
    cooling = 0.999
    
    while time.time() < deadline - 5:
        if not current_primes:
            break
            
        idx = random.randint(0, len(current_primes) - 1)
        p = current_primes[idx]
        old_r = current_residues[idx]
        
        # Try a few random residues
        found_improvement = False
        for _ in range(5): # Try 5 candidates per step
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
                
            test_residues = current_residues[:]
            test_residues[idx] = new_r
            
            new_cov = _compute_coverage(current_primes, test_residues, 100000)
            
            if new_cov > current_coverage:
                current_coverage = new_cov
                current_residues = test_residues
                found_improvement = True
                
                if current_coverage > best_coverage:
                    best_coverage = current_coverage
                    best_classes = [[pp, rr] for pp, rr in zip(current_primes, current_residues)]
                    
                break # Move to next step
            elif new_cov == current_coverage:
                # Accept with some probability to explore
                if random.random() < 0.5:
                    current_residues = test_residues
                    # current_coverage stays same
                    break # Move to next step
            # else reject
        
        if not found_improvement and random.random() < 0.1:
             # Occasionally try to swap a prime?
             # If we have budget slack, we can add a prime.
             # If we are over budget, we must remove.
             # For now, stick to residue optimization.
             pass

        T *= cooling
        
    # Final Check
    best_primes = [c[0] for c in best_classes]
    best_residues = [c[1] for c in best_classes]
    
    # Verify budget
    budget = sum(Fraction(1, p) for p in best_primes)
    if budget > 1:
        # This shouldn't happen with the greedy construction, but just in case
        # Fall back to a valid seed-like configuration
        fallback_primes = [p for p in primes_all if p >= 13]
        fallback_residues = [(i + 1) % p for i, p in enumerate(fallback_primes)]
        return [[p, r] for p, r in zip(fallback_primes, fallback_residues)]
        
    return best_classes

# EVOLVE-BLOCK-END
