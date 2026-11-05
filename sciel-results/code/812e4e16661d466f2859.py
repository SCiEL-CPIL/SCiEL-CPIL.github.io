"""Fresh design: local search with incremental coverage counting to extend the doubly-covered prefix."""
import time
import numpy as np

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    # Generate primes up to 1000
    sieve = [True] * 1001
    sieve[0] = sieve[1] = False
    for i in range(2, int(1000**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, 1001, i):
                sieve[j] = False
    primes = [p for p in range(2, 1001) if sieve[p]]
    n = len(primes)  # 168
    ncap = 100000
    p_arr = np.array(primes, dtype=np.int64)
    # Initial residues: seed-like
    residues = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)], dtype=np.int64)

    def compute_counts(res):
        counts = np.zeros(ncap + 1, dtype=np.int32)
        for r, p in zip(res, p_arr):
            if r <= ncap:
                counts[r::p] += 1
        return counts

    def compute_L(counts):
        # largest L such that all 1..L have count >= 2
        # find first index where count < 2
        bad = np.where(counts[1:] < 2)[0]
        if len(bad) == 0:
            return ncap
        return bad[0]  # since bad[0] is 0-indexed offset from 1, the failing integer is bad[0]+1, so L = bad[0]

    def score_of(res):
        counts = compute_counts(res)
        return compute_L(counts), counts

    best_res = residues.copy()
    best_L, best_counts = score_of(best_res)

    # Local search: iteratively improve the most-deficient position
    # Strategy: find the first failing integer x (where count < 2), then try to
    # increase its coverage by changing residues of primes that don't already cover it.
    # Also try to extend beyond the current L.

    # We'll use a hill-climbing approach: at each step, find the current L,
    # then try to fix the position L+1 (or the first deficient position) by
    # reassigning a prime's residue.
    
    # For efficiency, we maintain counts incrementally.
    # When we change residue of prime i from old_r to new_r:
    #   subtract 1 from positions old_r, old_r+p, old_r+2p, ...
    #   add 1 to positions new_r, new_r+p, new_r+2p, ...
    
    def change_residue(counts, i, old_r, new_r):
        p = p_arr[i]
        if old_r <= ncap:
            counts[old_r::p] -= 1
        if new_r <= ncap:
            counts[new_r::p] += 1

    # Main search loop
    while time.time() < deadline:
        # Find current L
        cur_L = compute_L(best_counts)
        
        # If we've reached the cap, we're done
        if cur_L >= ncap:
            break
        
        # The first deficient position is cur_L + 1
        target = cur_L + 1
        cur_count = best_counts[target]
        
        # We need to increase coverage at target (and ideally not decrease elsewhere)
        # Try each prime: if prime i does NOT currently cover target, we can reassign it to cover target.
        # If it does cover target, reassigning it away would hurt, so skip those (or handle carefully).
        
        # For primes that don't cover target: new_r = target % p
        # For primes that do cover target: we might still reassign if it helps extend further
        #   but that's riskier. Let's focus on primes that don't cover target first.
        
        improved = False
        best_new_L = cur_L
        best_new_res = None
        best_new_counts = None
        
        # Try reassigning each prime that doesn't currently cover target
        for i in range(n):
            p = p_arr[i]
            old_r = best_res[i]
            # Does prime i currently cover target?
            if (target - old_r) % p == 0:
                continue  # already covers, skip for now
            
            new_r = target % p
            if new_r == old_r:
                continue
            
            # Apply change
            change_residue(best_counts, i, old_r, new_r)
            new_L = compute_L(best_counts)
            
            if new_L > best_new_L:
                best_new_L = new_L
                best_new_res = best_res.copy()
                best_new_res[i] = new_r
                best_new_counts = best_counts.copy()
            
            # Revert
            change_residue(best_counts, i, new_r, old_r)
        
        if best_new_res is not None and best_new_L > cur_L:
            best_res = best_new_res
            best_counts = best_new_counts
            improved = True
        else:
            # No improvement by covering target. Try a more aggressive approach:
            # Try all primes with all possible new residues near target, looking for
            # any that extends L.
            # Actually, let's try: for each prime, try new_r = target % p even if it
            # already covers (might shift coverage pattern helpfully).
            # Or try a random restart / perturbation.
            
            # Try: for each prime, try new_r = (target + k) % p for small k
            for i in range(n):
                p = p_arr[i]
                old_r = best_res[i]
                # Try a few candidate residues
                for delta in range(-3, 4):
                    new_r = (target + delta) % p
                    if new_r == old_r:
                        continue
                    change_residue(best_counts, i, old_r, new_r)
                    new_L = compute_L(best_counts)
                    if new_L > best_new_L:
                        best_new_L = new_L
                        best_new_res = best_res.copy()
                        best_new_res[i] = new_r
                        best_new_counts = best_counts.copy()
                    change_residue(best_counts, i, new_r, old_r)
            
            if best_new_res is not None and best_new_L > cur_L:
                best_res = best_new_res
                best_counts = best_new_counts
                improved = True
            else:
                # Stuck. Try a random perturbation to escape local optimum.
                # Pick 3-5 random primes and randomize their residues.
                num_perturb = np.random.randint(2, 6)
                indices = np.random.choice(n, num_perturb, replace=False)
                for i in indices:
                    p = p_arr[i]
                    old_r = best_res[i]
                    new_r = np.random.randint(0, p)
                    if new_r != old_r:
                        change_residue(best_counts, i, old_r, new_r)
                        best_res[i] = new_r
                
                new_L = compute_L(best_counts)
                if new_L < cur_L - 20:
                    # Perturbation made things worse; revert and try a different approach
                    # Just break to avoid wasting time
                    pass
        
        if not improved and time.time() > deadline - 10:
            break

    return best_res.tolist()
