"""Structural approach: greedy gap-closure with fast numpy coverage + local search."""
import time
import numpy as np

def construct():
    deadline = time.time() + 130  # safety margin before 150s limit
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    n_primes = len(primes)  # 168
    
    # Use a larger range to allow L to grow
    MAX_L = 20000
    
    # NumPy coverage array
    coverage = np.zeros(MAX_L + 1, dtype=np.int32)
    residues = np.zeros(n_primes, dtype=np.int32)
    
    def find_first_gap(cov):
        """Find the first index >= 1 where coverage < 2. Returns 0 if none."""
        # We care about indices 1..MAX_L
        # cov[0] is unused for the prefix logic (starts at 1)
        gaps = np.where(cov[1:] < 2)[0]
        if len(gaps) == 0:
            return MAX_L + 1
        return gaps[0] + 1
    
    def add_congruence(idx, r, cov):
        p = primes[idx]
        # x in [1, MAX_L] such that x % p == r
        # First x >= 1: if r == 0, x=p; else x=r (if r>=1)
        # Actually, range(start, stop, step)
        # We want to increment cov at indices r, r+p, r+2p...
        # But indices must be >= 1.
        # If r == 0, indices are p, 2p, ...
        # If r > 0, indices are r, r+p, ... (r is in 1..p-1)
        start = r if r > 0 else p
        cov[start::p] += 1
        
    def remove_congruence(idx, r, cov):
        p = primes[idx]
        start = r if r > 0 else p
        cov[start::p] -= 1

    # Phase 1: Greedy Construction
    # Process primes in increasing order.
    # For each prime, choose the residue that best extends the double-covered prefix.
    
    for idx, p in enumerate(primes):
        if time.time() > deadline - 30:
            break
            
        # Find current gap
        gap = find_first_gap(coverage)
        
        # If gap is beyond MAX_L, we are safe for now
        if gap > MAX_L:
            # Just pick a random or 0? Let's pick 0 for simplicity or random to avoid bias
            r = np.random.randint(0, p)
            residues[idx] = r
            add_congruence(idx, r, coverage)
            continue
            
        # We want to cover the gap and positions around it.
        # The gap is at `gap`.
        # We can choose r such that gap % p == r, which covers the gap.
        # But we also want to cover other under-covered spots.
        
        # Strategy: Try all residues if p is small, otherwise sample.
        # Score: how many positions in [1, gap + window] become double-covered?
        
        # To be fast, let's focus on the immediate neighborhood of the gap.
        # If we pick r = gap % p, we cover the gap.
        # Does this help extend L?
        
        # Let's try a few candidates:
        # 1. r = gap % p (covers the gap)
        # 2. r = (gap + 1) % p
        # 3. r = (gap - 1) % p
        # 4. Random samples
        
        candidates = set()
        candidates.add(gap % p)
        candidates.add((gap + 1) % p)
        candidates.add((gap - 1) % p)
        candidates.add(0)
        
        # Add some randoms
        if p < 100:
            candidates.update(np.random.randint(0, p, 10))
        else:
            candidates.update(np.random.randint(0, p, 20))
        
        best_r = gap % p
        best_score = -np.inf
        
        # We evaluate by temporarily adding the congruence and checking the new gap.
        # This is expensive if done for many candidates.
        # Optimization: Only check if the candidate covers the current gap.
        # If it doesn't cover the gap, it can't extend L past the gap.
        # So only candidates with r == gap % p can extend L.
        # Wait, if r != gap % p, the gap remains. L stays the same.
        # So we MUST pick r = gap % p to make progress?
        # Yes, to increase L, we must cover the current gap.
        # So the only useful candidate is r = gap % p.
        
        # Is it always optimal to cover the gap?
        # Yes, because L is defined by the first gap.
        # So we set r = gap % p.
        
        r = gap % p
        residues[idx] = r
        add_congruence(idx, r, coverage)
        
        # Check if we improved
        new_gap = find_first_gap(coverage)
        # If new_gap == gap, it means the gap wasn't covered?
        # gap % p == r means coverage[gap] increases by 1.
        # If coverage[gap] was 0, it becomes 1. Still < 2.
        # If coverage[gap] was 1, it becomes 2. Gap moves.
        # If coverage[gap] was >= 2, it stays >= 2. Gap moves.
        
    # Phase 2: Local Search / Hill Climbing
    # Now we have a baseline. Try to improve L by changing residues.
    # We can change a residue to cover a *new* gap that appeared later,
    # or to allow removing a "wasted" coverage? No, we can't remove.
    # We can only change residues.
    
    current_L = find_first_gap(coverage) - 1
    if current_L > MAX_L:
        current_L = MAX_L
        
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        
        # Find the current gap
        gap = find_first_gap(coverage)
        
        if gap > MAX_L:
            break
            
        # We want to cover `gap`.
        # Which prime can cover it?
        # Any prime p can cover gap if we set r_p = gap % p.
        # But we already set r_p for all primes.
        # We need to find a prime p where changing r_p to gap % p
        # improves the situation (e.g., covers gap without breaking earlier coverage).
        
        # Actually, the current gap exists because no prime currently covers it with sufficient multiplicity.
        # We can pick ANY prime index idx, and change its residue to gap % primes[idx].
        # This will increment coverage[gap] by 1.
        # If coverage[gap] was 1, it becomes 2 -> Gap resolved.
        # If coverage[gap] was 0, it becomes 1 -> Gap still there (needs 2).
        
        # So we should look for a prime that already covers gap (coverage contribution 1)
        # and change it? No, if it already covers it, coverage[gap] includes it.
        
        # Let's try changing the residue of each prime to gap % p.
        # If this increases coverage[gap] from 1 to 2, we win.
        # If it increases from 0 to 1, we don't win immediately, but maybe it helps later?
        # Probably not directly.
        
        # So, for each prime idx:
        # 1. Calculate new_r = gap % p.
        # 2. If new_r == old_r, skip.
        # 3. Check if coverage[gap] is currently 1.
        #    If yes, and prime idx currently covers gap (old_r == gap % p), then we already have 1 contribution from it.
        #    Wait, if old_r == gap % p, then it ALREADY covers gap.
        #    So changing it to new_r (which is same) does nothing.
        
        # This logic is tricky.
        # Let's just try to change residues to close the gap.
        
        # For each prime idx:
        #   target_r = gap % p
        #   if residues[idx] != target_r:
        #       # Changing to target_r will add 1 to coverage[gap].
        #       # Will it break anything?
        #       # It removes 1 from coverage at positions covered by old_r.
        #       # We need to ensure no position x < gap drops below 2.
        
        # This is expensive to check.
        # Approximation: Just try it and check if L increases.
        
        for idx in range(n_primes):
            if time.time() > deadline - 10:
                break
                
            p = primes[idx]
            target_r = gap % p
            old_r = residues[idx]
            
            if old_r == target_r:
                continue
                
            # Try change
            remove_congruence(idx, old_r, coverage)
            add_congruence(idx, target_r, coverage)
            
            new_L = find_first_gap(coverage) - 1
            if new_L > current_L:
                current_L = new_L
                residues[idx] = target_r
                improved = True
            else:
                # Revert
                remove_congruence(idx, target_r, coverage)
                add_congruence(idx, old_r, coverage)
                
            if improved:
                break # Restart loop with new gap
    
    return residues.tolist()
# EVOLVE-BLOCK-END
