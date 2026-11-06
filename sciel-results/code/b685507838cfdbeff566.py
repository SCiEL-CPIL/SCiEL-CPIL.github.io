"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 140
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]
    
    primes = sieve(1000)
    n_primes = len(primes)
    
    # We want to maximize L such that every integer 1..L is covered by at least 2 congruences.
    # Strategy: 
    # 1. Start with a reasonable initial assignment.
    # 2. Use hill-climbing / local search to fix gaps.
    # 3. Use simulated annealing or random restarts if time permits.
    
    # Initial assignment: try to distribute residues to cover small numbers.
    # A good heuristic: for prime p_i, assign r_i such that it covers a small number.
    # Let's try r_i = i % p, or something that ensures coverage of 1..N.
    
    # Let's try a greedy approach for initialization:
    # We want to cover 1, 2, 3, ...
    # For each integer k, we want at least 2 primes p_i such that k = r_i (mod p_i).
    # This is equivalent to k - r_i is divisible by p_i.
    
    # Let's start with r_i = 0 for all, which covers multiples of p_i.
    # But we need to cover 1..L.
    # Let's try a random initialization first, then hill-climb.
    
    def compute_L(residues):
        # Returns the largest L such that 1..L are all doubly covered.
        # We can't check up to 10^5 easily every time, so we check up to a limit.
        # But the score is the exact L capped at 10^5.
        # For speed, we can maintain a coverage array.
        pass
    
    # Let's maintain a coverage count array.
    # coverage[x] = number of congruences satisfied by x.
    # We need coverage[x] >= 2 for all x in 1..L.
    
    # Since L can be up to 10^5, we can maintain an array of size 100001.
    # Updating coverage when a residue changes:
    # If residue r_i changes from old to new for prime p_i:
    #   For x in 1..100000 such that x % p_i == old: coverage[x] -= 1
    #   For x in 1..100000 such that x % p_i == new: coverage[x] += 1
    # This is O(100000 / p_i) per update. For small p_i, this is expensive.
    # For p_i = 2, it's 50000 operations. For p_i = 1000, it's 100 operations.
    
    # Alternative: compute L from scratch? No, that's O(100000 * 168) which is too slow.
    # We need incremental updates.
    
    # Let's implement incremental coverage.
    
    MAX_L = 100000
    coverage = [0] * (MAX_L + 1)
    
    def update_coverage(i, old_r, new_r):
        p = primes[i]
        if old_r == new_r:
            return
        # Remove old coverage
        for x in range(old_r if old_r > 0 else p, MAX_L + 1, p):
            coverage[x] -= 1
        # Add new coverage
        for x in range(new_r if new_r > 0 else p, MAX_L + 1, p):
            coverage[x] += 1
    
    def get_L():
        L = 0
        for x in range(1, MAX_L + 1):
            if coverage[x] >= 2:
                L = x
            else:
                break
        return L
    
    # Initial assignment
    residues = [0] * n_primes
    for i, p in enumerate(primes):
        # Try to cover small numbers. Let's set r_i = i % p?
        # Or maybe r_i = 0?
        # Let's try a random start first.
        residues[i] = random.randint(0, p - 1)
        for x in range(residues[i] if residues[i] > 0 else p, MAX_L + 1, p):
            coverage[x] += 1
    
    best_L = get_L()
    best_residues = residues[:]
    
    # Hill climbing: find the first gap, try to fix it.
    # Actually, let's try to improve L step by step.
    
    # Simulated annealing or random local search.
    # Let's try: pick a random prime, change its residue, if L increases, accept.
    # If L decreases, accept with probability exp(-delta/T).
    
    # But computing L is O(MAX_L). That's 100000 per step. 140s / 100000 ~ 1400 steps? Too few.
    # We need a faster way to estimate L or only update when necessary.
    
    # Alternative: maintain the current L, and when a change is made, only check around the current L.
    # But if we break a coverage before L, we need to know that.
    
    # Let's try a different approach: 
    # 1. Find the first gap (smallest x with coverage[x] < 2).
    # 2. Try to fix it by changing residues of primes that could cover x.
    
    # Actually, let's just do random hill climbing with a fast L check.
    # We can cache the coverage array and only recompute L when a change is made.
    # But even then, L check is O(MAX_L).
    
    # Let's try to optimize: 
    # We can store the coverage array, and when we change a residue, we update the array.
    # Then we can find the first gap by scanning from 1.
    # But if the gap is far, we scan a lot.
    
    # Let's try a greedy approach: 
    # For each prime, assign a residue that covers a small number.
    # Specifically, let's try to cover 1, 2, 3, ... with at least 2 primes each.
    
    # Let's try a constructive approach:
    # For k = 1, 2, 3, ..., assign residues to cover k.
    # But we have 168 primes. We can assign each prime to cover a specific k.
    
    # Let's try: r_i = k_i % p_i, where k_i is a target number.
    # We want to choose k_i such that each small number is covered by at least 2 k_i's.
    
    # This is like a covering design.
    
    # Let's try a simple heuristic: 
    # For i in 0..167, set r_i = (i+1) % p_i.
    # This covers 1, 2, ..., 168 (mod p_i).
    
    # Let's test this initial assignment.
    
    def try_initial(residues):
        nonlocal coverage, best_L, best_residues
        # Reset coverage
        coverage = [0] * (MAX_L + 1)
        for i, p in enumerate(primes):
            r = residues[i]
            for x in range(r if r > 0 else p, MAX_L + 1, p):
                coverage[x] += 1
        L = get_L()
        if L > best_L:
            best_L = L
            best_residues = residues[:]
        return L
    
    # Try the simple heuristic
    residues_simple = [(i + 1) % p for i, p in enumerate(primes)]
    try_initial(residues_simple)
    
    # Now do local search
    # Pick a random prime, try all residues? No, too many.
    # Try random residues.
    
    # Let's do simulated annealing.
    T = 10.0
    cooling = 0.999
    
    # We need to be careful with the time.
    # Each step: pick i, pick new_r, update coverage, compute L.
    # Update coverage: O(MAX_L / p_i)
    # Compute L: O(MAX_L)
    # Total: O(MAX_L) ~ 100000 operations.
    # 140s / 100000 ~ 1400 steps. That's not enough.
    
    # We need a faster L computation.
    # Idea: maintain a set of gaps (x where coverage[x] < 2).
    # When coverage[x] changes, update the set.
    # Then L is the smallest gap - 1.
    
    # Let's implement this.
    
    # gaps = sorted set of x where coverage[x] < 2
    # We can use a heap or a sorted list.
    # But updating a sorted list is O(n).
    
    # Alternative: maintain an array of coverage, and a variable L.
    # When we update coverage, we can check if the new coverage breaks the prefix.
    
    # Let's try a different strategy: 
    # Use a bitset or numpy array for coverage, and vectorized operations.
    
    import numpy as np
    
    # Let's redo with numpy.
    
    def construct_numpy():
        deadline = time.time() + 140
        primes = sieve(1000)
        n_primes = len(primes)
        MAX_L = 100000
        
        # Coverage array
        coverage = np.zeros(MAX_L + 1, dtype=np.int32)
        
        def update_coverage_np(i, old_r, new_r):
            p = primes[i]
            if old_r == new_r:
                return
            # Remove old
            if old_r > 0:
                start = old_r
            else:
                start = p
            coverage[start::p] -= 1
            # Add new
            if new_r > 0:
                start = new_r
            else:
                start = p
            coverage[start::p] += 1
        
        def get_L_np():
            # Find first index where coverage < 2
            # coverage[1:] < 2
            mask = coverage[1:] < 2
            if not np.any(mask):
                return MAX_L
            first_gap = np.argmax(mask) + 1
            return first_gap - 1
        
        # Initial assignment
        residues = [(i + 1) % p for i, p in enumerate(primes)]
        for i, p in enumerate(primes):
            r = residues[i]
            if r > 0:
                start = r
            else:
                start = p
            coverage[start::p] += 1
        
        best_L = get_L_np()
        best_residues = residues[:]
        
        # Local search
        T = 10.0
        cooling = 0.9999
        step = 0
        
        while time.time() < deadline:
            step += 1
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            old_r = residues[i]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            
            # Update coverage
            update_coverage_np(i, old_r, new_r)
            new_L = get_L_np()
            
            if new_L > best_L:
                best_L = new_L
                best_residues = residues[:]
                # Reset temperature?
                T = 10.0
            else:
                # Check if we should accept
                delta = best_L - new_L
                if delta == 0:
                    # Same L, accept with some probability
                    if random.random() < 0.1:
                        pass
                    else:
                        # Revert
                        update_coverage_np(i, new_r, old_r)
                        residues[i] = old_r
                        continue
                else:
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        pass
                    else:
                        # Revert
                        update_coverage_np(i, new_r, old_r)
                        residues[i] = old_r
                        continue
            
            residues[i] = new_r
            T *= cooling
        
        return best_residues
    
    # Try numpy version
    try:
        result = construct_numpy()
        return result
    except:
        return best_residues
# EVOLVE-BLOCK-END
