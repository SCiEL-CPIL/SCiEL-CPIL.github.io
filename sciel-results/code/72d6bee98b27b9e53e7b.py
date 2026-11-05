import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i:limit+1:i] = [False] * len(sieve[i*i:limit+1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]
        
    primes = get_primes(1000)
    n_primes = len(primes)
    MAX_L = 100000
    
    # 1. Greedy Initialization
    residues = [0] * n_primes
    counts = np.zeros(MAX_L + 1, dtype=np.int32)
    
    # Helper to find L
    def get_L(counts):
        # Find first index j >= 1 where counts[j] < 2
        # counts[0] is unused
        bad = np.where(counts[1:] < 2)[0]
        if len(bad) == 0:
            return MAX_L
        return bad[0] # This is index in 0-based slice of 1..L, so actual index is bad[0] + 1?
        # Wait. counts[1] is index 1.
        # If counts[1] < 2, bad[0]=0. L should be 0.
        # If counts[2] < 2, bad[0]=1. L should be 1.
        # So L = bad[0].
        
    current_L = 0
    
    for i, p in enumerate(primes):
        # Find current first hole
        # We know current_L is the last valid index.
        # So the first hole is at current_L + 1.
        # Unless current_L == MAX_L, then no hole.
        if current_L >= MAX_L:
            residues[i] = 0
            continue
            
        x_crit = current_L + 1
        # We must choose r such that x_crit % p == r
        r = x_crit % p
        residues[i] = r
        
        # Update counts
        start = r if r > 0 else p
        if start <= MAX_L:
            counts[start::p] += 1
            
        # Recalculate L
        # Optimization: We only added to indices >= start.
        # If start > current_L + 1, L didn't change?
        # No, if start <= current_L + 1, it might fill the hole.
        # Actually, since we picked r to cover x_crit, counts[x_crit] increased.
        # It might become >= 2.
        # We need to check if the hole is filled and if subsequent holes are filled.
        # Just recalculate L from scratch? 
        # Or scan forward from current_L?
        # Since we only increased counts, L can only increase or stay same.
        # We can scan from current_L + 1 upwards until we hit a hole.
        
        # But wait, did we break a previous valid spot? No, counts only increase.
        # So L is monotonic increasing in greedy phase.
        
        # Efficient L update:
        new_L = current_L
        # Check if x_crit is now covered
        if counts[x_crit] >= 2:
            new_L = x_crit
            # Check further
            while new_L < MAX_L and counts[new_L + 1] >= 2:
                new_L += 1
        current_L = new_L
        
    best_L = current_L
    best_residues = residues.copy()
    
    # 2. Simulated Annealing
    T = 1.0
    cooling = 0.9999
    
    # Precompute start indices for quick revert?
    # We will just update counts directly.
    
    while time.time() < deadline:
        i = np.random.randint(n_primes)
        p = primes[i]
        old_r = residues[i]
        new_r = np.random.randint(p)
        
        if new_r == old_r:
            continue
            
        # Apply change
        start_old = old_r if old_r > 0 else p
        if start_old <= MAX_L:
            counts[start_old::p] -= 1
            
        start_new = new_r if new_r > 0 else p
        if start_new <= MAX_L:
            counts[start_new::p] += 1
            
        # Calculate new L
        # Since counts changed, L can go up or down.
        # We need to find the first hole.
        # Optimization: 
        # If the change was "bad" (removed coverage from a critical spot), L might drop.
        # If "good", L might rise.
        # Simplest: Find first hole from 1.
        # This is O(MAX_L) per step. 
        # 100,000 steps * 100,000 size = 10^10 ops. Too slow.
        
        # Better L calculation:
        # Maintain a set of "bad" indices? 
        # Or use np.where. 
        # np.where(counts[1:] < 2)[0] is O(MAX_L).
        # With 115 seconds, and NumPy, maybe we can do ~10,000 to 50,000 steps?
        # 50,000 steps * 100,000 (vectorized) ~ 5 * 10^9 ops. 
        # NumPy is fast, but 5 * 10^9 is borderline for 115s.
        
        # Let's optimize the L check.
        # We only need to check if the *current* L is still valid, and if it can be extended.
        # If we remove coverage, L might drop.
        # If we add coverage, L might rise.
        
        # Heuristic:
        # 1. Check if current_L is still valid (counts[1..current_L] all >= 2).
        #    We can check min(counts[1:current_L+1]) >= 2.
        # 2. If valid, try to extend from current_L + 1.
        # 3. If invalid, find the first hole in [1, current_L].
        
        # This is still O(current_L) in worst case.
        # But typically L is small (hundreds/thousands).
        # So it's fast.
        
        # Let's implement this logic.
        
        # Check prefix validity
        if current_L > 0:
            # Check if any count in 1..current_L is < 2
            # This is a min operation.
            if np.min(counts[1:current_L+1]) < 2:
                # Find first bad
                bad = np.where(counts[1:current_L+1] < 2)[0]
                new_L = bad[0] # L drops to bad[0]
            else:
                # Current prefix is valid. Try to extend.
                new_L = current_L
                if new_L < MAX_L:
                    # Check from new_L + 1
                    # We can check a window or just scan.
                    # Since L is likely small, scanning is fine.
                    # Or use np.where on the rest?
                    # rest = counts[new_L+1:]
                    # bad_rest = np.where(rest < 2)[0]
                    # if len(bad_rest) == 0: new_L = MAX_L
                    # else: new_L = new_L + bad_rest[0]
                    
                    # This is O(MAX_L - current_L).
                    # If current_L is small, this is O(MAX_L).
                    # If current_L is large, this is small.
                    
                    bad_rest = np.where(counts[current_L+1:] < 2)[0]
                    if len(bad_rest) == 0:
                        new_L = MAX_L
                    else:
                        new_L = current_L + bad_rest[0]
        else:
            # L is 0
            bad = np.where(counts[1:] < 2)[0]
            if len(bad) == 0:
                new_L = MAX_L
            else:
                new_L = bad[0]
                
        # Acceptance
        if new_L >= current_L:
            residues[i] = new_r
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
        else:
            delta = new_L - current_L
            prob = np.exp(delta / T)
            if np.random.random() < prob:
                residues[i] = new_r
                current_L = new_L
            else:
                # Revert
                residues[i] = old_r
                if start_new <= MAX_L:
                    counts[start_new::p] -= 1
                if start_old <= MAX_L:
                    counts[start_old::p] += 1
                
        T *= cooling
        if T < 0.01:
            T = 0.01
            
    return best_residues
