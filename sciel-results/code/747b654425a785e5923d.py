import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                sieve[i * i:limit + 1:i] = [False] * len(sieve[i * i:limit + 1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    n_primes = len(primes)
    MAX_L = 100000
    
    # Use int8 for counts to save memory and improve cache locality
    # Max possible count is 168, which fits in int8? No, 168 > 127.
    # So use int16.
    counts = np.zeros(MAX_L + 1, dtype=np.int16)
    residues = [0] * n_primes

    def apply_residue(idx, new_r):
        p = primes[idx]
        old_r = residues[idx]
        if old_r == new_r:
            return
        # Remove old
        start_old = old_r if old_r > 0 else p
        if start_old <= MAX_L:
            counts[start_old::p] -= 1
        # Add new
        start_new = new_r if new_r > 0 else p
        if start_new <= MAX_L:
            counts[start_new::p] += 1
        residues[idx] = new_r

    def find_L():
        # Find first index >= 1 where count < 2
        # We check 1..MAX_L
        # np.argmax returns 0 if all are >= 2. 
        # We need to be careful: if counts[1:] all >= 2, argmax of (counts[1:] < 2) is 0.
        # But we want to know if there is ANY < 2.
        bad_mask = counts[1:MAX_L+1] < 2
        if not np.any(bad_mask):
            return MAX_L
        first_bad = np.argmax(bad_mask)
        return first_bad # This is 0-based index in the slice, so actual index is first_bad + 1? 
        # Wait. 
        # If counts[1] < 2, bad_mask[0] is True. argmax is 0. L should be 0? 
        # Definition: L is largest such that 1..L are covered.
        # If 1 is not covered, L=0.
        # If 1 is covered, 2 is not, L=1.
        # bad_mask[0] corresponds to x=1.
        # If bad_mask[0] is True, x=1 fails. L=0.
        # If bad_mask[0] is False, bad_mask[1] is True (x=2 fails). L=1.
        # So L = first_bad_index_in_slice.
        return first_bad

    # --- Phase 1: Greedy Repair Initialization ---
    # Start with all 0s
    for i in range(n_primes):
        p = primes[i]
        residues[i] = 0
        start = p # 0 mod p is p in 1..MAX_L context? 
        # r=0 means x=p, 2p, ...
        # start = p
        if start <= MAX_L:
            counts[start::p] += 1

    current_L = find_L()
    best_L = current_L
    best_residues = residues.copy()
    
    # Greedy Repair: Iteratively fix the first bad spot
    # This might be slow if done naively, but let's try a few passes
    # We limit this phase to 10 seconds
    repair_deadline = start_time + 10.0
    while time.time() < repair_deadline:
        current_L = find_L()
        if current_L == MAX_L:
            break
        
        x_bad = current_L + 1 # The first integer that fails
        # We need to increase count at x_bad
        # Find primes that do NOT cover x_bad
        # x_bad % p != residues[i]
        candidates = []
        for i in range(n_primes):
            p = primes[i]
            if x_bad % p != residues[i]:
                candidates.append(i)
        
        if not candidates:
            break # All primes cover x_bad, but count < 2? Impossible if n_primes >= 2.
            # Wait, if all primes cover x_bad, count is n_primes.
            # If count < 2, it means 0 or 1 primes cover it.
            # So candidates will not be empty.
            
        # Pick a random candidate, or the one with smallest p? 
        # Smaller p has larger impact. Let's pick randomly to diversify.
        idx = candidates[np.random.randint(len(candidates))]
        p = primes[idx]
        new_r = x_bad % p
        apply_residue(idx, new_r)
        
        new_L = find_L()
        if new_L > current_L:
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
        else:
            # Revert if it didn't help? 
            # In greedy repair, we usually only accept if it improves L.
            # But sometimes a move that doesn't improve L immediately might help later?
            # For initialization, let's stick to strict improvement to avoid wandering.
            apply_residue(idx, residues[idx]) # Wait, apply_residue updates residues.
            # We need to revert.
            # Let's store old_r before applying.
            pass # The logic above is slightly flawed with in-place update without revert logic in the loop.
            # Let's restart the repair logic properly.
            
    # Proper Greedy Repair
    counts = np.zeros(MAX_L + 1, dtype=np.int16)
    residues = [0] * n_primes
    for i in range(n_primes):
        p = primes[i]
        start = p
        if start <= MAX_L:
            counts[start::p] += 1
            
    current_L = find_L()
    best_L = current_L
    best_residues = residues.copy()
    
    repair_deadline = start_time + 15.0
    while time.time() < repair_deadline:
        current_L = find_L()
        if current_L >= MAX_L:
            break
        
        x_bad = current_L + 1
        candidates = []
        for i in range(n_primes):
            p = primes[i]
            if x_bad % p != residues[i]:
                candidates.append(i)
                
        if not candidates:
            break
            
        # Try a few random candidates to see which one improves L the most
        best_candidate_idx = -1
        best_candidate_L = current_L
        
        # Sample up to 10 candidates
        sample_size = min(10, len(candidates))
        sample_indices = np.random.choice(candidates, sample_size, replace=False)
        
        for idx in sample_indices:
            p = primes[idx]
            old_r = residues[idx]
            new_r = x_bad % p
            
            # Apply
            start_old = old_r if old_r > 0 else p
            start_new = new_r if new_r > 0 else p
            
            if start_old <= MAX_L:
                counts[start_old::p] -= 1
            if start_new <= MAX_L:
                counts[start_new::p] += 1
            residues[idx] = new_r
            
            new_L = find_L()
            
            if new_L > best_candidate_L:
                best_candidate_L = new_L
                best_candidate_idx = idx
                best_old_r = old_r
                best_new_r = new_r
                # We keep this change applied for now
            else:
                # Revert
                if start_new <= MAX_L:
                    counts[start_new::p] -= 1
                if start_old <= MAX_L:
                    counts[start_old::p] += 1
                residues[idx] = old_r
        
        if best_candidate_idx != -1:
            current_L = best_candidate_L
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
        else:
            # No improvement found in sample, break to avoid infinite loop
            break

    # --- Phase 2: Simulated Annealing ---
    # Start from best_residues
    residues = best_residues.copy()
    counts = np.zeros(MAX_L + 1, dtype=np.int16)
    for i in range(n_primes):
        p = primes[i]
        r = residues[i]
        start = r if r > 0 else p
        if start <= MAX_L:
            counts[start::p] += 1
            
    current_L = find_L()
    best_L = current_L
    best_residues = residues.copy()
    
    T = 50.0
    cooling_rate = 0.999
    min_T = 1.0
    
    while time.time() < deadline:
        # Generate Move
        if np.random.random() < 0.7:
            # Fix Move: Target x_bad
            current_L = find_L()
            if current_L >= MAX_L:
                break
            x_bad = current_L + 1
            
            # Find primes not covering x_bad
            candidates = []
            for i in range(n_primes):
                p = primes[i]
                if x_bad % p != residues[i]:
                    candidates.append(i)
            
            if candidates:
                idx = candidates[np.random.randint(len(candidates))]
                p = primes[idx]
                new_r = x_bad % p
                old_r = residues[idx]
                
                # Apply
                start_old = old_r if old_r > 0 else p
                start_new = new_r if new_r > 0 else p
                if start_old <= MAX_L:
                    counts[start_old::p] -= 1
                if start_new <= MAX_L:
                    counts[start_new::p] += 1
                residues[idx] = new_r
                
                new_L = find_L()
                delta = new_L - current_L
                
                if delta > 0 or np.random.random() < np.exp(delta / T):
                    current_L = new_L
                    if new_L > best_L:
                        best_L = new_L
                        best_residues = residues.copy()
                else:
                    # Revert
                    if start_new <= MAX_L:
                        counts[start_new::p] -= 1
                    if start_old <= MAX_L:
                        counts[start_old::p] += 1
                    residues[idx] = old_r
        else:
            # Random Move
            idx = np.random.randint(n_primes)
            p = primes[idx]
            old_r = residues[idx]
            new_r = np.random.randint(p)
            
            if old_r == new_r:
                continue
                
            start_old = old_r if old_r > 0 else p
            start_new = new_r if new_r > 0 else p
            
            if start_old <= MAX_L:
                counts[start_old::p] -= 1
            if start_new <= MAX_L:
                counts[start_new::p] += 1
            residues[idx] = new_r
            
            new_L = find_L()
            delta = new_L - current_L
            
            if delta > 0 or np.random.random() < np.exp(delta / T):
                current_L = new_L
                if new_L > best_L:
                    best_L = new_L
                    best_residues = residues.copy()
            else:
                # Revert
                if start_new <= MAX_L:
                    counts[start_new::p] -= 1
                if start_old <= MAX_L:
                    counts[start_old::p] += 1
                residues[idx] = old_r
        
        T *= cooling_rate
        if T < min_T:
            T = min_T

    return best_residues
