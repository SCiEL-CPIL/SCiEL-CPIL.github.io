import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 128
    min_sum = 128
    max_sum = 256
    
    # Preallocate indicator array size
    # We will reuse this to avoid allocation overhead in the inner loop if possible, 
    # but numpy convolve is fast enough.
    
    def get_counts(A):
        """
        Returns an array of counts for sums 0..256.
        A is a list or np.array of integers in [0, 128].
        """
        # Ensure A is unique and sorted for consistency, though indicator handles it
        # Using np.unique to be safe if A has duplicates (shouldn't, but just in case)
        A = np.unique(np.array(A, dtype=np.int64))
        indicator = np.zeros(n + 1, dtype=np.int64)
        indicator[A] = 1
        # Convolution of indicator with itself
        # Length is 2*n + 1 = 257
        conv = np.convolve(indicator, indicator, mode='full')
        
        # Convert ordered counts to unordered counts
        # conv[s] = sum_{a+b=s} 1_{a in A} 1_{b in A}
        # This is the number of ordered pairs (a,b).
        # Unordered pairs {a,b} with a<=b:
        # If a != b, (a,b) and (b,a) are in conv, so count is 2. Unordered count is 1.
        # If a == b, (a,a) is in conv, count is 1. Unordered count is 1.
        # Formula: Unordered(s) = (conv[s] + diag[s]) / 2
        # where diag[s] = 1 if s is even and s/2 in A, else 0.
        
        # Vectorized calculation for s in 0..256
        s_range = np.arange(257)
        # diag is 1 where s is even and s//2 is in A
        # s//2 is in A means indicator[s//2] == 1
        # We need to handle the parity.
        # Let's compute diag vector
        half_s = s_range // 2
        # For odd s, diag is 0. For even s, diag = indicator[half_s]
        # But we must ensure half_s is within bounds 0..128. 
        # s max 256 -> half 128. OK.
        diag = np.where(s_range % 2 == 0, indicator[half_s], 0)
        
        unordered_counts = (conv + diag) // 2
        return unordered_counts

    def calculate_score(A):
        """
        Returns (score, max_mult, num_uncovered).
        Lower score is better.
        """
        counts = get_counts(A)
        target_counts = counts[min_sum:max_sum+1]
        
        num_uncovered = np.sum(target_counts == 0)
        if num_uncovered > 0:
            # Huge penalty for not covering all targets
            # We also want to minimize max mult among covered ones? 
            # Actually, if not covered, it's invalid. 
            # Score = 1e9 + max_mult
            max_mult = np.max(target_counts)
            return 10**9 + int(max_mult), int(max_mult), int(num_uncovered)
        
        max_mult = np.max(target_counts)
        return int(max_mult), int(max_mult), 0

    best_score = float('inf')
    best_max_mult = float('inf')
    best_set = None

    def try_set(A):
        nonlocal best_score, best_max_mult, best_set
        # A should be a list of unique ints
        if len(A) == 0:
            return
        score, max_mult, _ = calculate_score(A)
        if score < best_score:
            best_score = score
            best_max_mult = max_mult
            best_set = sorted(A)
            # Early exit if we found the theoretical minimum?
            # Theoretical min is likely 2. If we find 2, we can keep searching for robustness or stop if time is tight?
            # Better to keep searching to ensure we don't have a bug, but 2 is very likely optimal.
            # If we find 2, we can switch to a mode that tries to find 1 (impossible) or just refine.
            # Actually, if we find 2, we should stop expensive search and just return, 
            # unless we suspect 1 is possible (it's not).
            # However, to be safe against "invalid" edge cases, we return 2.

    # Initial solutions
    # 1. Random sets of various sizes
    random.seed(42)
    
    # Try sizes from 16 to 25
    for size in range(16, 26):
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            A = random.sample(range(0, n+1), size)
            try_set(A)
        
        if best_max_mult == 2:
            # Found a solution with peak 2. 
            # We can try to verify or just break if we want to save time, 
            # but let's do a few more iterations to see if we can find 1 (unlikely) or just stabilize.
            # Actually, 2 is the lower bound for N=16. 
            # If we found 2, we are likely optimal.
            # Let's do a quick local search to ensure it's valid and maybe find a smaller set? 
            # No, task asks to minimize peak, not set size.
            # So if we have peak 2, we are done? 
            # Wait, is peak 1 possible? 
            # Sum of 129 targets. 
            # If peak is 1, total pairs >= 129. 
            # N=16 -> 120 pairs. Not enough. 
            # N=17 -> 153 pairs. 
            # Can 17 elements cover 129 sums with multiplicity 1? 
            # This requires a Sidon set that covers an interval. 
            # It is known that such "perfect" additive bases are rare or non-existent for these parameters. 
            # Most likely optimal is 2.
            
            # Let's run a local search on the best set found so far to ensure it's robust.
            if best_set is not None:
                current = best_set[:]
                in_set = set(current)
                current_score, current_max, _ = calculate_score(current)
                
                # Simple local search to confirm
                improved = True
                while improved and time.time() < deadline - 5:
                    improved = False
                    # Try swaps
                    for i in range(len(current)):
                        if time.time() > deadline - 5: break
                        old_val = current[i]
                        for new_val in range(0, n+1):
                            if new_val in in_set or new_val == old_val:
                                continue
                            trial = current[:]
                            trial[i] = new_val
                            s, m, _ = calculate_score(trial)
                            if s < current_score:
                                current = trial
                                current_score = s
                                current_max = m
                                in_set.discard(old_val)
                                in_set.add(new_val)
                                improved = True
                                break
                        if improved: break
                    
                    if not improved:
                        # Try adds/removes if size is not fixed? 
                        # But we want to minimize peak. 
                        # If peak is 2, removing might increase it or uncover. 
                        # Adding might lower it? Unlikely if we are already at 2.
                        # So we can stop.
                        break
                
                if current_score < best_score:
                    best_score = current_score
                    best_max_mult = current_max
                    best_set = sorted(current)
            
            # If we found 2, we can break the main loop if we are confident.
            # But let's continue to see if we can find 1 (just in case my theory is wrong).
            # Or just break to save time for other random seeds?
            # Given 60s limit, and we are at ~10s, we have time.
            # But 1 is highly unlikely. 
            # Let's just break out of the size loop if we found 2, and do a final polish.
            break

    if best_max_mult != 2:
        # If we didn't find 2, we need to search harder.
        # Run a more aggressive local search from the best set found.
        if best_set is not None:
            current = best_set[:]
            in_set = set(current)
            current_score, current_max, _ = calculate_score(current)
            
            # Simulated Annealing / Random Restart Local Search
            # We will do this for the remaining time.
            while time.time() < deadline - 5:
                improved = False
                
                # 1. Try to reduce peak by swapping
                # Pick a random element to swap
                if len(current) > 0:
                    i = random.randint(0, len(current) - 1)
                    old_val = current[i]
                    # Try a few random new values
                    for _ in range(100):
                        new_val = random.randint(0, n)
                        if new_val in in_set or new_val == old_val:
                            continue
                        trial = current[:]
                        trial[i] = new_val
                        s, m, _ = calculate_score(trial)
                        if s < current_score:
                            current = trial
                            current_score = s
                            current_max = m
                            in_set.discard(old_val)
                            in_set.add(new_val)
                            improved = True
                            break
                        # Accept worse move with small probability? 
                        # Not necessary if we have many restarts.
                
                # 2. Try to add/remove if it helps
                # Remove
                if len(current) > 1:
                    i = random.randint(0, len(current) - 1)
                    val = current[i]
                    trial = current[:i] + current[i+1:]
                    s, m, _ = calculate_score(trial)
                    if s < current_score:
                        current = trial
                        current_score = s
                        current_max = m
                        in_set.discard(val)
                        improved = True
                
                # Add
                if not improved:
                    new_val = random.randint(0, n)
                    if new_val not in in_set:
                        trial = current + [new_val]
                        s, m, _ = calculate_score(trial)
                        if s < current_score:
                            current = trial
                            current_score = s
                            current_max = m
                            in_set.add(new_val)
                            improved = True
                
                if not improved:
                    # Restart from a perturbed version of best
                    # Or just break and try a new random set?
                    # Let's just break this inner loop and try a new random seed set
                    break

            if current_score < best_score:
                best_score = current_score
                best_max_mult = current_max
                best_set = sorted(current)
        
        # Try more random sets if time remains
        while time.time() < deadline - 2:
            size = random.randint(16, 22)
            A = random.sample(range(0, n+1), size)
            try_set(A)
            
            # Quick local search on this new set
            if best_set is None:
                continue
            # Use the best set found so far as a reference for local search?
            # No, just refine the current A.
            current = A[:]
            in_set = set(current)
            current_score, current_max, _ = calculate_score(current)
            
            for _ in range(500): # Fixed steps
                if time.time() > deadline - 2: break
                if current_score == 2: break
                
                i = random.randint(0, len(current) - 1)
                old_val = current[i]
                new_val = random.randint(0, n)
                if new_val in in_set or new_val == old_val:
                    continue
                trial = current[:]
                trial[i] = new_val
                s, m, _ = calculate_score(trial)
                if s < current_score:
                    current = trial
                    current_score = s
                    current_max = m
                    in_set.discard(old_val)
                    in_set.add(new_val)
            
            if current_score < best_score:
                best_score = current_score
                best_max_mult = current_max
                best_set = sorted(current)

    if best_set is None:
        # Fallback: return a valid set, even if peak is high
        # [64, 128] works
        return list(range(64, 129))
    
    return best_set

# EVOLVE-BLOCK-END
