import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s margin for 60s limit
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1  # 129 sums
    MAX_VAL = 128
    
    # Precompute FFT size
    conv_size = 129 + 129 - 1  # 257
    fft_n = 1
    while fft_n < conv_size:
        fft_n *= 2
        
    def calc_reps(A_set):
        """Compute unordered pair representation counts for sums 128..256 using FFT."""
        ind = np.zeros(129, dtype=np.float64)
        for a in A_set:
            ind[a] = 1.0
        
        f = np.fft.rfft(ind, n=fft_n)
        conv = np.fft.irfft(f * f, n=fft_n)[:conv_size].round().astype(np.int64)
        
        ordered = conv[N_LOW:N_HIGH+1]
        
        reps = np.zeros(RANGE_SIZE, dtype=np.int64)
        for idx in range(RANGE_SIZE):
            s = N_LOW + idx
            if s % 2 == 0:
                half = s // 2
                diag = 1.0 if (0 <= half <= 128 and ind[half] > 0) else 0.0
                reps[idx] = (ordered[idx] + diag) // 2
            else:
                reps[idx] = ordered[idx] // 2
        return reps

    def get_peak_info(reps):
        """Returns (peak_value, count_of_peaks, second_peak_value)"""
        if len(reps) == 0:
            return 0, 0, 0
        max_val = np.max(reps)
        count_max = np.sum(reps == max_val)
        # Find second highest
        unique_vals = np.unique(reps)
        if len(unique_vals) > 1:
            second_val = unique_vals[-2]
        else:
            second_val = 0
        return int(max_val), int(count_max), int(second_val)

    def is_covered(reps):
        return np.all(reps > 0)

    # Phase 1: Find an initial valid set
    # Strategy: Random search with greedy hole-filling
    best_A = None
    best_peak = float('inf')
    
    # Try to find a valid set quickly
    for trial in range(100):
        if time.time() > deadline - 40:
            break
            
        # Random initialization
        size = random.randint(30, 50)
        A_init = set(random.sample(range(0, 129), size))
        A_init.add(128) # Ensure 256 is covered
        A_init.add(0)   # Ensure 128 is covered (0+128)
        
        reps = calc_reps(A_init)
        
        if not is_covered(reps):
            # Greedy hole filling
            holes = np.where(reps == 0)[0]
            while len(holes) > 0 and len(A_init) < 100:
                # Pick a random hole
                h_idx = random.choice(holes)
                target_sum = N_LOW + h_idx
                
                # Find candidates that cover this sum
                # a + b = target_sum
                # a in [0, 128], b in [0, 128]
                # b = target_sum - a
                # We want to add an element 'a' such that target_sum - a is in A_init
                # Or just add any pair? No, we add one element.
                # If we add x, it covers sums x + y for y in A_init.
                # We need x + y = target_sum for some y in A_init.
                # So x = target_sum - y.
                
                candidates = []
                for y in A_init:
                    x = target_sum - y
                    if 0 <= x <= 128 and x not in A_init:
                        candidates.append(x)
                
                if not candidates:
                    # If no candidate covers this hole with existing elements, 
                    # we might need to add an element that pairs with 0 or 128?
                    # Or just break and restart
                    break
                
                # Choose candidate that minimizes peak increase
                best_cand = None
                best_cand_peak = float('inf')
                for x in random.sample(candidates, min(10, len(candidates))):
                    new_A = A_init | {x}
                    new_reps = calc_reps(new_A)
                    if not is_covered(new_reps):
                        # Check if it covers the specific hole
                        if new_reps[h_idx] > 0:
                            p = get_peak_info(new_reps)[0]
                            if p < best_cand_peak:
                                best_cand_peak = p
                                best_cand = x
                
                if best_cand is not None:
                    A_init.add(best_cand)
                    reps = calc_reps(A_init)
                    holes = np.where(reps == 0)[0]
                else:
                    break
            
            if is_covered(reps):
                peak = get_peak_info(reps)[0]
                if peak < best_peak:
                    best_peak = peak
                    best_A = list(A_init)
        
        else:
            peak = get_peak_info(reps)[0]
            if peak < best_peak:
                best_peak = peak
                best_A = list(A_init)
        
        if best_peak <= 3: # Very good start
            break

    if best_A is None:
        # Fallback: Dense set
        return list(range(64, 129))

    current_A = set(best_A)
    current_reps = calc_reps(current_A)
    current_peak, current_count_peak, current_second_peak = get_peak_info(current_reps)
    
    # Phase 2: Targeted Local Search (Peak Reduction)
    # We target the sum with the highest multiplicity and try to swap elements to reduce it.
    
    stuck_count = 0
    no_improve_count = 0
    
    while time.time() < deadline:
        peak, count_peak, second_peak = get_peak_info(current_reps)
        
        # Find the sums with the peak multiplicity
        peak_sums_indices = np.where(current_reps == peak)[0]
        
        if len(peak_sums_indices) == 0:
            break
            
        # Pick a random peak sum to attack
        target_idx = random.choice(peak_sums_indices)
        target_sum = N_LOW + target_idx
        
        # Find pairs (a, b) in current_A such that a + b = target_sum
        pairs = []
        A_list = list(current_A)
        for i, a in enumerate(A_list):
            b = target_sum - a
            if b in current_A:
                # Avoid duplicate pairs (a,b) and (b,a)
                if a < b or (a == b):
                    pairs.append((a, b))
        
        improved = False
        
        # Try to swap one element from a pair
        for a, b in pairs:
            # Try to replace a with a'
            # Constraints:
            # 1. a' not in A
            # 2. 0 <= a' <= 128
            # 3. Coverage must be maintained.
            #    Sums that were covered ONLY by a (count 1) must be covered by a' or others.
            #    Specifically, for any c in A, if current_reps[a+c] == 1, 
            #    then a' + c must be a valid sum (i.e., a' + c is in range and 
            #    we don't break it). Actually, if reps[a+c]==1, removing a makes it 0.
            #    Adding a' makes reps[a'+c] increase by 1.
            #    So we need a' + c = a + c => a' = a. Contradiction.
            #    Wait. If reps[s] == 1 and s = a + c, then s is covered ONLY by pair (a,c).
            #    If we remove a, s becomes 0.
            #    To keep s covered, we need a' + c = s OR some other pair (x,y) in A\{a} sums to s.
            #    But if reps[s]==1, no other pair exists.
            #    So we MUST have a' + c = s => a' = s - c = a + c - c = a.
            #    This implies we cannot replace a if it is part of a unique representation of any sum.
            #    
            #    Let's refine: 
            #    Let CriticalSums(a) = { s | s = a + c, c in A, reps[s] == 1 }
            #    If we remove a, all s in CriticalSums(a) become 0.
            #    If we add a', they become reps_new[s] = 1 if a' + c = s for some c in A.
            #    So for all c such that a+c in CriticalSums(a), we need a' + c = a + c => a' = a.
            #    This means if a has ANY unique representation, we cannot swap it out 
            #    unless we also swap c? No, we are swapping a for a'.
            #    
            #    Therefore, we can ONLY swap out elements that are NOT part of any unique representation.
            #    i.e., for all c in A, reps[a+c] > 1.
            
            is_safe_to_remove = True
            for c in current_A:
                s = a + c
                if N_LOW <= s <= N_HIGH:
                    if current_reps[s - N_LOW] == 1:
                        is_safe_to_remove = False
                        break
                # If s is outside range, it doesn't matter for coverage, 
                # but removing a might affect future additions? No, we only care about [128, 256].
            
            if not is_safe_to_remove:
                continue
                
            # Now find candidates a'
            # a' should be in [0, 128], not in A.
            # To reduce the peak of target_sum, we want a' + b != target_sum 
            # (which is automatic if a' != a) OR we just remove one count from target_sum.
            # Since a' != a, a' + b != a + b = target_sum.
            # So the count of target_sum will decrease by 1.
            # The count of other sums will change.
            # We want to minimize the new global peak.
            
            # Generate a few random candidates
            # Ideally, a' should "fill" holes or go to low-density areas.
            # But we must ensure we don't create a NEW peak higher than current peak - 1.
            
            candidates = []
            # Sample random values
            for _ in range(20):
                a_prime = random.randint(0, 128)
                if a_prime in current_A:
                    continue
                candidates.append(a_prime)
            
            best_a_prime = None
            best_new_peak = peak # We want strictly lower or equal
            best_new_count = count_peak
            best_new_second = second_peak
            
            for a_prime in candidates:
                new_A = current_A - {a} | {a_prime}
                new_reps = calc_reps(new_A)
                
                if not is_covered(new_reps):
                    continue
                    
                new_peak, new_count, new_second = get_peak_info(new_reps)
                
                # Acceptance criteria:
                # 1. Lower peak is best.
                # 2. Same peak, but fewer sums at that peak (density reduction).
                # 3. Same peak, same count, lower second peak.
                
                if new_peak < best_new_peak:
                    best_new_peak = new_peak
                    best_new_count = new_count
                    best_new_second = new_second
                    best_a_prime = a_prime
                elif new_peak == best_new_peak:
                    if new_count < best_new_count:
                        best_new_count = new_count
                        best_new_second = new_second
                        best_a_prime = a_prime
                    elif new_count == best_new_count and new_second < best_new_second:
                        best_new_second = new_second
                        best_a_prime = a_prime
            
            if best_a_prime is not None:
                # Apply swap
                current_A.remove(a)
                current_A.add(best_a_prime)
                current_reps = calc_reps(current_A)
                improved = True
                stuck_count = 0
                no_improve_count = 0
                break # Restart loop to re-evaluate peaks
        
        if not improved:
            stuck_count += 1
            
            # If stuck for a while, do a random perturbation
            if stuck_count > 50:
                # Remove a random non-critical element and add a random one
                # Even if it increases peak slightly
                A_list = list(current_A)
                if len(A_list) > 0:
                    a_rem = random.choice(A_list)
                    # Check if safe to remove temporarily (allow peak increase)
                    current_A.remove(a_rem)
                    a_add = random.randint(0, 128)
                    while a_add in current_A:
                        a_add = random.randint(0, 128)
                    current_A.add(a_add)
                    current_reps = calc_reps(current_A)
                    
                    # If this broke coverage, restore
                    if not is_covered(current_reps):
                        current_A.remove(a_add)
                        current_A.add(a_rem)
                        current_reps = calc_reps(current_A)
                        # Restart stuck counter
                        stuck_count = 0
                        continue
                        
                stuck_count = 0
                no_improve_count += 1
                
                if no_improve_count > 100:
                    # Full restart with new random seed or different initialization
                    # For now, just break to avoid wasting time
                    break
                    
    # Final check
    if not is_covered(current_reps):
        # Fallback to best known valid if current is invalid
        if best_A is not None:
            return best_A
        return list(range(64, 129))
        
    return list(current_A)
