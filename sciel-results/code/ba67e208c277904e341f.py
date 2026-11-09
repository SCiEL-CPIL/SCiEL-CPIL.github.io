import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 58  # 2s margin for safety
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1
    MAX_VAL = 128
    
    # Precompute masks and indices for vectorized evaluation
    sums = np.arange(N_LOW, N_HIGH + 1)
    even_mask = (sums % 2 == 0)
    mid_indices = sums // 2
    # Mask for valid mid indices (0..128)
    valid_mid_mask = (mid_indices >= 0) & (mid_indices <= MAX_VAL)
    
    def evaluate(A):
        """
        Evaluates the set A.
        Returns (peak, sum_sq, is_covered).
        """
        if not A:
            return float('inf'), float('inf'), False
            
        A_arr = np.array(list(A), dtype=np.int32)
        indicator = np.zeros(MAX_VAL + 1, dtype=np.int32)
        indicator[A_arr] = 1
        
        # Convolution gives count of ordered pairs (a, b) with a + b = s
        conv = np.convolve(indicator, indicator, mode='full')
        
        # Extract the relevant slice [128, 256]
        conv_slice = conv[N_LOW:N_HIGH+1]
        
        # Calculate unordered pair counts
        # For even s: (conv[s] + indicator[s//2]) // 2
        # For odd s: conv[s] // 2
        
        # We need indicator at mid_indices. 
        # mid_indices can be out of bounds for the indicator array if not careful, 
        # but sums are 128..256, so mid is 64..128. All are valid.
        indicator_mid = indicator[mid_indices]
        
        # Vectorized calculation
        counts = np.zeros(RANGE_SIZE, dtype=np.int32)
        # Odd sums
        counts[~even_mask] = conv_slice[~even_mask] // 2
        # Even sums
        counts[even_mask] = (conv_slice[even_mask] + indicator_mid[even_mask]) // 2
        
        if not np.all(counts > 0):
            return float('inf'), float('inf'), False
            
        peak = int(np.max(counts))
        sum_sq = int(np.sum(counts * counts))
        
        return peak, sum_sq, True

    best_peak = float('inf')
    best_sq = float('inf')
    best_A = None

    # Try several different initial sizes
    # Based on experience, optimal size is often around 40-60
    sizes_to_try = [30, 40, 50, 60, 70]
    random.shuffle(sizes_to_try)

    for target_size in sizes_to_try:
        if time.time() > deadline - 15:
            break
            
        # Initialize
        A_init = set()
        # Anchor points to help coverage
        A_init.add(0)
        A_init.add(128)
        
        # Add random elements to reach target size
        candidates = list(range(1, 128))
        random.shuffle(candidates)
        
        for c in candidates:
            if len(A_init) >= target_size:
                break
            A_init.add(c)
            
        current_A = A_init
        current_peak, current_sq, is_covered = evaluate(current_A)
        
        if is_covered:
            if current_peak < best_peak or (current_peak == best_peak and current_sq < best_sq):
                best_peak = current_peak
                best_sq = current_sq
                best_A = list(current_A)

        # If not covered, we still run SA to try to fix it, 
        # but we need a strategy to accept uncovered states if they make progress.
        # For simplicity and speed, we will only run SA if we start covered or 
        # quickly become covered. If not covered, we do a quick greedy repair.
        
        if not is_covered:
            # Quick greedy repair
            repair_limit = 50
            for _ in range(repair_limit):
                if time.time() > deadline - 15:
                    break
                reps, _, cov = evaluate(current_A)
                if cov:
                    break
                # Find missing sums
                # Re-evaluating to get counts is expensive, let's just pick random add
                # Better: check which sums are missing
                # To save time, we just add random elements hoping to cover
                # Or, we can do a targeted add
                # Let's do a targeted add: pick a missing sum, add a partner
                # We need the counts to know what's missing.
                # Since evaluate returns inf for peak if not covered, we need a variant
                # or just accept that we might not fix it quickly and move to next size.
                # Given the small domain, random add usually fixes it fast.
                to_add = random.randint(0, 128)
                if to_add not in current_A:
                    current_A.add(to_add)
            
            current_peak, current_sq, is_covered = evaluate(current_A)
            if is_covered:
                if current_peak < best_peak or (current_peak == best_peak and current_sq < best_sq):
                    best_peak = current_peak
                    best_sq = current_sq
                    best_A = list(current_A)
            else:
                continue # Skip SA if we can't even cover

        # Simulated Annealing
        T = 5.0
        cooling_rate = 0.995
        min_T = 0.05
        
        # Track best in this run
        local_best_peak = current_peak
        local_best_sq = current_sq
        local_best_A = list(current_A)
        
        all_vals = list(range(MAX_VAL + 1))
        
        while time.time() < deadline - 5 and T > min_T:
            # Choose move
            move_type = random.random()
            
            new_A = set(current_A)
            
            if move_type < 0.7:
                # Swap
                if not new_A:
                    continue
                a = random.choice(list(new_A))
                b = random.choice(all_vals)
                if a == b:
                    continue
                new_A.remove(a)
                new_A.add(b)
            elif move_type < 0.85:
                # Add
                if len(new_A) > 128:
                    continue
                b = random.choice(all_vals)
                if b in new_A:
                    continue
                new_A.add(b)
            else:
                # Remove
                if len(new_A) <= 2:
                    continue
                a = random.choice(list(new_A))
                new_A.remove(a)
                
            new_peak, new_sq, is_covered_new = evaluate(new_A)
            
            if is_covered_new:
                # Calculate delta for acceptance
                # Composite objective: Primarily Peak, secondarily SumSq
                # We use Peak for the main SA step, but update best if (Peak, Sq) improves
                if current_peak == float('inf'):
                    # Was not covered (shouldn't happen in this branch if we skipped uncovered)
                    accept = True
                else:
                    delta_peak = new_peak - current_peak
                    if delta_peak < 0:
                        accept = True
                    elif delta_peak == 0:
                        # Accept if SumSq improves, or with some probability
                        if new_sq < current_sq:
                            accept = True
                        else:
                            accept = random.random() < 0.5
                    else:
                        # delta_peak > 0
                        accept = random.random() < np.exp(-delta_peak / T)
                
                if accept:
                    current_A = new_A
                    current_peak = new_peak
                    current_sq = new_sq
                    
                    if current_peak < local_best_peak or (current_peak == local_best_peak and current_sq < local_best_sq):
                        local_best_peak = current_peak
                        local_best_sq = current_sq
                        local_best_A = list(current_A)
                        
                    if current_peak < best_peak or (current_peak == best_peak and current_sq < best_sq):
                        best_peak = current_peak
                        best_sq = current_sq
                        best_A = list(current_A)
            else:
                # Not covered
                # If we were covered, reject (or accept with very low prob to escape? 
                # Usually rejecting is better to maintain coverage)
                # If we were not covered, we might accept if it reduces "missingness"
                # But we simplified to only run SA on covered sets.
                # So if we start covered, and move makes it uncovered, we reject.
                pass
            
            T *= cooling_rate

        # Update global best from local best
        if local_best_A is not None:
            if local_best_peak < best_peak or (local_best_peak == best_peak and local_best_sq < best_sq):
                best_peak = local_best_peak
                best_sq = local_best_sq
                best_A = local_best_A

    # Phase 3: Local Search Polish on the best A found
    if best_A is not None and time.time() < deadline:
        current_A = set(best_A)
        current_peak, current_sq, _ = evaluate(current_A)
        
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            
            # 1. Greedy Removals
            for a in list(current_A):
                if time.time() > deadline - 1:
                    break
                new_A = current_A - {a}
                if not new_A:
                    continue
                new_peak, new_sq, cov = evaluate(new_A)
                if cov:
                    if new_peak < current_peak or (new_peak == current_peak and new_sq < current_sq):
                        current_A = new_A
                        current_peak = new_peak
                        current_sq = new_sq
                        improved = True
                        # Restart loop to check for more removals
                        break
            
            if time.time() > deadline - 1:
                break
            
            # 2. Greedy Swaps
            A_list = list(current_A)
            non_A = [v for v in range(MAX_VAL + 1) if v not in current_A]
            
            # Shuffle to avoid bias
            random.shuffle(A_list)
            random.shuffle(non_A)
            
            for a in A_list:
                if time.time() > deadline - 1:
                    break
                for b in non_A:
                    if time.time() > deadline - 1:
                        break
                    new_A = current_A - {a}
                    new_A.add(b)
                    
                    new_peak, new_sq, cov = evaluate(new_A)
                    if cov:
                        if new_peak < current_peak or (new_peak == current_peak and new_sq < current_sq):
                            current_A = new_A
                            current_peak = new_peak
                            current_sq = new_sq
                            improved = True
                            # Restart loops
                            A_list = list(current_A)
                            non_A = [v for v in range(MAX_VAL + 1) if v not in current_A]
                            break # Restart inner loop
                if improved:
                    break # Restart outer loop
            
            # Update global best
            if current_peak < best_peak or (current_peak == best_peak and current_sq < best_sq):
                best_peak = current_peak
                best_sq = current_sq
                best_A = list(current_A)

    # Fallback if no valid A found
    if best_A is None:
        # A simple valid set: all even numbers? 
        # 0, 2, ..., 128. Sum of two evens is even. Doesn't cover odds.
        # We need to cover [128, 256].
        # A = [0, 128] covers 128. 
        # A = [0..128] covers everything.
        best_A = list(range(0, 129))
        
    return best_A
