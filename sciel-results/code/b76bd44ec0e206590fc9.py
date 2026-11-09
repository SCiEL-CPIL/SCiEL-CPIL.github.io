import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin for 90s limit
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1
    MAX_VAL = 128
    
    # Pre-compute indices for sums
    s_vals = np.arange(N_LOW, N_HIGH + 1)
    even_mask = (s_vals % 2 == 0)
    odd_mask = ~even_mask
    even_s = s_vals[even_mask]
    odd_s = s_vals[odd_mask]
    mid_even = even_s // 2
    
    def get_counts(A_set):
        """
        Returns (counts_array, is_covered, peak_value)
        counts_array is length 129, indexed by sum s in [128, 256]
        """
        if not A_set:
            return None, False, float('inf')
            
        indicator = np.zeros(MAX_VAL + 1, dtype=np.int32)
        # Using a list conversion is fast enough for small sets
        indicator[list(A_set)] = 1
        
        conv = np.convolve(indicator, indicator, mode='full')
        
        counts = np.empty(RANGE_SIZE, dtype=np.int32)
        
        # Odd sums: count = conv[s] // 2
        counts[odd_mask] = conv[odd_s] // 2
        
        # Even sums: count = (conv[s] + indicator[s//2]) // 2
        # Note: indicator[mid] is 1 if mid is in A, else 0
        counts[even_mask] = (conv[even_s] + indicator[mid_even]) // 2
        
        is_covered = bool(np.all(counts > 0))
        peak = int(np.max(counts)) if is_covered else float('inf')
        
        return counts, is_covered, peak

    # Strategy 1: Greedy Peeling from Full Set
    A_full = set(range(MAX_VAL + 1))
    counts, covered, peak = get_counts(A_full)
    
    if not covered:
        # Should not happen for full set
        return list(A_full)
        
    best_peak = peak
    best_A = list(A_full)
    current_A = A_full
    current_counts = counts
    current_peak = peak

    # Greedy removal loop
    # We try to remove elements. We prioritize elements that are part of high-count sums.
    removed_any = True
    while removed_any and time.time() < deadline - 30:
        removed_any = False
        # Identify elements contributing to the current peak sums
        peak_sums = np.where(current_counts == current_peak)[0]
        if len(peak_sums) == 0:
            break
            
        # Build a set of "critical" elements: those in pairs summing to peak_sums
        critical_elements = set()
        for idx in peak_sums:
            s = N_LOW + idx
            for a in current_A:
                b = s - a
                if a <= b and b in current_A:
                    critical_elements.add(a)
                    critical_elements.add(b)
        
        # Try removing elements from critical_elements first, then others
        candidates = list(critical_elements)
        random.shuffle(candidates)
        # Add remaining elements
        remaining = list(current_A - critical_elements)
        random.shuffle(remaining)
        candidates.extend(remaining)
        
        for elem in candidates:
            if elem not in current_A:
                continue
            if time.time() > deadline - 30:
                break
                
            new_A = current_A - {elem}
            new_counts, new_covered, new_peak = get_counts(new_A)
            
            if new_covered:
                if new_peak < current_peak:
                    # Strict improvement
                    current_A = new_A
                    current_counts = new_counts
                    current_peak = new_peak
                    best_peak = new_peak
                    best_A = list(current_A)
                    removed_any = True
                    break # Restart the pass with new state
                elif new_peak == current_peak:
                    # Same peak, check if we can reduce "excess" counts
                    # e.g., sum of counts, or variance
                    # If we remove an element, total counts decrease.
                    # This might help lower the peak later.
                    # Let's accept if it reduces the sum of counts (which it always does)
                    # To be safe, only accept if it doesn't increase the number of sums at peak?
                    # Actually, just accepting any removal that keeps peak same is a good greedy heuristic
                    # because it makes the set smaller, potentially allowing more removals later.
                    current_A = new_A
                    current_counts = new_counts
                    current_peak = new_peak
                    best_peak = new_peak
                    best_A = list(current_A)
                    removed_any = True
                    break # Restart the pass

    # Strategy 2: Local Search / Simulated Annealing
    # We might be stuck at a local minimum where no single removal works.
    # Try swaps: remove x, add y.
    
    current_A = set(best_A)
    current_counts, current_covered, current_peak = get_counts(current_A)
    
    if not current_covered:
        # Fallback
        return best_A

    T = 5.0
    cooling_rate = 0.9995
    min_T = 0.1
    stuck_count = 0
    
    while time.time() < deadline - 5 and T > min_T:
        new_A = set(current_A)
        
        # Generate a move
        move_type = random.random()
        if move_type < 0.8:
            # Swap
            if len(current_A) > 0:
                to_remove = random.choice(list(current_A))
                to_add = random.randint(0, MAX_VAL)
                if to_remove != to_add:
                    new_A.discard(to_remove)
                    new_A.add(to_add)
                else:
                    continue
        elif move_type < 0.9:
            # Add
            if len(new_A) < MAX_VAL + 1:
                to_add = random.randint(0, MAX_VAL)
                if to_add not in new_A:
                    new_A.add(to_add)
                else:
                    continue
        else:
            # Remove
            if len(new_A) > 1:
                to_remove = random.choice(list(new_A))
                new_A.discard(to_remove)
            else:
                continue
        
        new_counts, new_covered, new_peak = get_counts(new_A)
        
        if new_covered:
            delta = new_peak - current_peak
            if delta < 0:
                # Improvement
                current_A = new_A
                current_counts = new_counts
                current_peak = new_peak
                if new_peak < best_peak:
                    best_peak = new_peak
                    best_A = list(current_A)
                stuck_count = 0
            else:
                # Acceptance probability
                if stuck_count > 100:
                    prob = np.exp(-delta / (T * 2))
                else:
                    prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_A = new_A
                    current_counts = new_counts
                    current_peak = new_peak
                else:
                    stuck_count += 1
        else:
            # Not covered
            if current_covered:
                stuck_count += 1
                # Strong penalty: reject
            else:
                # Both not covered, try to improve coverage
                # Count how many sums are uncovered
                old_uncov = int(np.sum(current_counts == 0))
                new_uncov = int(np.sum(new_counts == 0))
                if new_uncov < old_uncov:
                    current_A = new_A
                    current_counts = new_counts
                    current_peak = new_peak
                    stuck_count = 0
                else:
                    stuck_count += 1
        
        T *= cooling_rate
        
        # Restart if stuck
        if stuck_count > 300:
            # Perturb current solution
            current_A = set(best_A)
            for _ in range(random.randint(5, 15)):
                if random.random() < 0.5 and len(current_A) > 5:
                    current_A.discard(random.choice(list(current_A)))
                else:
                    current_A.add(random.randint(0, MAX_VAL))
            current_counts, current_covered, current_peak = get_counts(current_A)
            T = 2.0
            stuck_count = 0

    # Final Polish: Targeted removals
    if time.time() < deadline - 2:
        polish_A = set(best_A)
        polish_counts, polish_covered, polish_peak = get_counts(polish_A)
        
        if polish_covered:
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                polish_counts, polish_covered, polish_peak = get_counts(polish_A)
                if not polish_covered:
                    break
                peak_val = int(np.max(polish_counts))
                peak_sums = np.where(polish_counts == peak_val)[0]
                
                # Try removing elements involved in peak sums
                candidates = set()
                for idx in peak_sums:
                    s = N_LOW + idx
                    for a in polish_A:
                        b = s - a
                        if a <= b and b in polish_A:
                            candidates.add(a)
                            candidates.add(b)
                
                random_list = list(candidates)
                random.shuffle(random_list)
                
                for elem in random_list:
                    if elem not in polish_A:
                        continue
                    new_A = polish_A - {elem}
                    new_counts, new_covered, new_peak = get_counts(new_A)
                    if new_covered and new_peak < peak_val:
                        polish_A = new_A
                        improved = True
                        if new_peak < best_peak:
                            best_peak = new_peak
                            best_A = list(polish_A)
                        break

    return best_A
