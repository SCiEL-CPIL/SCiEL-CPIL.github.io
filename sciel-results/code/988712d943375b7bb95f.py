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

    # Precompute targets for vectorized count computation
    target_sums = np.arange(N_LOW, N_HIGH + 1, dtype=np.int32)

    def get_counts_vectorized(A):
        """Get the count array for evaluation using vectorized operations."""
        if not A:
            return None
            
        A_arr = np.array(list(A), dtype=np.int32)
        indicator = np.zeros(MAX_VAL + 1, dtype=np.int32)
        indicator[A_arr] = 1
        
        conv = np.convolve(indicator, indicator, mode='full')
        
        # Vectorized computation of unordered pair counts
        s_vals = target_sums
        # For even s: (conv[s] + indicator[s//2]) // 2
        # For odd s: conv[s] // 2
        is_even = (s_vals % 2 == 0)
        mid_vals = s_vals // 2
        # Clamp mid to valid range
        mid_clamped = np.clip(mid_vals, 0, MAX_VAL)
        mid_indicator = np.where(is_even, indicator[mid_clamped], 0)
        
        counts = (conv[s_vals] + mid_indicator) // 2
        # For odd sums, conv[s] is always even (since a+b=s with a!=b when s is odd), so //2 is exact
        # For even sums, we added indicator to handle the a==b case
        
        return counts

    def evaluate(A):
        """
        Evaluates the set A using convolution for speed.
        Returns (peak, is_covered).
        peak: max multiplicity in [128, 256]
        is_covered: True if all sums in [128, 256] have count > 0
        """
        if not A:
            return float('inf'), False
            
        counts = get_counts_vectorized(A)
        if counts is None:
            return float('inf'), False
            
        if not np.all(counts > 0):
            return float('inf'), False
            
        return int(np.max(counts)), True

    def greedy_peeling_init():
        """
        Start from full set [0,128] and greedily remove elements
        that don't hurt coverage, preferring to remove elements that
        contribute to high-multiplicity sums.
        """
        A = set(range(0, MAX_VAL + 1))
        counts = get_counts_vectorized(A)
        if counts is None:
            return set()
        
        # Try to remove elements in order of their contribution to peak sums
        # We want to remove elements that are "least essential" for coverage
        # but also help reduce peak multiplicity
        
        removed = []
        for _ in range(100):  # Try up to 100 removals
            if time.time() > deadline - 30:
                break
            if not A:
                break
            
            best_remove = None
            best_score = float('inf')
            
            # Sample a subset of candidates to try (for speed)
            candidates = list(A)
            random.shuffle(candidates)
            candidates = candidates[:min(20, len(candidates))]
            
            for elem in candidates:
                test_A = set(A)
                test_A.remove(elem)
                if len(test_A) < 2:
                    continue
                test_counts = get_counts_vectorized(test_A)
                if test_counts is None:
                    continue
                if np.all(test_counts > 0):
                    # Covered - score by peak (lower is better)
                    peak = np.max(test_counts)
                    # Also consider how much this element contributes to high sums
                    score = peak
                    if score < best_score:
                        best_score = score
                        best_remove = elem
            
            if best_remove is None:
                break
            
            A.remove(best_remove)
            removed.append(best_remove)
        
        return A

    def polish_pass(A, time_budget=10.0):
        """
        Final high-precision polish: target peak-multiplicity sums directly.
        Try swaps that specifically reduce the maximum count.
        """
        if not A:
            return A, float('inf')
        
        polish_deadline = time.time() + time_budget
        current_A = set(A)
        current_counts = get_counts_vectorized(current_A)
        if current_counts is None:
            return current_A, float('inf')
        
        if not np.all(current_counts > 0):
            return current_A, float('inf')
        
        current_peak = int(np.max(current_counts))
        best_peak = current_peak
        best_A = set(current_A)
        
        while time.time() < polish_deadline:
            # Find sums with maximum multiplicity
            max_val = np.max(current_counts)
            peak_indices = np.where(current_counts == max_val)[0]
            
            if len(peak_indices) == 0:
                break
            
            # Pick a peak sum
            target_idx = random.choice(peak_indices)
            target_sum = N_LOW + target_idx
            
            # Find pairs that contribute to this sum
            contributing_pairs = []
            A_list = list(current_A)
            for i, a in enumerate(A_list):
                b = target_sum - a
                if b in current_A and a <= b:
                    contributing_pairs.append((a, b))
            
            if not contributing_pairs:
                continue
            
            # Try to swap one element from a contributing pair
            a, b = random.choice(contributing_pairs)
            
            # Try swapping 'a' with a random element
            to_add = random.randint(0, MAX_VAL)
            if to_add == a or to_add == b:
                continue
            
            new_A = set(current_A)
            new_A.remove(a)
            new_A.add(to_add)
            
            new_counts = get_counts_vectorized(new_A)
            if new_counts is None:
                continue
            
            if np.all(new_counts > 0):
                new_peak = int(np.max(new_counts))
                if new_peak < best_peak:
                    best_peak = new_peak
                    best_A = set(new_A)
                    current_A = new_A
                    current_counts = new_counts
                    current_peak = new_peak
                elif new_peak == best_peak:
                    # Accept with some probability to escape local optima
                    if random.random() < 0.3:
                        current_A = new_A
                        current_counts = new_counts
                        current_peak = new_peak
            else:
                # Not covered - reject
                pass
        
        return best_A, best_peak

    best_peak = float('inf')
    best_A = None

    # Phase 1: Greedy peeling initialization
    A_greedy = greedy_peeling_init()
    if A_greedy:
        peak, covered = evaluate(A_greedy)
        if covered:
            best_peak = peak
            best_A = list(A_greedy)
    
    # Phase 2: Simulated Annealing with multiple restarts
    sizes_to_try = [35, 40, 45, 50, 55]
    random.shuffle(sizes_to_try)

    for target_size in sizes_to_try:
        if time.time() > deadline - 20:
            break
            
        # Initialize
        A_init = set()
        A_init.add(0)
        A_init.add(128)
        
        # Add random elements
        candidates = list(range(1, 128))
        random.shuffle(candidates)
        
        for c in candidates:
            if len(A_init) >= target_size:
                break
            A_init.add(c)
            
        current_A = A_init
        current_peak, is_covered = evaluate(current_A)
        
        if is_covered:
            if current_peak < best_peak:
                best_peak = current_peak
                best_A = list(current_A)

        # Simulated Annealing
        T = 3.0
        cooling_rate = 0.9995
        min_T = 0.05
        
        local_best_peak = current_peak if is_covered else float('inf')
        local_best_A = list(current_A) if is_covered else None
        stuck_count = 0
        
        while time.time() < deadline - 15 and T > min_T:
            # Choose move type with adaptive probabilities
            move_type = random.random()
            
            new_A = set(current_A)
            
            if move_type < 0.6:
                # Swap - try to swap elements that are in high-multiplicity sums
                if not new_A:
                    continue
                    
                # Get current counts to identify problematic elements
                counts = get_counts_vectorized(new_A)
                if counts is not None and is_covered:
                    # Find sums with high multiplicity
                    high_mult_sums = np.where(counts > np.median(counts))[0]
                    if len(high_mult_sums) > 0:
                        # Pick a sum with high multiplicity and try to reduce it
                        target_sum = N_LOW + random.choice(high_mult_sums)
                        # Find elements that contribute to this sum
                        contributing = [a for a in new_A if (target_sum - a) in new_A and a <= target_sum - a]
                        if contributing:
                            to_remove = random.choice(contributing)
                            to_add = random.randint(0, 128)
                            if to_remove != to_add:
                                new_A.remove(to_remove)
                                new_A.add(to_add)
                            else:
                                continue
                        else:
                            to_remove = random.choice(list(new_A))
                            to_add = random.randint(0, 128)
                            if to_remove == to_add:
                                continue
                            new_A.remove(to_remove)
                            new_A.add(to_add)
                    else:
                        to_remove = random.choice(list(new_A))
                        to_add = random.randint(0, 128)
                        if to_remove == to_add:
                            continue
                        new_A.remove(to_remove)
                        new_A.add(to_add)
                else:
                    to_remove = random.choice(list(new_A))
                    to_add = random.randint(0, 128)
                    if to_remove == to_add:
                        continue
                    new_A.remove(to_remove)
                    new_A.add(to_add)
            elif move_type < 0.8:
                # Add
                if len(new_A) > 128:
                    continue
                to_add = random.randint(0, 128)
                if to_add in new_A:
                    continue
                new_A.add(to_add)
            else:
                # Remove
                if len(new_A) <= 2:
                    continue
                to_remove = random.choice(list(new_A))
                new_A.remove(to_remove)
                
            new_peak, is_covered_new = evaluate(new_A)
            
            if is_covered_new:
                if new_peak < local_best_peak:
                    local_best_peak = new_peak
                    local_best_A = list(new_A)
                    stuck_count = 0
                    
                if new_peak < best_peak:
                    best_peak = new_peak
                    best_A = list(new_A)
                    stuck_count = 0
                
                # Acceptance criterion
                if is_covered:
                    delta = new_peak - current_peak
                    if delta < 0:
                        current_A = new_A
                        current_peak = new_peak
                    else:
                        # Prob of accepting worse - increase acceptance when stuck
                        if stuck_count > 100:
                            prob = np.exp(-delta / (T * 2))
                        else:
                            prob = np.exp(-delta / T)
                        if random.random() < prob:
                            current_A = new_A
                            current_peak = new_peak
                        else:
                            stuck_count += 1
                else:
                    # Was not covered, now covered: Accept
                    current_A = new_A
                    current_peak = new_peak
                    stuck_count = 0
            else:
                # Not covered
                if is_covered:
                    # Was covered, now not: Reject
                    stuck_count += 1
                else:
                    # Was not covered, still not covered:
                    # Check if we're making progress toward coverage
                    old_counts = get_counts_vectorized(current_A)
                    new_counts = get_counts_vectorized(new_A)
                    if old_counts is not None and new_counts is not None:
                        old_uncovered = np.sum(old_counts == 0)
                        new_uncovered = np.sum(new_counts == 0)
                        if new_uncovered < old_uncovered:
                            # Making progress toward coverage
                            current_A = new_A
                            stuck_count = 0
                        else:
                            stuck_count += 1
                    
            T *= cooling_rate
            
        # After SA for this size, update global best if local best is better
        if local_best_A is not None and local_best_peak < best_peak:
            best_peak = local_best_peak
            best_A = local_best_A

    # Phase 3: Final polish pass on the best solution found
    if best_A is not None and time.time() < deadline - 5:
        polish_time = min(10.0, deadline - time.time() - 3)
        if polish_time > 2.0:
            polished_A, polished_peak = polish_pass(set(best_A), polish_time)
            if polished_peak < best_peak:
                best_peak = polished_peak
                best_A = list(polished_A)

    # If no valid A was found, return a fallback
    if best_A is None:
        best_A = list(range(0, 129))

    return best_A
