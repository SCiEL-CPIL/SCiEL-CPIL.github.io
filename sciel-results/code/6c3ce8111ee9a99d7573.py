import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin (evaluator kills at 90s)
    random.seed(42)  # Reproducibility
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1
    MAX_VAL = 128
    DOMAIN = list(range(MAX_VAL + 1))

    def calc_reps(A):
        """Returns array of counts for sums N_LOW..N_HIGH using unordered pairs."""
        A_arr = np.array(sorted(A), dtype=np.int32)
        n = len(A_arr)
        if n == 0:
            return np.zeros(RANGE_SIZE, dtype=np.int32)
        
        # Use FFT convolution for speed: count ordered pairs (a,b) with a+b=s
        # Then adjust for unordered: unordered_count = (ordered_count + diagonal) / 2
        # where diagonal counts pairs where a==b (i.e., s is even and s/2 in A)
        
        # Create indicator vector
        size = 257  # max sum is 256
        ind = np.zeros(size, dtype=np.float64)
        ind[A_arr] = 1.0
        
        # Convolve to get ordered pair counts
        conv = np.convolve(ind, ind, mode='full')
        # conv[s] = number of ordered pairs (a,b) in A x A with a+b=s
        
        # For unordered pairs with repetition allowed:
        # If s is odd: all pairs have a != b, so unordered = ordered / 2
        # If s is even: pairs with a=b contribute 1 to ordered, but should count as 1 unordered
        #   ordered_count includes (a,b) and (b,a) for a!=b, and (a,a) once
        #   unordered_count = (ordered_count - diag_count) / 2 + diag_count
        #   where diag_count = 1 if s/2 in A else 0
        #   = ordered_count/2 + diag_count/2
        
        sums_range = np.arange(N_LOW, N_HIGH + 1)
        ordered = conv[sums_range].astype(np.float64)
        
        # Diagonal contribution: s is even and s/2 is in A
        half_sums = sums_range / 2.0
        is_even = (sums_range % 2 == 0)
        diag = np.zeros(RANGE_SIZE, dtype=np.float64)
        even_indices = np.where(is_even)[0]
        if len(even_indices) > 0:
            half_vals = sums_range[even_indices] // 2
            # Check which half_vals are in A
            a_set = set(A_arr.tolist())
            diag_mask = np.array([v in a_set for v in half_vals])
            diag[even_indices[diag_mask]] = 1.0
        
        unordered = ordered / 2.0 + diag / 2.0
        # Round to integer
        result = np.round(unordered).astype(np.int32)
        return result

    def is_covered(reps):
        return np.all(reps > 0)

    def get_peak(reps):
        return int(np.max(reps))

    def element_contribution(x, A, reps):
        """For element x in set A, return (num_peak_contrib, coverage_risk).
        num_peak_contrib: how many sums at peak level involve x
        coverage_risk: number of sums that would drop to 0 if x removed
        """
        A_arr = np.array(sorted(A), dtype=np.int32)
        peak = get_peak(reps)
        
        sums = x + A_arr
        mask = (sums >= N_LOW) & (sums <= N_HIGH)
        if not np.any(mask):
            return 0, 0
        
        valid_sums = sums[mask]
        indices = valid_sums - N_LOW
        rep_vals = reps[indices]
        
        peak_count = int(np.sum(rep_vals == peak))
        coverage_risk = int(np.sum(rep_vals == 1))
        
        return peak_count, coverage_risk

    def find_peak_elements(A, reps):
        """Find elements that contribute most to peak sums."""
        peak = get_peak(reps)
        if peak == 0:
            return []
        
        A_list = list(A)
        scores = {}
        for x in A_list:
            pc, _ = element_contribution(x, A, reps)
            scores[x] = pc
        
        # Return elements sorted by peak contribution (descending)
        sorted_elems = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return sorted_elems

    best_A = None
    best_peak = float('inf')
    
    def update_best(A, reps):
        nonlocal best_A, best_peak
        if is_covered(reps):
            peak = get_peak(reps)
            if peak < best_peak:
                best_peak = peak
                best_A = list(A)

    def random_init():
        A = set()
        A.add(128)
        A.add(0)
        # Add structured elements to ensure good coverage
        # Include elements that help cover low and high ends
        for x in [1, 2, 3, 4, 124, 125, 126, 127]:
            A.add(x)
        # Add some mid-range elements
        for x in [64, 63, 65]:
            A.add(x)
        # Fill with random
        while len(A) < 45:
            A.add(random.randint(0, 128))
        return A

    def repair_coverage(A):
        """Add elements until all sums in [128,256] are covered."""
        A = set(A)
        reps = calc_reps(A)
        tries = 0
        while not is_covered(reps) and tries < 100:
            missing = np.where(reps == 0)[0] + N_LOW
            # For each missing sum, find a good element to add
            best_candidate = None
            best_gain = 0
            for s in missing:
                # Find elements x such that s-x is in [0,128] and x not in A
                candidates = set()
                for x in range(0, 129):
                    y = s - x
                    if 0 <= y <= 128:
                        candidates.add(x)
                candidates -= A
                # Only evaluate a subset of candidates for speed
                for c in list(candidates)[:20]:
                    new_A = A | {c}
                    new_reps = calc_reps(new_A)
                    new_missing = np.sum(new_reps == 0)
                    gain = len(missing) - new_missing
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = c
            if best_candidate is not None:
                A.add(best_candidate)
                reps = calc_reps(A)
            else:
                # Random fallback
                A.add(random.randint(0, 128))
                reps = calc_reps(A)
            tries += 1
        return A, reps

    # Phase 1: Multi-start Local Search with peak-aware moves
    num_restarts = 60
    
    for r in range(num_restarts):
        if time.time() > deadline: break
        
        current_A = random_init()
        current_A, current_reps = repair_coverage(current_A)
        
        if not is_covered(current_reps):
            continue

        update_best(current_A, current_reps)
        
        # Local Search: Peak-aware Simulated Annealing
        T = 1.5
        cooling = 0.997
        min_T = 0.03
        stuck_counter = 0
        max_stuck = 200
        
        while time.time() < deadline and T > min_T:
            if stuck_counter > max_stuck:
                break
            
            current_peak = get_peak(current_reps)
            move_type = random.random()
            
            if len(current_A) > 15 and move_type < 0.4:
                # Try Remove: prioritize elements with high peak contribution
                peak_elems = find_peak_elements(current_A, current_reps)
                if peak_elems:
                    # Weighted selection: higher peak contribution -> higher chance
                    weights = []
                    for x in peak_elems:
                        pc, _ = element_contribution(x, current_A, current_reps)
                        weights.append(pc + 1)
                    total_w = sum(weights)
                    r = random.random() * total_w
                    cum = 0
                    item_to_remove = peak_elems[-1]
                    for i, x in enumerate(peak_elems):
                        cum += weights[i]
                        if r <= cum:
                            item_to_remove = x
                            break
                else:
                    item_to_remove = random.choice(list(current_A))
                
                new_A = set(current_A)
                new_A.remove(item_to_remove)
                new_reps = calc_reps(new_A)
                
                if is_covered(new_reps):
                    new_peak = get_peak(new_reps)
                    
                    if new_peak < current_peak:
                        current_A = new_A
                        current_reps = new_reps
                        update_best(current_A, current_reps)
                        stuck_counter = 0
                    elif new_peak == current_peak:
                        # Accept with some probability to explore
                        if random.random() < 0.3:
                            current_A = new_A
                            current_reps = new_reps
                            update_best(current_A, current_reps)
                            stuck_counter = 0
                        else:
                            stuck_counter += 1
                    else:
                        prob = np.exp(-(new_peak - current_peak) / T)
                        if random.random() < prob:
                            current_A = new_A
                            current_reps = new_reps
                            stuck_counter = 0
                        else:
                            stuck_counter += 1
                else:
                    stuck_counter += 1
            elif move_type < 0.7:
                # Try Swap: remove a high-peak element, add a random one
                peak_elems = find_peak_elements(current_A, current_reps)
                if peak_elems:
                    item_to_remove = peak_elems[0]  # Highest peak contributor
                else:
                    item_to_remove = random.choice(list(current_A))
                
                new_item = random.randint(0, 128)
                if new_item == item_to_remove:
                    continue
                    
                new_A = set(current_A)
                new_A.remove(item_to_remove)
                new_A.add(new_item)
                
                new_reps = calc_reps(new_A)
                
                if is_covered(new_reps):
                    new_peak = get_peak(new_reps)
                    
                    if new_peak < current_peak:
                        current_A = new_A
                        current_reps = new_reps
                        update_best(current_A, current_reps)
                        stuck_counter = 0
                    else:
                        prob = np.exp(-(new_peak - current_peak) / T)
                        if random.random() < prob:
                            current_A = new_A
                            current_reps = new_reps
                            stuck_counter = 0
                        else:
                            stuck_counter += 1
                else:
                    stuck_counter += 1
            else:
                # Try targeted swap: replace a peak element with one that helps low-coverage sums
                peak_elems = find_peak_elements(current_A, current_reps)
                if not peak_elems:
                    stuck_counter += 1
                    T *= cooling
                    continue
                    
                item_to_remove = peak_elems[0]
                
                # Find sums with low coverage (count == 1 or 2)
                low_cov_sums = np.where(current_reps <= 2)[0] + N_LOW
                if len(low_cov_sums) == 0:
                    stuck_counter += 1
                    T *= cooling
                    continue
                
                # Try candidates that help cover low-coverage sums
                found = False
                for _ in range(10):
                    s = random.choice(low_cov_sums.tolist())
                    # Find x such that s - x is in current_A
                    candidates = []
                    for y in current_A:
                        x = s - y
                        if 0 <= x <= 128 and x != item_to_remove:
                            candidates.append(x)
                    if candidates:
                        new_item = random.choice(candidates)
                        if new_item not in current_A:
                            new_A = set(current_A)
                            new_A.remove(item_to_remove)
                            new_A.add(new_item)
                            new_reps = calc_reps(new_A)
                            if is_covered(new_reps):
                                new_peak = get_peak(new_reps)
                                if new_peak <= current_peak:
                                    current_A = new_A
                                    current_reps = new_reps
                                    update_best(current_A, current_reps)
                                    stuck_counter = 0
                                    found = True
                                    break
                if not found:
                    stuck_counter += 1
            
            T *= cooling
            
    # Final Polish: Peak-targeted refinement
    if best_A:
        A = set(best_A)
        reps = calc_reps(A)
        
        # Phase 2a: Greedy removal of redundant elements
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for x in sorted(A, key=lambda e: -element_contribution(e, A, reps)[0]):
                new_A = set(A)
                new_A.remove(x)
                new_reps = calc_reps(new_A)
                if is_covered(new_reps):
                    new_peak = get_peak(new_reps)
                    old_peak = get_peak(reps)
                    if new_peak <= old_peak:
                        A = new_A
                        reps = new_reps
                        changed = True
                        update_best(A, reps)
                        break
        
        # Phase 2b: Targeted swaps to reduce peak
        if time.time() < deadline:
            for _ in range(2000):
                if time.time() > deadline:
                    break
                
                peak = get_peak(reps)
                peak_elems = find_peak_elements(A, reps)
                if not peak_elems:
                    break
                
                improved = False
                for x in peak_elems[:8]:  # Try top 8 peak contributors
                    if time.time() > deadline:
                        break
                    # Try swapping x with various candidates, prioritizing elements
                    # that don't contribute to peak sums
                    # First try elements that are NOT peak contributors
                    non_peak_elems = [e for e in range(129) if e not in A and e not in set(peak_elems[:8])]
                    # Also try random
                    candidates = non_peak_elems[:20] + [random.randint(0, 128) for _ in range(15)]
                    for new_item in candidates:
                        if time.time() > deadline:
                            break
                        if new_item == x or new_item in A:
                            continue
                        new_A = set(A)
                        new_A.remove(x)
                        new_A.add(new_item)
                        new_reps = calc_reps(new_A)
                        if is_covered(new_reps):
                            new_peak = get_peak(new_reps)
                            if new_peak < peak:
                                A = new_A
                                reps = new_reps
                                peak = new_peak
                                update_best(A, reps)
                                improved = True
                                break
                    if improved:
                        break
                
                if not improved:
                    # Try a different strategy: remove elements with low coverage contribution
                    if len(A) > 20:
                        low_contrib = []
                        for x in A:
                            pc, cr = element_contribution(x, A, reps)
                            if pc == 0 and cr == 0:
                                low_contrib.append(x)
                        if low_contrib:
                            x_to_remove = random.choice(low_contrib)
                            new_A = set(A)
                            new_A.remove(x_to_remove)
                            new_reps = calc_reps(new_A)
                            if is_covered(new_reps):
                                A = new_A
                                reps = new_reps
                                update_best(A, reps)
                        else:
                            break
                    else:
                        break

    if best_A is None:
        return list(range(64, 129))
        
    return best_A
