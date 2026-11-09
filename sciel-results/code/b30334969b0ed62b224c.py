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

    def get_counts_fast(A_set):
        """Vectorized count computation. Returns (counts, is_covered, peak)."""
        if not A_set:
            return None, False, float('inf')
            
        indicator = np.zeros(MAX_VAL + 1, dtype=np.int32)
        indicator[list(A_set)] = 1
        
        conv = np.convolve(indicator, indicator, mode='full')
        
        # For unordered pairs with a <= b:
        # s even: (conv[s] + indicator[s//2]) // 2
        # s odd: conv[s] // 2
        s_vals = np.arange(N_LOW, N_HIGH + 1)
        counts = np.empty(RANGE_SIZE, dtype=np.int32)
        
        even_mask = (s_vals % 2 == 0)
        odd_mask = ~even_mask
        
        counts[odd_mask] = conv[s_vals[odd_mask]] // 2
        
        even_s = s_vals[even_mask]
        mid = even_s // 2
        valid_mid = (mid >= 0) & (mid <= MAX_VAL)
        even_counts = conv[even_s] // 2
        even_counts[valid_mid] += indicator[mid[valid_mid]] // 2
        # Need to be careful: (conv[s] + indicator[mid]) // 2
        # If conv[s] is even, adding indicator[mid] (0 or 1) then //2
        # If conv[s] is odd, adding indicator[mid] then //2
        # Actually let me just do it directly:
        even_counts2 = (conv[even_s] + np.where(valid_mid, indicator[mid], 0)) // 2
        counts[even_mask] = even_counts2
        
        is_covered = bool(np.all(counts > 0))
        peak = int(np.max(counts)) if is_covered else float('inf')
        
        return counts, is_covered, peak

    def evaluate(A_set):
        counts, is_covered, peak = get_counts_fast(A_set)
        if not is_covered:
            return float('inf'), False
        return peak, True

    best_peak = float('inf')
    best_A = None

    def try_update(A_set):
        nonlocal best_peak, best_A
        peak, is_covered = evaluate(A_set)
        if is_covered and peak < best_peak:
            best_peak = peak
            best_A = list(A_set)
            return True
        return False

    # Strategy 1: Greedy peeling from full set [0, 128]
    # Start with all elements, remove those that don't hurt coverage
    # and reduce peak multiplicity
    A_full = set(range(0, 129))
    counts_full, covered_full, peak_full = get_counts_fast(A_full)
    
    if covered_full:
        best_peak = min(best_peak, peak_full)
        best_A = list(A_full)
    
    # Greedy removal: try removing each element, keep if still covered and peak doesn't increase
    A_peel = set(A_full)
    # Sort candidates by how "redundant" they seem (elements contributing to high-multiplicity sums)
    order = list(range(0, 129))
    random.shuffle(order)
    
    for elem in order:
        if time.time() > deadline - 30:
            break
        if elem not in A_peel:
            continue
        A_test = A_peel - {elem}
        counts_test, covered_test, peak_test = get_counts_fast(A_test)
        if covered_test and peak_test <= peak_full:
            A_peel = A_test
            peak_full = peak_test
    
    try_update(A_peel)
    
    # Now do local search from the peeled set
    current_A = set(A_peel)
    current_counts, current_covered, current_peak = get_counts_fast(current_A)
    
    if not current_covered:
        # Fallback to full set if peeling broke coverage
        current_A = set(range(0, 129))
        current_counts, current_covered, current_peak = get_counts_fast(current_A)
    
    # Simulated Annealing phase
    T = 2.0
    cooling_rate = 0.9997
    min_T = 0.03
    stuck_count = 0
    no_improve = 0
    
    while time.time() < deadline - 15 and T > min_T:
        move_type = random.random()
        new_A = set(current_A)
        
        if move_type < 0.7:
            # Swap: target elements contributing to high-multiplicity sums
            if current_counts is not None and current_covered:
                # Find sums at or near peak
                peak_val = int(np.max(current_counts))
                high_sums = np.where(current_counts >= peak_val - 1)[0]
                if len(high_sums) > 0:
                    target_sum = N_LOW + random.choice(high_sums)
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
        elif move_type < 0.85:
            # Add an element
            if len(new_A) > 128:
                continue
            to_add = random.randint(0, 128)
            if to_add in new_A:
                continue
            new_A.add(to_add)
        else:
            # Remove an element
            if len(new_A) <= 3:
                continue
            to_remove = random.choice(list(new_A))
            new_A.remove(to_remove)
        
        new_counts, new_covered, new_peak = get_counts_fast(new_A)
        
        if new_covered:
            if new_peak < best_peak:
                best_peak = new_peak
                best_A = list(new_A)
                no_improve = 0
            else:
                no_improve += 1
            
            # Acceptance
            if current_covered:
                delta = new_peak - current_peak
                if delta < 0:
                    current_A = new_A
                    current_counts = new_counts
                    current_peak = new_peak
                    stuck_count = 0
                else:
                    if stuck_count > 200:
                        prob = np.exp(-delta / (T * 3))
                    else:
                        prob = np.exp(-delta / T)
                    if random.random() < prob:
                        current_A = new_A
                        current_counts = new_counts
                        current_peak = new_peak
                    else:
                        stuck_count += 1
            else:
                current_A = new_A
                current_counts = new_counts
                current_peak = new_peak
                stuck_count = 0
        else:
            if current_covered:
                stuck_count += 1
                no_improve += 1
            else:
                # Compare uncovered counts
                if current_counts is not None and new_counts is not None:
                    old_uncov = int(np.sum(current_counts == 0))
                    new_uncov = int(np.sum(new_counts == 0))
                    if new_uncov < old_uncov:
                        current_A = new_A
                        current_counts = new_counts
                        stuck_count = 0
                    else:
                        stuck_count += 1
                        no_improve += 1
        
        T *= cooling_rate
        
        # Periodic restart if stuck too long
        if stuck_count > 500:
            # Try a fresh random perturbation
            current_A = set(A_peel)
            for _ in range(random.randint(5, 20)):
                if random.random() < 0.5 and len(current_A) > 3:
                    current_A.discard(random.choice(list(current_A)))
                else:
                    current_A.add(random.randint(0, 128))
            current_counts, current_covered, current_peak = get_counts_fast(current_A)
            T = 1.5
            stuck_count = 0

    # Final high-precision polish pass: iterate over peak sums and try targeted swaps
    if time.time() < deadline - 3:
        polish_A = set(best_A) if best_A else set(A_peel)
        polish_counts, polish_covered, polish_peak = get_counts_fast(polish_A)
        
        if polish_covered:
            polish_peak_val = int(np.max(polish_counts))
            polish_improved = True
            polish_iter = 0
            
            while polish_improved and time.time() < deadline - 3 and polish_iter < 200:
                polish_improved = False
                polish_iter += 1
                polish_counts, polish_covered, polish_peak = get_counts_fast(polish_A)
                if not polish_covered:
                    break
                polish_peak_val = int(np.max(polish_counts))
                
                # Find all sums at peak
                peak_sums = np.where(polish_counts == polish_peak_val)[0]
                
                for idx in peak_sums:
                    if time.time() > deadline - 3:
                        break
                    s = N_LOW + idx
                    # Find pairs contributing to this sum
                    contributing_pairs = []
                    for a in sorted(polish_A):
                        b = s - a
                        if a <= b and b in polish_A:
                            contributing_pairs.append((a, b))
                    
                    if not contributing_pairs:
                        continue
                    
                    # Try removing each element from a random contributing pair
                    random.shuffle(contributing_pairs)
                    for (a, b) in contributing_pairs[:3]:
                        if time.time() > deadline - 3:
                            break
                        
                        # Try removing 'a' and adding something else
                        for _ in range(5):
                            new_A = set(polish_A)
                            new_A.discard(a)
                            to_add = random.randint(0, 128)
                            if to_add not in new_A:
                                new_A.add(to_add)
                            
                            nc, ncov, npk = get_counts_fast(new_A)
                            if ncov and npk < polish_peak_val:
                                polish_A = new_A
                                polish_peak_val = npk
                                polish_improved = True
                                break
                            elif ncov and npk == polish_peak_val:
                                # Accept if it doesn't make things worse
                                if random.random() < 0.3:
                                    polish_A = new_A
                                    polish_improved = True
                                    break
                        
                        if polish_improved:
                            break
                        
                        # Also try removing 'b' if different from 'a'
                        if b != a:
                            for _ in range(5):
                                if time.time() > deadline - 3:
                                    break
                                new_A = set(polish_A)
                                new_A.discard(b)
                                to_add = random.randint(0, 128)
                                if to_add not in new_A:
                                    new_A.add(to_add)
                                
                                nc, ncov, npk = get_counts_fast(new_A)
                                if ncov and npk < polish_peak_val:
                                    polish_A = new_A
                                    polish_peak_val = npk
                                    polish_improved = True
                                    break
                                elif ncov and npk == polish_peak_val:
                                    if random.random() < 0.3:
                                        polish_A = new_A
                                        polish_improved = True
                                        break
                            
                            if polish_improved:
                                break
                
                # Update global best
                if polish_covered:
                    final_peak, _ = evaluate(polish_A)
                    if final_peak < best_peak:
                        best_peak = final_peak
                        best_A = list(polish_A)

    # If no valid A was found, return a fallback
    if best_A is None:
        best_A = list(range(0, 129))

    return best_A
