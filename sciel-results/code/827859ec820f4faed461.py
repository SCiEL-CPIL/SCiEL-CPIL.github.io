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
    
    # Precompute FFT size for convolution
    # We need convolution of length-129 vector with itself
    # Result length is 257, we need indices 128..256
    conv_size = 129 + 129 - 1  # 257
    # Use next power of 2 for FFT efficiency
    fft_n = 1
    while fft_n < conv_size:
        fft_n *= 2
    
    def calc_reps(A_set):
        """Compute unordered pair representation counts for sums 128..256 using FFT."""
        # Create indicator vector
        ind = np.zeros(129, dtype=np.float64)
        for a in A_set:
            ind[a] = 1.0
        
        # Compute ordered pair counts via FFT
        f = np.fft.rfft(ind, n=fft_n)
        conv = np.fft.irfft(f * f, n=fft_n)[:conv_size].round().astype(np.int64)
        
        # conv[s] = count_ordered[s] = number of ordered pairs (a,b) with a+b=s
        # Extract sums 128..256
        ordered = conv[N_LOW:N_HIGH+1]  # length 129
        
        # Convert to unordered
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
    
    def is_covered(reps):
        return np.all(reps > 0)
    
    def get_peak(reps):
        return int(np.max(reps))
    
    # Phase 1: Generate initial candidates
    best_A = None
    best_peak = float('inf')
    
    # Try various random initializations
    for trial in range(30):
        # Force include 128 (needed for 256=128+128) and 0 (helps for 128=0+128)
        size = random.randint(35, 55)
        A_init = set(random.sample(range(0, 129), size))
        A_init.add(128)
        A_init.add(0)
        
        reps = calc_reps(A_init)
        if is_covered(reps):
            peak = get_peak(reps)
            if peak < best_peak:
                best_peak = peak
                best_A = list(A_init)
        
        if time.time() > deadline - 30:
            break
    
    if best_A is None:
        # Fallback
        return list(range(64, 129))
    
    current_A = set(best_A)
    current_reps = calc_reps(current_A)
    current_peak = get_peak(current_reps)
    
    # Simulated Annealing with smarter move generation
    T = 1.0
    cooling_rate = 0.9997
    min_T = 0.01
    stuck_count = 0
    
    while time.time() < deadline - 10 and T > min_T:
        # Choose move type based on current state
        move_type = random.random()
        
        if move_type < 0.5:
            # Swap: remove one, add one (keep size) - focus on high-multiplicity contributors
            if len(current_A) == 0:
                break
            
            # Identify elements that contribute to peak sums
            peak_sums = np.where(current_reps == current_peak)[0]
            if len(peak_sums) > 0:
                # Try removing elements that are in pairs forming peak sums
                peak_sum_vals = [N_LOW + idx for idx in peak_sums]
                # Find elements that participate in these sums
                candidate_removals = []
                for s in peak_sum_vals[:3]:  # Limit to top 3 peak sums
                    for a in current_A:
                        b = s - a
                        if b in current_A:
                            candidate_removals.append(a)
                            break
                if candidate_removals:
                    item_to_remove = random.choice(candidate_removals)
                else:
                    item_to_remove = random.choice(list(current_A))
            else:
                item_to_remove = random.choice(list(current_A))
            
            new_item = random.randint(0, 128)
            
            if new_item == item_to_remove:
                T *= cooling_rate
                continue
                
            new_A = set(current_A)
            new_A.remove(item_to_remove)
            new_A.add(new_item)
            
        elif move_type < 0.75:
            # Remove: try to reduce set size
            if len(current_A) <= 20:
                T *= cooling_rate
                continue
            # Prefer removing elements that don't contribute to coverage
            removable = []
            for a in current_A:
                test_A = set(current_A)
                test_A.remove(a)
                test_reps = calc_reps(test_A)
                if is_covered(test_reps):
                    removable.append(a)
            
            if removable:
                item_to_remove = random.choice(removable)
                new_A = set(current_A)
                new_A.remove(item_to_remove)
            else:
                T *= cooling_rate
                continue
            
        else:
            # Add: increase set size (less common, may help coverage)
            # Try adding elements that help with uncovered or low-rep sums
            low_reps_idx = np.where(current_reps <= 1)[0]
            if len(low_reps_idx) > 0:
                target_sums = [N_LOW + idx for idx in low_reps_idx[:5]]
                candidates = []
                for s in target_sums:
                    for a in range(0, 129):
                        b = s - a
                        if 0 <= b <= 128 and a not in current_A and b not in current_A:
                            candidates.append(a)
                            break
                if candidates:
                    new_item = random.choice(candidates)
                else:
                    new_item = random.randint(0, 128)
            else:
                new_item = random.randint(0, 128)
            
            if new_item in current_A:
                T *= cooling_rate
                continue
            new_A = set(current_A)
            new_A.add(new_item)
        
        new_reps = calc_reps(new_A)
        
        if is_covered(new_reps):
            new_peak = get_peak(new_reps)
            delta_peak = new_peak - current_peak
            
            # Accept if better, or with probability exp(-delta/T) if worse
            if delta_peak < 0:
                current_A = new_A
                current_reps = new_reps
                current_peak = new_peak
                stuck_count = 0
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_A = list(current_A)
            else:
                prob = np.exp(-delta_peak / max(T, 0.01))
                if random.random() < prob:
                    current_A = new_A
                    current_reps = new_reps
                    current_peak = new_peak
                    stuck_count = 0
                else:
                    stuck_count += 1
        else:
            stuck_count += 1
        
        T *= cooling_rate
        
        # If stuck for too long, do a small random restart
        if stuck_count > 150:
            # Perturb current solution more aggressively
            for _ in range(3):
                if len(current_A) > 0:
                    item = random.choice(list(current_A))
                    current_A.remove(item)
                    new_item = random.randint(0, 128)
                    current_A.add(new_item)
            current_reps = calc_reps(current_A)
            if is_covered(current_reps):
                current_peak = get_peak(current_reps)
            else:
                current_A = set(best_A)
                current_reps = calc_reps(current_A)
                current_peak = get_peak(current_reps)
            stuck_count = 0
            T = max(T, 0.3)
    
    # Final local optimization: try removing elements greedily and doing targeted swaps
    if best_A is not None:
        final_A = set(best_A)
        final_reps = calc_reps(final_A)
        if is_covered(final_reps):
            # Try to remove elements one by one
            changed = True
            while changed and time.time() < deadline - 5:
                changed = False
                for a in list(final_A):
                    if len(final_A) <= 20:
                        break
                    test_A = set(final_A)
                    test_A.remove(a)
                    test_reps = calc_reps(test_A)
                    if is_covered(test_reps):
                        test_peak = get_peak(test_reps)
                        if test_peak <= get_peak(final_reps):
                            final_A = test_A
                            final_reps = test_reps
                            changed = True
                            break
            
            # Try targeted swaps to reduce peak
            if time.time() < deadline - 5:
                improved = True
                while improved and time.time() < deadline - 5:
                    improved = False
                    peak_idx = np.where(final_reps == get_peak(final_reps))[0]
                    if len(peak_idx) == 0:
                        break
                    # Try swapping elements involved in peak sums
                    for _ in range(50):
                        if time.time() > deadline - 5:
                            break
                        a = random.choice(list(final_A))
                        b = random.randint(0, 128)
                        if b == a:
                            continue
                        test_A = set(final_A)
                        test_A.remove(a)
                        test_A.add(b)
                        test_reps = calc_reps(test_A)
                        if is_covered(test_reps):
                            test_peak = get_peak(test_reps)
                            if test_peak < get_peak(final_reps):
                                final_A = test_A
                                final_reps = test_reps
                                improved = True
                                break
            
            if get_peak(final_reps) < best_peak:
                best_peak = get_peak(final_reps)
                best_A = list(final_A)
    
    if best_A is None:
        return list(range(64, 129))
        
    return best_A
