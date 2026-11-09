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
    conv_size = 129 + 129 - 1  # 257
    fft_n = 1
    while fft_n < conv_size:
        fft_n *= 2
    
    # Precompute even-sum halves for unordered conversion
    even_halves = np.array([s // 2 for s in range(N_LOW, N_HIGH + 1) if s % 2 == 0])
    even_mask = np.array([s % 2 == 0 for s in range(N_LOW, N_HIGH + 1)])
    
    def calc_reps_from_ind(ind):
        """Compute unordered pair representation counts for sums 128..256 from indicator vector."""
        f = np.fft.rfft(ind, n=fft_n)
        conv = np.fft.irfft(f * f, n=fft_n)[:conv_size].round().astype(np.int64)
        ordered = conv[N_LOW:N_HIGH+1]
        
        reps = np.zeros(RANGE_SIZE, dtype=np.int64)
        # Odd sums: just divide by 2
        reps[~even_mask] = ordered[~even_mask] // 2
        # Even sums: (ordered + diag) // 2
        diag_vals = ind[even_halves]
        reps[even_mask] = (ordered[even_mask] + diag_vals) // 2
        
        return reps
    
    def calc_reps(A_set):
        """Compute unordered pair representation counts for sums 128..256 using FFT."""
        ind = np.zeros(129, dtype=np.float64)
        for a in A_set:
            ind[a] = 1.0
        return calc_reps_from_ind(ind)
    
    def is_covered(reps):
        return np.all(reps > 0)
    
    def get_peak(reps):
        return int(np.max(reps))
    
    # Phase 1: Greedy peeling from full interval [0, 128]
    # Start with full set and remove elements that don't hurt coverage/peak
    best_A = None
    best_peak = float('inf')
    
    # Start with full interval
    A_full = set(range(0, 129))
    ind_full = np.zeros(129, dtype=np.float64)
    ind_full[:] = 1.0
    reps_full = calc_reps_from_ind(ind_full)
    
    if is_covered(reps_full):
        peak_full = get_peak(reps_full)
        best_peak = peak_full
        best_A = list(A_full)
    
    # Greedy peeling: iteratively remove elements
    current_ind = ind_full.copy()
    current_reps = reps_full.copy()
    current_set = set(A_full)
    
    # Multiple passes of greedy removal
    for pass_num in range(10):
        if time.time() > deadline - 25:
            break
        changed = False
        # Shuffle order for variety
        elements = list(current_set)
        random.shuffle(elements)
        for a in elements:
            if time.time() > deadline - 25:
                break
            if a not in current_set:
                continue
            # Try removing a
            current_ind[a] = 0.0
            test_reps = calc_reps_from_ind(current_ind)
            if is_covered(test_reps):
                test_peak = get_peak(test_reps)
                if test_peak <= get_peak(current_reps):
                    current_set.remove(a)
                    current_reps = test_reps
                    changed = True
                else:
                    current_ind[a] = 1.0
            else:
                current_ind[a] = 1.0
        
        if not changed:
            break
    
    if is_covered(current_reps):
        peak = get_peak(current_reps)
        if peak < best_peak:
            best_peak = peak
            best_A = list(current_set)
    
    # Phase 2: Random restarts with greedy peeling
    for trial in range(15):
        if time.time() > deadline - 25:
            break
        # Random subset with forced elements
        size = random.randint(40, 70)
        A_init = set(random.sample(range(0, 129), size))
        A_init.add(128)
        A_init.add(0)
        
        reps = calc_reps(A_init)
        if is_covered(reps):
            peak = get_peak(reps)
            
            # Greedy peeling from this initial set
            init_ind = np.zeros(129, dtype=np.float64)
            for a in A_init:
                init_ind[a] = 1.0
            init_reps = calc_reps_from_ind(init_ind)
            init_set = set(A_init)
            
            for _ in range(5):
                if time.time() > deadline - 25:
                    break
                elements = list(init_set)
                random.shuffle(elements)
                for a in elements:
                    if a not in init_set:
                        continue
                    init_ind[a] = 0.0
                    test_reps = calc_reps_from_ind(init_ind)
                    if is_covered(test_reps):
                        test_peak = get_peak(test_reps)
                        if test_peak <= get_peak(init_reps):
                            init_set.remove(a)
                            init_reps = test_reps
                        else:
                            init_ind[a] = 1.0
                    else:
                        init_ind[a] = 1.0
            
            if is_covered(init_reps):
                final_peak = get_peak(init_reps)
                if final_peak < best_peak:
                    best_peak = final_peak
                    best_A = list(init_set)
    
    if best_A is None:
        return list(range(64, 129))
    
    current_A = set(best_A)
    current_ind = np.zeros(129, dtype=np.float64)
    for a in current_A:
        current_ind[a] = 1.0
    current_reps = calc_reps_from_ind(current_ind)
    current_peak = get_peak(current_reps)
    
    # Phase 3: Simulated Annealing with incremental evaluation
    T = 2.0
    cooling_rate = 0.9997
    min_T = 0.01
    stuck_count = 0
    current_A_list = list(current_A)
    
    while time.time() < deadline and T > min_T:
        # Choose move type
        move_type = random.random()
        
        if move_type < 0.7:
            # Swap: remove one, add one (keep size)
            if len(current_A) == 0:
                break
            idx_remove = random.randint(0, len(current_A_list) - 1)
            item_to_remove = current_A_list[idx_remove]
            new_item = random.randint(0, 128)
            
            if new_item == item_to_remove or new_item in current_A:
                T *= cooling_rate
                continue
            
            # Incremental update
            old_ind = current_ind.copy()
            current_ind[item_to_remove] = 0.0
            current_ind[new_item] = 1.0
            new_reps = calc_reps_from_ind(current_ind)
            
            if is_covered(new_reps):
                new_peak = get_peak(new_reps)
                delta_peak = new_peak - current_peak
                
                if delta_peak < 0 or random.random() < np.exp(-delta_peak / max(T, 0.01)):
                    # Accept
                    current_A.remove(item_to_remove)
                    current_A.add(new_item)
                    current_A_list[idx_remove] = new_item
                    current_reps = new_reps
                    current_peak = new_peak
                    stuck_count = 0
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = list(current_A)
                else:
                    stuck_count += 1
                    current_ind = old_ind
            else:
                stuck_count += 1
                current_ind = old_ind
                
        elif move_type < 0.9:
            # Remove: try to reduce set size
            if len(current_A) <= 25:
                T *= cooling_rate
                continue
            idx_remove = random.randint(0, len(current_A_list) - 1)
            item_to_remove = current_A_list[idx_remove]
            
            old_ind = current_ind.copy()
            current_ind[item_to_remove] = 0.0
            new_reps = calc_reps_from_ind(current_ind)
            
            if is_covered(new_reps):
                new_peak = get_peak(new_reps)
                delta_peak = new_peak - current_peak
                
                if delta_peak < 0 or random.random() < np.exp(-delta_peak / max(T, 0.01)):
                    current_A.remove(item_to_remove)
                    current_A_list.pop(idx_remove)
                    current_reps = new_reps
                    current_peak = new_peak
                    stuck_count = 0
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = list(current_A)
                else:
                    stuck_count += 1
                    current_ind = old_ind
            else:
                stuck_count += 1
                current_ind = old_ind
                
        else:
            # Add: increase set size
            new_item = random.randint(0, 128)
            if new_item in current_A:
                T *= cooling_rate
                continue
            
            old_ind = current_ind.copy()
            current_ind[new_item] = 1.0
            new_reps = calc_reps_from_ind(current_ind)
            
            if is_covered(new_reps):
                new_peak = get_peak(new_reps)
                delta_peak = new_peak - current_peak
                
                if delta_peak < 0 or random.random() < np.exp(-delta_peak / max(T, 0.01)):
                    current_A.add(new_item)
                    current_A_list.append(new_item)
                    current_reps = new_reps
                    current_peak = new_peak
                    stuck_count = 0
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = list(current_A)
                else:
                    stuck_count += 1
                    current_ind = old_ind
            else:
                stuck_count += 1
                current_ind = old_ind
        
        T *= cooling_rate
        
        # If stuck for too long, do a small random restart
        if stuck_count > 150:
            # Perturb: do several random swaps
            for _ in range(8):
                if len(current_A) > 0:
                    idx = random.randint(0, len(current_A_list) - 1)
                    item = current_A_list[idx]
                    current_A.remove(item)
                    current_A_list.pop(idx)
                    new_item = random.randint(0, 128)
                    if new_item not in current_A:
                        current_A.add(new_item)
                        current_A_list.append(new_item)
            
            current_ind = np.zeros(129, dtype=np.float64)
            for a in current_A:
                current_ind[a] = 1.0
            current_reps = calc_reps_from_ind(current_ind)
            if is_covered(current_reps):
                current_peak = get_peak(current_reps)
            else:
                # Restore best
                current_A = set(best_A)
                current_A_list = list(current_A)
                current_ind = np.zeros(129, dtype=np.float64)
                for a in current_A:
                    current_ind[a] = 1.0
                current_reps = calc_reps_from_ind(current_ind)
                current_peak = get_peak(current_reps)
            stuck_count = 0
            T = max(T, 0.5)
    
    # Phase 4: Final greedy peeling polish
    if best_A is not None:
        final_set = set(best_A)
        final_ind = np.zeros(129, dtype=np.float64)
        for a in final_set:
            final_ind[a] = 1.0
        final_reps = calc_reps_from_ind(final_ind)
        
        if is_covered(final_reps):
            changed = True
            while changed and time.time() < deadline:
                changed = False
                elements = list(final_set)
                random.shuffle(elements)
                for a in elements:
                    if a not in final_set:
                        continue
                    final_ind[a] = 0.0
                    test_reps = calc_reps_from_ind(final_ind)
                    if is_covered(test_reps):
                        test_peak = get_peak(test_reps)
                        if test_peak <= get_peak(final_reps):
                            final_set.remove(a)
                            final_reps = test_reps
                            changed = True
                        else:
                            final_ind[a] = 1.0
                    else:
                        final_ind[a] = 1.0
            
            if get_peak(final_reps) < best_peak:
                best_peak = get_peak(final_reps)
                best_A = list(final_set)
    
    if best_A is None:
        return list(range(64, 129))
        
    return best_A
