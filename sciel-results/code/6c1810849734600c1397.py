import time
import random
import numpy as np
from scipy.signal import fftconvolve

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    random.seed(42)
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1
    MAX_VAL = 128
    DOMAIN_SIZE = 257 # 0..256

    def calc_reps(A):
        """Returns array of counts for sums N_LOW..N_HIGH using unordered pairs."""
        if not A:
            return np.zeros(RANGE_SIZE, dtype=np.int32)
        
        v = np.zeros(DOMAIN_SIZE, dtype=np.float64)
        for x in A:
            v[x] = 1.0
        
        # Ordered pairs count
        c = fftconvolve(v, v, mode='full')
        
        reps = np.zeros(RANGE_SIZE, dtype=np.int32)
        # Pre-compute which sums in range have even s/2 in A?
        # Just check in loop, it's small.
        A_set = A # Already a set
        
        for i, s in enumerate(range(N_LOW, N_HIGH + 1)):
            ord_count = int(c[s])
            eq_count = 1 if (s % 2 == 0 and s//2 in A_set) else 0
            reps[i] = (ord_count + eq_count) // 2
        return reps

    def get_peak_contributions(A, reps):
        """Returns array of size 129 where contrib[x] is the number of peak sums involving x."""
        peak = np.max(reps)
        if peak == 0:
            return np.zeros(MAX_VAL + 1, dtype=np.int32)
        
        M = np.zeros(DOMAIN_SIZE, dtype=np.float64)
        for i, s in enumerate(range(N_LOW, N_HIGH + 1)):
            if reps[i] == peak:
                M[s] = 1.0
        
        vA = np.zeros(DOMAIN_SIZE, dtype=np.float64)
        for x in A:
            vA[x] = 1.0
        
        # Cross correlation C[x] = sum_y M[x+y] vA[y]
        # Using convolution with flipped vA
        vA_flip = vA[::-1]
        corr = fftconvolve(M, vA_flip, mode='full')
        
        # The valid range for x is 0..128
        contribs = corr[:MAX_VAL + 1].astype(np.int32)
        return contribs

    def is_covered(reps):
        return np.all(reps > 0)

    def get_peak(reps):
        return int(np.max(reps))

    def find_peak_elements(A, reps):
        """Find elements that contribute most to peak sums."""
        contribs = get_peak_contributions(A, reps)
        # Return elements sorted by contribution (descending)
        # Only consider elements in A
        A_list = list(A)
        scores = {x: contribs[x] for x in A_list}
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
        for x in [1, 2, 3, 4, 124, 125, 126, 127]:
            A.add(x)
        for x in [64, 63, 65]:
            A.add(x)
        while len(A) < 45:
            A.add(random.randint(0, 128))
        return A

    def repair_coverage(A):
        A = set(A)
        reps = calc_reps(A)
        tries = 0
        while not is_covered(reps) and tries < 100:
            missing = np.where(reps == 0)[0] + N_LOW
            best_candidate = None
            best_gain = 0
            for s in missing:
                candidates = set()
                for x in range(0, 129):
                    y = s - x
                    if 0 <= y <= 128:
                        candidates.add(x)
                candidates -= A
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
                A.add(random.randint(0, 128))
                reps = calc_reps(A)
            tries += 1
        return A, reps

    num_restarts = 50 # Increased due to speedup
    
    for r in range(num_restarts):
        if time.time() > deadline: break
        
        current_A = random_init()
        current_A, current_reps = repair_coverage(current_A)
        
        if not is_covered(current_reps):
            continue

        update_best(current_A, current_reps)
        
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
                # Remove
                peak_elems = find_peak_elements(current_A, current_reps)
                if peak_elems:
                    # Weighted selection
                    contribs = get_peak_contributions(current_A, current_reps)
                    weights = []
                    for x in peak_elems:
                        weights.append(contribs[x] + 1)
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
                # Swap
                peak_elems = find_peak_elements(current_A, current_reps)
                if peak_elems:
                    item_to_remove = peak_elems[0]
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
                # Targeted swap
                peak_elems = find_peak_elements(current_A, current_reps)
                if not peak_elems:
                    stuck_counter += 1
                    T *= cooling
                    continue
                    
                item_to_remove = peak_elems[0]
                
                low_cov_sums = np.where(current_reps <= 2)[0] + N_LOW
                if len(low_cov_sums) == 0:
                    stuck_counter += 1
                    T *= cooling
                    continue
                
                found = False
                for _ in range(10):
                    s = random.choice(low_cov_sums.tolist())
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
            
    # Final Polish
    if best_A:
        A = set(best_A)
        reps = calc_reps(A)
        
        # Greedy removal
        changed = True
        while changed and time.time() < deadline:
            changed = False
            contribs = get_peak_contributions(A, reps)
            # Sort by contribution descending
            sorted_A = sorted(A, key=lambda x: contribs[x], reverse=True)
            for x in sorted_A:
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
        
        # Targeted swaps
        if time.time() < deadline:
            for _ in range(500):
                if time.time() > deadline:
                    break
                
                peak = get_peak(reps)
                peak_elems = find_peak_elements(A, reps)
                if not peak_elems:
                    break
                
                improved = False
                for x in peak_elems[:5]:
                    if time.time() > deadline:
                        break
                    for _ in range(10):
                        new_item = random.randint(0, 128)
                        if new_item == x:
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
                    if len(A) > 20:
                        contribs = get_peak_contributions(A, reps)
                        low_contrib = [x for x in A if contribs[x] == 0]
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
