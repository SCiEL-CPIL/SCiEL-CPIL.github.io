import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85

    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1
    MAX_VAL = 128

    def calc_reps_fast(A_set):
        """Fast representation count using FFT convolution."""
        if not A_set:
            return np.zeros(RANGE_SIZE, dtype=np.int32)
        A_arr = np.array(sorted(A_set), dtype=np.int32)
        n = len(A_arr)
        # Unordered pairs: use triu
        idx = np.triu_indices(n)
        upper_sums = A_arr[idx[0]] + A_arr[idx[1]]
        mask = (upper_sums >= N_LOW) & (upper_sums <= N_HIGH)
        if not np.any(mask):
            return np.zeros(RANGE_SIZE, dtype=np.int32)
        valid_sums = upper_sums[mask]
        indices = (valid_sums - N_LOW).astype(np.int32)
        counts = np.bincount(indices, minlength=RANGE_SIZE)
        return counts[:RANGE_SIZE].astype(np.int32)

    def is_covered(reps):
        return np.all(reps > 0)

    def get_peak(reps):
        return int(np.max(reps))

    def peak_contrib_map(A_set, reps):
        """For each element, count how many times it appears in peak-representation sums."""
        peak = get_peak(reps)
        A_list = sorted(A_set)
        contrib = {}
        for x in A_list:
            contrib[x] = 0
        A_arr = np.array(A_list, dtype=np.int32)
        n = len(A_arr)
        idx = np.triu_indices(n)
        upper_sums = A_arr[idx[0]] + A_arr[idx[1]]
        mask = (upper_sums >= N_LOW) & (upper_sums <= N_HIGH) & (upper_sums - N_LOW >= 0)
        valid_mask = mask & (reps[upper_sums[mask] - N_LOW] == peak)
        if np.any(valid_mask):
            for k in range(np.sum(valid_mask)):
                i = idx[0][mask][k]
                j = idx[1][mask][k]
                contrib[A_list[i]] += 1
                contrib[A_list[j]] += 1
        return contrib

    def greedy_minimax_peel(A_set, reps):
        """Iteratively remove element with highest peak contribution if coverage maintained."""
        A_set = set(A_set)
        reps = calc_reps_fast(A_set)
        if not is_covered(reps):
            return A_set, reps
        peak = get_peak(reps)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            contrib = peak_contrib_map(A_set, reps)
            if not contrib:
                break
            # Sort by contribution descending, then by value for determinism
            sorted_elems = sorted(contrib.keys(), key=lambda x: (-contrib[x], x))
            for x in sorted_elems:
                if time.time() > deadline:
                    break
                if x not in A_set:
                    continue
                new_A = A_set - {x}
                if not new_A:
                    continue
                new_reps = calc_reps_fast(new_A)
                if is_covered(new_reps):
                    new_peak = get_peak(new_reps)
                    if new_peak <= peak:
                        A_set = new_A
                        reps = new_reps
                        peak = new_peak
                        improved = True
                        break
        return A_set, reps

    def targeted_local_search(A_set, reps):
        """Local search: try swaps and removals that reduce peak or maintain with fewer elements."""
        A_set = set(A_set)
        reps = calc_reps_fast(A_set)
        peak = get_peak(reps)
        best_A = set(A_set)
        best_reps = reps.copy()
        best_peak = peak
        best_size = len(A_set)
        random.seed(42)
        
        no_improve = 0
        max_no_improve = 200
        
        while time.time() < deadline and no_improve < max_no_improve:
            no_improve += 1
            current_peak = get_peak(reps)
            
            # Try removal
            if len(A_set) > 10:
                contrib = peak_contrib_map(A_set, reps)
                if contrib:
                    # Pick element with highest contribution
                    candidates = sorted(contrib.keys(), key=lambda x: -contrib[x])[:5]
                    random.shuffle(candidates)
                    for x in candidates:
                        if x not in A_set:
                            continue
                        new_A = A_set - {x}
                        if not new_A:
                            continue
                        new_reps = calc_reps_fast(new_A)
                        if is_covered(new_reps):
                            new_peak = get_peak(new_reps)
                            if new_peak < current_peak or (new_peak == current_peak and len(new_A) < len(A_set)):
                                A_set = new_A
                                reps = new_reps
                                no_improve = 0
                                break
            
            # Try swap
            if len(A_set) > 5:
                a = random.choice(list(A_set))
                b = random.randint(0, MAX_VAL)
                if b not in A_set:
                    new_A = set(A_set)
                    new_A.discard(a)
                    new_A.add(b)
                    new_reps = calc_reps_fast(new_A)
                    if is_covered(new_reps):
                        new_peak = get_peak(new_reps)
                        if new_peak < current_peak:
                            A_set = new_A
                            reps = new_reps
                            no_improve = 0
                        elif new_peak == current_peak and random.random() < 0.2:
                            A_set = new_A
                            reps = new_reps
        
        # Track best
        if is_covered(reps):
            p = get_peak(reps)
            if p < best_peak or (p == best_peak and len(A_set) < best_size):
                best_A = set(A_set)
                best_reps = reps.copy()
                best_peak = p
                best_size = len(A_set)
        
        return best_A, best_reps

    def repair_coverage(A_set):
        """Add elements until all sums in [128,256] are covered."""
        A_set = set(A_set)
        reps = calc_reps_fast(A_set)
        tries = 0
        while not is_covered(reps) and tries < 200:
            missing = np.where(reps == 0)[0] + N_LOW
            best_candidate = None
            best_gain = 0
            for s in missing[:10]:  # Limit to first 10 missing for speed
                for x in range(0, 129):
                    y = s - x
                    if 0 <= y <= 128 and x not in A_set:
                        new_A = A_set | {x}
                        new_reps = calc_reps_fast(new_A)
                        new_missing = np.sum(new_reps == 0)
                        gain = len(missing) - new_missing
                        if gain > best_gain:
                            best_gain = gain
                            best_candidate = x
            if best_candidate is not None:
                A_set.add(best_candidate)
                reps = calc_reps_fast(A_set)
            else:
                A_set.add(random.randint(0, 128))
                reps = calc_reps_fast(A_set)
            tries += 1
        return A_set, reps

    best_A = None
    best_peak = float('inf')
    best_size = float('inf')

    def update_best(A_set, reps):
        nonlocal best_A, best_peak, best_size
        if is_covered(reps):
            peak = get_peak(reps)
            size = len(A_set)
            if peak < best_peak or (peak == best_peak and size < best_size):
                best_peak = peak
                best_size = size
                best_A = set(A_set)

    # Phase 1: Structured initialization + minimax peel
    # Start with a structured set that has good coverage properties
    init_sets = [
        set(range(0, 129, 4)),           # Every 4th number
        set(range(0, 129, 3)),           # Every 3rd number
        set(range(0, 129, 2)),           # Every 2nd number
        set(range(16, 129, 4)),          # Offset every 4th
        set(range(8, 129, 4)),           # Offset every 4th
        set(list(range(0, 64, 2)) + list(range(64, 129, 2))),  # Even numbers
        set(list(range(0, 64, 3)) + list(range(64, 129, 3))),  # Every 3rd
    ]
    
    for init_set in init_sets:
        if time.time() > deadline:
            break
        A_set, reps = repair_coverage(set(init_set))
        if not is_covered(reps):
            continue
        A_set, reps = greedy_minimax_peel(A_set, reps)
        update_best(A_set, reps)
        A_set, reps = targeted_local_search(A_set, reps)
        update_best(A_set, reps)

    # Phase 2: Randomized starts with minimax peel
    random.seed(123)
    num_random_starts = 20
    for r in range(num_random_starts):
        if time.time() > deadline:
            break
        A_set = set()
        # Random initialization with some structure
        A_set.add(0)
        A_set.add(128)
        for _ in range(random.randint(20, 40)):
            A_set.add(random.randint(0, 128))
        
        A_set, reps = repair_coverage(A_set)
        if not is_covered(reps):
            continue
        A_set, reps = greedy_minimax_peel(A_set, reps)
        update_best(A_set, reps)
        A_set, reps = targeted_local_search(A_set, reps)
        update_best(A_set, reps)

    # Phase 3: Final polish on best
    if best_A is not None:
        best_A, _ = greedy_minimax_peel(best_A, calc_reps_fast(best_A))
        best_A, _ = targeted_local_search(best_A, calc_reps_fast(best_A))

    if best_A is None:
        return list(range(64, 129))

    return list(best_A)

if __name__ == "__main__":
    t0 = time.time()
    A = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s, |A|={len(A)}")
    reps = [0]*129
    for i in range(len(A)):
        for j in range(i, len(A)):
            s = A[i] + A[j]
            if 128 <= s <= 256:
                reps[s-128] += 1
    print(f"Covered: {all(r>0 for r in reps)}, Peak: {max(reps)}")
