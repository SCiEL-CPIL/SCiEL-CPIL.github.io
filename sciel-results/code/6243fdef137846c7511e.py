import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1  # 129
    MAX_VAL = 128
    
    def calc_reps(A):
        """Compute representation counts for sums N_LOW..N_HIGH using unordered pairs."""
        reps = np.zeros(RANGE_SIZE, dtype=np.int32)
        A_list = np.array(sorted(A), dtype=np.int32)
        n = len(A_list)
        if n == 0:
            return reps
        # Use outer product approach: compute all i+j for i<=j
        # For efficiency, we process in chunks to avoid memory issues for large n
        # But n is at most ~80, so n*(n+1)/2 <= 3240, which is fine
        i_idx = np.arange(n)
        j_idx = np.arange(n)
        # Create upper triangle indices (i <= j)
        tri_i, tri_j = np.triu_indices(n)
        sums = A_list[tri_i] + A_list[tri_j]
        mask = (sums >= N_LOW) & (sums <= N_HIGH)
        if np.any(mask):
            indices = sums[mask] - N_LOW
            np.add.at(reps, indices, 1)
        return reps
    
    def is_covered(reps):
        return np.all(reps > 0)
    
    def get_peak(reps):
        return int(np.max(reps))
    
    # --- Phase 1: Construct a structured initial candidate ---
    # We need A ⊆ [0,128] such that A+A (unordered, with repetition) covers [128,256].
    # Strategy: Use a grid-like structure to spread sums evenly.
    
    best_A = None
    best_peak = float('inf')
    
    # Try several structured constructions
    # Construction 1: Arithmetic progression
    # If A = {a, a+d, a+2d, ..., a+kd}, then A+A covers [2a, 2a+2kd] with step d
    # We need 2a <= 128 and 2a+2kd >= 256, and d must divide the gaps properly.
    
    # For d=1: A = {a, a+1, ..., a+k}. Sums cover [2a, 2a+2k] continuously.
    # Need 2a <= 128 and 2a+2k >= 256 => a+k >= 128-a => k >= 128-2a
    # To minimize size, maximize a. If a=64, k>=0, but we need enough elements for coverage.
    # Actually with d=1 and a=64, A={64,...,64+k}, sums cover [128, 128+2k].
    # Need 128+2k >= 256 => k >= 64. So A = {64,...,128}, size 65.
    # Peak multiplicity: for sum s in [128,256], # of pairs (i,j) with i<=j, i+j=s, i,j in A
    # This is a classic problem. For A = {64,...,128}, sum s has:
    #   #pairs = number of i in [64,128] such that i <= s-i and s-i in [64,128]
    #   i.e., 64 <= i <= min(128, s-64) and i <= s-i => i <= s/2
    #   So i ranges from 64 to min(128, floor(s/2), s-64)
    # For s=128: i=64 only (64+64), count=1
    # For s=192: i from 64 to 96, count=33
    # For s=256: i=128 only (128+128), count=1
    # Peak is around s=192, count ~ 33. That's high.
    
    # Construction 2: Two interleaved progressions
    # A = even numbers in some range + odd numbers in some range?
    # Better: Use a set that's "balanced" in the sense that sums are spread out.
    
    # Let's try: A = {0, 128} ∪ {x : x in [1, 127] with some pattern}
    # Actually, let's use a greedy approach: start with required elements, then add elements
    # that best balance the representation counts.
    
    # Greedy construction:
    # Start with A = {0, 128} (needed for edges)
    # Then iteratively add the element that most improves coverage/balance
    
    A_current = {0, 128}
    reps = calc_reps(A_current)
    
    # We need to cover [128, 256]. Currently only 128 and 256 are covered.
    # Add elements one by one to cover all targets.
    
    uncovered = set(range(N_LOW, N_HIGH + 1))
    for s in range(N_LOW, N_HIGH + 1):
        if reps[s - N_LOW] > 0:
            uncovered.discard(s)
    
    # Greedily add elements
    while len(uncovered) > 0 and len(A_current) < 80:  # cap size
        # For each candidate x in [0,128], compute how many new sums it would cover
        best_x = None
        best_new_covered = 0
        best_reps = None
        best_peak_val = float('inf')
        
        candidates = list(range(0, MAX_VAL + 1))
        # Shuffle for randomness in ties
        random.shuffle(candidates)
        
        for x in candidates:
            if x in A_current:
                continue
            new_A = A_current | {x}
            new_reps = calc_reps(new_A)
            
            if not is_covered(new_reps):
                # Count how many newly covered
                new_covered_count = 0
                for s in range(N_LOW, N_HIGH + 1):
                    if reps[s - N_LOW] == 0 and new_reps[s - N_LOW] > 0:
                        new_covered_count += 1
                # Also check if it helps reduce peak among covered
                peak = get_peak(new_reps)
                
                if new_covered_count > best_new_covered or (new_covered_count == best_new_covered and peak < best_peak_val):
                    best_new_covered = new_covered_count
                    best_x = x
                    best_reps = new_reps
                    best_peak_val = peak
            else:
                # Fully covered! Great
                peak = get_peak(new_reps)
                if peak < best_peak_val or best_reps is None:
                    best_x = x
                    best_reps = new_reps
                    best_peak_val = peak
                    best_new_covered = 1000  # priority
        
        if best_x is None:
            break
        
        A_current.add(best_x)
        reps = best_reps if best_reps is not None else calc_reps(A_current)
        
        for s in range(N_LOW, N_HIGH + 1):
            if reps[s - N_LOW] > 0:
                uncovered.discard(s)
        
        # Early exit if covered
        if len(uncovered) == 0:
            break
    
    if is_covered(reps):
        peak = get_peak(reps)
        if peak < best_peak:
            best_peak = peak
            best_A = list(A_current)
    else:
        # Fallback
        A_current = set(range(64, 129))
        reps = calc_reps(A_current)
        if is_covered(reps):
            best_peak = get_peak(reps)
            best_A = list(A_current)
    
    # --- Phase 2: Try to improve by removing elements ---
    if best_A is not None:
        current_A = set(best_A)
        current_reps = calc_reps(current_A)
        current_peak = get_peak(current_reps)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try removing each element
            for x in list(current_A):
                test_A = current_A - {x}
                if len(test_A) == 0:
                    continue
                test_reps = calc_reps(test_A)
                if is_covered(test_reps):
                    test_peak = get_peak(test_reps)
                    if test_peak < current_peak:
                        current_A = test_A
                        current_reps = test_reps
                        current_peak = test_peak
                        improved = True
                        break
                    elif test_peak == current_peak:
                        # Keep it if same peak but smaller set (fewer total pairs)
                        current_A = test_A
                        current_reps = test_reps
                        improved = True
                        break
        
        if is_covered(current_reps):
            peak = get_peak(current_reps)
            if peak < best_peak:
                best_peak = peak
                best_A = list(current_A)
    
    # --- Phase 3: Local search to balance representations ---
    if best_A is not None and time.time() < deadline - 10:
        current_A = set(best_A)
        current_reps = calc_reps(current_A)
        current_peak = get_peak(current_reps)
        
        T = 1.0
        cooling_rate = 0.9995
        min_T = 0.05
        
        while time.time() < deadline - 5 and T > min_T:
            if len(current_A) == 0:
                break
            
            item_to_remove = random.choice(list(current_A))
            new_item = random.randint(0, 128)
            
            if new_item == item_to_remove:
                T *= cooling_rate
                continue
            
            new_A = set(current_A)
            new_A.remove(item_to_remove)
            new_A.add(new_item)
            
            if len(new_A) == 0:
                T *= cooling_rate
                continue
            
            new_reps = calc_reps(new_A)
            
            if is_covered(new_reps):
                new_peak = get_peak(new_reps)
                delta = new_peak - current_peak
                
                if delta < 0:
                    current_A = new_A
                    current_reps = new_reps
                    current_peak = new_peak
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = list(current_A)
                else:
                    prob = np.exp(-delta / max(T, 0.001))
                    if random.random() < prob:
                        current_A = new_A
                        current_reps = new_reps
                        current_peak = new_peak
            else:
                # Penalize uncovered: reject
                pass
            
            T *= cooling_rate
        
        # Try removal after local search
        if time.time() < deadline - 5:
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for x in list(current_A):
                    test_A = current_A - {x}
                    if len(test_A) == 0:
                        continue
                    test_reps = calc_reps(test_A)
                    if is_covered(test_reps):
                        test_peak = get_peak(test_reps)
                        if test_peak < current_peak:
                            current_A = test_A
                            current_reps = test_reps
                            current_peak = test_peak
                            improved = True
                            break
                        elif test_peak == current_peak:
                            current_A = test_A
                            current_reps = test_reps
                            improved = True
                            break
            
            if is_covered(current_reps):
                peak = get_peak(current_reps)
                if peak < best_peak:
                    best_peak = peak
                    best_A = list(current_A)
    
    if best_A is None:
        return list(range(64, 129))
    
    return best_A
