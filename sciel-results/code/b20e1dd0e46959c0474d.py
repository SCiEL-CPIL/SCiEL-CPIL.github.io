import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    random.seed(42)
    
    n = 128
    target_low = 128
    target_high = 256
    num_targets = target_high - target_low + 1  # 129 targets
    
    def compute_reps(A_set):
        """Compute representation counts for all targets [128,256]."""
        reps = np.zeros(num_targets, dtype=np.int32)
        A_list = sorted(A_set)
        m = len(A_list)
        for i in range(m):
            for j in range(i, m):
                s = A_list[i] + A_list[j]
                if target_low <= s <= target_high:
                    reps[s - target_low] += 1
        return reps
    
    def get_peak(reps):
        """Get peak multiplicity, return inf if any target uncovered."""
        if np.any(reps == 0):
            return float('inf')
        return int(np.max(reps))
    
    def compute_contribution(a, A_set):
        """For element a in A_set, compute how many pairs involving a contribute to each target.
        This includes pairs (a,b) for b in A_set with b >= a, plus (a,a).
        Actually, we need to be careful: we count each unordered pair once.
        For element a, its contribution is:
        - For each b in A_set with b > a: pair (a,b) contributes to target a+b
        - Pair (a,a) contributes to target 2a
        But wait, if we're computing the contribution that should be subtracted when removing a,
        we need all pairs that include a. That's:
        - (a, b) for all b in A_set where b > a
        - (a, a)
        - (b, a) for all b in A_set where b < a  -- but these are already counted as (b,a) with b<a
        Actually no. The unordered pair (x,y) with x<=y includes a if either x=a or y=a.
        Pairs involving a: (a, b) for b >= a where b is in A_set, plus (b, a) for b < a where b is in A_set.
        But (b,a) with b<a is the same as (a,b) with a>b. So the pairs involving a are:
        (min(a,b), max(a,b)) for all b in A_set.
        That's |A_set| pairs (including (a,a) when b=a).
        """
        contrib = np.zeros(num_targets, dtype=np.int32)
        for b in A_set:
            s = a + b
            if target_low <= s <= target_high:
                contrib[s - target_low] += 1
        return contrib
    
    best_A = None
    best_peak = float('inf')
    
    # Phase 1: Structured initializations
    initializations = []
    
    # Consecutive blocks
    for start in range(0, 70):
        for length in range(35, 70, 5):
            if start + length <= 129:
                initializations.append(set(range(start, start + length)))
    
    # Arithmetic progressions
    for start in range(0, 20):
        for step in range(1, 5):
            A = set()
            v = start
            while v <= 128:
                A.add(v)
                v += step
            if len(A) >= 30:
                initializations.append(A)
    
    # Random samples
    for _ in range(30):
        size = random.randint(40, 65)
        initializations.append(set(random.sample(range(0, 129), size)))
    
    # Two-block: low and high
    for low_end in range(20, 50, 5):
        for high_start in range(80, 129, 5):
            A = set(range(0, low_end)) | set(range(high_start, 129))
            if len(A) >= 30:
                initializations.append(A)
    
    # Evaluate initializations
    for A_init in initializations:
        if time.time() > deadline:
            break
        reps = compute_reps(A_init)
        peak = get_peak(reps)
        if peak < best_peak:
            best_peak = peak
            best_A = set(A_init)
    
    # Phase 2: Local search with incremental updates
    if best_A is not None:
        A = set(best_A)
        reps = compute_reps(A)
        peak = get_peak(reps)
        
        A_list = list(A)
        
        for iteration in range(100000):
            if time.time() > deadline:
                break
            
            # Random swap
            old_val = random.choice(A_list)
            new_val = random.randint(0, 128)
            
            if new_val == old_val or new_val in A:
                continue
            
            # Compute contributions
            old_contrib = compute_contribution(old_val, A)
            
            # Tentatively swap
            A.remove(old_val)
            A.add(new_val)
            A_list.remove(old_val)
            A_list.append(new_val)
            
            new_contrib = compute_contribution(new_val, A)
            
            new_reps = reps - old_contrib + new_contrib
            new_peak = get_peak(new_reps)
            
            if new_peak <= peak:
                reps = new_reps
                peak = new_peak
                if peak < best_peak:
                    best_peak = peak
                    best_A = set(A)
            else:
                # Revert
                A.remove(new_val)
                A.add(old_val)
                A_list.remove(new_val)
                A_list.append(old_val)
    
    # Phase 3: Try to remove elements
    if best_A is not None and time.time() < deadline - 3:
        A = set(best_A)
        reps = compute_reps(A)
        peak = get_peak(reps)
        
        A_list = list(A)
        
        for attempt in range(5000):
            if time.time() > deadline:
                break
            if len(A) <= 20:
                break
            
            elem = random.choice(A_list)
            contrib = compute_contribution(elem, A)
            
            A.remove(elem)
            A_list.remove(elem)
            
            new_reps = reps - contrib
            new_peak = get_peak(new_reps)
            
            if new_peak <= peak:
                reps = new_reps
                peak = new_peak
                if peak < best_peak:
                    best_peak = peak
                    best_A = set(A)
            else:
                A.add(elem)
                A_list.append(elem)
    
    if best_A is None:
        return list(range(64, 129))
    
    return list(best_A)
