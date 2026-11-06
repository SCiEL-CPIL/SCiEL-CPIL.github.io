import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 82  # 8s safety margin before 90s kill
    
    random.seed(42)
    
    n = 128
    target_low = 128
    target_high = 256
    num_targets = target_high - target_low + 1  # 129 targets
    
    def compute_reps_full(A_list):
        """Compute representation counts for all targets [128,256].
        A_list is a sorted list of set elements.
        Returns reps array of length 129 (index 0 = target 128, index 128 = target 256)."""
        reps = np.zeros(num_targets, dtype=np.int32)
        m = len(A_list)
        for i in range(m):
            ai = A_list[i]
            for j in range(i, m):
                s = ai + A_list[j]
                if target_low <= s <= target_high:
                    reps[s - target_low] += 1
        return reps
    
    def get_peak(reps):
        """Get peak multiplicity, return inf if any target uncovered."""
        if np.any(reps == 0):
            return float('inf')
        return int(np.max(reps))
    
    def incremental_swap_update(A_list, reps, old_val, new_val):
        """Compute new reps after swapping old_val with new_val in A_list.
        A_list is sorted. Returns (new_reps, new_peak)."""
        # Find index of old_val
        idx = A_list.index(old_val)
        
        # Build new list
        A_new = A_list[:]
        A_new[idx] = new_val
        A_new.sort()
        
        # Incremental update:
        # Remove contributions of old_val with all elements
        # Add contributions of new_val with all elements
        new_reps = reps.copy()
        
        # For each element a in A_new (which includes new_val, not old_val):
        # We need to handle pairs carefully since we're doing unordered pairs with i<=j
        
        # Strategy: recompute from scratch but only for affected pairs
        # Actually, let's just do a full recompute but it's O(m^2) which for m~40-60 is fast
        # m^2/2 = 800-1800 operations, very fast in Python
        
        # Better: incremental
        # Old pairs involving old_val: (old_val, a) for all a in A_list where a >= old_val (since sorted, i<=j)
        # New pairs involving new_val: (new_val, a) for all a in A_new where a >= new_val
        
        # Remove old_val's contributions
        for a in A_list:
            s = old_val + a
            if target_low <= s <= target_high:
                new_reps[s - target_low] -= 1
            # Handle the case where old_val == a (double counted in full recompute)
            # In full recompute with i<=j, when i==j we count once
            # When we iterate "for a in A_list", we count (old_val, a) once for each a
            # But if a == old_val, that's the (old_val, old_val) pair counted once
            # This is correct: we're removing all pairs that include old_val
        
        # Add new_val's contributions
        for a in A_new:
            s = new_val + a
            if target_low <= s <= target_high:
                new_reps[s - target_low] += 1
        
        # Check for negative values (shouldn't happen if logic is correct)
        if np.any(new_reps < 0):
            # Fallback: full recompute
            new_reps = compute_reps_full(A_new)
        
        new_peak = get_peak(new_reps)
        return A_new, new_reps, new_peak
    
    best_A = None
    best_peak = float('inf')
    
    # Try various sizes
    for size in [30, 35, 40, 45, 50]:
        if time.time() > deadline:
            break
        
        # Multiple restarts for each size
        max_restarts = 15 if size <= 40 else 8
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialize: random sample
            A = sorted(random.sample(range(0, n + 1), size))
            reps = compute_reps_full(A)
            peak = get_peak(reps)
            
            if peak < best_peak:
                best_peak = peak
                best_A = list(A)
            
            # Local search with incremental updates
            A_set = set(A)
            
            for iteration in range(5000):
                if time.time() > deadline:
                    break
                
                # Random swap: pick one element to remove, one to add
                old_val = random.choice(A)
                new_val = random.randint(0, n)
                
                if new_val == old_val or new_val in A_set:
                    continue
                
                # Try the swap with incremental update
                A_new, new_reps, new_peak = incremental_swap_update(A, reps, old_val, new_val)
                
                if new_peak < peak:
                    peak = new_peak
                    A = A_new
                    reps = new_reps
                    A_set = set(A)
                    if peak < best_peak:
                        best_peak = peak
                        best_A = list(A)
                elif new_peak == peak and random.random() < 0.2:
                    A = A_new
                    reps = new_reps
                    A_set = set(A)
    
    # Additional phase: try to reduce size of best_A found
    if best_A is not None:
        if time.time() < deadline - 5:
            # Try removing elements
            A = list(best_A)
            reps = compute_reps_full(A)
            peak = get_peak(reps)
            
            for attempt in range(300):
                if time.time() > deadline:
                    break
                if len(A) <= 15:
                    break
                
                # Try removing a random element
                idx = random.randint(0, len(A) - 1)
                A_trial = A[:idx] + A[idx+1:]
                new_reps = compute_reps_full(A_trial)
                new_peak = get_peak(new_reps)
                
                if new_peak <= peak:
                    A = A_trial
                    reps = new_reps
                    peak = new_peak
                    if peak < best_peak:
                        best_peak = peak
                        best_A = list(A)
    
    if best_A is None:
        # Fallback: a guaranteed valid set
        return list(range(64, 129))
    
    return best_A
