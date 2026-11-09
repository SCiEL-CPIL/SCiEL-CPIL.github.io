import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin for 90s limit
    
    N_LOW = 128
    N_HIGH = 256
    MAX_VAL = 128
    
    # Precompute all possible pairs (a, b) with 0 <= a <= b <= 128
    # There are (129*130)/2 = 8385 pairs
    # This allows O(1) lookup for pair contributions if needed, but we'll use vectorized updates.
    
    def calc_reps(A_set):
        """Calculate reps for a given set A. Returns array of length 257."""
        reps = np.zeros(257, dtype=np.int32)
        A_list = np.array(list(A_set), dtype=np.int32)
        n = len(A_list)
        if n == 0:
            return reps
        # Use broadcasting to compute all sums
        # This creates an n x n matrix, which for n=60 is 3600 ints. Very fast.
        sums = A_list[:, None] + A_list[None, :]
        # We only care about unordered pairs with a <= b, which corresponds to upper triangle including diagonal
        # But np.bincount on the whole matrix counts ordered pairs.
        # If we count ordered pairs, reps_ordered[s] = 2 * reps_unordered[s] for s != 2a, and reps_ordered[2a] = reps_unordered[2a] + 1 (if a in A)
        # Actually, let's just count ordered pairs and divide? No, that's messy for integer division.
        # Better: use the upper triangle.
        iu = np.triu_indices(n)
        upper_sums = sums[iu]
        # Count occurrences of each sum in upper_sums
        # np.bincount is efficient
        reps = np.bincount(upper_sums, minlength=257)
        return reps.astype(np.int32)

    def get_objective(reps):
        """Calculate objective: penalty for uncovered + peak."""
        target_reps = reps[N_LOW:N_HIGH+1]
        uncovered = np.sum(target_reps == 0)
        peak = np.max(target_reps) if len(target_reps) > 0 else 0
        return uncovered * 10000 + int(peak)

    def is_valid(reps):
        return np.all(reps[N_LOW:N_HIGH+1] > 0)

    def get_peak(reps):
        return int(np.max(reps[N_LOW:N_HIGH+1]))

    # --- Phase 1: Greedy Construction ---
    # Goal: Build a set A that covers [128, 256] with minimal peak.
    # Strategy: Start with empty set. At each step, add the element that maximizes coverage of currently uncovered sums, or minimizes peak if all covered.
    
    A = set()
    reps = np.zeros(257, dtype=np.int32)
    
    # To speed up greedy, we can precompute for each candidate x, which sums it covers when added to current A.
    # Adding x to A adds pairs (x, y) for all y in A, and (x, x).
    
    # We will iterate until all sums in [128, 256] are covered.
    # To avoid getting stuck with a very high peak, we might stop greedy once coverage is complete, then refine.
    
    # Initial seed: Add 0 and 128 to help extremes?
    # 128 is needed for 256 (128+128). 0 is good for 128 (0+128).
    # Let's start with a small random set to break symmetry, or just start empty.
    
    # Let's try a few random initial seeds for the greedy phase to find a good starting point.
    best_A_greedy = None
    best_obj_greedy = float('inf')
    
    random.seed(42)
    
    for restart in range(20):
        if time.time() > deadline - 30:
            break
            
        # Random start
        A = set()
        # Add a few random elements
        for _ in range(5):
            A.add(random.randint(0, 128))
        # Ensure 128 is in A to cover 256 easily?
        # If 128 not in A, we need a+b=256 with a,b <= 128. Only (128,128) works. So 128 MUST be in A.
        A.add(128)
        
        reps = calc_reps(A)
        curr_obj = get_objective(reps)
        
        # Greedy expansion
        while not is_valid(reps) and time.time() < deadline - 30:
            best_candidate = None
            best_score = float('inf')
            
            # Evaluate all candidates not in A
            # To speed this up, we can't just try all 128 candidates with full recalculation each time if we do many steps.
            # But greedy steps are limited (maybe 30-50 steps). 128 * 50 = 6400 calcs.
            # Each calc is O(|A|^2). |A| ~ 30. 900 ops. 6400 * 900 = 5.7M ops. Very fast.
            
            candidates = [x for x in range(129) if x not in A]
            
            for x in candidates:
                # Calculate new reps if we add x
                # New pairs: (x, y) for y in A, and (x, x)
                # We can update a copy of reps
                # Actually, let's just compute the delta.
                # Delta for sum s: count of y in A such that x+y=s. Plus 1 if x+x=s.
                
                # Efficient delta calculation:
                # For each y in A, s = x+y.
                # We can just do a full recalculation for simplicity in the inner loop if it's fast enough.
                # Let's try full recalculation first.
                
                new_A = set(A)
                new_A.add(x)
                new_reps = calc_reps(new_A)
                
                # Objective for greedy:
                # Primary: Minimize uncovered
                # Secondary: Minimize peak
                # Tertiary: Maximize "spread" or something?
                
                target_reps = new_reps[N_LOW:N_HIGH+1]
                uncovered = np.sum(target_reps == 0)
                peak = np.max(target_reps)
                
                # Score: prioritize covering new sums, then keeping peak low
                # If uncovered < current_uncovered, it's good.
                # Let's use a lexicographic order: (uncovered, peak)
                # But we need to compare with current state.
                
                curr_target = reps[N_LOW:N_HIGH+1]
                curr_uncovered = np.sum(curr_target == 0)
                curr_peak = np.max(curr_target)
                
                # Score to minimize
                # If we reduce uncovered, great.
                # If uncovered is same, minimize peak.
                # If peak is same, maybe maximize min reps (robustness)?
                
                score = (uncovered, peak)
                
                if score < best_score:
                    best_score = score
                    best_candidate = x
                    best_new_reps = new_reps
            
            if best_candidate is None:
                break
                
            A.add(best_candidate)
            reps = best_new_reps
            
            # Check if we improved
            curr_target = reps[N_LOW:N_HIGH+1]
            curr_uncovered = np.sum(curr_target == 0)
            curr_peak = np.max(curr_target)
            curr_obj = curr_uncovered * 10000 + curr_peak
            
            if curr_obj < best_obj_greedy:
                best_obj_greedy = curr_obj
                best_A_greedy = set(A)
                
            # Early stop if we have a very good peak?
            # If peak is 2, it's likely optimal or close.
            if is_valid(reps) and get_peak(reps) <= 3:
                break

    if best_A_greedy is None:
        # Fallback if greedy failed completely (shouldn't happen)
        return list(range(0, 129, 2))

    # --- Phase 2: Targeted Local Search ---
    # Start from the best greedy solution
    A = set(best_A_greedy)
    reps = calc_reps(A)
    curr_peak = get_peak(reps)
    
    if not is_valid(reps):
        # If greedy didn't finish, use SA to finish
        # But greedy should always finish.
        pass

    # Local Search Strategy:
    # 1. Identify peak sums.
    # 2. Identify elements that contribute to peak sums.
    # 3. Try swapping such elements with others that do NOT contribute to peak sums, or contribute to less critical sums.
    # 4. Accept if peak decreases.
    # 5. If peak doesn't decrease, accept if peak stays same and "variance" decreases or random move to escape.
    
    best_A_final = set(A)
    best_peak_final = curr_peak
    
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    while time.time() < deadline - 5 and T > min_T:
        # Find peak sums
        target_reps = reps[N_LOW:N_HIGH+1]
        peak_val = np.max(target_reps)
        peak_sums = np.where(target_reps == peak_val)[0] + N_LOW
        
        # If no peak (shouldn't happen if valid), break
        if len(peak_sums) == 0:
            break
            
        # Pick a random peak sum
        s = random.choice(peak_sums)
        
        # Find pairs (a, b) in A such that a + b = s
        # We want to remove one of these elements.
        # To be efficient, we can precompute for each sum s, the list of pairs.
        # But A changes. Let's just scan A.
        
        candidate_pairs = []
        A_list = list(A)
        for i, a in enumerate(A_list):
            b = s - a
            if b in A:
                # Ensure unordered: a <= b
                if a <= b:
                    candidate_pairs.append((a, b))
        
        if not candidate_pairs:
            # Should not happen if s is a peak sum in reps
            continue
            
        # Pick a random pair from candidate_pairs
        u, v = random.choice(candidate_pairs)
        
        # We will try to swap u with a new element w.
        # w should be chosen to reduce the count of s, or at least not increase it.
        # Ideally, w should not form pairs with other elements in A that sum to peak values.
        
        # Strategy for w:
        # Random w from [0, 128] \ A.
        # Or, bias w towards values that are "far" from other elements to avoid creating new peaks?
        # Let's just try random w first.
        
        # Try a few random w's
        found_improvement = False
        for _ in range(5):
            w = random.randint(0, 128)
            if w in A:
                continue
                
            # Perform swap: remove u, add w
            new_A = set(A)
            new_A.remove(u)
            new_A.add(w)
            
            new_reps = calc_reps(new_A)
            
            if not is_valid(new_reps):
                continue
                
            new_peak = get_peak(new_reps)
            
            # Objective: Minimize peak.
            # If new_peak < curr_peak, accept.
            # If new_peak == curr_peak, accept with probability based on T (to explore).
            # If new_peak > curr_peak, reject (unless T is high and we want to escape, but let's be conservative).
            
            if new_peak < curr_peak:
                A = new_A
                reps = new_reps
                curr_peak = new_peak
                found_improvement = True
                break
            elif new_peak == curr_peak:
                # Check if this move reduces the number of peak sums or something?
                # For now, accept with prob 0.5 to allow wandering
                if random.random() < 0.5:
                    A = new_A
                    reps = new_reps
                    # curr_peak unchanged
                    found_improvement = True
                    break
            else:
                # new_peak > curr_peak
                # Accept with very low probability if T is high?
                # Let's try to escape local minima.
                if random.random() < np.exp(-(new_peak - curr_peak) / T):
                    A = new_A
                    reps = new_reps
                    curr_peak = new_peak
                    found_improvement = True
                    break
        
        if found_improvement:
            # Update best
            if curr_peak < best_peak_final:
                best_peak_final = curr_peak
                best_A_final = set(A)
            # Reset temperature on improvement? Or keep cooling?
            # Let's keep cooling but maybe reset T slightly to explore around new peak.
            # T = max(T, 0.5)
            pass
        else:
            # No improvement found in this iteration
            pass
            
        T *= cooling

    # Final Check
    if best_A_final is None:
        return list(A)
        
    return list(best_A_final)

# Test run
if __name__ == "__main__":
    import time
    start = time.time()
    A = construct()
    end = time.time()
    print(f"Time: {end-start:.2f}s")
    print(f"Set A: {sorted(A)}")
    print(f"Size: {len(A)}")
    
    # Verify
    reps = np.zeros(257, dtype=np.int32)
    A_list = np.array(sorted(A), dtype=np.int32)
    n = len(A_list)
    sums = A_list[:, None] + A_list[None, :]
    iu = np.triu_indices(n)
    upper_sums = sums[iu]
    reps = np.bincount(upper_sums, minlength=257).astype(np.int32)
    
    target_reps = reps[128:257]
    print(f"Covered: {np.all(target_reps > 0)}")
    print(f"Peak: {np.max(target_reps)}")
    print(f"Min: {np.min(target_reps)}")
