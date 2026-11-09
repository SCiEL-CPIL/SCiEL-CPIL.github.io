import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1
    MAX_VAL = 128
    
    # Precompute pair sums for all possible x in [0, 128]
    # For a given x, the sums it forms with a set A are {x+a | a in A} union {2x}
    # But we need to update counts dynamically.
    
    def calc_reps(A):
        """Compute representation counts for sums N_LOW..N_HIGH."""
        # Use numpy for speed
        ind = np.zeros(MAX_VAL + 1, dtype=np.int32)
        for a in A:
            ind[a] = 1
        conv = np.convolve(ind, ind, mode='full')
        reps = np.zeros(RANGE_SIZE, dtype=np.int32)
        for idx, s in enumerate(range(N_LOW, N_HIGH + 1)):
            ordered = conv[s]
            diag = 1 if (s % 2 == 0 and ind[s // 2] == 1) else 0
            reps[idx] = (ordered + diag) // 2
        return reps

    def is_covered(reps):
        return np.all(reps > 0)

    def get_peak(reps):
        return int(np.max(reps))

    def objective(reps):
        """Lower is better. Penalize uncovered sums heavily."""
        if not is_covered(reps):
            uncovered = np.sum(reps == 0)
            return 1000 + uncovered * 10 + get_peak(reps)
        return get_peak(reps)

    best_peak = float('inf')
    best_A = None
    best_reps = None

    # Phase 1: Try to find a valid cover with small peak via random restarts
    # We hypothesize that a small set (size 20-40) can cover the range with low peak.
    
    num_restarts = 200
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Choose a random size for A
        # Start with smaller sizes to aim for low peak
        min_size = 18
        max_size = 35
        if r > 100:
            min_size = 30
            max_size = 50
            
        size = random.randint(min_size, max_size)
        
        # Random initialization
        # Bias towards elements that help cover edges?
        # 0 and 128 are good for edges 128 and 256.
        # But 0+0=0, 128+128=256.
        # If we include 0, we cover sums 0..128 (if A is dense).
        # We need [128, 256].
        
        A = set()
        # Force inclusion of some strategic points?
        # Let's just do pure random first, then SA.
        
        # To ensure we don't miss the edges, maybe start with {0, 128} and fill rest randomly?
        # Or just random.
        
        # Strategy: Random sample
        A = set(random.sample(range(MAX_VAL + 1), size))
        
        # Ensure we have potential to cover 128 and 256?
        # 128 can be 0+128, 1+127, etc.
        # 256 must be 128+128. So 128 MUST be in A?
        # If 128 is not in A, max sum is < 128+128=256?
        # If max(A) < 128, max sum < 256.
        # So 128 MUST be in A.
        if 128 not in A:
            # Replace a random element with 128
            A.remove(random.choice(list(A)))
            A.add(128)
            
        # Also, to cover 128, we need pairs summing to 128.
        # If 0 is in A, 128 is covered.
        # If not, we need other pairs.
        
        reps = calc_reps(A)
        curr_obj = objective(reps)
        
        if is_covered(reps):
            peak = get_peak(reps)
            if peak < best_peak:
                best_peak = peak
                best_A = list(A)
                best_reps = reps
                # If we get peak 2 or 3, it's excellent. Stop early if peak <= 2?
                if best_peak <= 2:
                    # Maybe try to get 1? Unlikely for this range.
                    # But let's continue a bit to see if 1 is possible.
                    pass

        # Simulated Annealing
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        
        curr_A = set(A)
        curr_reps = reps
        curr_obj = objective(curr_reps)
        
        max_sa_steps = 5000
        steps = 0
        while steps < max_sa_steps and T > min_T:
            if time.time() > deadline - 5:
                break
            
            steps += 1
            
            # Move: Swap, Add, or Remove
            move_type = random.random()
            
            if len(curr_A) == 0:
                break
                
            if move_type < 0.6:
                # Swap
                if len(curr_A) < 2:
                    continue
                old_x = random.choice(list(curr_A))
                new_x = random.randint(0, MAX_VAL)
                if new_x == old_x:
                    continue
                new_A = set(curr_A)
                new_A.remove(old_x)
                new_A.add(new_x)
                if len(new_A) == 0:
                    continue
            elif move_type < 0.8:
                # Add
                new_x = random.randint(0, MAX_VAL)
                if new_x in curr_A:
                    continue
                new_A = set(curr_A)
                new_A.add(new_x)
            else:
                # Remove
                if len(curr_A) <= 10: # Don't remove too many
                    continue
                old_x = random.choice(list(curr_A))
                new_A = set(curr_A)
                new_A.remove(old_x)
                if len(new_A) == 0:
                    continue
            
            new_reps = calc_reps(new_A)
            new_obj = objective(new_reps)
            
            delta = new_obj - curr_obj
            
            if delta < 0 or random.random() < np.exp(-delta / max(T, 0.001)):
                curr_A = new_A
                curr_reps = new_reps
                curr_obj = new_obj
                
                if is_covered(curr_reps):
                    peak = get_peak(curr_reps)
                    if peak < best_peak:
                        best_peak = peak
                        best_A = list(curr_A)
                        best_reps = curr_reps
                        if best_peak <= 2:
                            # Great improvement, restart SA with lower T?
                            # Or just continue.
                            pass
            
            T *= cooling

    # Phase 2: Polish the best found
    if best_A is not None:
        curr_A = set(best_A)
        curr_reps = calc_reps(curr_A)
        curr_peak = get_peak(curr_reps)
        
        # Try removing elements to reduce size (might increase peak, so check)
        # Actually, we want to minimize peak. Removing elements might uncover sums.
        # But if there's redundancy, we can remove.
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for x in list(curr_A):
                test_A = curr_A - {x}
                if len(test_A) == 0:
                    continue
                test_reps = calc_reps(test_A)
                if is_covered(test_reps):
                    test_peak = get_peak(test_reps)
                    if test_peak < curr_peak:
                        curr_A = test_A
                        curr_reps = test_reps
                        curr_peak = test_peak
                        improved = True
                        break
                    elif test_peak == curr_peak:
                        # Prefer smaller set if peak is same
                        curr_A = test_A
                        curr_reps = test_reps
                        improved = True
                        break
        
        if is_covered(curr_reps):
            peak = get_peak(curr_reps)
            if peak < best_peak:
                best_peak = peak
                best_A = list(curr_A)

    # Fallback if no valid set found
    if best_A is None:
        # Use the interval [64, 128] which is guaranteed to cover, though peak is high
        best_A = list(range(64, 129))

    return best_A
