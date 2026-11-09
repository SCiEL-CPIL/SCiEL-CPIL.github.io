import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # 2s margin for 60s limit
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1
    MAX_VAL = 128
    
    # Precompute valid pairs for each sum in [128, 256]
    # valid_pairs[s] is a list of (a, b) such that a+b=s, a<=b, a,b in [0, 128]
    # This helps in repair steps to find candidates that cover a specific missing sum.
    valid_pairs = [[] for _ in range(RANGE_SIZE)]
    for s in range(N_LOW, N_HIGH + 1):
        # a ranges from max(0, s-128) to min(128, s//2)
        a_min = max(0, s - 128)
        a_max = min(128, s // 2)
        for a in range(a_min, a_max + 1):
            b = s - a
            if a <= b:
                valid_pairs[s - N_LOW].append((a, b))
                
    def calc_reps(A_set):
        """
        Calculate representation counts for sums in [128, 256].
        Returns a numpy array of length 129.
        """
        reps = np.zeros(RANGE_SIZE, dtype=np.int32)
        A_list = list(A_set)
        A_arr = np.array(A_list, dtype=np.int32)
        if len(A_arr) == 0:
            return reps
        
        # Efficient calculation:
        # For each a in A, for each b in A where b >= a:
        # sum = a + b
        # We can vectorize this partially, but for incremental updates, 
        # we will maintain reps manually. 
        # For initialization, let's do a bulk calculation.
        
        # Outer sum
        sums = np.add.outer(A_arr, A_arr)
        # Upper triangle indices
        i_idx, j_idx = np.triu_indices(len(A_arr))
        valid_sums = sums[i_idx, j_idx]
        
        mask = (valid_sums >= N_LOW) & (valid_sums <= N_HIGH)
        valid_sums = valid_sums[mask]
        
        if len(valid_sums) > 0:
            indices = valid_sums - N_LOW
            reps += np.bincount(indices, minlength=RANGE_SIZE)
            
        return reps

    def get_peak(reps):
        return int(np.max(reps))
        
    def is_covered(reps):
        return np.all(reps > 0)
        
    # Phase 1: Find a valid covering set
    # Strategy: Greedy set cover approximation or random init + repair.
    # Let's try a few random initializations and repair them.
    
    best_A = None
    best_reps = None
    best_peak = float('inf')
    
    # Try to find a good starting point
    for _ in range(50):
        # Random set
        k = random.randint(40, 80)
        A_init = set(random.sample(range(0, 129), k))
        A_init.add(0)
        A_init.add(128)
        
        reps = calc_reps(A_init)
        
        if is_covered(reps):
            peak = get_peak(reps)
            if peak < best_peak:
                best_peak = peak
                best_A = A_init.copy()
                best_reps = reps.copy()
        
    if best_A is None:
        # Fallback: Use a known valid structure (e.g., arithmetic progression)
        # range(64, 129) covers [128, 256]
        A_fallback = set(range(64, 129))
        best_A = A_fallback
        best_reps = calc_reps(A_fallback)
        best_peak = get_peak(best_reps)

    current_A = best_A.copy()
    current_reps = best_reps.copy()
    current_peak = best_peak
    
    # Phase 2: Local Search to minimize peak
    # We use a "diversification" strategy: 
    # 1. If peak is high, try to remove elements that contribute to the peak sums.
    # 2. If coverage is lost, repair by adding elements that cover the missing sums.
    
    # To make this efficient, we track which elements contribute to which sums.
    # However, maintaining this incrementally is complex. 
    # Given the small size (128 values), O(|A|) update per move is feasible if we precompute contributions.
    
    # Let's define a helper to update reps when adding/removing x.
    # Adding x: for all y in A where y >= x (to avoid double counting? No, unordered pairs).
    # Actually, the representation count for sum s is the number of pairs {a,b} in A such that a+b=s.
    # If we add x, new pairs are {x, y} for all y in A.
    # If y == x, it's one pair (x,x). If y != x, it's one pair {x,y}.
    # Wait, the count is just the number of pairs.
    # If we add x to A, the new pairs involving x are (x, y) for all y in A_old.
    # We must ensure we don't double count. The set A_old does not contain x.
    # So for each y in A_old:
    #   sum = x + y
    #   If sum in range, increment reps[sum] by 1.
    #   (Since x is new, the pair {x,y} is new. If y==x, it wasn't in A_old, so {x,x} is new.)
    
    # Removing x:
    # For each y in A_new (which is A_old \ {x}):
    #   sum = x + y
    #   If sum in range, decrement reps[sum] by 1.
    
    # This is O(|A|) per move. |A| ~ 60. 60 operations per move.
    # We can do ~100,000 moves in 1 second. 58 seconds -> 5.8M moves.
    # This is much faster than O(|A|^2) ~ 3600 ops per move.
    
    # Let's implement the incremental update.
    
    def add_to_reps(reps, x, A_set):
        # x is being added to A_set (x is NOT in A_set yet)
        # Pairs formed: (x, y) for y in A_set
        for y in A_set:
            s = x + y
            if N_LOW <= s <= N_HIGH:
                reps[s - N_LOW] += 1
                
    def remove_from_reps(reps, x, A_set):
        # x is being removed from A_set (x IS in A_set)
        # Pairs lost: (x, y) for y in A_set \ {x}
        # Note: The pair (x, x) is also lost if x is in A_set.
        # So we iterate over all y in A_set.
        # If y == x, we remove one count for sum 2x.
        # If y != x, we remove one count for sum x+y.
        for y in A_set:
            s = x + y
            if N_LOW <= s <= N_HIGH:
                reps[s - N_LOW] -= 1

    # Simulated Annealing with incremental updates
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.1
    no_improve = 0
    max_no_improve = 2000
    
    best_peak_found = current_peak
    best_A_found = current_A.copy()
    best_reps_found = current_reps.copy()
    
    while time.time() < deadline:
        if T < min_T or no_improve > max_no_improve:
            break
            
        # Generate move
        move_type = random.random()
        new_A = current_A.copy()
        new_reps = current_reps.copy()
        
        if move_type < 0.6:
            # Swap: Remove x, Add y
            if len(new_A) == 0: continue
            x = random.choice(list(new_A))
            y = random.randint(0, 128)
            if y in new_A: continue # No-op
            
            # Remove x
            remove_from_reps(new_reps, x, new_A)
            new_A.remove(x)
            
            # Add y
            add_to_reps(new_reps, y, new_A)
            new_A.add(y)
            
        elif move_type < 0.8:
            # Remove
            if len(new_A) == 0: continue
            x = random.choice(list(new_A))
            remove_from_reps(new_reps, x, new_A)
            new_A.remove(x)
            
        else:
            # Add
            y = random.randint(0, 128)
            if y in new_A: continue
            add_to_reps(new_reps, y, new_A)
            new_A.add(y)
            
        if new_A == current_A:
            continue
            
        # Evaluate
        # Check coverage
        if np.any(new_reps == 0):
            # Invalid move, penalize heavily
            new_obj = 1000 + int(np.sum(new_reps == 0))
        else:
            new_peak = int(np.max(new_reps))
            new_obj = new_peak
            
        current_peak = int(np.max(current_reps)) if np.all(current_reps > 0) else 1000 + int(np.sum(current_reps == 0))
        
        delta = new_obj - current_peak
        
        accept = False
        if delta < 0:
            accept = True
        else:
            # Accept worse with probability exp(-delta/T)
            # Cap delta to avoid underflow
            if delta > 100:
                prob = 0
            else:
                prob = np.exp(-delta / T)
            if random.random() < prob:
                accept = True
                
        if accept:
            current_A = new_A
            current_reps = new_reps
            
            if new_obj < 1000: # Valid
                if new_obj < best_peak_found:
                    best_peak_found = new_obj
                    best_A_found = current_A.copy()
                    best_reps_found = current_reps.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
        else:
            no_improve += 1
            
        T *= cooling_rate
        
    # Return the best valid solution found
    if best_peak_found < 1000:
        return list(best_A_found)
    else:
        # Fallback to the initial valid solution
        return list(best_A)
