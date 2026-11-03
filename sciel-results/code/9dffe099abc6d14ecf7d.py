"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    # Initial set: [0, 1, 3] which scores ~1.0
    # Let's start with a slightly better known configuration if possible, or just [0,1,3]
    current_set = np.array([0, 1, 3], dtype=np.int64)
    
    def score_set(arr):
        """
        Calculate the score for a set of non-negative integers.
        Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
        """
        if len(arr) == 0:
            return 0.0
        max_u = arr.max()
        if max_u == 0:
            return 0.0
        
        # Use broadcasting to compute sumset and diffset
        # To save memory/time for large sets, we might need to be careful, 
        # but n is expected to be small (< 100) for high scores usually.
        # If n gets very large, this O(n^2) memory might be an issue.
        # However, the task mentions sizes 2003 and 54265 in the paper, 
        # but the seed starts small. Let's assume n < 500 for safety in this loop.
        
        # Sumset: a + b for all a,b in arr
        # Diffset: a - b for all a,b in arr
        
        # To avoid huge memory allocation if n is large, we can chunk or use sets if n > 200.
        # But for simulated annealing, we want speed.
        # Let's use a hybrid: if n < 300, use numpy. Else use sets (slower but safe).
        
        n = len(arr)
        if n < 300:
            # Vectorized approach
            # a+b: shape (n, n)
            sums = arr[:, None] + arr[None, :]
            diffs = arr[:, None] - arr[None, :]
            
            # Count unique elements
            # np.unique on flattened array is O(n^2 log n^2) which is okay for n<300
            num_sums = len(np.unique(sums))
            num_diffs = len(np.unique(diffs))
        else:
            # Fallback to sets for large n
            arr_set = set(arr.tolist())
            sumset = set()
            diffset = set()
            for a in arr_set:
                for b in arr_set:
                    sumset.add(a + b)
                    diffset.add(a - b)
            num_sums = len(sumset)
            num_diffs = len(diffset)
            
        if num_sums == 0:
            return 0.0
            
        ratio = num_diffs / num_sums
        if ratio <= 0:
            return 0.0
        
        try:
            exponent = 1 + math.log(ratio) / math.log(2 * max_u + 1)
        except:
            return 0.0
            
        return exponent

    current_score = score_set(current_set)
    best_set = current_set.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    T = 1.0
    cooling_rate = 0.999
    min_T = 1e-6
    
    # Max steps based on time
    # Estimate cost of one iteration: score_set is the bottleneck.
    # For n=10, it's fast. For n=100, slower.
    
    step = 0
    while time.time() < deadline:
        step += 1
        
        # Generate neighbor
        # Strategies:
        # 1. Add a random element
        # 2. Remove a random element (if size > 2)
        # 3. Replace an element with a random one
        # 4. Shift an element
        
        action = random.random()
        new_set = current_set.copy()
        
        max_val = current_set.max() if len(current_set) > 0 else 0
        
        if action < 0.4:
            # Add element
            # Range: 1 to 2*max_val + 10
            upper_bound = max(10, 2 * max_val + 10)
            cand = random.randint(1, upper_bound)
            if cand not in new_set:
                new_set = np.append(new_set, cand)
        elif action < 0.6 and len(new_set) > 2:
            # Remove element
            idx = random.randint(1, len(new_set) - 1) # Don't remove 0 (index 0 usually, but let's be safe)
            # Ensure 0 stays. 0 is always at some index. Let's find index of 0.
            idx_zero = np.where(new_set == 0)[0]
            if len(idx_zero) > 0:
                idx_zero = idx_zero[0]
            # Pick a random index that is not idx_zero
            if len(new_set) > 1:
                removable_indices = [i for i in range(len(new_set)) if i != idx_zero]
                if removable_indices:
                    idx = random.choice(removable_indices)
                    new_set = np.delete(new_set, idx)
        elif action < 0.8:
            # Replace element
            if len(new_set) > 1:
                idx_zero = np.where(new_set == 0)[0]
                idx_zero = idx_zero[0] if len(idx_zero) > 0 else 0
                removable_indices = [i for i in range(len(new_set)) if i != idx_zero]
                if removable_indices:
                    idx = random.choice(removable_indices)
                    upper_bound = max(10, 2 * max_val + 10)
                    new_val = random.randint(1, upper_bound)
                    if new_val not in new_set:
                        new_set[idx] = new_val
        else:
            # Perturb an element
            if len(new_set) > 1:
                idx_zero = np.where(new_set == 0)[0]
                idx_zero = idx_zero[0] if len(idx_zero) > 0 else 0
                removable_indices = [i for i in range(len(new_set)) if i != idx_zero]
                if removable_indices:
                    idx = random.choice(removable_indices)
                    delta = random.randint(-5, 5)
                    new_val = new_set[idx] + delta
                    if new_val >= 0 and new_val not in new_set:
                        new_set[idx] = new_val
        
        if len(new_set) == len(current_set) and np.array_equal(new_set, current_set):
            continue
            
        new_score = score_set(new_set)
        
        # Acceptance criterion
        if new_score > current_score:
            current_set = new_set
            current_score = new_score
            if current_score > best_score:
                best_set = current_set.copy()
                best_score = current_score
        else:
            # Metropolis criterion
            delta = current_score - new_score
            if delta < 0:
                accept = True
            else:
                # exp(-delta / T)
                if T > 0:
                    prob = math.exp(-delta / T)
                    accept = random.random() < prob
                else:
                    accept = False
                    
            if accept:
                current_set = new_set
                current_score = new_score
        
        # Cool down
        T *= cooling_rate
        if T < min_T:
            T = min_T
            
        # Periodic restarts if stuck? 
        # If best_score hasn't improved in a long time, maybe restart from best_set
        if step % 1000 == 0:
             # Check if we are stuck
             if current_score < best_score - 0.01:
                 current_set = best_set.copy()
                 current_score = best_score
                 T = 0.5 # Reheat slightly

    return list(best_set)
# EVOLVE-BLOCK-END
