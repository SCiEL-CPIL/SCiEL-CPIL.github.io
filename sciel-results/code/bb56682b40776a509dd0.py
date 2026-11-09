import time
import random
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    time_limit = 58.0  # Safety margin

    # 1. Precompute APs
    aps = []
    aps_containing = [[] for _ in range(n)]
    
    # Generate all APs of length >= 2
    for d in range(1, n):
        for s in range(d): # Starts 0 to d-1 cover all APs with step d
            # Actually, starts can be 0 to n-1, but APs with start >= d are subsets of APs with start < d?
            # No, AP is defined by start and step.
            # If we fix step d, the APs are determined by their start s in [0, d-1] if we consider the "base" AP.
            # But the problem says "any start".
            # An AP is a sequence s, s+d, s+2d...
            # For a fixed d, the possible starts s are 0, 1, ..., n-1.
            # However, the AP starting at s and step d is the same set of elements as the AP starting at s % d and step d?
            # No. The AP is the *sequence*. But the sum is over the elements.
            # The set of elements $\{s, s+d, \dots\} \cap [0, n)$ depends on $s \pmod d$.
            # Wait, if we have indices 0..n-1.
            # AP defined by start $s$ and step $d$. Elements: $s, s+d, s+2d, \dots < n$.
            # If we change start to $s' = s + kd$, the elements are a suffix of the original AP.
            # The problem asks for "ANY arithmetic progression contained in [1,256]".
            # This usually means any subsequence $a, a+d, a+2d, \dots, a+kd$ where $a+kd < n$.
            # So we must consider all $s \in [0, n)$ and all $d \in [1, n)$.
            
            # Optimization:
            # For a fixed step d, the APs starting at s and s+d are related.
            # The AP starting at s is $s, s+d, \dots$.
            # The AP starting at s+d is $s+d, s+2d, \dots$.
            # These are different APs (different sets of elements, one is a subset of the other).
            # So we must iterate all s.
            
            # However, we can optimize the generation.
            pass

    # Let's just generate all unique APs (by their tuple of indices) to avoid duplicates?
    # No, "any start" implies distinct starts are distinct APs even if elements overlap?
    # "contained in [1,256]" usually defines the AP by its elements.
    # If AP1 = {1, 3, 5} and AP2 = {3, 5}, they are different APs.
    # We need to check all of them.
    
    # Re-generate properly
    aps = []
    aps_containing = [[] for _ in range(n)]
    seen = set()
    
    for d in range(1, n):
        for s in range(n):
            curr = s
            prefix = []
            while curr < n:
                prefix.append(curr)
                curr += d
            if len(prefix) >= 2:
                # Check if this AP is already in aps?
                # Since start and step define it uniquely, no duplicates if we iterate s, d uniquely.
                # But do we need length 1? "including one".
                # Singletons have sum 1. Max will be >= 1.
                # If we find max > 1, singletons don't matter.
                # We will assume max > 1.
                ap_id = len(aps)
                aps.append(prefix)
                for idx in prefix:
                    aps_containing[idx].append(ap_id)
    
    num_aps = len(aps)
    
    # Convert to numpy arrays for fast indexing
    # We need to update sums for a given index i.
    # We store sums as a numpy array.
    
    # Helper to compute initial sums
    def compute_sums(x):
        sums = np.zeros(num_aps, dtype=np.int32)
        for k, ap in enumerate(aps):
            s = 0
            for idx in ap:
                s += x[idx]
            sums[k] = s
        return sums

    # Initial random solution
    rng = random.Random(42)
    x = [rng.choice([-1, 1]) for _ in range(n)]
    sums = compute_sums(x)
    current_max = int(np.max(np.abs(sums)))
    
    best_x = x[:]
    best_max = current_max
    
    # Precompute aps_containing as numpy arrays
    aps_containing_np = [np.array(lst, dtype=np.int32) for lst in aps_containing]
    
    # Iterated Local Search with Target Tightening
    # We will try to beat the current best.
    # Target: best_max - 1
    
    while time.time() - start_time < time_limit:
        target_max = best_max - 1
        if target_max < 1:
            break
            
        # Local Search Phase
        # Start from a random or perturbed solution
        # Let's try starting from a random solution each iteration, or from best?
        # Starting from best might get stuck. Random restarts are better.
        # But we can use the best solution to seed a "warm" restart.
        
        # Strategy: 
        # 1. Pick a random starting point.
        # 2. Run SA/Local Search for a fixed time budget (e.g., 1s) or step count.
        # 3. If we find a solution with max < target_max, update best and loop.
        # 4. If not, try again.
        
        # To make it more efficient, we can do a single long SA run but with "reheat" if stuck.
        # Let's stick to a robust SA with adaptive restarts.
        
        # Initialize SA state
        x_curr = [rng.choice([-1, 1]) for _ in range(n)]
        sums_curr = compute_sums(x_curr)
        curr_max = int(np.max(np.abs(sums_curr)))
        
        T = 5.0
        cooling = 0.999
        min_T = 0.01
        
        # If we start random, we are likely far from best.
        # We want to drive towards low max.
        
        steps_since_improve = 0
        max_steps_no_improve = 10000
        
        while time.time() - start_time < time_limit:
            if T < min_T:
                # Reheat or restart
                # If we haven't improved global best, restart this SA chain
                if curr_max > best_max:
                    break
                else:
                    T = 5.0
                    # Perturb
                    for _ in range(5):
                        i = rng.randint(0, n - 1)
                        x_curr[i] = -x_curr[i]
                    sums_curr = compute_sums(x_curr)
                    curr_max = int(np.max(np.abs(sums_curr)))
                    continue

            i = rng.randint(0, n - 1)
            old_val = x_curr[i]
            new_val = -old_val
            delta = -2 * old_val
            
            indices = aps_containing_np[i]
            if len(indices) == 0:
                continue
                
            # Update sums
            sums_curr[indices] += delta
            
            # Calculate new max
            # Optimization: Only recalculate max if necessary?
            # For now, np.max is fast enough.
            new_max = int(np.max(np.abs(sums_curr)))
            delta_max = new_max - curr_max
            
            if delta_max < 0:
                # Improvement
                x_curr[i] = new_val
                curr_max = new_max
                steps_since_improve = 0
                
                if curr_max < best_max:
                    best_max = curr_max
                    best_x = x_curr[:]
                    # Update target for next outer loop iteration?
                    # We are in the inner loop. We can just continue searching for even lower.
                    # The outer loop will pick up the new best_max.
            else:
                # Accept with probability
                if rng.random() < np.exp(-delta_max / T):
                    x_curr[i] = new_val
                    curr_max = new_max
                else:
                    # Revert
                    sums_curr[indices] -= delta
            
            T *= cooling
            
            steps_since_improve += 1
            if steps_since_improve > max_steps_no_improve:
                break

    return best_x
