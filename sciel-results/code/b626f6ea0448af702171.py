import numpy as np
import random
import time

def construct():
    N = 128
    # Precompute all non-constant 4-term arithmetic progressions
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    ap_array = np.array(aps)
    
    def count_violations(colors):
        # colors is a numpy array
        # Extract values for all APs
        vals = colors[ap_array]
        # Check if any row has < 3 distinct values
        # Distinct count < 3 means all same (1) or two same (2)
        # We can check:
        # 1. All same: vals[:, 0] == vals[:, 1] == vals[:, 2] == vals[:, 3]
        # 2. Two distinct: 
        #    This is harder to vectorize directly.
        #    Alternative: sort each row and check if any adjacent pair is different.
        #    If sorted row is [a, a, a, a] -> 1 distinct.
        #    If sorted row is [a, a, a, b] -> 2 distinct.
        #    If sorted row is [a, a, b, b] -> 2 distinct.
        #    If sorted row is [a, a, b, c] -> 3 distinct.
        #    So, valid if:
        #    sorted[:, 0] != sorted[:, 1] AND sorted[:, 1] != sorted[:, 2] (implies 3 or 4)
        #    OR sorted[:, 1] != sorted[:, 2] AND sorted[:, 2] != sorted[:, 3]
        #    Basically, we need at least 2 "changes" in the sorted array?
        #    No.
        #    [a, a, b, c] -> changes at 1 and 2.
        #    [a, b, b, c] -> changes at 0 and 2.
        #    [a, b, c, c] -> changes at 0 and 1.
        #    [a, b, c, d] -> changes at 0, 1, 2.
        #    Invalid:
        #    [a, a, a, a] -> 0 changes.
        #    [a, a, a, b] -> 1 change.
        #    [a, a, b, b] -> 1 change.
        #    So, valid iff number of distinct adjacent pairs in sorted row >= 2?
        #    Let's check [a, a, b, b]. Sorted: a, a, b, b.
        #    Pairs: (a,a) same, (a,b) diff, (b,b) same. 1 diff. Invalid. Correct.
        #    [a, a, b, c]. Sorted: a, a, b, c.
        #    Pairs: (a,a) same, (a,b) diff, (b,c) diff. 2 diffs. Valid. Correct.
        #    [a, b, b, c]. Sorted: a, b, b, c.
        #    Pairs: (a,b) diff, (b,b) same, (b,c) diff. 2 diffs. Valid. Correct.
        
        sorted_vals = np.sort(vals, axis=1)
        diffs = np.diff(sorted_vals, axis=1) != 0
        num_diffs = np.sum(diffs, axis=1)
        # Valid if num_diffs >= 2
        violations = np.sum(num_diffs < 2)
        return violations

    start_time = time.time()
    time_limit = 55.0
    
    # Try 4 colors first
    for k in range(4, 9):
        # Random restarts
        found = False
        colors = np.random.randint(0, k, size=N)
        current_violations = count_violations(colors)
        
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.995
        no_improve_steps = 0
        max_no_improve = 1000
        
        while time.time() - start_time < time_limit:
            if current_violations == 0:
                found = True
                break
                
            idx = np.random.randint(N)
            old_color = colors[idx]
            new_color = np.random.randint(k)
            
            if old_color == new_color:
                continue
                
            colors[idx] = new_color
            new_violations = count_violations(colors)
            
            delta = new_violations - current_violations
            
            if delta <= 0 or random.random() < np.exp(-delta / T):
                current_violations = new_violations
                no_improve_steps = 0
            else:
                colors[idx] = old_color
                no_improve_steps += 1
                
            T *= cooling_rate
            if T < T_min:
                T = T_min
                # Restart if stuck
                if no_improve_steps > max_no_improve:
                    colors = np.random.randint(0, k, size=N)
                    current_violations = count_violations(colors)
                    T = 1.0
                    no_improve_steps = 0
                    
        if found:
            return colors.tolist()
            
    # If we run out of time or fail to find valid coloring for k=4..8
    # Return a random 4-coloring (will likely be invalid, score 0)
    # But we should try to return the best valid one if we found one with higher k?
    # The loop above returns immediately on first valid.
    # If no valid found, we return random.
    
    return np.random.randint(0, 4, size=N).tolist()
