import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 58.0  # Safety margin for 60s limit
    N = 128

    # Pre-compute 4-term APs
    # d is the difference. i is the start index.
    # AP: i, i+d, i+2d, i+3d
    # i + 3d < N => i < N - 3d
    fours = []
    for d in range(1, (N - 3) // 3 + 1):
        for i in range(0, N - 3 * d):
            fours.append((i, i + d, i + 2 * d, i + 3 * d))
    
    if not fours:
        return [0] * N
        
    fours = np.array(fours, dtype=np.int32)
    num_aps = len(fours)
    
    # Extract indices for vectorized operations
    idx0 = fours[:, 0]
    idx1 = fours[:, 1]
    idx2 = fours[:, 2]
    idx3 = fours[:, 3]

    def count_violations(coloring):
        """
        Returns the number of 4-term APs that do NOT have at least 3 distinct colors.
        A 4-term AP is invalid if it has 1 or 2 distinct colors.
        """
        a = coloring[idx0]
        b = coloring[idx1]
        c = coloring[idx2]
        d = coloring[idx3]
        
        # A 4-AP is valid if it has >= 3 distinct colors.
        # It is invalid if:
        # 1. All 4 are same (1 distinct)
        # 2. Only 2 distinct colors are present.
        
        # Let's identify invalid ones.
        # If a==b==c==d, it's invalid.
        # If not all same, but only 2 distinct, it's invalid.
        
        # Vectorized check for "has at least 3 distinct":
        # There must exist a triplet among (a,b,c,d) that are pairwise distinct.
        # Triplets: (a,b,c), (a,b,d), (a,c,d), (b,c,d)
        
        abc_distinct = (a != b) & (a != c) & (b != c)
        abd_distinct = (a != b) & (a != d) & (b != d)
        acd_distinct = (a != c) & (a != d) & (c != d)
        bcd_distinct = (b != c) & (b != d) & (c != d)
        
        has_three_distinct = abc_distinct | abd_distinct | acd_distinct | bcd_distinct
        
        return np.sum(~has_three_distinct)

    def is_valid(coloring):
        return count_violations(coloring) == 0

    # Step 1: Greedy Construction to get a valid starting point
    # This is fast and ensures we start with a valid coloring (violations = 0)
    def greedy_construct():
        colors = np.zeros(N, dtype=np.int32)
        for x in range(1, N):
            found = False
            # Try colors 0..127
            # We want to minimize colors, so try 0,1,2... first
            for color in range(128):
                valid = True
                # Check 4-term APs ending at x (i.e., x is the 4th term)
                # AP: x-3d, x-2d, x-d, x
                for d in range(1, (x) // 3 + 1):
                    i = x - 3 * d
                    if i < 0: continue
                    # Indices: i, i+d, i+2d, x
                    c0 = colors[i]
                    c1 = colors[i + d]
                    c2 = colors[i + 2 * d]
                    c3 = color
                    if len({c0, c1, c2, c3}) < 3:
                        valid = False
                        break
                if valid:
                    colors[x] = color
                    found = True
                    break
            if not found:
                # Fallback: assign a random high color, though greedy should work
                colors[x] = 127
        return colors

    # Initial Coloring
    coloring = greedy_construct()
    
    # Verify it's valid (it should be)
    if not is_valid(coloring):
        # If greedy failed (unlikely), do a quick random init + local search to fix violations
        coloring = np.random.randint(0, 10, N).astype(np.int32)
        # Simple hill climb to fix violations
        for _ in range(1000):
            if time.time() > DEADLINE - 2: break
            v = count_violations(coloring)
            if v == 0: break
            # Pick random position
            pos = random.randint(0, N-1)
            old_c = coloring[pos]
            new_c = random.randint(0, 10)
            coloring[pos] = new_c
            new_v = count_violations(coloring)
            if new_v > v:
                coloring[pos] = old_c
            else:
                v = new_v

    best_coloring = coloring.copy()
    best_num_colors = len(set(best_coloring.tolist()))

    # Step 2: Local Search to Minimize Distinct Colors
    # Strategy:
    # 1. We have a valid coloring with K colors.
    # 2. We want to reduce K.
    # 3. We can do this by trying to "merge" two colors.
    #    Pick color A and color B. Try to recolor all A's to B (or some mix) while maintaining validity.
    #    Actually, simpler: Try to eliminate the least frequent color.
    
    # To allow more exploration, we can also use a general local search that minimizes
    # a combined objective: violations + lambda * (num_distinct_colors)
    # But since we start valid, we can just focus on elimination.
    
    # Let's use a loop that tries to eliminate one color at a time.
    # If elimination fails, we might do a "shake" (random perturbation) to escape local optima.
    
    current_colors = set(coloring.tolist())
    
    # Pre-compute indices of each color for faster access? 
    # For N=128, simple list comprehension is fine.
    
    while time.time() < DEADLINE - 1.0:
        current_unique = list(set(coloring.tolist()))
        if len(current_unique) <= 3:
            # Can we go to 2? 
            # If we have 3 colors, try to eliminate one to get 2.
            # If we have 2 colors, check if valid. If valid, try to get 1 (impossible for 4-APs).
            pass
            
        # Sort colors by frequency (least frequent first)
        color_counts = {c: np.sum(coloring == c) for c in current_unique}
        # Sort keys by count
        sorted_colors = sorted(current_unique, key=lambda c: color_counts[c])
        
        eliminated_any = False
        
        for target_color in sorted_colors:
            if time.time() > DEADLINE - 1.0:
                break
                
            # Try to eliminate target_color
            positions_to_change = np.where(coloring == target_color)[0]
            if len(positions_to_change) == 0:
                continue
                
            # We will try to recolor these positions to other colors in current_unique \ {target_color}
            # This is a CSP. We'll use a simple backtracking/greedy approach with local search.
            
            # To make it robust, let's try a few random orders/shuffles if the first attempt fails.
            success = False
            
            for attempt in range(5):
                new_coloring = coloring.copy()
                # Shuffle positions to try different orders
                shuffled_pos = positions_to_change.copy()
                random.shuffle(shuffled_pos)
                
                valid_elimination = True
                
                for pos in shuffled_pos:
                    # Try each available color
                    found_new_color = False
                    # Sort candidate colors by frequency? Or just iterate.
                    # Let's try colors that are already present.
                    for c_new in current_unique:
                        if c_new == target_color:
                            continue
                        # Check if assigning c_new to pos keeps all APs valid
                        # We need to check APs involving pos in the context of new_coloring
                        # But new_coloring is being modified.
                        
                        # Optimization: Only check APs involving pos.
                        # We can pre-compute which AP indices involve which positions?
                        # For now, N is small, so checking all APs is ~1500 checks.
                        # 128 positions * 1500 checks = 192k checks per elimination attempt.
                        # This is fast enough in Python if vectorized? No, this loop is Python.
                        
                        # Let's use the vectorized check on the whole array?
                        # That's O(N_aps) per position change.
                        # Total O(N_pos * N_aps).
                        # If we do this for each position, it's expensive.
                        
                        # Better: Check only APs containing pos.
                        # Let's pre-compute ap_indices_by_pos.
                        # We didn't do that in the first block. Let's do it now if not done.
                        
                        # Actually, let's just use the vectorized count_violations on the whole array
                        # but only if we are close to success? No, that's too slow per step.
                        
                        # Let's stick to the Python loop for the specific APs, but optimize it.
                        # Or, simpler: Just apply the change, run count_violations, if > 0, revert.
                        # This is O(N_aps) per position.
                        # If we have 10 positions to change, that's 10 * 1500 = 15k ops.
                        # This is very fast in NumPy.
                        
                        old_val = new_coloring[pos]
                        new_coloring[pos] = c_new
                        
                        if count_violations(new_coloring) == 0:
                            found_new_color = True
                            break
                        else:
                            new_coloring[pos] = old_val
                            
                    if not found_new_color:
                        valid_elimination = False
                        break
                        
                if valid_elimination:
                    # Check final validity
                    if count_violations(new_coloring) == 0:
                        coloring = new_coloring
                        success = True
                        break
                        
            if success:
                eliminated_any = True
                current_unique = list(set(coloring.tolist()))
                # Update best
                if len(current_unique) < best_num_colors:
                    best_num_colors = len(current_unique)
                    best_coloring = coloring.copy()
                break # Restart the while loop with new set of colors
            
        if not eliminated_any:
            # If we can't eliminate any color, we might be stuck.
            # Try a "shake": Randomly perturb the coloring to escape local minima,
            # then run a short local search to restore validity, then try elimination again.
            # This is risky but might find a better configuration.
            
            # Simple shake: change 5 random positions to random colors
            for _ in range(5):
                pos = random.randint(0, N-1)
                coloring[pos] = random.randint(0, 10)
            
            # Restore validity with hill climbing
            for _ in range(200):
                if time.time() > DEADLINE - 1.0:
                    break
                v = count_violations(coloring)
                if v == 0:
                    break
                pos = random.randint(0, N-1)
                old_c = coloring[pos]
                new_c = random.randint(0, 10)
                coloring[pos] = new_c
                new_v = count_violations(coloring)
                if new_v > v:
                    coloring[pos] = old_c
                else:
                    v = new_v

    # Final check and return
    # Ensure the best coloring found is valid
    if is_valid(best_coloring):
        return [int(x) for x in best_coloring]
    else:
        # Fallback to current coloring if it's valid, else the initial greedy one
        if is_valid(coloring):
            return [int(x) for x in coloring]
        return [int(x) for x in best_coloring] # Should be valid

if __name__ == "__main__":
    res = construct()
    print(f"Distinct colors: {len(set(res))}")
