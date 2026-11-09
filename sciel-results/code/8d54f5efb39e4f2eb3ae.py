import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 seconds limit, keep 10s margin
    
    N = 128
    
    # Precompute all 4-term arithmetic progressions
    aps = []
    # a, d such that a+3d <= N
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(aps)
    
    # Map position to list of (ap_index, position_in_ap)
    pos_to_aps = {i: [] for i in range(1, N + 1)}
    for idx, ap in enumerate(aps):
        for j, pos in enumerate(ap):
            pos_to_aps[pos].append((idx, j))
    
    def count_distinct_colors(ap_colors):
        return len(set(ap_colors))
    
    def get_ap_color_count(colors, ap):
        return len(set([colors[ap[0]-1], colors[ap[1]-1], colors[ap[2]-1], colors[ap[3]-1]]))
    
    def initial_cost(colors):
        cost = 0
        for ap in aps:
            if get_ap_color_count(colors, ap) < 3:
                cost += 1
        return cost
    
    def delta_cost(colors, pos, old_c, new_c):
        """
        Calculate change in cost when changing color of 'pos' from old_c to new_c.
        colors is the current state (0-indexed internally, but pos is 1-indexed).
        """
        delta = 0
        idx = pos - 1
        for ap_idx, j in pos_to_aps[pos]:
            ap = aps[ap_idx]
            # Get current colors of the AP
            c0 = colors[ap[0]-1]
            c1 = colors[ap[1]-1]
            c2 = colors[ap[2]-1]
            c3 = colors[ap[3]-1]
            
            old_colors = [c0, c1, c2, c3]
            old_count = len(set(old_colors))
            
            # Create new colors list
            new_colors = old_colors[:]
            new_colors[j] = new_c
            
            new_count = len(set(new_colors))
            
            # Cost is 1 if count < 3, else 0
            old_bad = 1 if old_count < 3 else 0
            new_bad = 1 if new_count < 3 else 0
            
            delta += new_bad - old_bad
            
        return delta

    def solve_for_k(k):
        # Try to find a valid coloring with k colors
        # Use Simulated Annealing
        if time.time() > deadline - 5:
            return None
            
        colors = np.random.randint(0, k, size=N)
        current_cost = initial_cost(colors)
        
        if current_cost == 0:
            return colors.copy()
            
        best_colors = colors.copy()
        best_cost = current_cost
        
        T = 1.0
        T_min = 0.01
        alpha = 0.995
        
        # Number of iterations per temperature step
        # Adjust based on time
        steps_per_temp = 10000
        
        while T > T_min:
            for _ in range(steps_per_temp):
                if time.time() > deadline - 5:
                    return None
                    
                pos = random.randint(1, N)
                old_c = colors[pos - 1]
                
                if k == 1:
                    new_c = 0
                else:
                    new_c = random.randint(0, k - 1)
                    if new_c == old_c:
                        continue
                        
                d_cost = delta_cost(colors, pos, old_c, new_c)
                
                if d_cost < 0 or random.random() < np.exp(-d_cost / T):
                    colors[pos - 1] = new_c
                    current_cost += d_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = colors.copy()
                        
                    if current_cost == 0:
                        return colors.copy()
                        
            T *= alpha
            
        return None

    # Try k = 3, 4, 5...
    # We know k must be at least 3.
    for k in range(3, 10):
        if time.time() > deadline - 10:
            break
            
        # Run multiple restarts for each k
        # If k=3 is hard, we might need more time.
        # Let's allocate time based on k.
        # Actually, just loop until deadline.
        
        # To improve success rate, we can use multiple random restarts within the time budget
        restarts = 0
        max_restarts = 50
        
        while time.time() < deadline - 10 and restarts < max_restarts:
            res = solve_for_k(k)
            if res is not None:
                # Found valid coloring with k colors
                # Map colors to 0..k-1 (already done)
                # Return as list of integers
                return res.tolist()
            restarts += 1
            
        # If we failed for this k, try next k
        # Note: If k=3 fails, it might be impossible, or just hard.
        # Given N=128, k=3 is likely possible or k=4.
        
    # Fallback: Return a greedy solution if no optimized one found
    # This ensures we return *something* valid if possible, or at least the seed's logic
    # But we should try to return the best found so far?
    # The problem asks to minimize colors. If we fail to find a valid one for small k,
    # we must return a valid one for larger k.
    
    # If the loop above finished without returning, it means we didn't find a solution
    # in the time limit for k=3..9.
    # This is unlikely for k>=4.
    # Let's try a quick greedy or random search for a larger k as a last resort.
    
    # Actually, if we didn't return, we should just return a valid coloring with more colors.
    # Let's try k=10 with a short SA.
    if time.time() < deadline - 2:
         res = solve_for_k(10)
         if res is not None:
             return res.tolist()
             
    # Last resort: The seed program logic (greedy)
    # Re-implementing seed to ensure validity if SA fails completely
    colors=[]
    for x in range(1,129):
        found = False
        for color in range(128):
            valid = True
            # Check all APs ending at x
            # x is the 4th term? No, x can be any term.
            # The seed checks "all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 ..."
            # This checks if x is the LAST term of an AP.
            # But x can be the 1st, 2nd, or 3rd term too.
            # The seed logic seems flawed or only checks partial constraints?
            # "Every nonconstant four-term progression of positions must use AT LEAST three distinct colors."
            # If we build left-to-right, we only know colors for < x.
            # We can't fully validate an AP where x is not the last term until we assign later terms.
            # So a simple greedy left-to-right is hard unless we look ahead or the constraint is local.
            
            # However, the seed program is given as "score 1.0000", implying it works.
            # Let's look at the seed code again.
            # It checks:
            # 1. For d in range(1, (x-1)//3 + 1): 
            #    This implies x-3d >= 1. So x is the 4th term.
            #    It checks if adding 'color' to {c[x-1], c[x-1-d], c[x-1-2d], c[x-1-3d]} gives >= 3 colors.
            #    Wait, the set is {color} U {colors[x-1-j*d] for j in (1,2,3)}.
            #    The indices are x-1, x-1-d, x-1-2d, x-1-3d.
            #    This corresponds to positions x, x-d, x-2d, x-3d.
            #    So it checks APs where x is the LAST term.
            # 2. For d in range(1, (x-1)//2 + 1):
            #    It checks if any of the previous 2 terms is different from 'color'.
            #    This seems to be a heuristic to avoid 2-color patterns?
            
            # If the seed works, it implies that satisfying the "x is last term" constraint is sufficient?
            # No, that's not true. An AP (1, 2, 3, 4) is checked when x=4.
            # But what about (2, 3, 4, 5)? Checked when x=5.
            # Every AP has a last term. So if we ensure that for every x, and every d such that x-3d>=1,
            # the set {C(x), C(x-d), C(x-2d), C(x-3d)} has >= 3 colors,
            # then EVERY 4-AP satisfies the condition.
            # Yes! Because every 4-AP (a, a+d, a+2d, a+3d) is checked when we assign a+3d.
            # So the greedy approach IS valid if we can always find a color that satisfies the condition.
            # The condition is: for all d, |{C(x)} U {C(x-d), C(x-2d), C(x-3d)}| >= 3.
            # This means C(x) cannot be such that the total set size is < 3.
            # If the previous 3 colors are all the same (say A), we need C(x) != A? No, if C(x)=A, size is 1. Bad.
            # If previous are A, A, B. Set is {A, B}. If C(x)=A, size 2. Bad. If C(x)=B, size 2. Bad.
            # We need C(x) to be a new color C, or if we pick A, size is 2.
            # So if the previous 3 have <= 2 distinct colors, we MUST pick a color that is NOT in that set?
            # No. If previous are A, A, B (set {A,B}), and we pick C, set is {A,B,C} (size 3). OK.
            # If we pick A, set is {A,B} (size 2). Bad.
            # So if the set of previous 3 has size 1 or 2, we are forced to pick a color outside that set?
            # If set size is 2, we MUST pick a 3rd distinct color.
            # If set size is 3, we can pick anything (even one of them, size stays 3 or 4).
            
            # So the greedy algorithm works if we have enough colors available.
            # If we are limited to K colors, we might get stuck if we need a 3rd color but only have 2 available in the "pool" of choices?
            # No, we can use ANY color from 0..127.
            # The goal is to MINIMIZE the number of colors used.
            # So we should try to reuse colors.
            # If we are forced to pick a new color, we pick the smallest unused one?
            # Or we pick a color that minimizes future conflicts?
            
            # The seed program just picks the first valid color in 0..127.
            # This minimizes the label value, but not necessarily the COUNT of distinct labels used?
            # Actually, if it picks 0, then 1, then 2, it uses 3 colors.
            # If it picks 0, 0, 0... it uses 1 color (but invalid).
            # If it picks 0, 1, 0, 1... it uses 2 colors (but invalid for 4-APs with 3 distinct req).
            # So it will be forced to pick new colors when necessary.
            # The number of distinct colors used will be the maximum number of distinct colors needed at any step?
            # Not exactly, because a color used early might not be needed later.
            # But if we pick the smallest available valid color, we tend to reuse small indices.
            
            # The SA approach allows us to optimize the global count.
            # But the greedy approach is a valid fallback.
            
            pass # Skip re-implementing seed in fallback, use simple random if SA fails
            
    # If we reach here, we failed to find a solution with SA.
    # Return a random valid coloring?
    # Let's try a simple greedy with K=5 as a last resort.
    
    colors = [0] * N
    for x in range(1, N + 1):
        for c in range(5):
            valid = True
            for d in range(1, (x - 1) // 3 + 1):
                prev_colors = {colors[x - 1 - j * d] for j in (1, 2, 3)}
                if len(prev_colors | {c}) < 3:
                    valid = False
                    break
            if valid:
                colors[x - 1] = c
                break
        else:
            # If no color in 0..4 works, try 5, 6...
            for c in range(5, 128):
                valid = True
                for d in range(1, (x - 1) // 3 + 1):
                    prev_colors = {colors[x - 1 - j * d] for j in (1, 2, 3)}
                    if len(prev_colors | {c}) < 3:
                        valid = False
                        break
                if valid:
                    colors[x - 1] = c
                    break
    return colors

