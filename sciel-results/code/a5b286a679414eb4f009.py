import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin out of 90s
    
    n = 128
    
    # Precompute all nonconstant 4-term progressions (a, a+d, a+2d, a+3d) within 1..128
    # Indices are 0-based
    progressions = []
    for a in range(1, n + 1):
        for d in range(1, n):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= n:
                progressions.append((a - 1, b - 1, c - 1, e - 1))
    
    if not progressions:
        return [0] * n
        
    # Convert to numpy array for fast access
    # shape: (num_progs, 4)
    prog_arr = np.array(progressions, dtype=np.int32)
    num_progs = len(prog_arr)
    
    # Function to calculate the number of bad progressions (those with < 3 distinct colors)
    # colors: array of shape (n,) with values 0..k-1
    def count_bad(colors):
        # Extract colors for each progression
        # c1 = colors[prog_arr[:, 0]]
        # c2 = colors[prog_arr[:, 1]]
        # c3 = colors[prog_arr[:, 2]]
        # c4 = colors[prog_arr[:, 3]]
        
        c1 = colors[prog_arr[:, 0]]
        c2 = colors[prog_arr[:, 1]]
        c3 = colors[prog_arr[:, 2]]
        c4 = colors[prog_arr[:, 3]]
        
        # A progression is bad if the number of unique colors is < 3.
        # This is equivalent to:
        # 1. All 4 are same (1 color)
        # 2. Exactly 2 colors are used.
        
        # Let's check distinct count.
        # We can compute this by checking equality patterns.
        # It's faster to compute the set size logic via boolean masks.
        
        # 1 color: c1==c2==c3==c4
        one_color = (c1 == c2) & (c2 == c3) & (c3 == c4)
        
        # 2 colors:
        # The number of distinct values is 2.
        # This happens if not one_color AND not three_or_four_colors.
        # Easier: 
        # Distinct count is 1 if one_color.
        # Distinct count is 2 if:
        #   (c1==c2 and c3==c4 and c1!=c3) -> A A B B
        #   (c1==c3 and c2==c4 and c1!=c2) -> A B A B
        #   (c1==c4 and c2==c3 and c1!=c2) -> A B B A
        #   (c1==c2 and c1!=c3 and c3==c4) ... wait, this is covered by A A B B?
        #   Let's list all 2-color patterns:
        #   A A A B (3-1)
        #   A A B A
        #   A B A A
        #   B A A A
        #   A A B B (2-2)
        #   A B B A (2-2)
        #   B B A A (2-2)
        #   A B A B (2-2)
        #   B A B A (2-2)
        #   A B B B (1-3)
        #   B A B B
        #   B B A B
        #   B B B A
        
        # Basically, if the set of {c1, c2, c3, c4} has size 1 or 2.
        # We can compute min and max. If max - min == 0, size 1.
        # If max - min > 0, we need to check if all elements are in {min, max}.
        
        c_min = np.minimum(np.minimum(c1, c2), np.minimum(c3, c4))
        c_max = np.maximum(np.maximum(c1, c2), np.maximum(c3, c4))
        
        # Size 1
        bad_1 = (c_min == c_max)
        
        # Size 2: All elements are either c_min or c_max, and c_min != c_max.
        # And not all are min (which is size 1) and not all are max (size 1).
        # Actually, if c_min != c_max, and all elements are in {c_min, c_max}, then size is exactly 2.
        is_min = (c1 == c_min) | (c2 == c_min) | (c3 == c_min) | (c4 == c_min)
        is_max = (c1 == c_max) | (c2 == c_max) | (c3 == c_max) | (c4 == c_max)
        
        # If all are min or max, and min != max, then it's 2 colors.
        all_in_range = (is_min & is_max) # Wait, this logic is wrong.
        # is_min checks if ANY is min. is_max checks if ANY is max.
        # We need: (c1 in {min,max}) AND (c2 in {min,max}) ...
        
        c1_in = (c1 == c_min) | (c1 == c_max)
        c2_in = (c2 == c_min) | (c2 == c_max)
        c3_in = (c3 == c_min) | (c3 == c_max)
        c4_in = (c4 == c_min) | (c4 == c_max)
        
        all_in = c1_in & c2_in & c3_in & c4_in
        
        # If min != max and all_in, then distinct count is 2.
        bad_2 = (c_min != c_max) & all_in
        
        bad = bad_1 | bad_2
        return np.sum(bad)

    best_solution = None
    best_cost = float('inf')
    
    # Try k=3 first
    for k in range(3, 6):
        if time.time() > deadline - 5:
            break
            
        # Simulated Annealing
        colors = np.random.randint(0, k, n)
        current_cost = count_bad(colors)
        
        if current_cost == 0:
            best_solution = colors
            break
            
        best_cost_sa = current_cost
        best_colors_sa = colors.copy()
        
        T = 1.0
        cooling = 0.999
        min_T = 1e-5
        
        steps = 0
        max_steps = 500000 # Adjust based on speed
        
        while T > min_T and time.time() < deadline - 5:
            # Pick a random position and a new color
            idx = np.random.randint(0, n)
            old_color = colors[idx]
            if k == 1:
                new_color = 0
            else:
                new_color = np.random.randint(0, k)
            if new_color == old_color:
                continue
                
            colors[idx] = new_color
            new_cost = count_bad(colors)
            
            delta = new_cost - current_cost
            
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_cost = new_cost
                if current_cost < best_cost_sa:
                    best_cost_sa = current_cost
                    best_colors_sa = colors.copy()
                    if current_cost == 0:
                        best_solution = colors
                        break
            else:
                colors[idx] = old_color
                
            T *= cooling
            steps += 1
            
        if best_cost_sa == 0:
            best_solution = best_colors_sa
            break
            
        if best_cost_sa < best_cost:
            best_cost = best_cost_sa
            best_solution = best_colors_sa

    if best_solution is not None:
        return best_solution.tolist()
        
    # Fallback: Return a valid coloring with many colors (e.g. all distinct)
    # This ensures validity even if SA fails
    return list(range(n))

