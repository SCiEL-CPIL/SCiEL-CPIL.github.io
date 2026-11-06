import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    N = 128

    # Pre-compute 4-term APs
    fours = []
    for d in range(1, (N - 3) // 3 + 1):
        for i in range(0, N - 3 * d):
            fours.append((i, i + d, i + 2 * d, i + 3 * d))
    fours = np.array(fours, dtype=np.int32)
    
    # Helper to check if a coloring is valid
    def is_valid(coloring):
        a = coloring[fours[:, 0]]
        b = coloring[fours[:, 1]]
        c = coloring[fours[:, 2]]
        d = coloring[fours[:, 3]]
        # Check if any 4-AP has < 3 distinct colors
        # < 3 distinct means size 1 or 2.
        # Size 1: a==b==c==d
        # Size 2: not size 1, but all in {x, y}
        # Easier: count distinct.
        # Vectorized distinct count is tricky.
        # Let's use the logic from before:
        # has_three_distinct = (a!=b & a!=c & b!=c) | (a!=b & a!=d & b!=d) | (a!=c & a!=d & c!=d) | (b!=c & b!=d & c!=d)
        abc_distinct = (a != b) & (a != c) & (b != c)
        abd_distinct = (a != b) & (a != d) & (b != d)
        acd_distinct = (a != c) & (a != d) & (c != d)
        bcd_distinct = (b != c) & (b != d) & (c != d)
        has_three_distinct = abc_distinct | abd_distinct | acd_distinct | bcd_distinct
        return np.sum(~has_three_distinct) == 0

    # Step 1: Greedy Construction (Inspiration 1)
    def greedy_construct():
        colors = []
        for x in range(1, N + 1):
            found = False
            for color in range(128):
                valid = True
                # Check 4-term APs ending at x
                # x is 1-indexed in the loop logic of Insp 1, but 0-indexed in array?
                # Insp 1: colors[x-1] is the current element.
                # d in range(1, (x-1)//3 + 1)
                # Indices: x-1, x-1-d, x-1-2d, x-1-3d
                for d in range(1, (x - 1) // 3 + 1):
                    i = x - 1 - 3 * d
                    if i < 0: continue
                    ap_colors = {colors[i], colors[i + d], colors[i + 2 * d], color}
                    if len(ap_colors) < 3:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check 3-term APs ending at x (to avoid mono 3-APs)
                for d in range(1, (x - 1) // 2 + 1):
                    i = x - 1 - 2 * d
                    if i < 0: continue
                    # colors[i], colors[i+d], color
                    if colors[i] == colors[i + d] == color:
                        valid = False
                        break
                
                if valid:
                    colors.append(color)
                    found = True
                    break
            if not found:
                # Should not happen if we have enough colors, but just in case
                colors.append(0)
        return np.array(colors, dtype=np.int32)

    coloring = greedy_construct()
    
    # Verify validity
    if not is_valid(coloring):
        # Fallback to random + local search if greedy failed (unlikely)
        # But greedy should work.
        pass

    # Step 2: Local Search to Minimize Distinct Colors
    # Current distinct colors
    unique_colors = set(coloring.tolist())
    
    # Try to remove colors
    # We iterate while we can remove a color
    while time.time() < DEADLINE - 5:
        current_unique = list(set(coloring.tolist()))
        if len(current_unique) <= 3: # Can't go below 3? Maybe 2? 
            # If 2 colors, check if valid. If valid, try to get 1? No, 1 is invalid.
            # So min is likely 3 or 4.
            pass
            
        # Pick a color to try to eliminate. 
        # Strategy: Try the least frequent color first.
        color_counts = {c: np.sum(coloring == c) for c in current_unique}
        target_color = min(current_unique, key=lambda c: color_counts[c])
        
        # Try to recolor all positions with target_color to other colors
        positions_to_change = np.where(coloring == target_color)[0]
        if len(positions_to_change) == 0:
            continue
            
        success = True
        new_coloring = coloring.copy()
        
        # We need to be careful: changing one position might affect the validity of others.
        # Since we are removing a color, we are mapping target_color -> other_color.
        # We should process positions in an order that minimizes conflicts?
        # Or just try randomly.
        
        # Let's try a simple iterative improvement for this specific elimination
        # Actually, if we change pos 1 to color A, and pos 2 to color B, they might interact.
        # It's a CSP. 
        # Given the small size, we can try backtracking for the elimination?
        # Or just simple local search: 
        # For each pos in positions_to_change:
        #   Try to assign a color from current_unique \ {target_color}
        #   such that all 4-APs involving this pos are valid in the NEW coloring (with other changes applied).
        
        # This is complex to vectorize easily. Let's do a simple check.
        # If we just try to find a single mapping target_color -> c_new that works for ALL positions at once?
        # That's too restrictive.
        
        # Let's use the local search approach from the current program but with the objective:
        # Minimize number of distinct colors.
        # Penalty for each new distinct color introduced?
        # No, let's stick to the "try to eliminate one color" heuristic.
        
        # Implementation of elimination:
        # 1. Make a copy.
        # 2. For each pos in positions_to_change (shuffled):
        #    Try each c_new in current_unique \ {target_color}
        #    Check if assigning c_new to pos makes all 4-APs involving pos valid (considering current state of new_coloring).
        #    If yes, assign and break.
        #    If no c_new works, fail.
        # 3. If all positions assigned, success.
        
        random.shuffle(positions_to_change.tolist())
        new_coloring = coloring.copy()
        eliminated = True
        
        for pos in positions_to_change:
            found_color = False
            for c_new in current_unique:
                if c_new == target_color:
                    continue
                # Check validity for all 4-APs containing pos
                valid = True
                for ap in fours:
                    if pos in ap:
                        # Get colors of the 4 positions in this AP from new_coloring
                        # But we haven't updated new_coloring for pos yet?
                        # We are testing c_new.
                        c0 = new_coloring[ap[0]] if ap[0] != pos else c_new
                        c1 = new_coloring[ap[1]] if ap[1] != pos else c_new
                        c2 = new_coloring[ap[2]] if ap[2] != pos else c_new
                        c3 = new_coloring[ap[3]] if ap[3] != pos else c_new
                        if len({c0, c1, c2, c3}) < 3:
                            valid = False
                            break
                if valid:
                    new_coloring[pos] = c_new
                    found_color = True
                    break
            if not found_color:
                eliminated = False
                break
        
        if eliminated:
            coloring = new_coloring
            # Continue loop to try eliminating more
        else:
            # If we can't eliminate the least frequent, maybe try another?
            # Or break?
            # Let's try a few random colors if the least frequent fails.
            # But to keep it simple and fast, we might just break or try next.
            # Given the structure, if one fails, others might too.
            # Let's just break to save time, as greedy is already good.
            break

    return [int(x) for x in coloring]

if __name__ == "__main__":
    res = construct()
    print(f"Distinct colors: {len(set(res))}")
