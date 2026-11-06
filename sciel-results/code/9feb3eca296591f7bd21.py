import time
import random
import math

def construct():
    n = 128
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    # Precompute all nonconstant 4-term APs (0-indexed positions)
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for i in range(0, n - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
            
    num_aps = len(aps)
    
    # Map each position to the APs it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)

    def is_ap_valid(colors, ap):
        c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        # Need at least 3 distinct colors
        # Invalid if:
        # 1. All 4 same
        # 2. 3 same, 1 diff
        # 3. 2 pairs (a,a,b,b)
        # 4. a,b,a,b
        # Basically, len({c0,c1,c2,c3}) < 3
        
        # Fast check
        if c0 == c1 == c2 == c3:
            return False
        if c0 == c1 == c2 or c0 == c1 == c3 or c0 == c2 == c3 or c1 == c2 == c3:
            return False
        if c0 == c1 and c2 == c3:
            return False
        if c0 == c2 and c1 == c3:
            return False
        if c0 == c3 and c1 == c2:
            return False
        return True

    def solve_with_k_colors(k, time_limit):
        # Initialize random coloring
        colors = [random.randint(0, k - 1) for _ in range(n)]
        
        # Initial status
        ap_status = [True] * num_aps
        violating_set = set()
        
        for idx, ap in enumerate(aps):
            if not is_ap_valid(colors, ap):
                ap_status[idx] = False
                violating_set.add(idx)
        
        if len(violating_set) == 0:
            return colors

        temp = 5.0
        cooling = 0.995
        min_temp = 0.01
        
        best_colors = colors[:]
        best_violations = len(violating_set)
        
        while time.time() < time_limit and temp > min_temp:
            if len(violating_set) < best_violations:
                best_violations = len(violating_set)
                best_colors = colors[:]
                if best_violations == 0:
                    break
            
            if len(violating_set) == 0:
                break
                
            # Pick a random position
            pos = random.randint(0, n - 1)
            old_color = colors[pos]
            
            # Pick a random new color
            new_color = random.randint(0, k - 1)
            if new_color == old_color:
                temp *= cooling
                continue
                
            size_before = len(violating_set)
            
            # Apply change
            colors[pos] = new_color
            
            # Update status for affected APs
            for ap_idx in pos_to_aps[pos]:
                ap = aps[ap_idx]
                new_valid = is_ap_valid(colors, ap)
                old_valid = ap_status[ap_idx]
                
                if new_valid != old_valid:
                    ap_status[ap_idx] = new_valid
                    if new_valid:
                        # Became valid
                        if ap_idx in violating_set:
                            violating_set.remove(ap_idx)
                    else:
                        # Became invalid
                        violating_set.add(ap_idx)
            
            size_after = len(violating_set)
            delta = size_after - size_before
            
            # Acceptance criteria
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = True
            else:
                # delta > 0
                if temp > 0:
                    prob = math.exp(-delta / temp)
                    accept = random.random() < prob
            
            if not accept:
                # Undo change
                colors[pos] = old_color
                for ap_idx in pos_to_aps[pos]:
                    ap = aps[ap_idx]
                    # Revert status
                    # Note: is_ap_valid is deterministic based on colors.
                    # Since colors are reverted, we just need to revert ap_status and violating_set
                    # It's safer to recalculate or store previous state.
                    # Recalculating is cheap enough for few APs?
                    # Let's just re-evaluate to be safe, or trust the revert logic.
                    # Actually, if we revert colors, the validity is the same as before.
                    # We need to restore ap_status[ap_idx] to its previous value.
                    # We didn't store previous value.
                    # Let's just recalculate validity and update set.
                    new_valid = is_ap_valid(colors, ap)
                    # ap_status[ap_idx] is currently the 'new_valid' from the failed move?
                    # No, we updated it.
                    # If we undo, we must restore ap_status and violating_set to pre-move state.
                    # This is tricky without storing history.
                    
                    # Alternative: Don't update in-place for the check.
                    # Calculate delta without modifying global state?
                    # That requires checking all affected APs against a hypothetical color.
                    
                    # Let's refactor: 
                    # 1. Store old_color.
                    # 2. For each affected AP, determine if it was valid before.
                    # 3. Determine if it would be valid with new_color.
                    # 4. Calculate delta.
                    # 5. If accept, apply. Else, do nothing.
                    
                    pass # This logic is flawed in the 'undo' block above.
            
            # REFACTORING THE ACCEPTANCE LOGIC TO BE SAFE
            
            # Let's restart the inner loop logic properly.
            
            # We will use a helper to calculate delta without modifying state.
            
            break # Break out of this messy implementation and write it correctly below.
        
        return best_colors

    # Correct Implementation of solve_with_k_colors
    
    def solve(k, time_limit):
        colors = [random.randint(0, k - 1) for _ in range(n)]
        
        ap_status = [True] * num_aps
        violating_set = set()
        
        for idx, ap in enumerate(aps):
            if not is_ap_valid(colors, ap):
                ap_status[idx] = False
                violating_set.add(idx)
                
        if len(violating_set) == 0:
            return colors

        temp = 5.0
        cooling = 0.995
        min_temp = 0.01
        
        best_colors = colors[:]
        best_violations = len(violating_set)
        
        while time.time() < time_limit and temp > min_temp:
            if len(violating_set) < best_violations:
                best_violations = len(violating_set)
                best_colors = colors[:]
                if best_violations == 0:
                    break
            
            if len(violating_set) == 0:
                break
                
            pos = random.randint(0, n - 1)
            old_color = colors[pos]
            new_color = random.randint(0, k - 1)
            
            if new_color == old_color:
                temp *= cooling
                continue
                
            # Calculate delta
            delta = 0
            for ap_idx in pos_to_aps[pos]:
                ap = aps[ap_idx]
                old_valid = ap_status[ap_idx]
                
                # Simulate new color
                # We need to check validity with new_color at pos
                c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                # Replace the color at pos
                if ap[0] == pos: c0 = new_color
                elif ap[1] == pos: c1 = new_color
                elif ap[2] == pos: c2 = new_color
                elif ap[3] == pos: c3 = new_color
                
                # Check validity
                new_valid = True
                if c0 == c1 == c2 == c3:
                    new_valid = False
                elif c0 == c1 == c2 or c0 == c1 == c3 or c0 == c2 == c3 or c1 == c2 == c3:
                    new_valid = False
                elif c0 == c1 and c2 == c3:
                    new_valid = False
                elif c0 == c2 and c1 == c3:
                    new_valid = False
                elif c0 == c3 and c1 == c2:
                    new_valid = False
                    
                if old_valid and not new_valid:
                    delta += 1
                elif not old_valid and new_valid:
                    delta -= 1
            
            # Acceptance
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = True
            else:
                prob = math.exp(-delta / temp)
                accept = random.random() < prob
            
            if accept:
                colors[pos] = new_color
                # Update state
                for ap_idx in pos_to_aps[pos]:
                    ap = aps[ap_idx]
                    c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                    new_valid = True
                    if c0 == c1 == c2 == c3:
                        new_valid = False
                    elif c0 == c1 == c2 or c0 == c1 == c3 or c0 == c2 == c3 or c1 == c2 == c3:
                        new_valid = False
                    elif c0 == c1 and c2 == c3:
                        new_valid = False
                    elif c0 == c2 and c1 == c3:
                        new_valid = False
                    elif c0 == c3 and c1 == c2:
                        new_valid = False
                        
                    old_valid = ap_status[ap_idx]
                    if old_valid != new_valid:
                        ap_status[ap_idx] = new_valid
                        if new_valid:
                            violating_set.discard(ap_idx)
                        else:
                            violating_set.add(ap_idx)
            
            temp *= cooling
            
        return best_colors

    # Try 3 colors first
    t_start = time.time()
    time_per_try = 20.0
    
    result = solve(3, t_start + time_per_try)
    if result is not None and all(is_ap_valid(result, ap) for ap in aps):
        return result
        
    # Try 4 colors
    t_start = time.time()
    result = solve(4, t_start + time_per_try)
    if result is not None and all(is_ap_valid(result, ap) for ap in aps):
        return result

    # Fallback: Greedy construction with many colors
    # This ensures we return a valid solution
    colors = [None] * n
    for x in range(n):
        for color in range(128):
            valid = True
            for ap_idx in pos_to_aps[x]:
                ap = aps[ap_idx]
                cset = set()
                for pos in ap:
                    if pos == x:
                        cset.add(color)
                    else:
                        if colors[pos] is None:
                            valid = False
                            break
                        cset.add(colors[pos])
                if not valid:
                    break
                if len(cset) < 3:
                    valid = False
                    break
            if valid:
                colors[x] = color
                break
        if colors[x] is None:
            colors[x] = x
            
    return colors
