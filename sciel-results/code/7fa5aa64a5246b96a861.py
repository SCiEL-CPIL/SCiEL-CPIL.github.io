import time
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s deadline, 5s margin
    
    n = 128
    
    # Precompute all 4-APs and their relationships
    # For each position i, store which APs it participates in and its role
    # ap_roles[i] = list of (ap_index, role) where role is 0,1,2,3 (position in AP)
    
    # Generate all 4-APs
    aps = []
    ap_by_pos = [[] for _ in range(n + 1)]  # 1-indexed
    ap_idx = 0
    for d in range(1, n//3 + 1):
        for a in range(1, n - 3*d + 1):
            terms = (a, a+d, a+2*d, a+3*d)
            aps.append(terms)
            for role, pos in enumerate(terms):
                ap_by_pos[pos].append((ap_idx, role))
            ap_idx += 1
    
    # For backtracking with color masks
    # colors[i] = color assigned to position i, or -1 if unassigned
    colors = [-1] * (n + 1)
    
    # Symmetry breaking: fix c[1] = 0
    colors[1] = 0
    
    # For each position, maintain a bitmask of valid colors
    # valid_mask[i] = bitmask where bit c is 1 if color c is valid for position i
    # Initially all colors 0-127 are valid
    all_colors_mask = (1 << 128) - 1
    valid_mask = [all_colors_mask] * (n + 1)
    valid_mask[0] = 0
    valid_mask[1] = 1  # Only color 0 for position 1
    
    # Precompute for each AP and each role, what constraints it imposes
    # When position i is assigned color c, check all APs where i appears
    
    def check_assignment(i, c):
        """Check if assigning color c to position i is consistent with previous assignments.
        Returns True if valid, False otherwise."""
        for ap_idx, role in ap_by_pos[i]:
            terms = aps[ap_idx]
            # Get colors of the other 3 positions
            other_colors = []
            for r in range(4):
                if r == role:
                    continue
                pos = terms[r]
                if pos < i:  # Already assigned
                    other_colors.append(colors[pos])
                else:
                    other_colors.append(-1)
            
            # Count how many of the other 3 are assigned
            assigned_count = sum(1 for oc in other_colors if oc != -1)
            
            if assigned_count == 3:
                # All other 3 are assigned. Check if {c, oc1, oc2, oc3} has >= 3 distinct colors
                distinct = len(set([c] + other_colors))
                if distinct < 3:
                    return False
            elif assigned_count == 2:
                # Two assigned, one unassigned
                assigned_oc = [oc for oc in other_colors if oc != -1]
                # If the two assigned are the same color, and c is also that color,
                # then even if the 4th is different, we have at most 2 colors. FAIL.
                if assigned_oc[0] == assigned_oc[1] == c:
                    return False
            elif assigned_count == 1:
                # One assigned, two unassigned
                # No immediate violation possible
                pass
        
        return True
    
    def backtrack(i):
        if time.time() > deadline:
            return False
        if i > n:
            return True
        
        # Get valid colors for position i
        mask = valid_mask[i]
        if mask == 0:
            return False
        
        # Try each valid color
        c = 0
        while c < 128:
            if mask & (1 << c):
                if check_assignment(i, c):
                    colors[i] = c
                    # Update valid masks for future positions
                    saved_masks = []
                    for j in range(i + 1, n + 1):
                        saved_masks.append(valid_mask[j])
                    
                    # Propagate constraints
                    for ap_idx, role in ap_by_pos[i]:
                        terms = aps[ap_idx]
                        for r in range(4):
                            if r == role:
                                continue
                            pos = terms[r]
                            if pos > i:
                                # This position is affected by our assignment
                                # Check which colors are still valid for pos
                                new_mask = all_colors_mask
                                for c2 in range(128):
                                    if not (new_mask & (1 << c2)):
                                        continue
                                    # Check if assigning c2 to pos would be valid given current state
                                    # We need to check all APs involving pos
                                    # For efficiency, only check APs that are now "tight"
                                    # Simplified: just check the AP we're in
                                    other_colors = []
                                    for r2 in range(4):
                                        if r2 == r:
                                            continue
                                        p2 = terms[r2]
                                        if p2 == i:
                                            other_colors.append(c)
                                        elif p2 < i:
                                            other_colors.append(colors[p2])
                                        else:
                                            other_colors.append(-1)
                                    
                                    assigned_count = sum(1 for oc in other_colors if oc != -1)
                                    
                                    if assigned_count == 3:
                                        distinct = len(set([c2] + other_colors))
                                        if distinct < 3:
                                            new_mask &= ~(1 << c2)
                                    elif assigned_count == 2:
                                        assigned_oc = [oc for oc in other_colors if oc != -1]
                                        if assigned_oc[0] == assigned_oc[1] == c2:
                                            new_mask &= ~(1 << c2)
                                
                                valid_mask[pos] &= new_mask
                    
                    if backtrack(i + 1):
                        return True
                    
                    # Backtrack: restore masks
                    for j in range(i + 1, n + 1):
                        valid_mask[j] = saved_masks[j - i - 1]
                
                colors[i] = -1
            
            c += 1
            if c < 128:
                mask &= ~(1 << (c - 1))
        
        return False
    
    if backtrack(2):
        return colors[1:n+1]
        
    # Fallback to greedy if backtracking fails or times out
    result = []
    for x in range(1, 129):
        assigned = False
        for color in range(128):
            valid = True
            # Check all 4-APs where x participates
            for ap_idx, role in ap_by_pos[x]:
                terms = aps[ap_idx]
                other_colors = []
                for r in range(4):
                    if r == role:
                        continue
                    pos = terms[r]
                    if pos < x:
                        other_colors.append(result[pos - 1])
                    else:
                        other_colors.append(-1)
                
                assigned_count = sum(1 for oc in other_colors if oc != -1)
                
                if assigned_count == 3:
                    distinct = len(set([color] + other_colors))
                    if distinct < 3:
                        valid = False
                        break
                elif assigned_count == 2:
                    assigned_oc = [oc for oc in other_colors if oc != -1]
                    if assigned_oc[0] == assigned_oc[1] == color:
                        valid = False
                        break
            
            if valid:
                result.append(color)
                assigned = True
                break
        
        if not assigned:
            result.append(0)
    
    return result
