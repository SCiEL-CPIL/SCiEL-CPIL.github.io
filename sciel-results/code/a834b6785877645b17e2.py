import time
import random
import sys
from collections import defaultdict

def construct():
    n = 128
    deadline = time.time() + 85.0  # Safety margin before 90s kill

    # Precompute all 4-term APs (0-indexed): (i, i+d, i+2d, i+3d)
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for i in range(0, n - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))

    # For each position, list of (ap_index, role) where role is 0,1,2,3
    pos_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for role, pos in enumerate(ap):
            pos_aps[pos].append((idx, role))

    # For each AP, precompute the 4 positions
    ap_positions = [ap for ap in aps]

    def count_violations(colors):
        v = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                v += 1
        return v

    def backtrack_solve(k, time_limit, seed=0):
        """
        Backtracking with forward checking to find a valid k-coloring.
        Returns colors list if found, else None.
        """
        rng = random.Random(seed)
        colors = [-1] * n  # -1 means unassigned

        # For each AP, track how many distinct colors are assigned and what they are
        # ap_color_counts[ap_idx] = dict mapping color -> count of positions with that color
        ap_color_counts = [dict() for _ in range(len(aps))]
        # ap_assigned[ap_idx] = number of positions assigned in this AP
        ap_assigned = [0] * len(aps)

        # For forward checking: for each unassigned position, track which colors are still "safe"
        # A color is unsafe for position p if assigning it would make some AP containing p invalid
        # An AP becomes invalid if all 4 positions are assigned and use < 3 distinct colors
        # But also: if 3 positions are assigned using only 1 or 2 colors, and the 4th must avoid
        # making it worse... actually the constraint is: at least 3 distinct colors among 4.
        
        # Let me think more carefully. The constraint is: for every 4-AP, the 4 colors must have
        # at least 3 distinct values. This is violated when:
        # - All 4 same
        # - 3 same + 1 different (wait, that's 2 distinct colors - VIOLATED)
        # - 2 same + 2 same (different pairs) - 2 distinct colors - VIOLATED
        # - 2 same + 1 + 1 - 3 distinct - OK
        # - 1 + 1 + 1 + 1 - 4 distinct - OK
        # - 3 same + 1 different - 2 distinct - VIOLATED
        # So valid patterns: those with exactly 3 or 4 distinct colors.
        # Invalid: exactly 1 or exactly 2 distinct colors.

        # For forward checking, when we assign a color to a position, we need to check
        # all APs containing that position. For each such AP:
        # - If the AP is now fully assigned, check if it's valid.
        # - If not fully assigned, check if the remaining unassigned positions can still
        #   make the AP valid.
        
        # When 3 of 4 positions are assigned:
        #   - If those 3 use 3 distinct colors: the 4th can be anything (still >= 3)
        #   - If those 3 use 2 distinct colors (e.g., A,A,B): the 4th must NOT be A or B
        #     (must be a new color C to get 3 distinct)
        #   - If those 3 use 1 distinct color (A,A,A): the 4th must NOT be A
        #     (must be B or C to get 2 distinct... wait, that's only 2 distinct. VIOLATED!)
        #     Actually if 3 are A,A,A and 4th is B: colors are A,A,A,B -> 2 distinct -> VIOLATED
        #     If 4th is C: A,A,A,C -> 2 distinct -> VIOLATED
        #     So if 3 positions all have the same color, the AP is ALREADY doomed!
        #     No matter what the 4th color is, we get at most 2 distinct colors.
        #     
        # Wait, let me recount. A,A,A,B: distinct colors are {A, B} = 2 distinct. VIOLATED.
        # A,A,A,C: distinct colors are {A, C} = 2 distinct. VIOLATED.
        # So yes, if 3 positions in an AP all have the same color, it's impossible to fix.
        
        # When 2 of 4 positions are assigned:
        #   - If both same color (A,A): remaining 2 must include at least 2 new colors? No.
        #     We need total >= 3 distinct. With A,A, we need the other 2 to provide at least 2 new colors.
        #     So the other 2 must have different colors, and neither can be A.
        #     Actually: A,A,B,C -> {A,B,C} = 3 distinct OK. A,A,B,B -> {A,B} = 2 VIOLATED.
        #     A,A,B,A -> {A,B} = 2 VIOLATED. A,A,A,B -> {A,B} = 2 VIOLATED.
        #     So after A,A: the remaining two must be two distinct colors, both different from A.
        #   - If different colors (A,B): remaining 2 can be anything as long as total >= 3 distinct.
        #     A,B,C,D -> 4 distinct OK. A,B,C,C -> {A,B,C} = 3 OK. A,B,A,B -> {A,B} = 2 VIOLATED.
        #     A,B,A,C -> {A,B,C} = 3 OK. A,B,B,C -> {A,B,C} = 3 OK. A,B,A,A -> {A,B} = 2 VIOLATED.
        #     So after A,B: we can't have the remaining two both be A, both be B, or one A and one B
        #     (in any combination that gives only {A,B}).
        #     Specifically: forbidden if remaining two are both from {A,B} and don't introduce a new color.
        #     i.e., remaining must not be (A,A), (B,B), (A,B), (B,A). So at least one of the remaining
        #     two must be a color other than A or B.

        # This is getting complex. Let me use a simpler forward checking:
        # After assigning a position, for each affected AP, check if it's still "possible"
        # to complete validly. If not, prune.

        def is_ap_possible(ap_idx, colors):
            """Check if an AP can still be completed to be valid given current assignments."""
            ap = ap_positions[ap_idx]
            assigned_colors = []
            unassigned_count = 0
            for pos in ap:
                if colors[pos] == -1:
                    unassigned_count += 1
                else:
                    assigned_colors.append(colors[pos])
            
            if unassigned_count == 0:
                # Fully assigned, check validity
                return len(set(assigned_colors)) >= 3
            
            distinct_assigned = set(assigned_colors)
            num_distinct = len(distinct_assigned)
            
            if unassigned_count == 1:
                # One position left. We need final distinct count >= 3.
                # If num_distinct >= 3: any color works for the last one.
                # If num_distinct == 2: the last color must be new (not in distinct_assigned)
                # If num_distinct == 1: the last color must be new, but even then we only get 2 distinct. VIOLATED.
                # If num_distinct == 0: impossible since we said unassigned_count==1 means 3 assigned.
                if num_distinct >= 3:
                    return True
                elif num_distinct == 2:
                    return True  # As long as the last color is not in the set, but we'll handle that in domain
                else:  # num_distinct <= 1
                    return False
            
            elif unassigned_count == 2:
                # Two positions left. We need to be able to assign them to get >= 3 distinct total.
                if num_distinct >= 3:
                    return True
                elif num_distinct == 2:
                    # Need at least one of the two new colors to be different from both assigned colors
                    # This is always possible if k >= 3 (we can pick a new color for one of them)
                    return k >= 3
                elif num_distinct == 1:
                    # Two positions left, need to add 2 new distinct colors (both different from the one assigned)
                    # Possible if k >= 3
                    return k >= 3
                else:  # num_distinct == 0
                    # All 4 unassigned, trivially possible
                    return True
            
            else:  # unassigned_count >= 3
                return True

        def get_domain(pos, colors):
            """Get the set of valid colors for position pos given current assignments."""
            if colors[pos] != -1:
                return {colors[pos]}
            
            valid_colors = set(range(k))
            
            for ap_idx, role in pos_aps[pos]:
                ap = ap_positions[ap_idx]
                # Check if this AP constrains the color at pos
                assigned_colors = []
                unassigned_in_ap = 0
                for p in ap:
                    if p == pos:
                        continue
                    if colors[p] == -1:
                        unassigned_in_ap += 1
                    else:
                        assigned_colors.append(colors[p])
                
                distinct_assigned = set(assigned_colors)
                num_distinct = len(distinct_assigned)
                
                if unassigned_in_ap == 0:
                    # All other 3 positions assigned. pos is the last one.
                    if num_distinct >= 3:
                        pass  # Any color works
                    elif num_distinct == 2:
                        # Last color must be new
                        valid_colors -= distinct_assigned
                    else:  # num_distinct <= 1
                        # Impossible, but let's still filter
                        valid_colors -= distinct_assigned
                        # Actually if num_distinct == 1, even a new color gives only 2 distinct. 
                        # The AP is already doomed. But we can't fix that here.
                        # Mark as no valid colors?
                        if num_distinct == 1:
                            valid_colors = set()
                            break
                
                elif unassigned_in_ap == 1:
                    # One other position unassigned. 
                    # If num_distinct == 1: we need pos and the other unassigned to provide 2 new colors.
                    # pos must be a new color (not in distinct_assigned), and the other must be different from pos.
                    # So pos must not be in distinct_assigned.
                    if num_distinct == 1:
                        valid_colors -= distinct_assigned
                    elif num_distinct == 2:
                        # We need at least one of {pos, other_unassigned} to be a new color.
                        # If pos is new, other can be anything. If pos is not new, other must be new.
                        # So pos can be anything, BUT if pos is in distinct_assigned, other must be new.
                        # This doesn't restrict pos directly, but we should note it.
                        # For now, don't restrict pos.
                        pass
                    # num_distinct >= 3: no restriction
                
                elif unassigned_in_ap == 2:
                    # Two other positions unassigned.
                    if num_distinct == 0:
                        pass  # No restriction
                    elif num_distinct == 1:
                        # Need 2 more distinct colors from {pos, other1, other2}. 
                        # pos can be anything, but if pos is in distinct_assigned, we need other1 and other2
                        # to be two new distinct colors. This is always possible if k >= 3.
                        # So no direct restriction on pos.
                        pass
                    else:
                        pass
                
                # unassigned_in_ap >= 3: no restriction (plenty of room)
            
            return valid_colors

        # Variable ordering: pick the unassigned position with the smallest domain (MRV heuristic)
        # But computing full domains is expensive. Use a simpler heuristic:
        # Pick the position that appears in the most "tight" APs (APs with few unassigned positions)
        
        assigned_count = 0
        solution = [None] * n

        def solve():
            nonlocal assigned_count, solution
            
            if time.time() > time_limit:
                return False
            
            if assigned_count == n:
                # Verify solution
                for ap in aps:
                    if len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}) < 3:
                        return False
                solution[:] = colors[:]
                return True
            
            # Find unassigned position with most constraints (heuristic)
            best_pos = -1
            best_score = -1
            for pos in range(n):
                if colors[pos] != -1:
                    continue
                # Score: number of APs where this position is critical
                score = 0
                for ap_idx, role in pos_aps[pos]:
                    ap = ap_positions[ap_idx]
                    unassigned = sum(1 for p in ap if colors[p] == -1)
                    # More critical when fewer unassigned
                    score += (4 - unassigned)
                if score > best_score:
                    best_score = score
                    best_pos = pos
            
            if best_pos == -1:
                return False  # Shouldn't happen
            
            # Try colors for best_pos
            domain = get_domain(best_pos, colors)
            if not domain:
                return False
            
            color_list = list(domain)
            rng.shuffle(color_list)
            
            for c in color_list:
                colors[best_pos] = c
                assigned_count += 1
                
                # Check if any AP containing best_pos is now impossible
                valid = True
                for ap_idx, role in pos_aps[best_pos]:
                    if not is_ap_possible(ap_idx, colors):
                        valid = False
                        break
                
                if valid:
                    if solve():
                        return True
                
                colors[best_pos] = -1
                assigned_count -= 1
            
            return False

        # Initialize with some random assignments to break symmetry
        # Assign a few positions randomly
        initial_assignments = min(8, n)
        positions = list(range(n))
        rng.shuffle(positions)
        for i in range(initial_assignments):
            pos = positions[i]
            c = rng.randint(0, k - 1)
            colors[pos] = c
            assigned_count += 1
            for ap_idx, role in pos_aps[pos]:
                if not is_ap_possible(ap_idx, colors):
                    # Bad initial assignment, reset
                    colors[pos] = -1
                    assigned_count -= 1
                    break
            else:
                continue
            break

        # Hmm, the initial assignment logic is messy. Let me simplify: just start with all unassigned.
        # Reset
        colors[:] = [-1] * n
        assigned_count = 0

        if solve():
            return solution[:]
        return None

    # Try increasing k
    k_values = [3, 4, 5, 6, 7, 8]
    
    for k in k_values:
        if time.time() > deadline - 5.0:
            break
        
        # Give each k a time slice
        time_slice = 12.0 if k <= 4 else 8.0
        if time.time() + time_slice > deadline - 2.0:
            time_slice = max(1.0, deadline - 2.0 - time.time())
        
        # Try multiple seeds
        result = None
        start_time = time.time()
        seed = 0
        while time.time() < start_time + time_slice and result is None:
            result = backtrack_solve(k, time.time(), seed=seed)
            seed += 1
        
        if result is not None:
            return result
    
    # Fallback: Identity mapping (always valid, uses 128 colors)
    return list(range(n))
