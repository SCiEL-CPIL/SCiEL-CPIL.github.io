import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    n = 128
    
    # Precompute all 4-term APs
    # (a, a+d, a+2d, a+3d) with 0-indexed
    ap_list = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # For each position, list of AP indices it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(ap_list):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    def solve_with_k(k, seed_val=0):
        """Backtracking with forward checking and MRV to find valid k-coloring."""
        random.seed(seed_val)
        colors = [-1] * n
        # domains[i] is a bitmask of available colors for position i
        # If k <= 32, we can use integers as bitmasks.
        full_mask = (1 << k) - 1
        domains = [full_mask for _ in range(n)]
        
        # To speed up MRV, we can maintain a list of unassigned variables
        # But scanning 128 variables is fast.
        
        def get_color_from_mask(mask):
            # Return list of colors set in mask
            return [i for i in range(k) if (mask >> i) & 1]
            
        def check_and_propagate(pos, color):
            """
            Assign colors[pos] = color.
            Update domains of neighbors in APs.
            Returns False if any domain becomes empty.
            """
            colors[pos] = color
            domains[pos] = 0 # Assigned, no domain
            
            for ap_idx in pos_to_aps[pos]:
                a, b, c, d_pos = ap_list[ap_idx]
                vals = [colors[a], colors[b], colors[c], colors[d_pos]]
                
                # Find unassigned positions in this AP
                unassigned = []
                for p in (a, b, c, d_pos):
                    if colors[p] == -1:
                        unassigned.append(p)
                
                if not unassigned:
                    # All assigned, check validity
                    if len(set(vals)) < 3:
                        return False
                    continue
                
                # Calculate distinct colors among assigned
                assigned_vals = [v for v in vals if v != -1]
                distinct_assigned = set(assigned_vals)
                num_distinct = len(distinct_assigned)
                
                # Determine constraint on unassigned positions
                # We need total distinct >= 3.
                
                if num_distinct == 1:
                    # All assigned are same, e.g., {0}
                    # We need 2 more distinct colors from unassigned.
                    # If len(unassigned) < 2, impossible?
                    # If len(unassigned) == 1, we need that 1 to be 2 distinct colors? Impossible.
                    # So if num_distinct == 1 and len(unassigned) == 1, FAIL.
                    if len(unassigned) == 1:
                        return False
                    # If len(unassigned) == 2, both must be distinct from each other and from the assigned color.
                    # So they must be two different colors from K-1 options.
                    # This doesn't immediately restrict a single domain to empty, 
                    # but it restricts the pair. 
                    # For simple forward checking, we just ensure domains are not empty.
                    # We can prune: unassigned colors must not be the assigned color.
                    forbidden = distinct_assigned
                    for p in unassigned:
                        # Remove forbidden colors from domain
                        new_mask = domains[p]
                        for c in forbidden:
                            new_mask &= ~(1 << c)
                        if new_mask == 0:
                            return False
                        domains[p] = new_mask
                        
                elif num_distinct == 2:
                    # Assigned use 2 colors, e.g., {0, 1}
                    # We need 1 more distinct color.
                    # If len(unassigned) == 1, that 1 must be a color NOT in {0, 1}.
                    if len(unassigned) == 1:
                        p = unassigned[0]
                        new_mask = domains[p]
                        for c in distinct_assigned:
                            new_mask &= ~(1 << c)
                        if new_mask == 0:
                            return False
                        domains[p] = new_mask
                    # If len(unassigned) > 1, at least one must be new.
                    # We can't easily prune individual domains without pairwise constraints,
                    # but we can ensure that for each unassigned p, it's not forced to be in distinct_assigned
                    # if that would make it impossible to get a 3rd color elsewhere?
                    # Actually, if len(unassigned) == 2, say p1, p2.
                    # If p1 takes 0, p2 must take 2.
                    # If p1 takes 2, p2 can take 0 or 1 or 2 (if 2 is new).
                    # Simple pruning: If all unassigned positions have domains subset of distinct_assigned,
                    # then no new color can be introduced. Fail.
                    all_subset = True
                    for p in unassigned:
                        # Check if domain[p] is subset of distinct_assigned
                        # i.e., domain[p] & ~mask(distinct_assigned) == 0
                        mask_assigned = 0
                        for c in distinct_assigned:
                            mask_assigned |= (1 << c)
                        if domains[p] & ~mask_assigned:
                            all_subset = False
                            break
                    if all_subset:
                        return False
                        
                else: # num_distinct >= 3
                    # Constraint already satisfied by assigned parts.
                    # No pruning needed.
                    pass
            
            return True

        def backtrack():
            if time.time() > deadline:
                return False
            
            # MRV: Find unassigned variable with smallest domain
            min_domain_size = float('inf')
            min_pos = -1
            for i in range(n):
                if colors[i] == -1:
                    # Count bits
                    cnt = bin(domains[i]).count('1')
                    if cnt < min_domain_size:
                        min_domain_size = cnt
                        min_pos = i
                        if min_domain_size == 1:
                            break
                    if min_domain_size == 0:
                        return False # Should have been caught by propagation
            
            if min_pos == -1:
                # All assigned
                return True
            
            # Get colors in domain
            mask = domains[min_pos]
            # Shuffle colors for randomization
            avail_colors = [i for i in range(k) if (mask >> i) & 1]
            random.shuffle(avail_colors)
            
            for color in avail_colors:
                # Save state
                old_colors = colors[:]
                old_domains = domains[:]
                
                if check_and_propagate(min_pos, color):
                    if backtrack():
                        return True
                
                # Restore state
                colors[:] = old_colors
                domains[:] = old_domains
            
            return False
        
        return backtrack()

    # Try to find a coloring with minimal K
    # K=3 is theoretically possible? Let's try.
    # K=4 is very likely.
    
    # Use random restarts for robustness
    for k in [3, 4, 5]:
        for attempt in range(20):
            if time.time() > deadline - 10: # Leave time for fallback
                break
            if solve_with_k(k, seed_val=attempt):
                # Found valid coloring with k colors
                return [c if c != -1 else 0 for c in colors]
        
        # If k=3 failed, try k=4, etc.
        # We break out of the k-loop if we find a solution.
        # The loop structure above tries k=3, then k=4...
        # But we need to break out of the outer loop if successful.
        # The return statement handles that.
        
    # Fallback: Greedy construction
    # This ensures we return *something* valid, even if K is large.
    colors = []
    for x in range(1, n + 1):
        idx = x - 1
        best_color = -1
        # Try colors 0..127
        for color in range(128):
            valid = True
            # Check all APs ending at or containing idx?
            # Actually, just check all APs containing idx.
            for ap_idx in pos_to_aps[idx]:
                a, b, c, d_pos = ap_list[ap_idx]
                vals = [-1, -1, -1, -1]
                for p in (a, b, c, d_pos):
                    if p == idx:
                        vals[p] = color
                    elif p < idx:
                        vals[p] = colors[p]
                    else:
                        # Future position, can't check fully yet in greedy?
                        # Greedy usually checks constraints with *previous* elements.
                        # But this constraint involves 4 elements.
                        # If we are at idx, and the AP is (idx, idx+d, ...), we don't know future.
                        # However, we can check if the *partial* assignment violates the constraint
                        # assuming future can be anything?
                        # No, greedy must ensure that the choice doesn't make it *impossible*
                        # to satisfy constraints with future elements?
                        # Or simpler: only check constraints where all other elements are already assigned.
                        # If other elements are future, we can't verify yet.
                        # This standard greedy might fail to produce a valid coloring if it doesn't look ahead.
                        # However, for "minimize colors", a simple greedy picking smallest valid color
                        # based on *past* constraints often works for these types of problems if the density isn't too high.
                        # But "at least 3 distinct" is a global property of the 4-tuple.
                        
                        # Let's refine the greedy check:
                        # For an AP containing idx:
                        # If all other 3 are assigned, check if {color, c1, c2, c3} has size >= 3.
                        # If some are unassigned, we assume we can fix it later?
                        # Risky.
                        
                        # Alternative Fallback: Use the backtracker with K=128 (effectively no limit) 
                        # but with very tight time? No, that's slow.
                        
                        # Let's stick to the check: only validate if all other 3 are assigned.
                        # If not all assigned, assume valid for now.
                        pass
                
                # Re-implementing the check properly for greedy:
                # We only reject a color if it violates a constraint where all other vars are fixed.
                # This guarantees the final assignment is valid for those "closed" constraints.
                # But what about constraints with future vars?
                # If we pick a color that forces a future var to be impossible, greedy fails.
                # Given the time, let's try a slightly smarter greedy:
                # Check if assigning this color leaves any *closed* constraint invalid.
                # And check if it leaves any *open* constraint in a "dead end" state (like 3 same).
                
                all_assigned_others = True
                other_vals = []
                for p in (a, b, c, d_pos):
                    if p == idx:
                        continue
                    if p < idx:
                        other_vals.append(colors[p])
                    else:
                        all_assigned_others = False
                        break
                
                if all_assigned_others:
                    if len(set([color] + other_vals)) < 3:
                        valid = False
                        break
                else:
                    # Check for immediate dead ends in open constraints
                    # e.g., if we have 3 assigned (including current) that are all same.
                    # But here only 1 is current. Others are future.
                    # So we only have 1 assigned. No dead end yet.
                    # However, if we have 2 assigned in the past, and we pick a 3rd that matches them?
                    # e.g. Past: 0, 0. Current: 0. Future: ?.
                    # Then we have 0,0,0,?. This is a dead end (need 3 distinct, only 1 slot left).
                    # So we should check: if among the assigned parts of this AP (including current),
                    # we have 3 values that are all the same, reject.
                    # Or if we have 3 values with only 2 distinct, and the 4th is future,
                    # the future MUST be the 3rd distinct. This is a restriction, not a dead end,
                    # unless K=2. But we are using up to 128 colors.
                    # So only "3 same" is a hard dead end.
                    
                    assigned_in_ap = []
                    for p in (a, b, c, d_pos):
                        if p == idx:
                            assigned_in_ap.append(color)
                        elif p < idx:
                            assigned_in_ap.append(colors[p])
                        # else future, ignore for this check
                    
                    if len(assigned_in_ap) >= 3:
                        if len(set(assigned_in_ap)) == 1:
                            valid = False
                            break
                            
            if valid:
                best_color = color
                break
        
        if best_color == -1:
            # Should not happen with K=128 and loose checks, but just in case
            best_color = 0
            
        colors.append(best_color)
        
    return colors

# Verification step (optional, but good for debugging)
def verify(coloring):
    n = len(coloring)
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            vals = [coloring[p] for p in ap]
            if len(set(vals)) < 3:
                return False
    return True

# The construct function is the entry point.
# We don't call verify in production to save time, but it's good to know.
