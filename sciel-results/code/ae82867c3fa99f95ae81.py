import sys
import time

def construct():
    N = 128
    # Precompute all 4-APs
    # A 4-AP is (a, a+d, a+2d, a+3d) with a+3d < N
    aps = []
    for a in range(N):
        max_d = (N - 1 - a) // 3
        for d in range(1, max_d + 1):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Map position to list of AP indices containing it
    pos_to_aps = [[] for _ in range(N)]
    for i, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(i)
            
    def solve(k):
        # colors[i] = -1 if unassigned, else 0..k-1
        colors = [-1] * N
        # domains[i] = bitmask of available colors
        domains = [(1 << k) - 1] * N
        
        # Symmetry breaking: Fix first few positions
        # If k >= 1, c[0]=0
        # If k >= 2, c[1]=1
        # If k >= 3, c[2]=2
        # This reduces symmetry.
        
        def assign(pos, c):
            colors[pos] = c
            domains[pos] = 1 << c
            
        def unassign(pos):
            colors[pos] = -1
            domains[pos] = (1 << k) - 1
            
        # Forward checking / Propagation
        # Returns False if a contradiction is found
        def check_and_propagate(pos, c, depth):
            # Check all APs containing pos
            for ap_idx in pos_to_aps[pos]:
                a, b, cc, d = aps[ap_idx]
                # Get colors of the 4 positions
                # We know colors[pos] is now c.
                # Check the state of the AP
                vals = [colors[x] for x in (a, b, cc, d)]
                # Count assigned
                assigned_vals = [v for v in vals if v != -1]
                unassigned_pos = [x for x in (a, b, cc, d) if colors[x] == -1]
                
                if not unassigned_pos:
                    # All assigned
                    if len(set(assigned_vals)) < 3:
                        return False
                elif len(unassigned_pos) == 1:
                    u = unassigned_pos[0]
                    s_assigned = set(assigned_vals)
                    if len(s_assigned) == 1:
                        # 3 assigned have same color. 
                        # Final set will be {c1, cu}. Size <= 2. 
                        # This is a contradiction.
                        return False
                    elif len(s_assigned) == 2:
                        # 3 assigned have 2 distinct colors {c1, c2}.
                        # Final set {c1, c2, cu}. Size >= 3 iff cu not in {c1, c2}.
                        forbidden = s_assigned
                        # Check if u's domain allows any color not in forbidden
                        avail = domains[u] & ~((1 << k) - 1) # This is wrong logic for mask
                        # Let's just iterate
                        valid = False
                        for color in range(k):
                            if color not in forbidden:
                                if (domains[u] >> color) & 1:
                                    valid = True
                                    break
                        if not valid:
                            return False
                        # Propagate: restrict u's domain to colors not in forbidden
                        new_mask = 0
                        for color in range(k):
                            if color not in forbidden:
                                new_mask |= (1 << color)
                        if domains[u] != new_mask:
                            domains[u] = new_mask
                    else: # len(s_assigned) == 3
                        # Already satisfied
                        pass
                elif len(unassigned_pos) == 2:
                    u1, u2 = unassigned_pos
                    s_assigned = set(assigned_vals)
                    if len(s_assigned) == 1:
                        # 2 assigned have same color c1.
                        # Unassigned u1, u2.
                        # Final set {c1, cu1, cu2}. Size >= 3.
                        # Requires cu1 != c1, cu2 != c1, cu1 != cu2.
                        # So u1 and u2 must take distinct colors, neither being c1.
                        # This is a "distinct" constraint between u1 and u2, plus exclusion of c1.
                        # We can propagate exclusion of c1 to u1 and u2 domains.
                        mask_excl_c1 = domains[u1] & ~(1 << list(s_assigned)[0])
                        if mask_excl_c1 == 0: return False
                        domains[u1] = mask_excl_c1
                        mask_excl_c1_2 = domains[u2] & ~(1 << list(s_assigned)[0])
                        if mask_excl_c1_2 == 0: return False
                        domains[u2] = mask_excl_c1_2
                        # The distinctness cu1 != cu2 is handled by search, 
                        # but if domains become size 1 and equal, fail.
                        if domains[u1] == domains[u2] and bin(domains[u1]).count('1') == 1:
                            return False
                    elif len(s_assigned) == 2:
                        # 2 assigned have colors {c1, c2}.
                        # Unassigned u1, u2.
                        # Final set {c1, c2, cu1, cu2}. Size >= 3.
                        # Requires at least one of cu1, cu2 to be outside {c1, c2}.
                        # This is an OR constraint. Hard to propagate to single domains 
                        # without arc consistency. We skip tight propagation here.
                        pass
                    else: # len(s_assigned) == 0 (impossible since pos is assigned and in AP)
                        pass
                else:
                    # 3 or 4 unassigned. 
                    # If 4 unassigned, no constraint.
                    # If 3 unassigned, 1 assigned. No constraint on single domains yet.
                    pass
            return True

        def backtrack(depth):
            # Select variable with MRV
            min_size = k + 1
            var = -1
            for i in range(N):
                if colors[i] == -1:
                    size = bin(domains[i]).count('1')
                    if size < min_size:
                        min_size = size
                        var = i
                        if size == 1:
                            break
            if var == -1:
                return True # All assigned
            
            # If a domain is empty, fail
            if min_size == 0:
                return False
            
            # Try values
            # Heuristic: try colors in order, or maybe least constraining?
            # For now, simple order 0..k-1
            old_domains = domains[:]
            for c in range(k):
                if (domains[var] >> c) & 1:
                    # Assign
                    old_dom_var = domains[var]
                    assign(var, c)
                    
                    if check_and_propagate(var, c, depth):
                        if backtrack(depth + 1):
                            return True
                    
                    # Undo
                    unassign(var)
                    domains[var] = old_dom_var
                    domains[:] = old_domains[:] # Restore all domains
                    
            return False

        # Initial symmetry breaking
        # c[0] = 0
        if k > 0:
            assign(0, 0)
            if not check_and_propagate(0, 0, 0):
                return None
        # c[1] = 1 (if k > 1)
        if k > 1:
            if (domains[1] >> 1) & 1:
                assign(1, 1)
                if not check_and_propagate(1, 1, 0):
                    unassign(1)
                    return None
            else:
                return None
        # c[2] = 2 (if k > 2)
        if k > 2:
            if (domains[2] >> 2) & 1:
                assign(2, 2)
                if not check_and_propagate(2, 2, 0):
                    unassign(2)
                    return None
            else:
                return None
        
        if backtrack(3):
            return colors
        return None

    # Try increasing k
    # Theoretical min is likely 4 or 5.
    # Start from 3? 3 colors might be impossible.
    start_time = time.time()
    deadline = start_time + 85.0 # 5s margin
    
    for k in range(3, 10):
        if time.time() > deadline:
            break
        res = solve(k)
        if res is not None:
            return res
            
    # Fallback: Greedy construction (should not happen if k is high enough)
    # But if we run out of time, we need to return *something* valid?
    # The task says "Minimize colors". If we don't find the min, we return the best found?
    # The interface implies we must return a valid coloring.
    # If the solver fails due to time, we should run a fast greedy to ensure validity.
    
    colors = []
    for x in range(1, 129):
        # Try colors 0..127
        for color in range(128):
            valid = True
            # Check all 4-APs ending at x-1 (0-indexed) or involving x-1
            # We need to check all APs where x-1 is a member.
            idx = x - 1
            for ap_idx in pos_to_aps[idx]:
                a, b, cc, d = aps[ap_idx]
                # Build the set of colors for this AP
                # We know colors for indices < idx are in 'colors' list
                # color for idx is 'color'
                vals = []
                for p in (a, b, cc, d):
                    if p == idx:
                        vals.append(color)
                    else:
                        vals.append(colors[p])
                if len(set(vals)) < 3:
                    valid = False
                    break
            if valid:
                colors.append(color)
                break
        else:
            # Should not happen with 128 colors
            colors.append(0)
    return colors

# To ensure the fallback is fast and robust, we pre-compute pos_to_aps at module level
# but the function above defines it internally. 
# Let's move the precomputation to the top of the file for speed.

N = 128
aps = []
for a in range(N):
    max_d = (N - 1 - a) // 3
    for d in range(1, max_d + 1):
        aps.append((a, a + d, a + 2 * d, a + 3 * d))

pos_to_aps = [[] for _ in range(N)]
for i, ap in enumerate(aps):
    for pos in ap:
        pos_to_aps[pos].append(i)

def construct():
    N = 128
    
    def solve(k):
        colors = [-1] * N
        domains = [(1 << k) - 1] * N
        
        def assign(pos, c):
            colors[pos] = c
            domains[pos] = 1 << c
            
        def unassign(pos):
            colors[pos] = -1
            domains[pos] = (1 << k) - 1
            
        def check_and_propagate(pos, c):
            for ap_idx in pos_to_aps[pos]:
                a, b, cc, d = aps[ap_idx]
                vals = [colors[x] for x in (a, b, cc, d)]
                assigned_vals = [v for v in vals if v != -1]
                unassigned_pos = [x for x in (a, b, cc, d) if colors[x] == -1]
                
                if not unassigned_pos:
                    if len(set(assigned_vals)) < 3:
                        return False
                elif len(unassigned_pos) == 1:
                    u = unassigned_pos[0]
                    s_assigned = set(assigned_vals)
                    if len(s_assigned) == 1:
                        return False
                    elif len(s_assigned) == 2:
                        forbidden = s_assigned
                        valid = False
                        for color in range(k):
                            if color not in forbidden:
                                if (domains[u] >> color) & 1:
                                    valid = True
                                    break
                        if not valid:
                            return False
                        new_mask = 0
                        for color in range(k):
                            if color not in forbidden:
                                new_mask |= (1 << color)
                        if domains[u] != new_mask:
                            domains[u] = new_mask
                elif len(unassigned_pos) == 2:
                    u1, u2 = unassigned_pos
                    s_assigned = set(assigned_vals)
                    if len(s_assigned) == 1:
                        c1 = list(s_assigned)[0]
                        mask_excl = ~(1 << c1)
                        if domains[u1] & mask_excl == 0: return False
                        if domains[u2] & mask_excl == 0: return False
                        domains[u1] &= mask_excl
                        domains[u2] &= mask_excl
                        if domains[u1] == domains[u2] and bin(domains[u1]).count('1') == 1:
                            return False
                return True
            return True

        def backtrack(depth):
            min_size = k + 1
            var = -1
            for i in range(N):
                if colors[i] == -1:
                    size = bin(domains[i]).count('1')
                    if size < min_size:
                        min_size = size
                        var = i
                        if size == 1:
                            break
            if var == -1:
                return True
            if min_size == 0:
                return False
            
            old_domains = domains[:]
            for c in range(k):
                if (domains[var] >> c) & 1:
                    old_dom_var = domains[var]
                    assign(var, c)
                    if check_and_propagate(var, c):
                        if backtrack(depth + 1):
                            return True
                    unassign(var)
                    domains[var] = old_dom_var
                    domains[:] = old_domains[:]
            return False

        # Symmetry breaking
        if k > 0:
            assign(0, 0)
            if not check_and_propagate(0, 0): return None
        if k > 1:
            if (domains[1] >> 1) & 1:
                assign(1, 1)
                if not check_and_propagate(1, 1):
                    unassign(1)
                    return None
            else: return None
        if k > 2:
            if (domains[2] >> 2) & 1:
                assign(2, 2)
                if not check_and_propagate(2, 2):
                    unassign(2)
                    return None
            else: return None
        
        if backtrack(3):
            return colors
        return None

    start_time = time.time()
    deadline = start_time + 85.0
    
    for k in range(3, 10):
        if time.time() > deadline:
            break
        res = solve(k)
        if res is not None:
            return res
            
    # Fallback
    colors = []
    for x in range(1, 129):
        for color in range(128):
            valid = True
            idx = x - 1
            for ap_idx in pos_to_aps[idx]:
                a, b, cc, d = aps[ap_idx]
                vals = []
                for p in (a, b, cc, d):
                    if p == idx:
                        vals.append(color)
                    else:
                        vals.append(colors[p])
                if len(set(vals)) < 3:
                    valid = False
                    break
            if valid:
                colors.append(color)
                break
        else:
            colors.append(0)
    return colors
