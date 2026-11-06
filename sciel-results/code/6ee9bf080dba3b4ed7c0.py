"""Combined approach: Randomized repair and local search for 4-AP colorings."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    deadline = time.time() + 85  # Safety margin before 90s kill
    N = 128

    # Precompute all non-constant 4-term APs
    ap_list = []
    for a in range(N):
        for d in range(1, (N - 1 - a) // 3 + 1):
            b, c, e = a + d, a + 2 * d, a + 3 * d
            if e < N:
                ap_list.append((a, b, c, e))

    # For each position, precompute which APs it participates in
    pos_aps = [[] for _ in range(N)]
    for idx, (a, b, c, e) in enumerate(ap_list):
        pos_aps[a].append(idx)
        pos_aps[b].append(idx)
        pos_aps[c].append(idx)
        pos_aps[e].append(idx)

    def count_violations(colors):
        """Count number of 4-APs with < 3 distinct colors."""
        count = 0
        for a, b, c, e in ap_list:
            ca, cb, cc, ce = colors[a], colors[b], colors[c], colors[e]
            # Check if distinct count < 3
            # This happens if:
            # 1. All 4 same (1 color)
            # 2. 3 same, 1 diff (2 colors)
            # 3. 2 pairs (2 colors)
            # Fast check:
            if ca == cb:
                if cc == ce:
                    if ca == cc: # AAAA
                        count += 1
                    else: # AABB
                        count += 1
                else:
                    if ca == cc: # AAAB
                        count += 1
                    elif ca == ce: # AABB (covered? no, ca==cb, cc!=ce. if ca==ce then cb==ce. so AABA? No. ca=0, cb=0, cc=1, ce=0. 0,0,1,0. Distinct {0,1}. Yes.)
                        count += 1
                    # else AABC -> 3 colors. OK.
            else:
                if cc == ce:
                    if cc == ca: # ABAA
                        count += 1
                    elif cc == cb: # ABAB
                        count += 1
                    # else ABAC -> 3 colors. OK.
                else:
                    if ca == cc and cb == ce: # ABAB? No, ca!=cb. ca=cc, cb=ce. A B A B. Distinct {A,B}.
                        count += 1
                    elif ca == ce and cb == cc: # ABBA. Distinct {A,B}.
                        count += 1
        return count

    def get_violations(colors):
        """Return list of indices of APs that are violations."""
        v_list = []
        for idx, (a, b, c, e) in enumerate(ap_list):
            ca, cb, cc, ce = colors[a], colors[b], colors[c], colors[e]
            if ca == cb:
                if cc == ce:
                    v_list.append(idx)
                else:
                    if ca == cc or ca == ce:
                        v_list.append(idx)
            else:
                if cc == ce:
                    if cc == ca or cc == cb:
                        v_list.append(idx)
                else:
                    if (ca == cc and cb == ce) or (ca == ce and cb == cc):
                        v_list.append(idx)
        return v_list

    def is_valid(colors):
        return count_violations(colors) == 0

    # Strategy: Random Init + Repair + Reduce
    
    # 1. Random Init with 6 colors
    colors = [random.randint(0, 5) for _ in range(N)]
    
    # 2. Repair Loop
    max_repair_iters = 20000
    for _ in range(max_repair_iters):
        if time.time() > deadline - 10:
            break
        
        violations = get_violations(colors)
        if not violations:
            break
        
        # Pick a random violation
        v_idx = random.choice(violations)
        a, b, c, e = ap_list[v_idx]
        pos = random.choice([a, b, c, e])
        
        # Recolor pos to a random color
        # Try to pick a color that might help
        # Simple: random 0..5
        new_color = random.randint(0, 5)
        colors[pos] = new_color
        
    # If repair failed to make it valid, use a fallback greedy that is more robust
    if not is_valid(colors):
        # Greedy: for i in range(N), pick smallest color that doesn't violate completed APs
        # And also check "potential" conflicts? No, just completed.
        # Then repair again.
        colors = [0] * N
        for i in range(N):
            for c in range(6):
                colors[i] = c
                # Check APs where i is the max index
                ok = True
                for ap_idx in pos_aps[i]:
                    a, b, cc, e = ap_list[ap_idx]
                    # Only check if all are <= i
                    if max(a, b, cc, e) == i:
                        ca, cb, ccc, ce = colors[a], colors[b], colors[cc], colors[e]
                        # Check validity
                        if ca == cb:
                            if ccc == ce:
                                ok = False
                                break
                            else:
                                if ca == ccc or ca == ce:
                                    ok = False
                                    break
                        else:
                            if ccc == ce:
                                if ccc == ca or ccc == cb:
                                    ok = False
                                    break
                            else:
                                if (ca == ccc and cb == ce) or (ca == ce and cb == ccc):
                                    ok = False
                                    break
                if ok:
                    break
            # If no color worked (shouldn't happen with 6 colors), keep last tried or random
            if colors[i] == 0 and i > 0 and not ok:
                 colors[i] = random.randint(0, 5)
        
        # Repair again
        for _ in range(max_repair_iters):
            if time.time() > deadline - 5:
                break
            violations = get_violations(colors)
            if not violations:
                break
            v_idx = random.choice(violations)
            a, b, c, e = ap_list[v_idx]
            pos = random.choice([a, b, c, e])
            colors[pos] = random.randint(0, 5)

    # 3. Reduce Colors
    # Try to remove colors one by one
    # Start with the color that appears least? Or just 0, 1, 2...
    # We want to minimize the count.
    
    # Helper to check if a specific AP is valid
    def check_ap(ap_idx, colors):
        a, b, c, e = ap_list[ap_idx]
        ca, cb, cc, ce = colors[a], colors[b], colors[c], colors[e]
        if ca == cb:
            if cc == ce:
                return False
            else:
                if ca == cc or ca == ce:
                    return False
        else:
            if cc == ce:
                if cc == ca or cc == cb:
                    return False
            else:
                if (ca == cc and cb == ce) or (ca == ce and cb == cc):
                    return False
        return True

    # Try to merge color classes
    # We'll try to recolor all nodes of color 'target' to other colors.
    # To be safe, we do it one node at a time, and if we get stuck, we revert.
    # Or simpler: try to find a valid coloring with k colors for k=3,4,5...
    # But we already have a valid one. We just want to reduce the *used* count.
    
    # Let's try to reduce the number of distinct colors in the current valid solution.
    # Identify used colors.
    used = set(colors)
    
    # Try to remove each color from 'used' one by one.
    # If we can recolor all nodes of color 'c' to other colors without breaking validity, we do it.
    
    # To make it efficient, we can try to recolor nodes of 'c' to the most common other color first.
    
    # Sort used colors by frequency (ascending) to try removing rare ones first?
    # Or just iterate.
    
    list_used = list(used)
    # Shuffle to be random?
    random.shuffle(list_used)
    
    for target_color in list_used:
        if time.time() > deadline - 2:
            break
            
        # Find all nodes with target_color
        nodes = [i for i in range(N) if colors[i] == target_color]
        
        # Try to recolor them
        # We'll use a simple backtracking or greedy for these nodes.
        # Since N=128, and nodes count is small on average, greedy is fine.
        
        # Save original colors
        orig_colors = colors[:]
        
        success = True
        for node in nodes:
            # Try to assign a new color
            found = False
            # Try other used colors first
            for new_c in used:
                if new_c == target_color:
                    continue
                colors[node] = new_c
                # Check all APs involving node
                valid = True
                for ap_idx in pos_aps[node]:
                    if not check_ap(ap_idx, colors):
                        valid = False
                        break
                if valid:
                    found = True
                    break
            
            if not found:
                # Try a new color? No, we are minimizing.
                # If we can't find a valid existing color, we can't remove this color easily.
                # Revert and stop trying to remove this color.
                colors = orig_colors[:]
                success = False
                break
        
        if success:
            # Update used set
            used.remove(target_color)
            # Note: we might have introduced new colors? No, we only used existing ones.
            # So the count decreases.
        
        # If we failed to remove target_color, we keep the original colors for those nodes
        # (which is what we did by reverting).
        # But what if we removed some nodes and then failed on a later one?
        # We reverted all. So it's atomic.

    # Final validation
    if not is_valid(colors):
        # Emergency fallback: if something went wrong, return a known valid pattern
        # We know 6 colors works. Let's just return the current one if valid, else re-init.
        # But we checked is_valid above.
        # If it's invalid, we must fix it.
        # Use the repair loop again.
        for _ in range(1000):
            violations = get_violations(colors)
            if not violations:
                break
            v_idx = random.choice(violations)
            a, b, c, e = ap_list[v_idx]
            pos = random.choice([a, b, c, e])
            colors[pos] = random.randint(0, 5)

    # Relabel to 0..k-1
    mapping = {}
    new_colors = []
    next_id = 0
    for c in colors:
        if c not in mapping:
            mapping[c] = next_id
            next_id += 1
        new_colors.append(mapping[c])
    
    return new_colors
# EVOLVE-BLOCK-END
