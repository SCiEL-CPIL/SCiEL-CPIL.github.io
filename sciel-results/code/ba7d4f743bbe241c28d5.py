import time
import random

# Precompute all 4-term arithmetic progressions (APs)
# AP is (a, b, c, d) where b = a+d, c = a+2d, d = a+3d
# Positions are 1-indexed in problem, 0-indexed in array.
# Let's use 0-indexed positions 0..127 corresponding to 1..128.

APs = []
# ap_index: list of AP indices for each position
pos_to_aps = [[] for _ in range(128)]

for a in range(128):
    for d in range(1, (127 - a) // 3 + 1):
        b = a + d
        c = a + 2 * d
        e = a + 3 * d
        if e < 128:
            ap_idx = len(APs)
            APs.append((a, b, c, e))
            pos_to_aps[a].append(ap_idx)
            pos_to_aps[b].append(ap_idx)
            pos_to_aps[c].append(ap_idx)
            pos_to_aps[e].append(ap_idx)

def count_violations(colors):
    """Count number of APs with < 3 distinct colors."""
    count = 0
    for a, b, c, e in APs:
        ca, cb, cc, ce = colors[a], colors[b], colors[c], colors[e]
        # Check distinct colors
        # If all same: 1 distinct
        # If 2 same, 2 other same: 2 distinct
        # If 3 same, 1 other: 2 distinct
        # If all diff: 4 distinct (but max K colors)
        # We need >= 3 distinct.
        # So invalid if distinct count is 1 or 2.
        
        # Efficient check:
        # Sort or use set? Set is slow.
        # Logic:
        # If ca == cb == cc == ce: invalid
        # Else if (ca == cb and cb == cc and cc != ce) or ... 
        # Easier: 
        # distinct = len({ca, cb, cc, ce})
        # if distinct < 3: count += 1
        
        # Optimization without set:
        # If ca == cb:
        #   if cc == ca:
        #     if ce == ca: count += 1 (1 distinct)
        #     else: count += 1 (2 distinct: ca, ce)
        #   else:
        #     if ce == ca: count += 1 (2 distinct: ca, cc)
        #     elif ce == cc: count += 1 (2 distinct: ca, cc)
        #     else: pass (3 distinct: ca, cc, ce)
        # Else (ca != cb):
        #   if cc == ca:
        #     if ce == ca: count += 1 (2 distinct: ca, cb)
        #     elif ce == cb: count += 1 (2 distinct: ca, cb)
        #     else: pass (3 distinct: ca, cb, ce)
        #   elif cc == cb:
        #     if ce == ca: count += 1 (2 distinct: ca, cb)
        #     elif ce == cb: count += 1 (2 distinct: ca, cb)
        #     else: pass (3 distinct: ca, cb, ce)
        #   else (cc != ca and cc != cb):
        #     # ca, cb, cc are distinct
        #     # ce can be anything, distinct count is at least 3
        #     pass
        
        # This logic is correct and fast.
        if ca == cb:
            if cc == ca:
                count += 1 # ce is either ca (1 dist) or not (2 dist)
            else:
                if ce == ca or ce == cc:
                    count += 1 # 2 distinct
                # else 3 distinct
        else: # ca != cb
            if cc == ca:
                if ce == ca or ce == cb:
                    count += 1 # 2 distinct
            elif cc == cb:
                if ce == ca or ce == cb:
                    count += 1 # 2 distinct
            # else 3 distinct
    return count

def greedy_construct():
    """Fast greedy fallback."""
    colors = [0] * 128
    for i in range(128):
        # Try colors 0..127
        for c in range(128):
            colors[i] = c
            # Check validity with previous positions
            valid = True
            # Only need to check APs ending at i or involving i where all other positions are < i
            # Actually, just check all APs involving i where the other 3 are already assigned.
            # Since we assign left to right, if an AP involves i and positions > i, they are not assigned yet.
            # But the constraint is global.
            # The seed logic checks:
            # all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
            # This checks APs where x is the LAST term (a+3d).
            # AND
            # all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))
            # This checks APs where x is the SECOND or THIRD term?
            # Let's just use a simpler check: 
            # For all APs involving i, if the other 3 are assigned, check if adding c makes it invalid.
            # If other 3 are not assigned, we can't check yet.
            # But wait, if we assign left-to-right, an AP (a, b, c, d) is fully assigned only when d is assigned.
            # So we only need to check APs where i is the largest index (i.e., i = a+3d).
            # Is that sufficient?
            # If we ensure that when we assign the last element, the AP is valid, then the AP is valid.
            # Yes.
            # So for each i, check all APs where i is the 4th term.
            for ap_idx in pos_to_aps[i]:
                a, b, c, e = APs[ap_idx]
                if e == i:
                    ca, cb, cc, ce = colors[a], colors[b], colors[c], colors[i]
                    # Check distinct count
                    if ca == cb:
                        if cc == ca:
                            valid = False
                            break
                        else:
                            if ce == ca or ce == cc:
                                valid = False
                                break
                    else:
                        if cc == ca:
                            if ce == ca or ce == cb:
                                valid = False
                                break
                        elif cc == cb:
                            if ce == ca or ce == cb:
                                valid = False
                                break
            if valid:
                break
        else:
            # If no color works (should not happen if K is large enough, but with small K it might)
            # The seed uses range(128), so it will always find one.
            pass
    return colors

def local_search(K, time_limit):
    """Try to find a valid K-coloring using Simulated Annealing."""
    start_time = time.time()
    colors = [random.randint(0, K - 1) for _ in range(128)]
    current_cost = count_violations(colors)
    
    if current_cost == 0:
        return colors

    best_colors = colors[:]
    best_cost = current_cost
    
    temp = 10.0
    cooling = 0.995
    
    while time.time() - start_time < time_limit:
        # Pick random position
        i = random.randint(0, 127)
        old_color = colors[i]
        new_color = random.randint(0, K - 1)
        if new_color == old_color:
            continue
        
        # Calculate delta cost
        # Only APs involving i are affected
        delta = 0
        for ap_idx in pos_to_aps[i]:
            a, b, c, e = APs[ap_idx]
            # Get old colors
            old_ca, old_cb, old_cc, old_ce = colors[a], colors[b], colors[c], colors[e]
            # Get new colors (only i changes)
            if a == i:
                new_ca, new_cb, new_cc, new_ce = new_color, colors[b], colors[c], colors[e]
            elif b == i:
                new_ca, new_cb, new_cc, new_ce = colors[a], new_color, colors[c], colors[e]
            elif c == i:
                new_ca, new_cb, new_cc, new_ce = colors[a], colors[b], new_color, colors[e]
            else: # e == i
                new_ca, new_cb, new_cc, new_ce = colors[a], colors[b], colors[c], new_color
            
            # Check old validity
            old_valid = True
            if old_ca == old_cb:
                if old_cc == old_ca:
                    old_valid = False
                else:
                    if old_ce == old_ca or old_ce == old_cc:
                        old_valid = False
            else:
                if old_cc == old_ca:
                    if old_ce == old_ca or old_ce == old_cb:
                        old_valid = False
                elif old_cc == old_cb:
                    if old_ce == old_ca or old_ce == old_cb:
                        old_valid = False
            
            # Check new validity
            new_valid = True
            if new_ca == new_cb:
                if new_cc == new_ca:
                    new_valid = False
                else:
                    if new_ce == new_ca or new_ce == new_cc:
                        new_valid = False
            else:
                if new_cc == new_ca:
                    if new_ce == new_ca or new_ce == new_cb:
                        new_valid = False
                elif new_cc == new_cb:
                    if new_ce == new_ca or new_ce == new_cb:
                        new_valid = False
            
            if not old_valid and new_valid:
                delta -= 1
            elif old_valid and not new_valid:
                delta += 1
            
        colors[i] = new_color
        current_cost += delta
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_colors = colors[:]
            if best_cost == 0:
                return best_colors
        
        # Acceptance criterion
        if delta > 0:
            import math
            if random.random() < math.exp(-delta / temp):
                pass # Accept
            else:
                # Reject
                colors[i] = old_color
                current_cost -= delta
        temp *= cooling
        if temp < 0.01:
            temp = 10.0 # Restart annealing

    return best_colors if best_cost == 0 else None

def construct():
    start_time = time.time()
    
    # Try K=3
    result = local_search(3, 20)
    if result is not None:
        return result
        
    # Try K=4
    result = local_search(4, 20)
    if result is not None:
        return result
        
    # Fallback to Greedy
    # Greedy is guaranteed to produce a valid coloring with enough colors
    return greedy_construct()
