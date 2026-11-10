import time
import random
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    # Precompute all 3-APs and 4-APs
    three_aps = []
    three_aps_by_elem = [[] for _ in range(n + 1)]
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            idx = len(three_aps)
            three_aps.append((a, b, c))
            three_aps_by_elem[a].append(idx)
            three_aps_by_elem[b].append(idx)
            three_aps_by_elem[c].append(idx)

    four_aps = []
    four_aps_by_elem = [[] for _ in range(n + 1)]
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            dd = a + 3 * d
            idx = len(four_aps)
            four_aps.append((a, b, c, dd))
            four_aps_by_elem[a].append(idx)
            four_aps_by_elem[b].append(idx)
            four_aps_by_elem[c].append(idx)
            four_aps_by_elem[dd].append(idx)

    # Precompute for each element and each 4-AP containing it, the mask of the other 3 elements
    four_ap_other = {}
    for idx, (a, b, c, dd) in enumerate(four_aps):
        for y in (a, b, c, dd):
            others = [z for z in (a, b, c, dd) if z != y]
            m = 0
            for z in others:
                m |= (1 << (z - 1))
            four_ap_other[(y, idx)] = m

    # Precompute 3-AP other masks for each (element, 3-AP index)
    three_ap_other = {}
    for idx, (a, b, c) in enumerate(three_aps):
        for y in (a, b, c):
            others = [z for z in (a, b, c) if z != y]
            m = 0
            for z in others:
                m |= (1 << (z - 1))
            three_ap_other[(y, idx)] = m

    def mask_to_list(mask):
        return [i + 1 for i in range(n) if (mask >> i) & 1]

    def count_3aps(mask):
        count = 0
        for a, b, c in three_aps:
            if (mask >> (a - 1)) & 1 and (mask >> (b - 1)) & 1 and (mask >> (c - 1)) & 1:
                count += 1
        return count

    def has_4ap(mask):
        for a, b, c, dd in four_aps:
            if (mask >> (a - 1)) & 1 and (mask >> (b - 1)) & 1 and (mask >> (c - 1)) & 1 and (mask >> (dd - 1)) & 1:
                return True
        return False

    def adds_4ap(y, mask):
        bit = 1 << (y - 1)
        if mask & bit:
            return False
        for idx in four_aps_by_elem[y]:
            om = four_ap_other[(y, idx)]
            if (mask & om) == om:
                return True
        return False

    def gain_add(y, mask):
        bit = 1 << (y - 1)
        if mask & bit:
            return 0
        cnt = 0
        for idx in three_aps_by_elem[y]:
            om = three_ap_other[(y, idx)]
            if (mask & om) == om:
                cnt += 1
        return cnt

    def loss_rem(y, mask):
        bit = 1 << (y - 1)
        if not (mask & bit):
            return 0
        cnt = 0
        for idx in three_aps_by_elem[y]:
            om = three_ap_other[(y, idx)]
            if (mask & om) == om:
                cnt += 1
        return cnt

    def find_violated_4aps(mask):
        """Return a list of 4-AP indices that are fully contained in mask."""
        violated = []
        for idx, (a, b, c, dd) in enumerate(four_aps):
            if (mask >> (a - 1)) & 1 and (mask >> (b - 1)) & 1 and (mask >> (c - 1)) & 1 and (mask >> (dd - 1)) & 1:
                violated.append(idx)
        return violated

    best_score = -1
    best_mask = 0

    def update_best(mask, score):
        nonlocal best_score, best_mask
        if score > best_score:
            best_score = score
            best_mask = mask

    # Initialize with a few diverse starting points
    initial_masks = []
    
    # 1. Greedy construction to get a valid starting point
    def greedy_valid_start(order):
        mask = 0
        for x in order:
            if not adds_4ap(x, mask):
                mask |= (1 << (x - 1))
        return mask

    orders = [
        list(range(1, n + 1)),
        list(range(n, 0, -1)),
        [x for x in range(1, n + 1) if x % 2 == 1] + [x for x in range(1, n + 1) if x % 2 == 0]
    ]
    
    for _ in range(5):
        o = list(range(1, n + 1))
        random.shuffle(o)
        orders.append(o)

    for order in orders:
        if time.time() > deadline - 20:
            break
        m = greedy_valid_start(order)
        s = count_3aps(m)
        update_best(m, s)
        initial_masks.append(m)

    # WalkSAT Loop
    # We will run WalkSAT on each initial mask
    for init_mask in initial_masks:
        if time.time() > deadline - 5:
            break
            
        mask = init_mask
        score = count_3aps(mask)
        update_best(mask, score)
        
        # WalkSAT parameters
        # p_flip: probability to flip a variable that doesn't improve score (to escape local optima)
        # We tune this dynamically or keep it fixed.
        p_walk = 0.3 
        max_iter = 20000
        
        for it in range(max_iter):
            if time.time() > deadline - 2:
                break
            
            # 1. Check for hard constraint violations (4-APs)
            violated = find_violated_4aps(mask)
            
            if violated:
                # Must fix a violation. Pick a random violated 4-AP and flip one of its elements.
                # To minimize damage to 3-APs, we can pick the element in the 4-AP that has the lowest "cost" to remove.
                v_idx = random.choice(violated)
                a, b, c, dd = four_aps[v_idx]
                candidates = [a, b, c, dd]
                
                # Score each candidate by the loss in 3-APs if removed
                best_cand = None
                min_loss = float('inf')
                for y in candidates:
                    loss = loss_rem(y, mask)
                    # Also check if removing y breaks other 4-APs? No, removing always helps 4-AP constraints.
                    # But we want to minimize loss of 3-APs.
                    if loss < min_loss:
                        min_loss = loss
                        best_cand = y
                        
                y = best_cand
                # Flip y (remove it)
                mask &= ~(1 << (y - 1))
                score -= min_loss
                update_best(mask, score)
                
            else:
                # No violations. Try to improve 3-AP count.
                # Calculate best possible move (add or remove)
                best_gain = 0
                best_move = None
                
                # Try adds
                # Optimization: Only check adds that don't create 4-APs
                # We can sample or iterate. Iterating 256 is fast.
                for y in range(1, n + 1):
                    if (mask >> (y - 1)) & 1:
                        continue
                    if adds_4ap(y, mask):
                        continue
                    g = gain_add(y, mask)
                    if g > best_gain:
                        best_gain = g
                        best_move = ('add', y)
                        
                # Try removes
                # Only consider removing if it helps? Removing usually hurts 3-AP count unless it allows future adds?
                # In WalkSAT, we can flip any variable. But here, removing reduces 3-APs directly.
                # However, removing might allow adding a "better" element later?
                # For now, let's stick to adding if possible, or random walk.
                # Actually, removing an element that is part of many 3-APs but "blocks" no adds (since adds are constrained by 4-APs)
                # might not be directly beneficial for 3-AP count, but could be part of a swap.
                # Standard WalkSAT flips a variable. Let's allow removes if they lead to a state where we can add more?
                # Simpler: Just try adds. If no positive gain add, do a random walk.
                
                if best_gain > 0 and best_move:
                    # Accept the best move
                    y = best_move[1]
                    mask |= (1 << (y - 1))
                    score += best_gain
                    update_best(mask, score)
                else:
                    # No improving add. Perform a random walk to escape local optimum.
                    # Pick a random variable to flip.
                    # To maintain validity (no 4-APs), we can only flip to 1 if it doesn't create a 4-AP.
                    # Flipping to 0 is always safe for 4-APs.
                    
                    # Strategy: 
                    # 50% chance: Try to add a random valid element.
                    # 50% chance: Try to remove a random element (to allow different structure).
                    
                    if random.random() < 0.5:
                        # Try random add
                        cands = [y for y in range(1, n + 1) if not (mask >> (y - 1)) & 1 and not adds_4ap(y, mask)]
                        if cands:
                            y = random.choice(cands)
                            g = gain_add(y, mask)
                            # Accept if it doesn't decrease score too much? Or always accept to explore?
                            # WalkSAT usually accepts moves that don't increase violations.
                            # Here, adding never increases 4-AP violations (we checked).
                            # It might decrease 3-AP score? No, adding y adds g 3-APs. g >= 0.
                            # So random add is always non-decreasing for 3-APs.
                            mask |= (1 << (y - 1))
                            score += g
                            update_best(mask, score)
                    else:
                        # Try random remove
                        elems = mask_to_list(mask)
                        if elems:
                            y = random.choice(elems)
                            loss = loss_rem(y, mask)
                            # This decreases score. But might help escape local optima.
                            mask &= ~(1 << (y - 1))
                            score -= loss
                            # Don't update best here as it's worse, but keep exploring from this state.
                            
        # After WalkSAT, do a final hill climb polish
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            best_g = 0
            best_move = None
            for y in range(1, n + 1):
                if (mask >> (y - 1)) & 1:
                    continue
                if adds_4ap(y, mask):
                    continue
                g = gain_add(y, mask)
                if g > best_g:
                    best_g = g
                    best_move = y
            if best_move:
                mask |= (1 << (best_move - 1))
                score += best_g
                improved = True
                update_best(mask, score)
                
            # Also try swaps? WalkSAT doesn't do swaps natively, but hill climb can.
            # Let's stick to adds for polish, as swaps were in previous version and helped.
            # But the directive asked for WalkSAT. The polish is just cleanup.
            
    # Final verification
    if best_mask and has_4ap(best_mask):
        # Rebuild safely
        mask = 0
        for x in range(1, n + 1):
            if not adds_4ap(x, mask):
                mask |= (1 << (x - 1))
        best_mask = mask
        best_score = count_3aps(mask)

    # Ensure nonempty
    if best_mask == 0:
        best_mask = 1  # {1}
        best_score = 0

    return mask_to_list(best_mask)
