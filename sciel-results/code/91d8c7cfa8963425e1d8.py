"""Large four-term-progression-free integer sets — bitmask local search."""
import time
import random
import numpy as np

N = 512
DEADLINE = 75.0  # safety margin before 90s kill

# Precompute all 4-term APs: (a, a+d, a+2d, a+3d) with a>=1, a+3d<=512
# Represent each AP as a bitmask (int with 4 bits set)
def _build_aps():
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1))
            aps.append(mask)
    return aps

AP_MASKS = _build_aps()
AP_COUNT = len(AP_MASKS)

def _count_violations(sel_mask):
    """Count how many 4-APs are fully contained in sel_mask."""
    v = 0
    for m in AP_MASKS:
        if (sel_mask & m) == m:
            v += 1
    return v

def _greedy_initial():
    """Greedy: add elements in random order if no 4-AP is formed."""
    sel = 0
    order = list(range(1, N + 1))
    random.shuffle(order)
    for x in order:
        bit = 1 << (x - 1)
        # Check: adding x would complete a 4-AP if the other 3 are already in sel
        # For each 4-AP containing x, check if the other 3 bits are in sel
        ok = True
        for m in AP_MASKS:
            if m & bit:  # x is in this AP
                others = m & ~bit
                if (sel & others) == others:
                    ok = False
                    break
        if ok:
            sel |= bit
    return sel

def _local_search(sel_mask, time_limit):
    """WalkSAT-style local search minimizing violations. Returns best found."""
    best_mask = sel_mask
    best_viol = _count_violations(sel_mask)
    best_size = bin(best_mask).count('1')
    start = time.time()
    
    # Precompute for each element, the list of AP masks it participates in
    elem_aps = [[] for _ in range(N)]
    for i, m in enumerate(AP_MASKS):
        # Extract set bits
        tmp = m
        while tmp:
            lb = tmp & (-tmp)
            idx = lb.bit_length() - 1  # 0-based index
            elem_aps[idx].append(i)
            tmp ^= lb
    
    # For faster violation counting after a flip, we can use incremental updates
    # But for simplicity, let's do full recount periodically and use heuristic moves
    
    # Strategy: try random single flips (add or remove) that don't increase violations
    # Also try 2-flips occasionally
    
    cur_mask = sel_mask
    cur_viol = best_viol
    cur_size = best_size
    no_improve = 0
    max_no_improve = 50000
    
    while time.time() - start < time_limit:
        if no_improve > max_no_improve:
            # Restart with perturbation
            cur_mask = best_mask
            cur_viol = best_viol
            cur_size = best_size
            no_improve = 0
            # Random perturbation: flip 3-5 random bits
            for _ in range(random.randint(3, 5)):
                x = random.randint(1, N)
                bit = 1 << (x - 1)
                cur_mask ^= bit
            cur_viol = _count_violations(cur_mask)
            cur_size = bin(cur_mask).count('1')
            continue
        
        # Choose a move
        move_type = random.random()
        if move_type < 0.7:
            # Single flip
            if random.random() < 0.6 and cur_size < N:
                # Try adding
                candidates = [x for x in range(1, N + 1) if not (cur_mask >> (x-1)) & 1]
                if not candidates:
                    continue
                x = random.choice(candidates)
            else:
                # Try removing
                candidates = [x for x in range(1, N + 1) if (cur_mask >> (x-1)) & 1]
                if not candidates:
                    continue
                x = random.choice(candidates)
            
            bit = 1 << (x - 1)
            new_mask = cur_mask ^ bit
            
            # Compute violation change
            # Only APs containing x can change status
            delta_viol = 0
            for ap_idx in elem_aps[x - 1]:
                m = AP_MASKS[ap_idx]
                # Check status before and after
                was_viol = (cur_mask & m) == m
                now_viol = (new_mask & m) == m
                if was_viol and not now_viol:
                    delta_viol -= 1
                elif not was_viol and now_viol:
                    delta_viol += 1
            
            new_viol = cur_viol + delta_viol
            new_size = cur_size + (1 if (new_mask & bit) else -1)
            
            # Accept if violations decrease, or same violations with larger size
            if new_viol < cur_viol or (new_viol == cur_viol and new_size > cur_size):
                cur_mask = new_mask
                cur_viol = new_viol
                cur_size = new_size
                no_improve = 0
                if cur_viol < best_viol or (cur_viol == best_viol and cur_size > best_size):
                    best_mask = cur_mask
                    best_viol = cur_viol
                    best_size = cur_size
            else:
                no_improve += 1
        else:
            # 2-flip: add one, remove one (or two adds, two removes)
            if random.random() < 0.5:
                # Add one, remove one
                add_cands = [x for x in range(1, N + 1) if not (cur_mask >> (x-1)) & 1]
                rem_cands = [x for x in range(1, N + 1) if (cur_mask >> (x-1)) & 1]
                if not add_cands or not rem_cands:
                    continue
                xa = random.choice(add_cands)
                xr = random.choice(rem_cands)
                if xa == xr:
                    continue
                ba = 1 << (xa - 1)
                br = 1 << (xr - 1)
                new_mask = cur_mask ^ ba ^ br
                # Compute delta violations
                delta_viol = 0
                affected_aps = set(elem_aps[xa - 1]) | set(elem_aps[xr - 1])
                for ap_idx in affected_aps:
                    m = AP_MASKS[ap_idx]
                    was_viol = (cur_mask & m) == m
                    now_viol = (new_mask & m) == m
                    if was_viol and not now_viol:
                        delta_viol -= 1
                    elif not was_viol and now_viol:
                        delta_viol += 1
                new_viol = cur_viol + delta_viol
                new_size = cur_size  # one add, one remove
                if new_viol < cur_viol or (new_viol == cur_viol and new_size > cur_size):
                    cur_mask = new_mask
                    cur_viol = new_viol
                    cur_size = new_size
                    no_improve = 0
                    if cur_viol < best_viol or (cur_viol == best_viol and cur_size > best_size):
                        best_mask = cur_mask
                        best_viol = cur_viol
                        best_size = cur_size
                else:
                    no_improve += 1
            else:
                # Two adds
                add_cands = [x for x in range(1, N + 1) if not (cur_mask >> (x-1)) & 1]
                if len(add_cands) < 2:
                    continue
                xa, xb = random.sample(add_cands, 2)
                ba = 1 << (xa - 1)
                bb = 1 << (xb - 1)
                new_mask = cur_mask | ba | bb
                delta_viol = 0
                affected_aps = set(elem_aps[xa - 1]) | set(elem_aps[xb - 1])
                for ap_idx in affected_aps:
                    m = AP_MASKS[ap_idx]
                    was_viol = (cur_mask & m) == m
                    now_viol = (new_mask & m) == m
                    if was_viol and not now_viol:
                        delta_viol -= 1
                    elif not was_viol and now_viol:
                        delta_viol += 1
                new_viol = cur_viol + delta_viol
                new_size = cur_size + 2
                if new_viol < cur_viol or (new_viol == cur_viol and new_size > cur_size):
                    cur_mask = new_mask
                    cur_viol = new_viol
                    cur_size = new_size
                    no_improve = 0
                    if cur_viol < best_viol or (cur_viol == best_viol and cur_size > best_size):
                        best_mask = cur_mask
                        best_viol = cur_viol
                        best_size = cur_size
                else:
                    no_improve += 1
    
    return best_mask, best_viol, best_size

def construct():
    start = time.time()
    deadline = start + DEADLINE
    
    # Multiple random restarts
    best_mask = 0
    best_size = 0
    best_viol = float('inf')
    
    # First, get a good greedy solution
    for attempt in range(20):
        if time.time() > deadline - 5:
            break
        sel = _greedy_initial()
        size = bin(sel).count('1')
        viol = _count_violations(sel)
        if size > best_size or (size == best_size and viol < best_viol):
            best_mask = sel
            best_size = size
            best_viol = viol
    
    # Now do local search from the best greedy solution
    if time.time() < deadline - 5:
        # Run local search with remaining time budget
        remaining = deadline - 5 - time.time()
        # Split time among a few local search runs
        runs = max(1, int(remaining / 10))
        runs = min(runs, 5)
        per_run = remaining / runs
        
        for r in range(runs):
            if time.time() > deadline - 2:
                break
            # Perturb best solution
            cur = best_mask
            # Random perturbation
            for _ in range(random.randint(5, 15)):
                x = random.randint(1, N)
                bit = 1 << (x - 1)
                cur ^= bit
            
            bm, bv, bs = _local_search(cur, per_run * 0.9)
            if bs > best_size or (bs == best_size and bv < best_viol):
                best_mask = bm
                best_size = bs
                best_viol = bv
    
    # Final: verify and return
    if best_viol > 0:
        # Need to fix violations: remove elements from violating APs
        sel_mask = best_mask
        # Greedy removal of elements in violating APs
        changed = True
        while changed and time.time() < deadline - 1:
            changed = False
            for m in AP_MASKS:
                if (sel_mask & m) == m:
                    # Remove one element from this AP (the one with fewest other APs)
                    bits_in_ap = []
                    tmp = m
                    while tmp:
                        lb = tmp & (-tmp)
                        idx = lb.bit_length() - 1
                        bits_in_ap.append(idx)
                        tmp ^= lb
                    # Remove the element that participates in the fewest APs
                    min_count = float('inf')
                    remove_idx = bits_in_ap[0]
                    for idx in bits_in_ap:
                        if len(elem_aps[idx]) < min_count:
                            min_count = len(elem_aps[idx])
                            remove_idx = idx
                    sel_mask &= ~(1 << remove_idx)
                    changed = True
        best_mask = sel_mask
        best_size = bin(best_mask).count('1')
    
    # Convert to sorted list
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
    
    if not result:
        result = [1]
    
    return sorted(result)

# Precompute elem_aps at module level for use in _local_search
elem_aps = [[] for _ in range(N)]
for i, m in enumerate(AP_MASKS):
    tmp = m
    while tmp:
        lb = tmp & (-tmp)
        idx = lb.bit_length() - 1
        elem_aps[idx].append(i)
        tmp ^= lb
