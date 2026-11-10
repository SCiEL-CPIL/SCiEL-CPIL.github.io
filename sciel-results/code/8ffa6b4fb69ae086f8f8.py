import time
import numpy as np
import random

N = 256
DEADLINE = 85.0

# Precompute all 4-APs and 3-APs as lists of (a, b, c[, d]) tuples
# 4-APs: (a, a+d, a+2d, a+3d)
_4AP_LIST = []
for a in range(N):
    for d in range(1, (N - 1 - a) // 3 + 1):
        _4AP_LIST.append((a, a + d, a + 2 * d, a + 3 * d))

# For each element x, the bitmask of 4-AP indices it participates in
_4AP_MASKS = [0] * N
for idx, (a, b, c, d) in enumerate(_4AP_LIST):
    bit = 1 << idx
    _4AP_MASKS[a] |= bit
    _4AP_MASKS[b] |= bit
    _4AP_MASKS[c] |= bit
    _4AP_MASKS[d] |= bit

# 3-APs: (a, b, c) with a < b < c, b-a = c-b
_3AP_LIST = []
for b in range(N):
    max_d = min(b, N - 1 - b)
    for d in range(1, max_d + 1):
        _3AP_LIST.append((b - d, b, b + d))

# For each element x, the bitmask of 3-AP indices it participates in
_3AP_MASKS = [0] * N
for idx, (a, b, c) in enumerate(_3AP_LIST):
    bit = 1 << idx
    _3AP_MASKS[a] |= bit
    _3AP_MASKS[b] |= bit
    _3AP_MASKS[c] |= bit


def _count_3aps(mask):
    """Count 3-APs using precomputed list."""
    count = 0
    for (a, b, c) in _3AP_LIST:
        if mask[a] and mask[b] and mask[c]:
            count += 1
    return count


def _count_3aps_fast(mask):
    """Faster 3-AP count using numpy."""
    n = len(mask)
    count = 0
    for d in range(1, n // 2 + 1):
        a_part = mask[:n - 2 * d]
        b_part = mask[d:n - d]
        c_part = mask[2 * d:]
        count += np.sum(a_part & b_part & c_part)
    return int(count)


def _marginal_3ap_gain(mask, pos):
    """Compute how many new 3-APs are created by adding pos."""
    n = N
    gain = 0
    # pos as first (a): need b, c with b = pos+d, c = pos+2d
    max_d = (n - 1 - pos) // 2
    if max_d >= 1:
        d = np.arange(1, max_d + 1)
        b_pos = pos + d
        c_pos = pos + 2 * d
        gain += np.sum(mask[b_pos] & mask[c_pos])
    # pos as middle (b): need a, c with a = pos-d, c = pos+d
    max_d = min(pos, n - 1 - pos)
    if max_d >= 1:
        d = np.arange(1, max_d + 1)
        a_pos = pos - d
        c_pos = pos + d
        gain += np.sum(mask[a_pos] & mask[c_pos])
    # pos as last (c): need a, b with b = pos-d, a = pos-2d
    max_d = pos // 2
    if max_d >= 1:
        d = np.arange(1, max_d + 1)
        b_pos = pos - d
        a_pos = pos - 2 * d
        gain += np.sum(mask[b_pos] & mask[a_pos])
    return int(gain)


def _has_4ap_fast(mask, new_pos):
    """Optimized 4-AP check for adding new_pos."""
    n = N

    # Position 0: new_pos = a
    max_d = (n - 1 - new_pos) // 3
    if max_d >= 1:
        d = np.arange(1, max_d + 1)
        positions1 = new_pos + d
        positions2 = new_pos + 2 * d
        positions3 = new_pos + 3 * d
        if np.any(mask[positions1] & mask[positions2] & mask[positions3]):
            return True

    # Position 1: new_pos = a + d
    max_d = min(new_pos, (n - 1 - new_pos) // 2)
    if max_d >= 1:
        d = np.arange(1, max_d + 1)
        a = new_pos - d
        pos2 = new_pos + d
        pos3 = new_pos + 2 * d
        if np.any(mask[a] & mask[pos2] & mask[pos3]):
            return True

    # Position 2: new_pos = a + 2d
    max_d = min(new_pos // 2, n - 1 - new_pos)
    if max_d >= 1:
        d = np.arange(1, max_d + 1)
        a = new_pos - 2 * d
        pos1 = new_pos - d
        pos3 = new_pos + d
        if np.any(mask[a] & mask[pos1] & mask[pos3]):
            return True

    # Position 3: new_pos = a + 3d
    max_d = new_pos // 3
    if max_d >= 1:
        d = np.arange(1, max_d + 1)
        a = new_pos - 3 * d
        pos1 = a + d
        pos2 = a + 2 * d
        if np.any(mask[a] & mask[pos1] & mask[pos2]):
            return True

    return False


def _constraint_greedy(start_time, deadline):
    """
    Constraint-propagation greedy:
    - Maintain a set of 'active' 4-APs (those not yet fully blocked).
    - At each step, for each candidate, compute:
        * 3-AP gain
        * Number of active 4-APs that this candidate would block
    - Select candidate maximizing (gain, -blocked_count) lexicographically.
    """
    mask = np.zeros(N, dtype=bool)
    
    # Precompute for each 4-AP which elements it contains
    # active_4aps: set of 4-AP indices that are still "active" (could still be completed)
    # A 4-AP is active if at least 3 of its 4 elements are in the mask
    # Actually, a 4-AP is "threatened" if 3 of its elements are in the mask
    # We want to avoid completing any 4-AP, so we track which 4-APs have 3 elements present
    
    # For efficiency, track for each 4-AP how many of its elements are in the mask
    ap4_counts = np.zeros(len(_4AP_LIST), dtype=np.int32)
    
    # Precompute element -> list of 4-AP indices
    elem_to_4aps = [[] for _ in range(N)]
    for idx, (a, b, c, d) in enumerate(_4AP_LIST):
        elem_to_4aps[a].append(idx)
        elem_to_4aps[b].append(idx)
        elem_to_4aps[c].append(idx)
        elem_to_4aps[d].append(idx)
    
    remaining = set(range(N))
    
    while remaining:
        if time.time() > deadline - 5:
            break
        
        best_pos = -1
        best_key = None  # (gain, -blocked, pos) for tie-breaking
        
        for pos in remaining:
            if time.time() > deadline - 5:
                break
            
            # Check if adding pos creates a 4-AP
            if _has_4ap_fast(mask, pos):
                continue
            
            # Compute 3-AP gain
            gain = _marginal_3ap_gain(mask, pos)
            
            # Compute how many 4-APs this would block
            # A 4-AP is blocked if we add an element that is NOT in the 4-AP but makes it impossible
            # Actually, adding pos doesn't "block" 4-APs directly. The constraint is just that
            # we don't complete any 4-AP. The "blocking" is implicit in the feasibility check.
            # 
            # Better forward-looking metric: how many 4-APs have exactly 2 elements in mask
            # and pos is one of the other 2? No, that's complex.
            #
            # Simpler: just use gain as primary, and as secondary, prefer elements that
            # are in fewer 4-APs (less "dangerous" in future)
            danger = len(elem_to_4aps[pos])
            
            key = (gain, -danger, pos)
            if best_key is None or key > best_key:
                best_key = key
                best_pos = pos
        
        if best_pos == -1:
            break
        
        mask[best_pos] = True
        remaining.discard(best_pos)
        
        # Update 4-AP counts
        for ap_idx in elem_to_4aps[best_pos]:
            ap4_counts[ap_idx] += 1
    
    return mask


def _two_swap_local_search(mask, start_time, deadline):
    """
    Local search with 2-swap moves: try removing one element and adding another.
    Use delta evaluation for speed.
    """
    current = mask.copy()
    current_score = _count_3aps_fast(current)
    best = current.copy()
    best_score = current_score
    
    selected = set(np.where(current)[0].tolist())
    unselected = set(range(N)) - selected
    
    improved = True
    while improved:
        if time.time() > deadline - 5:
            break
        improved = False
        
        # Try all 2-swaps
        for rem in list(selected):
            if time.time() > deadline - 5:
                break
            
            # Temporarily remove rem
            temp_mask = current.copy()
            temp_mask[rem] = False
            temp_score = current_score - _marginal_3ap_gain_loss(temp_mask, rem)
            
            # Actually, let's just compute the score after removal
            # For speed, compute delta of removal
            # 3-APs lost by removing rem:
            loss = 0
            # rem as first
            max_d = (N - 1 - rem) // 2
            if max_d >= 1:
                d = np.arange(1, max_d + 1)
                b_pos = rem + d
                c_pos = rem + 2 * d
                loss += np.sum(current[b_pos] & current[c_pos])
            # rem as middle
            max_d = min(rem, N - 1 - rem)
            if max_d >= 1:
                d = np.arange(1, max_d + 1)
                a_pos = rem - d
                c_pos = rem + d
                loss += np.sum(current[a_pos] & current[c_pos])
            # rem as last
            max_d = rem // 2
            if max_d >= 1:
                d = np.arange(1, max_d + 1)
                b_pos = rem - d
                a_pos = rem - 2 * d
                loss += np.sum(current[b_pos] & current[a_pos])
            
            score_after_rem = current_score - int(loss)
            
            for add in unselected:
                if time.time() > deadline - 5:
                    break
                if add == rem:
                    continue
                
                # Check if adding add to (current - rem) creates 4-AP
                if _has_4ap_fast(temp_mask, add):
                    continue
                
                # Compute gain of adding add to temp_mask
                gain = _marginal_3ap_gain(temp_mask, add)
                
                new_score = score_after_rem + gain
                if new_score > best_score:
                    current = temp_mask.copy()
                    current[add] = True
                    current_score = new_score
                    best = current.copy()
                    best_score = new_score
                    selected.discard(rem)
                    selected.add(add)
                    unselected.discard(add)
                    unselected.add(rem)
                    improved = True
                    break  # Restart search with new state
            
            if improved:
                break
    
    return best, best_score


def _marginal_3ap_gain_loss(mask, pos):
    """Compute how many 3-APs are lost by removing pos from mask."""
    # Same as marginal gain but using mask where pos is still present
    # Actually, we want to count 3-APs that include pos and have all other elements in mask
    gain = 0
    # pos as first
    max_d = (N - 1 - pos) // 2
    if max_d >= 1:
        d = np.arange(1, max_d + 1)
        b_pos = pos + d
        c_pos = pos + 2 * d
        gain += np.sum(mask[b_pos] & mask[c_pos])
    # pos as middle
    max_d = min(pos, N - 1 - pos)
    if max_d >= 1:
        d = np.arange(1, max_d + 1)
        a_pos = pos - d
        c_pos = pos + d
        gain += np.sum(mask[a_pos] & mask[c_pos])
    # pos as last
    max_d = pos // 2
    if max_d >= 1:
        d = np.arange(1, max_d + 1)
        b_pos = pos - d
        a_pos = pos - 2 * d
        gain += np.sum(mask[b_pos] & mask[a_pos])
    return int(gain)


def _multi_start_constraint_greedy(start_time, deadline, n_starts=10):
    """Run constraint greedy multiple times with different random orderings."""
    best_mask = None
    best_score = -1
    
    for i in range(n_starts):
        if time.time() > deadline - 10:
            break
        
        # Randomize the order of candidates slightly
        random.shuffle_order = random.sample(range(N), N)
        
        mask = np.zeros(N, dtype=bool)
        remaining = set(range(N))
        
        # Precompute element -> 4-AP count for danger metric
        elem_danger = [len(elem_to_4aps[x]) for x in range(N)]
        
        while remaining:
            if time.time() > deadline - 10:
                break
            
            best_pos = -1
            best_key = None
            
            # Process in random order for diversity
            for pos in random.shuffle_order:
                if pos not in remaining:
                    continue
                if time.time() > deadline - 10:
                    break
                
                if _has_4ap_fast(mask, pos):
                    continue
                
                gain = _marginal_3ap_gain(mask, pos)
                danger = elem_danger[pos]
                
                key = (gain, -danger, pos)
                if best_key is None or key > best_key:
                    best_key = key
                    best_pos = pos
            
            if best_pos == -1:
                break
            
            mask[best_pos] = True
            remaining.discard(best_pos)
        
        score = _count_3aps_fast(mask)
        if score > best_score:
            best_score = score
            best_mask = mask.copy()
    
    return best_mask, best_score


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    random.seed(42)
    
    # Phase 1: Constraint-propagation greedy (deterministic)
    mask1 = _constraint_greedy(start_time, deadline)
    score1 = _count_3aps_fast(mask1)
    
    # Phase 2: Multi-start constraint greedy
    mask2, score2 = _multi_start_constraint_greedy(start_time, deadline, n_starts=8)
    if mask2 is not None and score2 > score1:
        mask1 = mask2
        score1 = score2
    
    # Phase 3: 2-swap local search
    if time.time() < deadline - 10:
        mask1, score1 = _two_swap_local_search(mask1, start_time, deadline)
    
    # Phase 4: Final quick local search (additions only)
    if time.time() < deadline - 5:
        current = mask1.copy()
        current_score = score1
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            improved = False
            for pos in range(N):
                if current[pos]:
                    continue
                if _has_4ap_fast(current, pos):
                    continue
                gain = _marginal_3ap_gain(current, pos)
                if gain > 0:
                    current[pos] = True
                    current_score += gain
                    improved = True
                    break
        if current_score > score1:
            mask1 = current
            score1 = current_score
    
    # Convert to 1-indexed list
    result = sorted([int(i) + 1 for i in np.where(mask1)[0]])
    
    if len(result) == 0:
        return [1]
    
    return result
