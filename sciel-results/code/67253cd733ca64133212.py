"""
4-AP-free subset of [1,512]: maximize cardinality.
Strategy:
1. Use base-3 (Cantor) construction as a strong structured seed (~128 elements).
2. Try a "base-4" style construction: numbers whose base-4 representation contains only 0,1,2 (no 3).
   This is also 4-AP-free by a digit argument (if a, a+d, a+2d, a+3d have no 3 in base-4,
   then the 4th term would require a carry pattern that forces a 3 somewhere).
   Wait - that's not quite right. Let me think more carefully.
   
   Actually, the standard result: numbers with no digit 2 in base 3 are 4-AP-free.
   For base 4: numbers with no digit 3 in base 4 are NOT necessarily 4-AP-free.
   
   Let me use a different approach: the "Stanley sequence" or greedy with a twist.
   
   Actually, a better direct construction: use the Cantor set (base 3, digits 0,1) which gives ~128 elements.
   Then use local search to add more elements.
   
   Alternative: Use a "modular" construction. For example, take numbers congruent to 0 or 1 mod 3,
   but this gives ~342 elements and is NOT 4-AP-free (e.g., 1,4,7,10 all ≡1 mod 3).
   
   So the Cantor set is the right structured seed. Let me also try:
   - Numbers with no 2 in base 3 (standard Cantor)
   - A modified construction: in base 3, allow digit 2 only in the least significant position? No, that doesn't work.
   
   Let me just use the Cantor set as seed and do aggressive local search.
   
   Key insight for local search: precompute all 4-APs, then use a conflict-based approach.
   When we want to add x, find all 4-APs that would be completed. If we remove one element from each
   such AP, we can add x. The goal is to minimize removals.
"""

import time
import random

N = 512

# Precompute all 4-APs as frozensets
_all_4aps = []
_ap_by_element = [[] for _ in range(N + 1)]  # _ap_by_element[x] = list of indices into _all_4aps

for a in range(1, N - 3):
    for d in range(1, (N - a) // 3 + 1):
        b = a + d
        c = a + 2 * d
        e = a + 3 * d
        if e > N:
            break
        ap = (a, b, c, e)
        idx = len(_all_4aps)
        _all_4aps.append(ap)
        for elem in ap:
            _ap_by_element[elem].append(idx)

# For each element, precompute the other three elements in each AP it belongs to
_ap_partners = [None] * (N + 1)
for x in range(1, N + 1):
    partners = []
    for idx in _ap_by_element[x]:
        a, b, c, e = _all_4aps[idx]
        others = tuple(t for t in (a, b, c, e) if t != x)
        partners.append(others)
    _ap_partners[x] = tuple(partners)

# Precompute conflict bitmask: for each x, a bitmask of all elements y such that
# there exists a 4-AP containing both x and y where the other two elements are also
# in the set. Actually, simpler: for each x, precompute a list of "conflict triples"
# but also a bitmask for fast checking.
# 
# Better approach: For each x, precompute a list of bitmasks where each bitmask
# represents one blocking AP (the other 3 elements). Then _can_add checks if any
# of these bitmasks is fully contained in the current set.
#
# Even faster: precompute for each x a single "conflict" bitmask that has bits set
# for elements that would conflict. But that doesn't work directly because we need
# ALL three other elements to be present.
#
# Let's use a different approach: for each x, store the list of (o1, o2, o3) tuples,
# and also precompute a combined bitmask where bit i is set if element i appears in
# any blocking AP for x. This gives a quick reject, but we still need the detailed check.
#
# Actually, the most effective speedup: use integer bitmasks for the set, and for
# each x, precompute a list of 3-element bitmasks (each with exactly 3 bits set).
# Then _can_add checks: for each 3-bitmask m, is (bits & m) == m?
_conflict_bitmasks = [0] * (N + 1)
_conflict_mask_list = [[] for _ in range(N + 1)]
for x in range(1, N + 1):
    for others in _ap_partners[x]:
        o1, o2, o3 = others
        mask = (1 << o1) | (1 << o2) | (1 << o3)
        _conflict_mask_list[x].append(mask)
        _conflict_bitmasks[x] |= mask


def _cantor_bits():
    """Base-3 Cantor set: numbers with no digit 2 in base 3."""
    bits = 0
    for x in range(1, N + 1):
        t = x
        valid = True
        while t > 0:
            if t % 3 == 2:
                valid = False
                break
            t //= 3
        if valid:
            bits |= (1 << x)
    return bits


def _base3_mod_bits():
    """
    Alternative: numbers whose base-3 representation has at most one '2' digit.
    This is NOT guaranteed 4-AP-free, so we can't use it directly.
    Skip this.
    """
    return 0


def _can_add(x, bits):
    """Check if x can be added to set represented by bits without creating a 4-AP."""
    # Quick reject: if no conflict element is in the set, we're fine
    if (bits & _conflict_bitmasks[x]) == 0:
        return True
    # Detailed check
    for mask in _conflict_mask_list[x]:
        if (bits & mask) == mask:
            return False
    return True


def _is_valid(bits):
    """Check if the set is 4-AP-free."""
    for x in range(1, N + 1):
        if not ((bits >> x) & 1):
            continue
        if (bits & _conflict_bitmasks[x]) == 0:
            continue
        for mask in _conflict_mask_list[x]:
            if (bits & mask) == mask:
                return False
    return True


def _bits_to_set(bits):
    return [i for i in range(1, N + 1) if (bits >> i) & 1]


def _greedy_forward(bits_start=0):
    """Forward greedy: add elements in order 1..N if possible."""
    bits = bits_start
    for x in range(1, N + 1):
        if not ((bits >> x) & 1) and _can_add(x, bits):
            bits |= (1 << x)
    return bits


def _greedy_random(bits_start=0, rng=None):
    """Random-order greedy."""
    if rng is None:
        rng = random.Random()
    order = list(range(1, N + 1))
    rng.shuffle(order)
    bits = bits_start
    for x in order:
        if not ((bits >> x) & 1) and _can_add(x, bits):
            bits |= (1 << x)
    return bits


def _local_search(bits, deadline, max_passes=20):
    """
    Local search: try to add elements, and if blocked, try swapping.
    Uses bitmask-based conflict checking for speed.
    """
    current = bits
    current_size = bin(current).count('1')
    
    for _pass in range(max_passes):
        if time.time() > deadline - 2:
            break
        improved = False
        
        # First try to add any element
        for x in range(1, N + 1):
            if time.time() > deadline - 2:
                return current
            if not ((current >> x) & 1) and _can_add(x, current):
                current |= (1 << x)
                current_size += 1
                improved = True
        
        if not improved:
            # Try swaps: for each x not in set, find minimal removal set
            for x in range(1, N + 1):
                if time.time() > deadline - 2:
                    return current
                if (current >> x) & 1:
                    continue
                
                # Find all APs that would be completed
                blocking_aps = []
                for mask in _conflict_mask_list[x]:
                    if (current & mask) == mask:
                        # Decode the mask back to elements
                        elems = []
                        for e in range(1, N + 1):
                            if (mask >> e) & 1:
                                elems.append(e)
                        blocking_aps.append(tuple(elems))
                
                if not blocking_aps:
                    if _can_add(x, current):
                        current |= (1 << x)
                        current_size += 1
                        improved = True
                    continue
                
                # Try removing 1 element from blocking APs
                candidates = set()
                for ap in blocking_aps:
                    for elem in ap:
                        candidates.add(elem)
                
                # Try each single removal
                found_swap = False
                for rem in candidates:
                    if time.time() > deadline - 2:
                        return current
                    trial = current & ~(1 << rem)
                    if _can_add(x, trial):
                        trial |= (1 << x)
                        trial = _greedy_forward(trial)
                        trial_size = bin(trial).count('1')
                        if trial_size > current_size:
                            current = trial
                            current_size = trial_size
                            improved = True
                            found_swap = True
                            break
                
                # If no single swap worked, try 2-element removals
                if not found_swap and len(blocking_aps) <= 4 and time.time() < deadline - 3:
                    all_elems = set()
                    for ap in blocking_aps:
                        all_elems.update(ap)
                    elem_list = list(all_elems)
                    if len(elem_list) <= 12:
                        for i in range(len(elem_list)):
                            if time.time() > deadline - 2:
                                return current
                            for j in range(i + 1, len(elem_list)):
                                trial = current & ~(1 << elem_list[i]) & ~(1 << elem_list[j])
                                if _can_add(x, trial):
                                    trial |= (1 << x)
                                    trial = _greedy_forward(trial)
                                    trial_size = bin(trial).count('1')
                                    if trial_size > current_size:
                                        current = trial
                                        current_size = trial_size
                                        improved = True
                                        found_swap = True
                                        break
                            if found_swap:
                                break
            
            if not improved:
                break
    
    return current


def _randomized_local_search(bits, deadline, rng):
    """
    Randomized version: shuffle the order in which we try to add elements,
    and randomize which removals to try first.
    """
    current = bits
    current_size = bin(current).count('1')
    
    for _pass in range(10):
        if time.time() > deadline - 2:
            break
        improved = False
        
        # Random order for trying additions
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        for x in order:
            if time.time() > deadline - 2:
                return current
            if not ((current >> x) & 1) and _can_add(x, current):
                current |= (1 << x)
                current_size += 1
                improved = True
        
        if not improved:
            # Try a few random swaps
            for _attempt in range(50):
                if time.time() > deadline - 2:
                    return current
                x = rng.randint(1, N)
                if (current >> x) & 1:
                    continue
                
                blocking_aps = []
                for mask in _conflict_mask_list[x]:
                    if (current & mask) == mask:
                        elems = []
                        for e in range(1, N + 1):
                            if (mask >> e) & 1:
                                elems.append(e)
                        blocking_aps.append(tuple(elems))
                
                if not blocking_aps:
                    continue
                
                candidates = set()
                for ap in blocking_aps:
                    candidates.update(ap)
                
                if not candidates:
                    continue
                
                rem = rng.choice(list(candidates))
                trial = current & ~(1 << rem)
                if _can_add(x, trial):
                    trial |= (1 << x)
                    trial = _greedy_forward(trial)
                    trial_size = bin(trial).count('1')
                    if trial_size > current_size:
                        current = trial
                        current_size = trial_size
                        improved = True
    
    return current


def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # 1. Cantor set seed
    cantor = _cantor_bits()
    best = cantor
    best_size = bin(cantor).count('1')
    
    # 2. Try forward greedy from scratch
    fwd = _greedy_forward()
    fwd_size = bin(fwd).count('1')
    if fwd_size > best_size:
        best = fwd
        best_size = fwd_size
    
    # 3. Local search on the best so far
    if time.time() < deadline - 30:
        improved = _local_search(best, deadline - 10)
        imp_size = bin(improved).count('1')
        if imp_size > best_size:
            best = improved
            best_size = imp_size
    
    # 4. Randomized restarts with more iterations (faster _can_add enables more)
    rng = random.Random(12345)
    restart_count = 0
    while time.time() < deadline - 15 and restart_count < 50:
        restart_count += 1
        
        # Different strategies
        r = rng.random()
        if r < 0.25:
            # Start from Cantor, add randomly
            bits = _greedy_random(cantor, rng)
        elif r < 0.5:
            # Pure random greedy
            bits = _greedy_random(0, rng)
        elif r < 0.75:
            # Start from best, perturb: remove a random element, then re-greedy
            bits = best
            if best_size > 1:
                elems = _bits_to_set(bits)
                if elems:
                    rem = rng.choice(elems)
                    bits = bits & ~(1 << rem)
                    bits = _greedy_forward(bits)
        else:
            # Start from best, perturb: remove 2 random elements, then re-greedy
            bits = best
            if best_size > 2:
                elems = _bits_to_set(bits)
                if len(elems) >= 2:
                    rem1 = rng.choice(elems)
                    elems.remove(rem1)
                    rem2 = rng.choice(elems)
                    bits = bits & ~(1 << rem1) & ~(1 << rem2)
                    bits = _greedy_forward(bits)
        
        size = bin(bits).count('1')
        if size > best_size:
            best = bits
            best_size = size
        
        # Local search on this result
        if time.time() < deadline - 10:
            improved = _local_search(bits, deadline - 8)
            imp_size = bin(improved).count('1')
            if imp_size > best_size:
                best = improved
                best_size = imp_size
    
    # 5. Final local search on best
    if time.time() < deadline - 5:
        improved = _local_search(best, deadline)
        imp_size = bin(improved).count('1')
        if imp_size > best_size:
            best = improved
            best_size = imp_size
    
    # 6. Verify
    if not _is_valid(best):
        best = cantor
    
    return sorted(_bits_to_set(best))


if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
