import time
import random

DEADLINE = 85.0  # safety margin below 90s

def _build_4ap_conflicts(N):
    """Precompute for each x the list of (y, z) pairs such that {x, y, z} is part of a 4-AP.
    Actually, we need: for each x, which other elements conflict with it given the current set.
    Better: precompute all 4-APs as tuples, then for each element, list of 4-APs containing it.
    """
    # For N=512, number of 4-APs is manageable
    # A 4-AP is (a, a+d, a+2d, a+3d) with a >= 0, a+3d < N
    ap_list = []
    for a in range(N):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < N:
                ap_list.append((a, b, c, e))
    
    # For each element, list of AP indices containing it
    elem_to_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(ap_list):
        for elem in ap:
            elem_to_aps[elem].append(idx)
    
    return ap_list, elem_to_aps


def _cantor_set_bits(N):
    """Construct the Cantor-like set: numbers in [0, N) with no digit '2' in base 3."""
    bits = 0
    for x in range(N):
        # Check if x has no digit 2 in base 3
        v = x
        ok = True
        while v > 0:
            if v % 3 == 2:
                ok = False
                break
            v //= 3
        if ok:
            bits |= (1 << x)
    return bits


def _has_4ap_in_set(bits, N, ap_list, elem_to_aps):
    """Check if the set (represented as bitset) contains any 4-AP. Return list of violating APs."""
    violations = []
    for idx, ap in enumerate(ap_list):
        a, b, c, e = ap
        if (bits >> a) & 1 and (bits >> b) & 1 and (bits >> c) & 1 and (bits >> e) & 1:
            violations.append(idx)
    return violations


def _can_add(bits, x, N, ap_list, elem_to_aps):
    """Check if adding x to bits creates a 4-AP."""
    for ap_idx in elem_to_aps[x]:
        a, b, c, e = ap_list[ap_idx]
        # Check if all other 3 elements are in bits
        others = [a, b, c, e]
        others.remove(x)
        all_present = True
        for o in others:
            if not ((bits >> o) & 1):
                all_present = False
                break
        if all_present:
            return False
    return True


def _greedy_add_from(bits, N, rng, ap_list, elem_to_aps, order=None):
    """Greedily add elements in given order that don't create 4-APs."""
    if order is None:
        order = list(range(N))
        rng.shuffle(order)
    for x in order:
        if (bits >> x) & 1:
            continue
        if _can_add(bits, x, N, ap_list, elem_to_aps):
            bits |= (1 << x)
    return bits


def _local_search_add(bits, N, rng, ap_list, elem_to_aps, max_steps=2000, deadline=None):
    """Try to add random elements that don't create 4-APs."""
    count = 0
    for _ in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        x = rng.randint(0, N - 1)
        if (bits >> x) & 1:
            continue
        if _can_add(bits, x, N, ap_list, elem_to_aps):
            bits |= (1 << x)
            count += 1
    return bits, count


def _local_search_swap(bits, N, rng, ap_list, elem_to_aps, max_steps=2000, deadline=None):
    """Try swap moves: remove one element, then try to add two others."""
    count = 0
    for _ in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        # Pick a random element to remove
        elems = [i for i in range(N) if (bits >> i) & 1]
        if not elems:
            break
        rem = rng.choice(elems)
        trial = bits & ~(1 << rem)
        
        # Try to add up to 2 new elements
        added = 0
        for _ in range(20):
            x = rng.randint(0, N - 1)
            if (trial >> x) & 1:
                continue
            if _can_add(trial, x, N, ap_list, elem_to_aps):
                trial |= (1 << x)
                added += 1
                if added >= 2:
                    break
        
        if added >= 1:
            bits = trial
            count += added - 1  # net gain
    
    return bits, count


def _repair_remove(bits, N, rng, ap_list, elem_to_aps, deadline=None):
    """Remove elements to eliminate all 4-APs, then re-add greedily."""
    # Find all violations
    violations = _has_4ap_in_set(bits, N, ap_list, elem_to_aps)
    while violations:
        if deadline is not None and time.time() > deadline:
            break
        # Pick a random violating AP
        ap_idx = rng.choice(violations)
        a, b, c, e = ap_list[ap_idx]
        # Remove one random element from this AP
        rem = rng.choice([a, b, c, e])
        bits &= ~(1 << rem)
        # Re-check
        violations = _has_4ap_in_set(bits, N, ap_list, elem_to_aps)
    
    # Now greedily re-add
    bits = _greedy_add_from(bits, N, rng, ap_list, elem_to_aps)
    return bits


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    N = 512
    rng = random.Random(12345)
    
    # Precompute 4-AP structure
    ap_list, elem_to_aps = _build_4ap_conflicts(N)
    
    best_bits = 0
    best_size = 0
    
    # Strategy 1: Cantor set initialization
    cantor_bits = _cantor_set_bits(N)
    cantor_size = bin(cantor_bits).count('1')
    # Ensure no 4-APs in Cantor set (it should be 4-AP-free by construction for small N,
    # but let's verify and repair if needed)
    violations = _has_4ap_in_set(cantor_bits, N, ap_list, elem_to_aps)
    if violations:
        cantor_bits = _repair_remove(cantor_bits, N, rng, ap_list, elem_to_aps, deadline - 1.0)
        cantor_size = bin(cantor_bits).count('1')
    
    # Local search on Cantor set
    cantor_bits, _ = _local_search_add(cantor_bits, N, rng, ap_list, elem_to_aps, max_steps=3000, deadline=deadline - 2.0)
    cantor_size = bin(cantor_bits).count('1')
    cantor_bits, _ = _local_search_swap(cantor_bits, N, rng, ap_list, elem_to_aps, max_steps=3000, deadline=deadline - 2.0)
    cantor_size = bin(cantor_bits).count('1')
    
    best_bits = cantor_bits
    best_size = cantor_size
    
    # Strategy 2: Multiple randomized greedy restarts from scratch
    restarts = 0
    while time.time() < deadline - 5.0:
        # Random order greedy
        order = list(range(N))
        rng.shuffle(order)
        bits = _greedy_add_from(0, N, rng, ap_list, elem_to_aps, order)
        sz = bin(bits).count('1')
        
        # Local search
        bits, _ = _local_search_add(bits, N, rng, ap_list, elem_to_aps, max_steps=2000, deadline=deadline - 3.0)
        bits, _ = _local_search_swap(bits, N, rng, ap_list, elem_to_aps, max_steps=2000, deadline=deadline - 3.0)
        sz = bin(bits).count('1')
        
        if sz > best_size:
            best_size = sz
            best_bits = bits
            restarts += 1
        
        # Occasionally perturb the best solution
        if restarts > 3 and rng.random() < 0.4 and time.time() < deadline - 4.0:
            perturbed = best_bits
            # Remove 3-5 random elements
            for _ in range(rng.randint(3, 5)):
                elems = [i for i in range(N) if (perturbed >> i) & 1]
                if elems:
                    rem = rng.choice(elems)
                    perturbed &= ~(1 << rem)
            # Re-add greedily
            perturbed = _greedy_add_from(perturbed, N, rng, ap_list, elem_to_aps)
            # Local search
            perturbed, _ = _local_search_add(perturbed, N, rng, ap_list, elem_to_aps, max_steps=1500, deadline=deadline - 3.0)
            perturbed, _ = _local_search_swap(perturbed, N, rng, ap_list, elem_to_aps, max_steps=1500, deadline=deadline - 3.0)
            sz = bin(perturbed).count('1')
            if sz > best_size:
                best_size = sz
                best_bits = perturbed
                restarts += 1
    
    # Final local search on best
    if time.time() < deadline - 2.0:
        best_bits, _ = _local_search_add(best_bits, N, rng, ap_list, elem_to_aps, max_steps=3000, deadline=deadline - 1.0)
        best_bits, _ = _local_search_swap(best_bits, N, rng, ap_list, elem_to_aps, max_steps=3000, deadline=deadline - 1.0)
        best_size = bin(best_bits).count('1')
    
    # Convert to 1-indexed list
    result = [i + 1 for i in range(N) if (best_bits >> i) & 1]
    
    if not result:
        result = [1]
    
    return result
