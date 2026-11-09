"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

START_TIME = time.time()
DEADLINE = 55.0  # Safety margin before 60s kill

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def try_add_fast(x, sums_np, current_size):
    """Try adding element x to the set represented by sums_np (numpy array of Python ints).
    Returns new_sums_np if valid, else None."""
    new_sums = sums_np.copy()
    for k in range(current_size, -1, -1):
        shifted = sums_np[k] << x
        candidate = new_sums[k + 1] | shifted
        # Check against all other cardinalities
        for m in range(current_size + 2):
            if m != k + 1 and (candidate & new_sums[m]):
                return None
        new_sums[k + 1] = candidate
    return new_sums

def try_add_batch(x, sums_list, current_size):
    """Try adding element x. Returns (new_sums_list, True) if valid, else (None, False)."""
    new_sums = list(sums_list)
    for k in range(current_size, -1, -1):
        shifted = sums_list[k] << x
        candidate = new_sums[k + 1] | shifted
        # Check against all other cardinalities
        for m in range(current_size + 2):
            if m != k + 1 and (candidate & new_sums[m]):
                return None, False
        new_sums[k + 1] = candidate
    return new_sums, True

def greedy_construct(order, sums=None, current_size=None, result=None):
    """Greedy: iterate through 'order', add each element if valid.
    Then do a second pass over all remaining candidates."""
    if sums is None:
        sums = [0] * (len(order) + 1)
        sums[0] = 1  # empty subset sum 0
        current_size = 0
        result = []
    used = set(result) if result else set()
    
    # First pass
    for x in order:
        if is_done():
            break
        if x in used:
            continue
        new_sums, valid = try_add_batch(x, sums, current_size)
        if valid:
            sums = new_sums
            current_size += 1
            result.append(x)
            used.add(x)
    
    # Second pass: try all remaining candidates in the order
    remaining = [x for x in order if x not in used]
    for x in remaining:
        if is_done():
            break
        new_sums, valid = try_add_batch(x, sums, current_size)
        if valid:
            sums = new_sums
            current_size += 1
            result.append(x)
            used.add(x)
    
    return result, sums, current_size

def local_search_improve(result, sums, current_size):
    """Try to add more elements after initial greedy."""
    if is_done():
        return result, sums, current_size
    
    all_candidates = list(range(1, 257))
    result_set = set(result)
    candidates = [x for x in all_candidates if x not in result_set]
    
    best_result = list(result)
    best_sums = sums
    best_size = current_size
    
    for x in candidates:
        if is_done():
            break
        new_sums, valid = try_add_batch(x, best_sums, best_size)
        if valid:
            best_sums = new_sums
            best_size += 1
            best_result.append(x)
    
    return best_result, best_sums, best_size

def construct():
    best_set = [1, 2, 4, 8, 16, 32, 64, 128, 256]  # Powers of 2 baseline (9 elements)
    best_sums = None
    best_size = 9
    
    rng = random.Random(42)
    
    # Generate diverse orderings
    orderings = []
    
    # Deterministic orders
    orderings.append(list(range(1, 257)))
    orderings.append(list(range(256, 0, -1)))
    
    # Prime-based orders
    for prime in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
        order = [x for x in range(1, 257) if x % prime == 0]
        order += [x for x in range(1, 257) if x % prime != 0]
        orderings.append(order)
    
    # Random orders - use time budget to generate more
    num_random = 50
    for i in range(num_random):
        if is_done():
            break
        order = list(range(1, 257))
        rng.shuffle(order)
        orderings.append(order)
    
    # Structured random: mix small and large
    for i in range(20):
        if is_done():
            break
        small = list(range(1, 50))
        medium = list(range(50, 150))
        large = list(range(150, 257))
        rng.shuffle(small)
        rng.shuffle(medium)
        rng.shuffle(large)
        # Interleave
        order = []
        si, mi, li = 0, 0, 0
        while si < len(small) or mi < len(medium) or li < len(large):
            if si < len(small):
                order.append(small[si]); si += 1
            if mi < len(medium):
                order.append(medium[mi]); mi += 1
            if li < len(large):
                order.append(large[li]); li += 1
        orderings.append(order)
    
    # Weighted random: bias toward certain ranges
    for i in range(20):
        if is_done():
            break
        weights = []
        for x in range(1, 257):
            # Higher weight for numbers that are powers of 2 or near them
            log2 = int(np.log2(x))
            closest_pow = 1 << log2
            if x == closest_pow:
                w = 10.0
            elif abs(x - closest_pow) <= 2:
                w = 5.0
            else:
                w = 1.0
            weights.append(w)
        order = list(range(1, 257))
        # Sample without replacement weighted
        remaining = list(range(256))
        weighted_order = []
        for _ in range(256):
            if not remaining:
                break
            total_w = sum(weights[x-1] for x in remaining)
            if total_w <= 0:
                weighted_order.extend(remaining)
                break
            r = rng.random() * total_w
            cum = 0
            chosen = remaining[0]
            for x in remaining:
                cum += weights[x-1]
                if r <= cum:
                    chosen = x
                    break
            weighted_order.append(chosen)
            remaining.remove(chosen)
        orderings.append(weighted_order)
    
    # Run greedy on each ordering
    for order in orderings:
        if is_done():
            break
        result, sums, current_size = greedy_construct(order)
        if len(result) > len(best_set):
            best_set = result
            best_sums = sums
            best_size = current_size
    
    # Local search improvement on best found
    if best_sums is not None and not is_done():
        best_set, best_sums, best_size = local_search_improve(best_set, best_sums, best_size)
    
    # Ensure Python ints in a list
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
