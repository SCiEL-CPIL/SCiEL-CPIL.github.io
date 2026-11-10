"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    N = 128

    # Precompute conflict structure for fast validity checks
    # For each candidate c, precompute all triples (a,b,c,s1,s2,s3) where c appears
    # as one of the six values, with all values in 1..128
    triples_by_element = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                s1 = a + b
                s2 = a + c
                s3 = b + c
                if s3 <= N:  # s3 is the largest sum
                    triple = (a, b, c, s1, s2, s3)
                    for val in (a, b, c, s1, s2, s3):
                        if val <= N:
                            triples_by_element[val].append(triple)

    def is_valid_addition(mask, c):
        """Fast check: adding c to mask keeps validity iff no triple involving c
        has all six values in mask ∪ {c}."""
        for (a, b, c2, s1, s2, s3) in triples_by_element[c]:
            # All six must be in mask ∪ {c}
            # Since c is being added, treat c as present
            ok = True
            for v in (a, b, c2, s1, s2, s3):
                if v == c:
                    continue
                if not (mask >> v) & 1:
                    ok = False
                    break
            if ok:
                return False
        return True

    # Base set: use sum-free upper block plus a carefully chosen lower block
    # {65..128} is sum-free (min pair sum = 131 > 128)
    # For the lower block, we want elements whose pairwise sums don't create
    # conflicts with the upper block or with each other
    # {17..32}: max pair sum = 64, min upper = 65, so no cross-sum in upper
    # Also need: for a,b in lower, a+b not in lower (sum-free within lower)
    # {17..32}: min pair sum = 35, max element = 32, so 35 > 32, sum-free ✓
    
    base_set = set(range(65, 129)) | set(range(17, 33))
    base_mask = 0
    for x in base_set:
        base_mask |= (1 << x)

    remaining = [x for x in range(1, N + 1) if x not in base_set]

    best_mask = base_mask
    best_size = len(base_set)

    # Multiple greedy strategies
    rng = random.Random(42)
    
    strategies = []
    strategies.append(sorted(remaining))
    strategies.append(sorted(remaining, reverse=True))
    # Alternating small/large
    half = len(remaining) // 2
    small = sorted(remaining[:half])
    large = sorted(remaining[half:], reverse=True)
    strategies.append(small + large)
    strategies.append(large + small)
    # Random shuffles
    for _ in range(60):
        c = remaining[:]
        rng.shuffle(c)
        strategies.append(c)

    for strat in strategies:
        if time.time() > deadline - 8:
            break
        mask = base_mask
        for x in strat:
            if time.time() > deadline - 8:
                break
            if (mask >> x) & 1:
                continue
            if is_valid_addition(mask, x):
                mask |= (1 << x)
        size = bin(mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = mask

    # Local improvement: 1-for-1 swaps
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            mask = best_mask
            in_mask = [x for x in range(1, N + 1) if (mask >> x) & 1]
            not_in_mask = [x for x in range(1, N + 1) if not (mask >> x) & 1]
            
            for x in in_mask:
                if time.time() > deadline - 5:
                    break
                mask_remove = mask & ~(1 << x)
                for y in not_in_mask:
                    if time.time() > deadline - 5:
                        break
                    if is_valid_addition(mask_remove, y):
                        mask_new = mask_remove | (1 << y)
                        # Try to grow from here
                        mask_grow = mask_new
                        for w in not_in_mask:
                            if w == y or (mask_grow >> w) & 1:
                                continue
                            if is_valid_addition(mask_grow, w):
                                mask_grow |= (1 << w)
                        size_new = bin(mask_grow).count('1')
                        if size_new > best_size:
                            best_mask = mask_grow
                            best_size = size_new
                            improved = True
                            break
                if improved:
                    break

    # 1-for-2 swaps: remove x, add y and z
    if time.time() < deadline - 3:
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            mask = best_mask
            in_mask = [x for x in range(1, N + 1) if (mask >> x) & 1]
            not_in_mask = [x for x in range(1, N + 1) if not (mask >> x) & 1]
            
            for x in in_mask:
                if time.time() > deadline - 3:
                    break
                mask_remove = mask & ~(1 << x)
                for y in not_in_mask:
                    if time.time() > deadline - 3:
                        break
                    if not is_valid_addition(mask_remove, y):
                        continue
                    mask_temp = mask_remove | (1 << y)
                    for z in not_in_mask:
                        if z == y:
                            continue
                        if (mask_temp >> z) & 1:
                            continue
                        if is_valid_addition(mask_temp, z):
                            mask_new = mask_temp | (1 << z)
                            size_new = bin(mask_new).count('1')
                            if size_new > best_size:
                                best_mask = mask_new
                                best_size = size_new
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break

    # Final polish: try adding any remaining elements
    if time.time() < deadline - 1:
        mask = best_mask
        for x in range(1, N + 1):
            if time.time() > deadline - 1:
                break
            if (mask >> x) & 1:
                continue
            if is_valid_addition(mask, x):
                mask |= (1 << x)
        size = bin(mask).count('1')
        if size > best_size:
            best_mask = mask
            best_size = size

    result = [int(x) for x in range(1, N + 1) if (best_mask >> x) & 1]
    return result
# EVOLVE-BLOCK-END
