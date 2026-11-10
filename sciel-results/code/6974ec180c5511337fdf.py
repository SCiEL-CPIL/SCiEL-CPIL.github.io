"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    N = 128

    # Precompute for each element c, the list of "other 5" masks for each forbidden triple
    # A triple (a,b,c,s1,s2,s3) forbids having all 6 in the set.
    # When we consider adding element c, we need to check: for each triple involving c,
    # are the other 5 elements all present?
    # Store as list of 5-bit masks (bits for the other 5 elements)
    triples_by_element = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                s1 = a + b
                s2 = a + c
                s3 = b + c
                if s3 <= N:
                    six = (a, b, c, s1, s2, s3)
                    for val in six:
                        if val <= N:
                            # Build mask of the other 5
                            other_mask = 0
                            for v in six:
                                if v != val:
                                    other_mask |= (1 << v)
                            triples_by_element[val].append(other_mask)

    def is_valid_addition(mask, c):
        """Check if adding c to mask keeps validity."""
        for other_mask in triples_by_element[c]:
            # Check if all bits in other_mask are set in mask
            if (mask & other_mask) == other_mask:
                return False
        return True

    # Better base: [63, 128] is valid (size 66)
    # For a<b<c all >= 63: b+c >= 64+65=129 > 128, so b+c never in set
    # Thus no triple can have all 3 pair sums in set
    base_set = set(range(63, 129))
    base_mask = 0
    for x in base_set:
        base_mask |= (1 << x)

    remaining = [x for x in range(1, 63)]

    best_mask = base_mask
    best_size = len(base_set)

    # Greedy: try to add elements from 1..62
    # An element x in 1..62 can be added if no forbidden triple is completed
    # Key insight: for x < 63, a triple (a,b,c,s1,s2,s3) with x in it requires
    # all 6 values in set. Since all elements in base are >= 63, the only way a triple
    # involving x is completed is if the other 5 are all in the set.
    
    # Greedy add from remaining
    rng = random.Random(42)
    
    # Try multiple orderings
    orderings = [
        sorted(remaining),
        sorted(remaining, reverse=True),
    ]
    for _ in range(30):
        c = remaining[:]
        rng.shuffle(c)
        orderings.append(c)

    for strat in orderings:
        if time.time() > deadline - 10:
            break
        mask = base_mask
        for x in strat:
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
                        # Try to grow
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

    # 1-for-2 swaps
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

    # Final polish
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
