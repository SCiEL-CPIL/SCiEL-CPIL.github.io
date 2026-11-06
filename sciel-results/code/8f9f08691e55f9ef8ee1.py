"""Combined: bitmask-accelerated local search with incremental 3-AP deltas."""
import time
import random

N = 256
DEADLINE = 52.0  # safety margin under 90s kill


def construct():
    start = time.time()
    random.seed(12345)

    # Bitmask: bit (x-1) set means element x is in the set
    # Python ints are arbitrary precision, so 256-bit masks are fine

    def mask_has_4ap(mask, x):
        """Return True if mask ∪ {x} contains a nonconstant 4-AP involving x."""
        # x as 1st term: x, x+d, x+2d, x+3d
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            if (mask >> (x + d - 1)) & 1 and (mask >> (x + 2*d - 1)) & 1 and (mask >> (x + 3*d - 1)) & 1:
                return True
        # x as 2nd term: x-d, x, x+d, x+2d
        max_d = min(x - 1, (N - x) // 2)
        for d in range(1, max_d + 1):
            if (mask >> (x - d - 1)) & 1 and (mask >> (x + d - 1)) & 1 and (mask >> (x + 2*d - 1)) & 1:
                return True
        # x as 3rd term: x-2d, x-d, x, x+d
        max_d = min((x - 1) // 2, N - x)
        for d in range(1, max_d + 1):
            if (mask >> (x - 2*d - 1)) & 1 and (mask >> (x - d - 1)) & 1 and (mask >> (x + d - 1)) & 1:
                return True
        # x as 4th term: x-3d, x-2d, x-d, x
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if (mask >> (x - 3*d - 1)) & 1 and (mask >> (x - 2*d - 1)) & 1 and (mask >> (x - d - 1)) & 1:
                return True
        return False

    def delta_add(mask, x):
        """Number of new 3-APs created by adding x to mask (x not in mask)."""
        cnt = 0
        # x as first: x, x+d, x+2d
        max_d = (N - x) // 2
        for d in range(1, max_d + 1):
            if (mask >> (x + d - 1)) & 1 and (mask >> (x + 2*d - 1)) & 1:
                cnt += 1
        # x as middle: x-d, x, x+d
        max_d = min(x - 1, N - x)
        for d in range(1, max_d + 1):
            if (mask >> (x - d - 1)) & 1 and (mask >> (x + d - 1)) & 1:
                cnt += 1
        # x as last: x-2d, x-d, x
        max_d = (x - 1) // 2
        for d in range(1, max_d + 1):
            if (mask >> (x - 2*d - 1)) & 1 and (mask >> (x - d - 1)) & 1:
                cnt += 1
        return cnt

    def count_3aps(mask):
        """Total number of distinct 3-term APs in mask."""
        total = 0
        for x in range(1, N + 1):
            if (mask >> (x - 1)) & 1:
                # Count 3-APs where x is involved, using mask without x
                m = mask & ~(1 << (x - 1))
                total += delta_add(m, x)
        return total

    # Greedy seed
    mask = 0
    for x in range(1, N + 1):
        if not mask_has_4ap(mask, x):
            mask |= (1 << (x - 1))

    current_3ap = count_3aps(mask)
    best_mask = mask
    best_3ap = current_3ap

    # Local search
    while time.time() - start < DEADLINE:
        improved = False
        elems = [x for x in range(1, N + 1) if (mask >> (x - 1)) & 1]
        missing = [x for x in range(1, N + 1) if not ((mask >> (x - 1)) & 1)]

        if not elems or not missing:
            break

        random.shuffle(elems)
        random.shuffle(missing)

        for rem in elems:
            if time.time() - start > DEADLINE:
                break
            for add in missing:
                if time.time() - start > DEADLINE:
                    break
                # Trial: remove rem, add add
                trial_mask = (mask & ~(1 << (rem - 1))) | (1 << (add - 1))
                # Check 4-AP for add in context of trial without add
                if mask_has_4ap(trial_mask & ~(1 << (add - 1)), add):
                    continue
                delta = delta_add(trial_mask & ~(1 << (add - 1)), add) - delta_add(mask & ~(1 << (rem - 1)), rem)
                if delta > 0:
                    mask = trial_mask
                    current_3ap += delta
                    if current_3ap > best_3ap:
                        best_3ap = current_3ap
                        best_mask = mask
                    improved = True
                    break
            if improved:
                break

        if not improved:
            # Try single additions
            found = False
            for add in missing:
                if time.time() - start > DEADLINE:
                    break
                if mask_has_4ap(mask, add):
                    continue
                delta = delta_add(mask, add)
                if delta > 0:
                    mask |= (1 << (add - 1))
                    current_3ap += delta
                    if current_3ap > best_3ap:
                        best_3ap = current_3ap
                        best_mask = mask
                    found = True
                    break
            if not found:
                # Perturbation: remove a random element to escape local optimum
                if elems:
                    rem = random.choice(elems)
                    mask &= ~(1 << (rem - 1))
                    current_3ap -= delta_add(mask | (1 << (rem - 1)), rem)

    # Convert best_mask to sorted list
    result = [x for x in range(1, N + 1) if (best_mask >> (x - 1)) & 1]
    return sorted(result)
