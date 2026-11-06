"""Fast local search using NumPy bitmasks for 4-AP and 3-AP checks."""
import time
import random
import numpy as np

N = 256
DEADLINE = 55.0  # safety margin under 90s kill


def construct():
    start = time.time()
    random.seed(42)

    # Bitmask representation: bit i (0-indexed) represents element (i+1)
    # So element x has bit (x-1)
    
    def mask_has_4ap(mask, x):
        """Check if adding x to mask creates a 4-AP involving x."""
        # x as 1st term: x, x+d, x+2d, x+3d
        # Constraints: x+3d <= N => d <= (N-x)//3
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            if (mask >> (x + d - 1) & 1) and (mask >> (x + 2*d - 1) & 1) and (mask >> (x + 3*d - 1) & 1):
                return True
        # x as 2nd term: x-d, x, x+d, x+2d
        # Constraints: x-d >= 1 => d <= x-1; x+2d <= N => d <= (N-x)//2
        max_d = min(x - 1, (N - x) // 2)
        for d in range(1, max_d + 1):
            if (mask >> (x - d - 1) & 1) and (mask >> (x + d - 1) & 1) and (mask >> (x + 2*d - 1) & 1):
                return True
        # x as 3rd term: x-2d, x-d, x, x+d
        # Constraints: x-2d >= 1 => d <= (x-1)//2; x+d <= N => d <= N-x
        max_d = min((x - 1) // 2, N - x)
        for d in range(1, max_d + 1):
            if (mask >> (x - 2*d - 1) & 1) and (mask >> (x - d - 1) & 1) and (mask >> (x + d - 1) & 1):
                return True
        # x as 4th term: x-3d, x-2d, x-d, x
        # Constraints: x-3d >= 1 => d <= (x-1)//3
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if (mask >> (x - 3*d - 1) & 1) and (mask >> (x - 2*d - 1) & 1) and (mask >> (x - d - 1) & 1):
                return True
        return False

    def delta_add(mask, x):
        """Number of new 3-APs created by adding x."""
        cnt = 0
        # x as first: x, x+d, x+2d
        # Constraints: x+2d <= N => d <= (N-x)//2
        max_d = (N - x) // 2
        for d in range(1, max_d + 1):
            if (mask >> (x + d - 1) & 1) and (mask >> (x + 2*d - 1) & 1):
                cnt += 1
        # x as middle: x-d, x, x+d
        # Constraints: x-d >= 1 => d <= x-1; x+d <= N => d <= N-x
        max_d = min(x - 1, N - x)
        for d in range(1, max_d + 1):
            if (mask >> (x - d - 1) & 1) and (mask >> (x + d - 1) & 1):
                cnt += 1
        # x as last: x-2d, x-d, x
        # Constraints: x-2d >= 1 => d <= (x-1)//2
        max_d = (x - 1) // 2
        for d in range(1, max_d + 1):
            if (mask >> (x - 2*d - 1) & 1) and (mask >> (x - d - 1) & 1):
                cnt += 1
        return cnt

    def delta_remove(mask, x):
        """Number of 3-APs destroyed by removing x."""
        # This is identical to delta_add because the condition "other two elements are in mask"
        # is the same whether we are adding x to a set or removing x from a set (assuming x is in set for remove).
        # Wait, delta_add checks if neighbors are in `mask`.
        # delta_remove should check if neighbors are in `mask` (where x is still present? No, usually we calculate contribution of x to the total count).
        # The number of 3-APs containing x is the number of pairs (a,b) in A\{x} such that a,x,b is a 3-AP.
        # This is exactly what delta_add(mask, x) computes if `mask` does NOT contain x.
        # If `mask` DOES contain x, we must ensure we don't count x itself as a neighbor.
        # In delta_add, we check `mask >> (x+d-1)`. If x is in mask, this doesn't affect the check for neighbors x+d.
        # So delta_add(mask, x) gives the number of 3-APs formed by x and elements in mask.
        # For removal, we want to know how many 3-APs are lost. These are the 3-APs that contain x.
        # These are exactly the 3-APs formed by x and the other elements in the set.
        # So delta_remove(mask, x) is effectively delta_add(mask, x) if we treat `mask` as the set of *other* elements.
        # But in the current call `delta_remove(mask, rem)`, `mask` contains `rem`.
        # The loop checks `mask >> (rem + d - 1)`. This is correct for neighbors.
        # The bounds are the same.
        cnt = 0
        max_d = (N - x) // 2
        for d in range(1, max_d + 1):
            if (mask >> (x + d - 1) & 1) and (mask >> (x + 2*d - 1) & 1):
                cnt += 1
        max_d = min(x - 1, N - x)
        for d in range(1, max_d + 1):
            if (mask >> (x - d - 1) & 1) and (mask >> (x + d - 1) & 1):
                cnt += 1
        max_d = (x - 1) // 2
        for d in range(1, max_d + 1):
            if (mask >> (x - 2*d - 1) & 1) and (mask >> (x - d - 1) & 1):
                cnt += 1
        return cnt

    def count_3aps(mask):
        """Total 3-APs in mask."""
        total = 0
        for x in range(1, N + 1):
            if mask >> (x - 1) & 1:
                total += delta_add(mask & ~(1 << (x - 1)), x)
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
        elems = [x for x in range(1, N + 1) if mask >> (x - 1) & 1]
        missing = [x for x in range(1, N + 1) if not (mask >> (x - 1) & 1)]
        
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
                if mask_has_4ap(trial_mask & ~(1 << (add - 1)), add):
                    continue
                delta = delta_add(trial_mask & ~(1 << (add - 1)), add) - delta_remove(mask, rem)
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
                # Perturbation
                if elems:
                    rem = random.choice(elems)
                    mask &= ~(1 << (rem - 1))
                    current_3ap -= delta_remove(mask | (1 << (rem - 1)), rem)

    # Convert best_mask to sorted list
    result = [x for x in range(1, N + 1) if best_mask >> (x - 1) & 1]
    return sorted(result)
