import time
from fractions import Fraction
import sys

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Precompute reciprocal values
    recips = [Fraction(0, 1)] + [Fraction(1, i) for i in range(1, N + 1)]

    # Precompute for each x: list of (a, b) pairs such that {a, b, x} forms a 3-AP
    # and a < b, a != x, b != x
    conflict_pairs = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        pairs = []
        for a in range(1, N + 1):
            if a == x:
                continue
            for b in range(a + 1, N + 1):
                if b == x:
                    continue
                if 2 * a == b + x or 2 * b == a + x or 2 * x == a + b:
                    pairs.append((a, b))
        conflict_pairs[x] = pairs

    # Helper: check if adding x to mask is valid
    def is_valid_add(mask, x):
        for (a, b) in conflict_pairs[x]:
            if (mask >> (a - 1)) & 1 and (mask >> (b - 1)) & 1:
                return False
        return True

    # Helper: check if mask represents a valid 3-AP-free set
    def is_valid_set(mask):
        for x in range(1, N + 1):
            if (mask >> (x - 1)) & 1:
                for (a, b) in conflict_pairs[x]:
                    if (mask >> (a - 1)) & 1 and (mask >> (b - 1)) & 1:
                        return False
        return True

    # Helper: compute sum from mask
    def mask_sum(mask):
        s = Fraction(0, 1)
        for x in range(1, N + 1):
            if (mask >> (x - 1)) & 1:
                s += recips[x]
        return s

    # Helper: get list from mask
    def mask_to_list(mask):
        return [x for x in range(1, N + 1) if (mask >> (x - 1)) & 1]

    # --- Construction 1: Simple greedy (smallest first) ---
    mask1 = 0
    for x in range(1, N + 1):
        if is_valid_add(mask1, x):
            mask1 |= (1 << (x - 1))

    # --- Construction 2: Base-p digit method ---
    # Numbers with no zero digit in base p form a 3-AP-free set
    # Try several primes and pick the best
    def base_p_set(p, max_val=N):
        """Return bitmask of numbers in [1, max_val] with no zero digit in base p."""
        mask = 0
        for x in range(1, max_val + 1):
            n = x
            has_zero = False
            while n > 0:
                if n % p == 0:
                    has_zero = True
                    break
                n //= p
            if not has_zero:
                mask |= (1 << (x - 1))
        return mask

    best_mask = mask1
    best_sum = mask_sum(mask1)

    for p in [2, 3, 5, 7, 11, 13]:
        pm = base_p_set(p)
        if pm == 0:
            continue
        ps = mask_sum(pm)
        if ps > best_sum:
            best_mask = pm
            best_sum = ps

    # Also try greedy from the base-p set: add remaining elements greedily
    for p in [3, 5, 7, 11]:
        pm = base_p_set(p)
        # Greedily add elements not in pm
        pm2 = pm
        for x in range(1, N + 1):
            if not ((pm2 >> (x - 1)) & 1):
                if is_valid_add(pm2, x):
                    pm2 |= (1 << (x - 1))
        ps2 = mask_sum(pm2)
        if ps2 > best_sum:
            best_mask = pm2
            best_sum = ps2

    # --- Local search with add, swap, and 2-for-1 moves ---
    improved = True
    while improved and time.time() < deadline:
        improved = False

        # Pass 1: Try all single additions
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if not ((best_mask >> (x - 1)) & 1):
                if is_valid_add(best_mask, x):
                    new_sum = best_sum + recips[x]
                    if new_sum > best_sum:
                        best_mask |= (1 << (x - 1))
                        best_sum = new_sum
                        improved = True

        if time.time() > deadline:
            break

        # Pass 2: Try swaps (remove x, add y)
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if (best_mask >> (x - 1)) & 1:
                trial_mask = best_mask & ~(1 << (x - 1))
                for y in range(1, N + 1):
                    if time.time() > deadline:
                        break
                    if y == x:
                        continue
                    if not ((trial_mask >> (y - 1)) & 1):
                        if is_valid_add(trial_mask, y):
                            new_sum = best_sum - recips[x] + recips[y]
                            if new_sum > best_sum:
                                best_mask = trial_mask | (1 << (y - 1))
                                best_sum = new_sum
                                improved = True
                                break
                if improved:
                    break

        if time.time() > deadline:
            break

        # Pass 3: Try 2-for-1 (remove x, y; add z)
        # This can be beneficial when removing two large-index elements frees up
        # a small-index element that has high reciprocal value
        current_list = mask_to_list(best_mask)
        for i in range(len(current_list)):
            if time.time() > deadline:
                break
            x = current_list[i]
            for j in range(i + 1, len(current_list)):
                if time.time() > deadline:
                    break
                y = current_list[j]
                trial_mask = best_mask & ~(1 << (x - 1)) & ~(1 << (y - 1))
                # Only try adding elements smaller than both x and y (higher reciprocal)
                max_idx = max(x, y)
                for z in range(1, max_idx):
                    if time.time() > deadline:
                        break
                    if z == x or z == y:
                        continue
                    if not ((trial_mask >> (z - 1)) & 1):
                        if is_valid_add(trial_mask, z):
                            new_sum = best_sum - recips[x] - recips[y] + recips[z]
                            if new_sum > best_sum:
                                best_mask = trial_mask | (1 << (z - 1))
                                best_sum = new_sum
                                improved = True
                                break
                if improved:
                    break
            if improved:
                break

        if time.time() > deadline:
            break

        # Pass 4: Try 1-for-2 (remove x; add y, z)
        # Remove one large element, add two smaller ones
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if (best_mask >> (x - 1)) & 1:
                trial_mask = best_mask & ~(1 << (x - 1))
                for y in range(1, N + 1):
                    if time.time() > deadline:
                        break
                    if y == x:
                        continue
                    if not ((trial_mask >> (y - 1)) & 1):
                        if is_valid_add(trial_mask, y):
                            trial_mask2 = trial_mask | (1 << (y - 1))
                            for z in range(1, N + 1):
                                if time.time() > deadline:
                                    break
                                if z == x or z == y:
                                    continue
                                if not ((trial_mask2 >> (z - 1)) & 1):
                                    if is_valid_add(trial_mask2, z):
                                        new_sum = best_sum - recips[x] + recips[y] + recips[z]
                                        if new_sum > best_sum:
                                            best_mask = trial_mask2 | (1 << (z - 1))
                                            best_sum = new_sum
                                            improved = True
                                            break
                            if improved:
                                break
                    if improved:
                        break
                if improved:
                    break

    # Extract final list
    final_list = []
    for x in range(1, N + 1):
        if (best_mask >> (x - 1)) & 1:
            final_list.append(x)

    # Ensure nonempty
    if not final_list:
        final_list = [1]

    return final_list
