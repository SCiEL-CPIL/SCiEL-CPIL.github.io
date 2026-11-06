"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    DEADLINE = 135.0  # safety margin before 150s limit

    # Prescribed elements
    prescribed = [1, 2, 5, 11]

    # Bitmask approach: sums range from 2 to 568 (284+284)
    # Use a Python int as a bitmask where bit s means sum s is used
    MAX_SUM = 568  # 284 + 284

    def init_mask(s):
        """Initialize bitmask for a set of elements."""
        mask = 0
        for i, a in enumerate(s):
            for b in s[i:]:
                mask |= (1 << (a + b))
        return mask

    def can_add_mask(s, mask, x):
        """Check if x can be added using bitmask. O(n) but with fast bit ops."""
        for y in s:
            if (mask >> (x + y)) & 1:
                return False
        if (mask >> (2 * x)) & 1:
            return False
        return True

    def add_element(s, mask, x):
        """Add element x to set s and update mask."""
        s.append(x)
        for y in s[:-1]:
            mask |= (1 << (x + y))
        mask |= (1 << (2 * x))
        return s, mask

    def conflict_count(s, mask, x):
        """Count how many conflicts x has with existing sums."""
        count = 0
        for y in s:
            if (mask >> (x + y)) & 1:
                count += 1
        if (mask >> (2 * x)) & 1:
            count += 1
        return count

    # Strategy 1: Simple sequential greedy - fast baseline
    s1 = [1, 2, 5, 11]
    mask1 = init_mask(s1)

    for x in range(12, 285):
        if can_add_mask(s1, mask1, x):
            s1, mask1 = add_element(s1, mask1, x)

    best_set = s1[:]
    best_size = len(best_set)
    best_mask = mask1

    # Strategy 2: Many randomized greedy restarts with bitmask speed
    random.seed(42)
    restart_count = 0
    while time.time() < DEADLINE - 30.0:
        restart_count += 1

        candidates = list(range(12, 285))
        random.shuffle(candidates)

        s = [1, 2, 5, 11]
        mask = init_mask(s)

        for x in candidates:
            if can_add_mask(s, mask, x):
                s, mask = add_element(s, mask, x)

        if len(s) > best_size:
            best_set = s[:]
            best_size = len(s)
            best_mask = mask

    # Strategy 3: Local search - swap moves with bitmask
    # Try to improve by removing one element and adding another
    time_budget_local = DEADLINE - 10.0
    local_iters = 0
    while time.time() < time_budget_local:
        local_iters += 1

        s = best_set[:]
        mask = best_mask

        # Pick a random non-prescribed element to try removing
        removable = [x for x in s if x not in prescribed]
        if not removable:
            break

        x_remove = random.choice(removable)
        idx = s.index(x_remove)
        s_trial = s[:idx] + s[idx+1:]
        mask_trial = init_mask(s_trial)

        # Try to add a candidate that improves or maintains size
        candidates = [x for x in range(12, 285) if x not in s_trial]
        random.shuffle(candidates)

        for y in candidates:
            if can_add_mask(s_trial, mask_trial, y):
                s_trial, mask_trial = add_element(s_trial, mask_trial, y)
                if len(s_trial) > best_size:
                    best_set = s_trial[:]
                    best_size = len(s_trial)
                    best_mask = mask_trial
                break

    # Strategy 4: Multi-swap local search - remove 2, add 1 or 2
    while time.time() < DEADLINE - 5.0:
        s = best_set[:]
        mask = best_mask

        removable = [x for x in s if x not in prescribed]
        if len(removable) < 2:
            break

        # Try removing 2 random non-prescribed elements
        x1, x2 = random.sample(removable, 2)
        s_trial = [x for x in s if x not in (x1, x2)]
        mask_trial = init_mask(s_trial)

        # Try to add up to 2 candidates
        candidates = [x for x in range(12, 285) if x not in s_trial]
        random.shuffle(candidates)

        added_count = 0
        for y in candidates:
            if added_count >= 2:
                break
            if can_add_mask(s_trial, mask_trial, y):
                s_trial, mask_trial = add_element(s_trial, mask_trial, y)
                added_count += 1

        if len(s_trial) > best_size:
            best_set = s_trial[:]
            best_size = len(s_trial)
            best_mask = mask_trial

    # Strategy 5: Greedy with dynamic conflict scoring (bitmask-accelerated)
    while time.time() < DEADLINE - 2.0:
        s = [1, 2, 5, 11]
        mask = init_mask(s)
        remaining = list(range(12, 285))

        while remaining:
            if time.time() > DEADLINE - 2.0:
                break

            # Find candidate with minimum conflicts
            min_conf = 999
            best_candidates = []
            for x in remaining:
                c = conflict_count(s, mask, x)
                if c < min_conf:
                    min_conf = c
                    best_candidates = [x]
                elif c == min_conf:
                    best_candidates.append(x)

            if not best_candidates or min_conf > 0:
                # No zero-conflict candidates; try adding the least conflicting
                if best_candidates and min_conf <= len(s) + 1:
                    x = random.choice(best_candidates)
                    if can_add_mask(s, mask, x):
                        s, mask = add_element(s, mask, x)
                        remaining.remove(x)
                    else:
                        remaining.remove(x)
                else:
                    break
            else:
                x = random.choice(best_candidates)
                s, mask = add_element(s, mask, x)
                remaining.remove(x)

        if len(s) > best_size:
            best_set = s[:]
            best_size = len(s)
            best_mask = mask

    # Final validation
    s_final = sorted(set(best_set))
    mask_final = 0
    valid = True
    for i, a in enumerate(s_final):
        for b in s_final[i:]:
            if (mask_final >> (a + b)) & 1:
                valid = False
                break
            mask_final |= (1 << (a + b))
        if not valid:
            break

    if not valid:
        # Fallback to sequential greedy
        s_final = [1, 2, 5, 11]
        mask_final = init_mask(s_final)
        for x in range(12, 285):
            if can_add_mask(s_final, mask_final, x):
                s_final, mask_final = add_element(s_final, mask_final, x)

    # Ensure prescribed elements are present
    for p in prescribed:
        if p not in s_final:
            s_final.append(p)
            mask_final = init_mask(s_final)

    s_final = sorted(set(s_final))
    return s_final

# EVOLVE-BLOCK-END
