"""Sidon set construction with bitmask conflict checking."""
import time
import random


def construct():
    start_time = time.time()
    DEADLINE = 135.0  # safety margin before 150s limit

    prescribed = [1, 2, 5, 11]
    MAX_VAL = 284

    def elapsed_ok():
        return time.time() < DEADLINE

    def compute_mask(s):
        m = 0
        for i, a in enumerate(s):
            for b in s[i:]:
                m |= 1 << (a + b)
        return m

    def conflict_mask(x, s, mask):
        """Bitmask of sums that would be created by adding x to s."""
        cm = 1 << (2 * x)
        for y in s:
            cm |= 1 << (x + y)
        return cm

    def can_add(x, s, mask):
        if (1 << x) in mask:  # x already in set (we'll track separately)
            return False
        cm = conflict_mask(x, s, mask)
        return (cm & mask) == 0

    def greedy_random(s, mask, rng, order):
        """Add elements from order (list of candidates) greedily."""
        s = list(s)
        mask = mask
        used = set(s)
        for x in order:
            if not elapsed_ok():
                break
            if x in used:
                continue
            cm = conflict_mask(x, s, mask)
            if (cm & mask) == 0:
                s.append(x)
                mask |= cm
                used.add(x)
        return s, mask

    def greedy_min_conflict(s, mask, rng, candidates):
        """At each step, pick the candidate with fewest conflicts."""
        s = list(s)
        mask = mask
        used = set(s)
        remaining = [c for c in candidates if c not in used]
        while remaining and elapsed_ok():
            # Score each candidate by number of conflicts
            best_x = None
            best_score = None
            # Sample if too many candidates
            if len(remaining) > 60:
                sample = rng.sample(remaining, 60)
            else:
                sample = remaining
            for x in sample:
                cm = conflict_mask(x, s, mask)
                score = bin(cm & mask).count('1')
                if score == 0:
                    # No conflict - great, but we want to pick one that's "safe"
                    # Actually if score==0 it's valid; prefer lower values to leave room?
                    # Let's just pick it
                    best_x = x
                    best_score = 0
                    break
                if best_score is None or score < best_score:
                    best_x = x
                    best_score = score
            if best_x is None or best_score > 0:
                # No valid candidate in sample; try full remaining if small
                if len(remaining) <= 60:
                    break
                # Otherwise, try to find any valid
                found = False
                for x in remaining:
                    cm = conflict_mask(x, s, mask)
                    if (cm & mask) == 0:
                        best_x = x
                        found = True
                        break
                if not found:
                    break
            cm = conflict_mask(best_x, s, mask)
            s.append(best_x)
            mask |= cm
            used.add(best_x)
            remaining.remove(best_x)
        return s, mask

    # Initial set
    init_s = [1, 2, 5, 11]
    init_mask = compute_mask(init_s)
    init_used = set(init_s)

    # Strategy 1: Simple forward greedy (baseline)
    s = init_s[:]
    mask = init_mask
    used = init_used.copy()
    for x in range(12, MAX_VAL + 1):
        cm = conflict_mask(x, s, mask)
        if (cm & mask) == 0:
            s.append(x)
            mask |= cm
            used.add(x)
    best_s = s[:]
    best_mask = mask
    best_size = len(s)

    # Strategy 2: Randomized greedy with multiple restarts
    rng = random.Random(12345)
    candidates_pool = list(range(12, MAX_VAL + 1))

    for restart in range(200):
        if not elapsed_ok():
            break
        order = candidates_pool[:]
        rng.shuffle(order)
        s, mask = greedy_random(init_s, init_mask, rng, order)
        if len(s) > best_size:
            best_s = s[:]
            best_mask = mask
            best_size = len(s)

    # Strategy 3: Min-conflict greedy
    for restart in range(100):
        if not elapsed_ok():
            break
        s, mask = greedy_min_conflict(init_s, init_mask, rng, candidates_pool)
        if len(s) > best_size:
            best_s = s[:]
            best_mask = mask
            best_size = len(s)

    # Strategy 4: Local search - swap elements
    # Try to improve by removing one element and adding another
    for _ in range(500):
        if not elapsed_ok():
            break
        s = best_s[:]
        mask = best_mask
        used = set(s)
        removable = [x for x in s if x not in prescribed]
        if not removable:
            break
        x_remove = rng.choice(removable)
        # Remove x_remove
        s.remove(x_remove)
        used.remove(x_remove)
        # Recompute mask
        mask = compute_mask(s)
        # Try to add a random candidate
        candidates = [x for x in range(12, MAX_VAL + 1) if x not in used]
        rng.shuffle(candidates)
        added = False
        for y in candidates:
            cm = conflict_mask(y, s, mask)
            if (cm & mask) == 0:
                s.append(y)
                mask |= cm
                used.add(y)
                if len(s) > best_size:
                    best_s = s[:]
                    best_mask = mask
                    best_size = len(s)
                added = True
                break

    # Strategy 5: Two-phase - build with random, then local search
    for restart in range(50):
        if not elapsed_ok():
            break
        order = candidates_pool[:]
        rng.shuffle(order)
        s, mask = greedy_random(init_s, init_mask, rng, order)
        # Local search on this
        for _ in range(20):
            if not elapsed_ok():
                break
            used = set(s)
            removable = [x for x in s if x not in prescribed]
            if not removable:
                break
            x_remove = rng.choice(removable)
            s.remove(x_remove)
            used.remove(x_remove)
            mask = compute_mask(s)
            candidates = [x for x in range(12, MAX_VAL + 1) if x not in used]
            rng.shuffle(candidates)
            for y in candidates:
                cm = conflict_mask(y, s, mask)
                if (cm & mask) == 0:
                    s.append(y)
                    mask |= cm
                    used.add(y)
                    break
        if len(s) > best_size:
            best_s = s[:]
            best_mask = mask
            best_size = len(s)

    # Final: try to extend best_s further
    s = best_s[:]
    mask = best_mask
    used = set(s)
    for x in range(12, MAX_VAL + 1):
        if x in used:
            continue
        cm = conflict_mask(x, s, mask)
        if (cm & mask) == 0:
            s.append(x)
            mask |= cm
            used.add(x)
    if len(s) > best_size:
        best_s = s[:]
        best_size = len(s)

    # Validate
    s_final = sorted(set(best_s))
    sums = set()
    valid = True
    for i, a in enumerate(s_final):
        for b in s_final[i:]:
            total = a + b
            if total in sums:
                valid = False
                break
            sums.add(total)
        if not valid:
            break

    if not valid:
        # Fallback to simple greedy
        s_final = init_s[:]
        mask = init_mask
        used = init_used.copy()
        for x in range(12, MAX_VAL + 1):
            cm = conflict_mask(x, s_final, mask)
            if (cm & mask) == 0:
                s_final.append(x)
                mask |= cm
                used.add(x)

    # Ensure prescribed elements
    for p in prescribed:
        if p not in s_final:
            s_final.append(p)
    s_final = sorted(set(s_final))

    return s_final
