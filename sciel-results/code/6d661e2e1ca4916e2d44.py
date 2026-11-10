import time
import random

def _get_sums(arr):
    """Return the set of all non-empty subset sums of arr."""
    sums = set()
    for x in arr:
        new_sums = {s + x for s in sums}
        new_sums.add(x)
        sums |= new_sums
    return sums

def _is_valid(A):
    """Check if set A is valid: all non-empty subset sums are distinct and no two divide each other."""
    if not A:
        return True
    sums = _get_sums(A)
    # Check divisibility: for each s, no multiple 2s, 3s, ... in sums
    max_s = max(sums)
    for s in sums:
        for k in range(2, max_s // s + 1):
            if s * k in sums:
                return False
    return True

def _can_add(arr, x, current_sums):
    """Check if x can be added to arr (with current_sums precomputed) while preserving validity."""
    if x in arr:
        return False
    # New sums: x and s+x for each existing sum s
    new_sums = {x}
    for s in current_sums:
        new_sums.add(s + x)
    # Uniqueness: no overlap between old and new sums
    if current_sums & new_sums:
        return False
    all_sums = current_sums | new_sums
    # Divisibility check: only need to check pairs involving new sums
    # (old sums were already valid)
    max_s = max(all_sums)
    # Check new sums against all sums for divisibility
    for ns in new_sums:
        for k in range(2, max_s // ns + 1):
            if ns * k in all_sums:
                return False
    # Also check: does any old sum divide a new sum?
    # (covered above if we check ns*k, but also need old*k == new)
    # Actually the above checks: for each new sum ns, is ns*k in all_sums?
    # This catches: old sum = ns*k for some k>=2, i.e., old sum is a multiple of new sum
    # But we also need: new sum = old*k for some k>=2
    for os in current_sums:
        for k in range(2, max_s // os + 1):
            if os * k in new_sums:
                return False
    return True

def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # 5s safety margin before 90s kill

    rng = random.Random(42)
    best = []
    best_size = 0

    # Pool: larger values are easier (fewer multiples fit in 1..256 range)
    # But we also want some smaller values for diversity
    pool = list(range(64, 257))

    # --- Phase 1: Incremental greedy with random restarts ---
    # Use larger pool, try descending order (greedy prefers large values)
    for restart in range(500):
        if time.time() > deadline:
            break

        order = pool[:]
        rng.shuffle(order)
        # Bias: sort descending to try large values first (they have fewer multiples)
        order.sort(reverse=True)

        cur = []
        cur_sums = set()

        for x in order:
            if time.time() > deadline:
                break
            if _can_add(cur, x, cur_sums):
                cur.append(x)
                # Update sums incrementally
                for s in cur_sums:
                    cur_sums.add(s + x)
                cur_sums.add(x)

        if len(cur) > best_size:
            best_size = len(cur)
            best = cur[:]

    # --- Phase 2: Targeted addition on best set ---
    if time.time() < deadline:
        cur = list(best)
        cur_sums = _get_sums(cur)
        used = set(cur)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in pool:
                if x in used:
                    continue
                if _can_add(cur, x, cur_sums):
                    cur.append(x)
                    used.add(x)
                    for s in cur_sums:
                        cur_sums.add(s + x)
                    cur_sums.add(x)
                    improved = True
                    if len(cur) > best_size:
                        best_size = len(cur)
                        best = cur[:]
                    break  # Restart loop with updated set

    # --- Phase 3: Randomized hill-climbing (add/remove) ---
    if time.time() < deadline:
        cur = list(best)
        cur_sums = _get_sums(cur)

        for step in range(50000):
            if time.time() > deadline:
                break

            used = set(cur)
            unused = [x for x in pool if x not in used]

            if unused:
                # Try to add a random unused candidate
                x = rng.choice(unused)
                if _can_add(cur, x, cur_sums):
                    cur.append(x)
                    for s in cur_sums:
                        cur_sums.add(s + x)
                    cur_sums.add(x)
                    if len(cur) > best_size:
                        best_size = len(cur)
                        best = cur[:]
            elif len(cur) > 2 and rng.random() < 0.3:
                # Remove a random element to escape local minimum
                idx = rng.randrange(len(cur))
                removed = cur.pop(idx)
                cur_sums = _get_sums(cur)  # Recompute (expensive but correct)
                # Try to add something back
                unused2 = [x for x in pool if x not in set(cur)]
                if unused2:
                    x2 = rng.choice(unused2)
                    if _can_add(cur, x2, cur_sums):
                        cur.append(x2)
                        for s in cur_sums:
                            cur_sums.add(s + x2)
                        cur_sums.add(x2)
                        if len(cur) > best_size:
                            best_size = len(cur)
                            best = cur[:]

    # Final safety: ensure distinct, in range, plain Python ints
    best = [int(x) for x in dict.fromkeys(best)]
    best = [x for x in best if 1 <= x <= 256]

    return best
