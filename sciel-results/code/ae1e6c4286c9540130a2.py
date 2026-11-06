import random
import time

N = 256
MAX_SUM = 2 * N  # 512


def _saturate(sel, sum_mask, deadline):
    """Add all addable elements in ascending order until none remain.
    Returns (sel, sum_mask)."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, N + 1):
            # Check if x is already in sel
            if any(v == x for v in sel):
                continue
            # Check if x can be added
            ok = True
            for a in sel:
                if (sum_mask >> (x + a)) & 1:
                    ok = False
                    break
            if not ok:
                continue
            if (sum_mask >> (2 * x)) & 1:
                continue
            # Add x
            sel.append(x)
            for a in sel[:-1]:
                sum_mask |= 1 << (x + a)
            sum_mask |= 1 << (2 * x)
            changed = True
    return sel, sum_mask


def _is_valid_sidon(sel):
    """Check if sel is a Sidon set."""
    sums = set()
    for i in range(len(sel)):
        a = sel[i]
        for j in range(i, len(sel)):
            s = a + sel[j]
            if s in sums:
                return False
            sums.add(s)
    return True


def _is_maximal(sel):
    """Check if sel is inclusion-maximal."""
    if not _is_valid_sidon(sel):
        return False
    sums = set()
    for i in range(len(sel)):
        a = sel[i]
        for j in range(i, len(sel)):
            sums.add(a + sel[j])
    sel_set = set(sel)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        ok = True
        for a in sel:
            if x + a in sums:
                ok = False
                break
        if not ok:
            continue
        if 2 * x in sums:
            continue
        # x can be added, so not maximal
        return False
    return True


def _greedy_min_block(deadline, seed=0):
    """Greedy: pick element that blocks the FEWEST remaining (to keep set small)."""
    rng = random.Random(seed)
    sel = []
    sum_mask = 0
    remaining = set(range(1, N + 1))

    while remaining and time.time() < deadline:
        best_x = None
        best_block = float('inf')

        # Sample candidates
        if len(remaining) > 60:
            candidates = rng.sample(list(remaining), 60)
        else:
            candidates = list(remaining)

        for x in candidates:
            # Check if x can be added
            ok = True
            for a in sel:
                if (sum_mask >> (x + a)) & 1:
                    ok = False
                    break
            if not ok:
                continue
            if (sum_mask >> (2 * x)) & 1:
                continue

            # Count how many remaining elements become un-addable
            block = 0
            for y in remaining:
                if y == x:
                    continue
                # y is blocked if after adding x, y cannot be added
                # New sums: x+a for a in sel, and 2x
                blocked = False
                # Check 2y
                if (sum_mask >> (2 * y)) & 1:
                    blocked = True
                elif (sum_mask >> (x + y)) & 1:
                    blocked = True
                else:
                    for a in sel:
                        if (sum_mask >> (y + a)) & 1:
                            blocked = True
                            break
                if blocked:
                    block += 1

            if block < best_block:
                best_block = block
                best_x = x

        if best_x is None:
            break

        sel.append(best_x)
        for a in sel[:-1]:
            sum_mask |= 1 << (best_x + a)
        sum_mask |= 1 << (2 * best_x)
        remaining.discard(best_x)

    # Saturate
    sel, sum_mask = _saturate(sel, sum_mask, deadline)
    return sel


def _greedy_random(deadline, seed=0):
    """Randomized greedy: pick from top-k candidates randomly."""
    rng = random.Random(seed)
    sel = []
    sum_mask = 0
    remaining = set(range(1, N + 1))

    while remaining and time.time() < deadline:
        # Compute blocking counts for all addable candidates
        scored = []
        for x in remaining:
            ok = True
            for a in sel:
                if (sum_mask >> (x + a)) & 1:
                    ok = False
                    break
            if not ok:
                continue
            if (sum_mask >> (2 * x)) & 1:
                continue

            block = 0
            for y in remaining:
                if y == x:
                    continue
                blocked = False
                if (sum_mask >> (2 * y)) & 1:
                    blocked = True
                elif (sum_mask >> (x + y)) & 1:
                    blocked = True
                else:
                    for a in sel:
                        if (sum_mask >> (y + a)) & 1:
                            blocked = True
                            break
                if blocked:
                    block += 1
            scored.append((block, x))

        if not scored:
            break

        # Sort by block count (fewest first) and pick from top-k
        scored.sort()
        k = min(5, len(scored))
        best_x = rng.choice([x for _, x in scored[:k]])

        sel.append(best_x)
        for a in sel[:-1]:
            sum_mask |= 1 << (best_x + a)
        sum_mask |= 1 << (2 * best_x)
        remaining.discard(best_x)

    # Saturate
    sel, sum_mask = _saturate(sel, sum_mask, deadline)
    return sel


def _local_search_reduce(sel, deadline, max_iters=30):
    """Try removing elements and re-saturating to reduce size."""
    best = list(sel)
    best_size = len(best)
    sel_set = set(best)

    for it in range(max_iters):
        if time.time() > deadline:
            break
        improved = False

        # Try removing each element
        for i in range(len(best)):
            if time.time() > deadline:
                break
            candidate = best[:i] + best[i + 1:]
            candidate_set = set(candidate)

            # Check if candidate is still Sidon
            sums = set()
            valid = True
            for a_idx in range(len(candidate)):
                a = candidate[a_idx]
                for b_idx in range(a_idx, len(candidate)):
                    s = a + candidate[b_idx]
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
                if not valid:
                    break
            if not valid:
                continue

            # Saturate candidate minimally
            cand_sel = list(candidate)
            cand_sum_mask = 0
            for a in cand_sel:
                for b in cand_sel:
                    if a <= b:
                        cand_sum_mask |= 1 << (a + b)
                cand_sum_mask |= 1 << (2 * a)

            changed = True
            while changed and time.time() < deadline:
                changed = False
                for x in range(1, N + 1):
                    if x in candidate_set:
                        continue
                    ok = True
                    for a in cand_sel:
                        if (cand_sum_mask >> (x + a)) & 1:
                            ok = False
                            break
                    if not ok:
                        continue
                    if (cand_sum_mask >> (2 * x)) & 1:
                        continue
                    cand_sel.append(x)
                    candidate_set.add(x)
                    for a in cand_sel[:-1]:
                        cand_sum_mask |= 1 << (x + a)
                    cand_sum_mask |= 1 << (2 * x)
                    changed = True

            if len(cand_sel) < best_size:
                best = cand_sel
                best_size = len(best)
                sel_set = set(best)
                improved = True
                break

        if not improved:
            break

    return best


def construct():
    start = time.time()
    deadline = start + 55  # safety margin before 60s kill

    best_set = None
    best_size = float('inf')

    # Phase 1: Multiple greedy constructions with different strategies
    for trial in range(300):
        if time.time() > deadline:
            break

        if trial % 2 == 0:
            sel = _greedy_min_block(deadline, seed=trial)
        else:
            sel = _greedy_random(deadline, seed=trial)

        if len(sel) < best_size:
            best_size = len(sel)
            best_set = list(sel)
            # Local search on this
            improved = _local_search_reduce(sel, deadline, max_iters=15)
            if len(improved) < best_size:
                best_size = len(improved)
                best_set = improved

    # Phase 2: More local search on the best
    if best_set is not None and time.time() < deadline - 5:
        improved = _local_search_reduce(best_set, deadline, max_iters=50)
        if len(improved) < best_size:
            best_set = improved

    # Fallback: simple greedy
    if best_set is None:
        sel = []
        sums = set()
        for x in range(1, N + 1):
            ok = True
            for a in sel:
                if x + a in sums:
                    ok = False
                    break
            if not ok:
                continue
            if 2 * x in sums:
                continue
            sel.append(x)
            for a in sel[:-1]:
                sums.add(x + a)
            sums.add(2 * x)
        best_set = sel

    return best_set
