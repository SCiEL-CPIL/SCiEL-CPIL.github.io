"""Combined program: graph-based structural optimization + fast maximality + disciplined local search
for small inclusion-maximal Sidon sets in [1,256]."""
import time
import random

N = 256
# Evaluator kills at 60s wall-clock. Use a tight internal deadline for safety.
START_TIME = time.time()
DEADLINE = START_TIME + 55  # 5s safety margin


# EVOLVE-BLOCK-START

def _build_sum_set(selected):
    """Build the set of all unordered pair sums (including diagonal) for a Sidon set."""
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        for j in range(i, n):
            sums.add(a + selected[j])
    return sums


def _is_sidon(selected):
    """Check if selected is a Sidon set (all unordered pair sums distinct)."""
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        for j in range(i, n):
            s = a + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True


def _fast_maximality(selected, sums, precomputed_blocked=None):
    """Check if selected is inclusion-maximal.
    If precomputed_blocked is provided (a set of x in [1,N] that are already
    blocked by the current sums via 2x or x+a), we only need to probe the
    remaining candidates for the x+a condition.
    """
    sset = set(selected)
    if precomputed_blocked is not None:
        # Only check elements NOT already blocked
        candidates = [x for x in range(1, N + 1) if x not in sset and x not in precomputed_blocked]
    else:
        candidates = [x for x in range(1, N + 1) if x not in sset]

    for x in candidates:
        # x is addable iff 2x not in sums AND for all a in selected, x+a not in sums
        # (and the new sums don't collide with each other, but since x is new,
        #  2x and x+a are all distinct from each other iff 2x != x+a for all a,
        #  which means x != a for all a, which is true since x not in sset)
        if 2 * x in sums:
            continue
        blocked = False
        for a in selected:
            if (x + a) in sums:
                blocked = True
                break
        if not blocked:
            return False
    return True


def _compute_blocked(selected, sums):
    """Compute the set of missing elements that are blocked by the current sums.
    An element x is blocked if 2x in sums OR exists a in selected with x+a in sums.
    """
    sset = set(selected)
    blocked = set()
    for x in range(1, N + 1):
        if x in sset:
            continue
        if 2 * x in sums:
            blocked.add(x)
        else:
            for a in selected:
                if (x + a) in sums:
                    blocked.add(x)
                    break
    return blocked


def _greedy_construct(order):
    """Greedy: process elements in given order, add if keeps Sidon property."""
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected, sums


def _greedy_randomized(rng):
    """Randomized greedy with diverse ordering strategies."""
    base = list(range(1, N + 1))
    strategy = rng.random()
    if strategy < 0.2:
        order = sorted(base)
    elif strategy < 0.4:
        order = sorted(base, reverse=True)
    elif strategy < 0.6:
        order = base[:]
        rng.shuffle(order)
    elif strategy < 0.8:
        mod = rng.choice([2, 3, 4, 5, 7, 8, 16])
        order = sorted(base, key=lambda x: x % mod)
    else:
        # Random with bias toward extremes
        order = base[:]
        rng.shuffle(order)
        smalls = [x for x in order if x <= N // 4]
        larges = [x for x in order if x > 3 * N // 4]
        rest = [x for x in order if N // 4 < x <= 3 * N // 4]
        rng.shuffle(smalls)
        rng.shuffle(larges)
        order = smalls[:rng.randint(2, 6)] + larges[:rng.randint(2, 6)] + rest
    return _greedy_construct(order)


def _local_search_remove(selected, sums, rng, deadline):
    """Iteratively try to remove elements while maintaining maximality.
    Uses fast maximality check with precomputed blocked set.
    """
    cur_sel = list(selected)
    cur_sums = set(sums)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Precompute blocked set for fast maximality check
        blocked = _compute_blocked(cur_sel, cur_sums)
        # Shuffle removal order
        indices = list(range(len(cur_sel)))
        rng.shuffle(indices)
        for i in indices:
            if time.time() > deadline:
                break
            new_sel = cur_sel[:i] + cur_sel[i + 1:]
            new_sums = set()
            n = len(new_sel)
            for a_idx in range(n):
                a = new_sel[a_idx]
                for b_idx in range(a_idx, n):
                    new_sums.add(a + new_sel[b_idx])
            # Fast maximality check: only probe elements not blocked by new_sums
            # But we need to recompute blocked for new_sums... 
            # Actually, let's just use the standard check but with early exit
            if _fast_maximality(new_sel, new_sums):
                cur_sel = new_sel
                cur_sums = new_sums
                improved = True
                break  # Restart after a removal
    return cur_sel, cur_sums


def _swap_search(selected, sums, rng, deadline):
    """Try swaps: remove one element, add another, then try removals."""
    cur_sel = list(selected)
    cur_sums = set(sums)
    best_sel = list(cur_sel)
    best_sums = set(cur_sums)

    for _ in range(30):
        if time.time() > deadline:
            break
        i = rng.randrange(len(cur_sel))
        # Remove element i
        new_sel = cur_sel[:i] + cur_sel[i + 1:]
        new_sums = set()
        n = len(new_sel)
        for a_idx in range(n):
            a = new_sel[a_idx]
            for b_idx in range(a_idx, n):
                new_sums.add(a + new_sel[b_idx])

        # Try adding a random element
        candidates = [x for x in range(1, N + 1) if x not in new_sel]
        if not candidates:
            continue
        # Bias toward elements that might block more (middle of range)
        if rng.random() < 0.5:
            cand = rng.choice(candidates)
        else:
            # Prefer elements in the middle third
            mid_candidates = [x for x in candidates if N // 4 <= x <= 3 * N // 4]
            if mid_candidates:
                cand = rng.choice(mid_candidates)
            else:
                cand = rng.choice(candidates)
        added = [2 * cand] + [cand + a for a in new_sel]
        if len(set(added)) != len(added) or new_sums.intersection(added):
            continue

        new_sel2 = new_sel + [cand]
        new_sums2 = new_sums.union(added)

        if _fast_maximality(new_sel2, new_sums2):
            # Try removals from this new state
            ls_sel, ls_sums = _local_search_remove(new_sel2, new_sums2, rng, min(deadline, time.time() + 1.0))
            if len(ls_sel) < len(best_sel):
                best_sel = ls_sel
                best_sums = ls_sums
                cur_sel = ls_sel
                cur_sums = ls_sums

    return best_sel, best_sums


def _structured_construction(rng):
    """Try structured constructions that might yield smaller maximal sets."""
    results = []

    # Strategy 1: Arithmetic progressions with perturbations
    for start in [1, 2, 3]:
        for step in [4, 5, 6, 7, 8]:
            base = [start + k * step for k in range(N // step + 1) if start + k * step <= N]
            if len(base) >= 3:
                extra = [x for x in range(1, N + 1) if x not in base]
                rng.shuffle(extra)
                order = base + extra[:rng.randint(0, 3)]
                sel, sums = _greedy_construct(order)
                if _fast_maximality(sel, sums):
                    results.append((sel, sums))

    # Strategy 2: Cluster-based - pick a few well-spread elements
    for _ in range(5):
        anchors = sorted(rng.sample(range(1, N + 1), rng.randint(3, 5)))
        order = anchors[:]
        remaining = [x for x in range(1, N + 1) if x not in set(anchors)]
        rng.shuffle(remaining)
        order += remaining
        sel, sums = _greedy_construct(order)
        if _fast_maximality(sel, sums):
            results.append((sel, sums))

    # Strategy 3: Power-of-2 based with modifications
    powers = [2 ** k for k in range(1, 9) if 2 ** k <= N]
    for _ in range(3):
        modified = powers[:]
        for _ in range(rng.randint(0, 3)):
            idx = rng.randrange(len(modified))
            modified[idx] = rng.randint(1, N)
        for _ in range(rng.randint(0, 2)):
            modified.append(rng.randint(1, N))
        modified = list(set(modified))
        rng.shuffle(modified)
        sel, sums = _greedy_construct(modified)
        if _fast_maximality(sel, sums):
            results.append((sel, sums))

    return results


def construct():
    rng = random.Random()
    best_size = N + 1
    best_result = None
    best_sums = None
    deadline = time.time() + 50  # 10s margin from 60s wall-clock

    # Phase 1: Structured constructions
    if time.time() < deadline - 15:
        structured = _structured_construction(rng)
        for sel, sums in structured:
            if len(sel) < best_size:
                best_size = len(sel)
                best_result = list(sel)
                best_sums = set(sums)

    # Phase 2: Randomized greedy with local search
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        selected, sums = _greedy_randomized(rng)

        if len(selected) >= best_size:
            continue

        if _fast_maximality(selected, sums):
            best_size = len(selected)
            best_result = list(selected)
            best_sums = set(sums)

            # Apply local search
            if time.time() < deadline - 3:
                ls_sel, ls_sums = _local_search_remove(selected, sums, rng, min(deadline - 2, time.time() + 2.0))
                if len(ls_sel) < best_size:
                    best_size = len(ls_sel)
                    best_result = ls_sel
                    best_sums = ls_sums

                # Try swap search
                if time.time() < deadline - 2:
                    sw_sel, sw_sums = _swap_search(selected, sums, rng, min(deadline - 1, time.time() + 1.5))
                    if len(sw_sel) < best_size:
                        best_size = len(sw_sel)
                        best_result = sw_sel
                        best_sums = sw_sums

            if best_size <= 12:
                break

    # Phase 3: Final local search on best result
    if best_result is not None and time.time() < deadline - 1:
        final_sel, final_sums = _local_search_remove(best_result, best_sums, rng, deadline - 0.5)
        if len(final_sel) < best_size:
            best_size = len(final_sel)
            best_result = final_sel
            best_sums = final_sums

    # Phase 4: Additional swap search on best
    if best_result is not None and time.time() < deadline - 0.5:
        sw_sel, sw_sums = _swap_search(best_result, best_sums, rng, deadline - 0.2)
        if len(sw_sel) < best_size:
            best_size = len(sw_sel)
            best_result = sw_sel
            best_sums = sw_sums

    if best_result is None:
        # Fallback: deterministic ascending greedy
        selected, sums = _greedy_construct(list(range(1, N + 1)))
        best_result = selected

    return best_result

# EVOLVE-BLOCK-END
