"""Combined program: order-greedy + max-blocking saturation + ILS + backtracking polish."""
import time
import random

N = 256
MAX_SUM = 2 * N


# ---------- Core predicates ----------

def _is_sidon(selected):
    sums = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True


def _get_sums(selected):
    sums = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            sums.add(a + b)
    return sums


def _can_add(sums, selected, x):
    # diagonal
    if 2 * x in sums:
        return False
    for a in selected:
        if (a + x) in sums:
            return False
    return True


def _is_maximal_fast(selected, sums):
    selected_set = set(selected)
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        if _can_add(sums, selected, x):
            return False
    return True


# ---------- Saturation: order-greedy (old) ----------

def _greedy_construct(order):
    selected = []
    sums = set()
    for x in order:
        if _can_add(sums, selected, x):
            selected.append(x)
            for a in selected[:-1]:
                sums.add(a + x)
            sums.add(2 * x)
    return selected


# ---------- Saturation: max-blocking greedy (inspiration) ----------

def _max_blocking_saturate(seed_elements=None, rng=None, time_budget=None, deadline=None):
    """
    Saturate a Sidon set by repeatedly adding the element that blocks the most
    remaining candidates. Returns a maximal Sidon set.
    """
    if deadline is None:
        deadline = time.time() + 27.0
    if time_budget is None:
        time_budget = max(0.0, deadline - time.time())

    selected = list(seed_elements) if seed_elements else []
    selected_set = set(selected)
    sums = _get_sums(selected)

    # Candidates that can still be added
    unblocked = [x for x in range(1, N + 1) if x not in selected_set]

    # Precompute current sum set for fast membership
    while unblocked:
        if time.time() >= deadline:
            break

        best_x = None
        best_score = -1
        best_tie = None

        # We want to pick x that maximizes number of z in unblocked that become blocked
        # after adding x (including x itself becoming selected, but we count others).
        # To keep this fast, we approximate blocking by counting how many z in unblocked
        # would violate Sidon if x were added.
        #
        # For a candidate x:
        #   new_sums = sums ∪ {2x} ∪ {x + a for a in selected}
        #   z is blocked if (2z in new_sums) or (∃ a in selected ∪ {x}: z + a in new_sums)
        #
        # We only need a *relative* score to pick the best x; ties can be randomized.
        #
        # For speed, we avoid materializing new_sums for every x. Instead:
        #   - If 2x in sums or (x+a) in sums for some a, x is not compatible.
        #   - For compatible x, we estimate blocked count by checking z in unblocked
        #     against a small set of new sums: {2x} ∪ {x+a for a in selected}.
        #
        # This is exact for the *new* sums; old sums are already in `sums`.
        #
        # We cap the per-candidate work by early-exiting the z-loop if score cannot
        # exceed current best.

        # Build the set of new sums for a compatible x quickly:
        #   new_sums_small = {2x} ∪ {x + a for a in selected}
        # We also need to know if x is compatible: 2x not in sums and x+a not in sums.

        # For each x in unblocked:
        for x in unblocked:
            # compatibility
            if 2 * x in sums:
                continue
            ok = True
            for a in selected:
                if (x + a) in sums:
                    ok = False
                    break
            if not ok:
                continue

            # new sums
            new_sums_small = {2 * x}
            for a in selected:
                new_sums_small.add(x + a)

            # Count blocked z in unblocked (z != x)
            score = 0
            # We can early-exit if even counting all remaining z can't beat best_score
            # (we don't know exactly, but we can bound by len(unblocked)-1)
            # For simplicity, we just compute; unblocked is <= 256, selected is small-ish.
            for z in unblocked:
                if z == x:
                    continue
                # z is blocked if 2z in (sums ∪ new_sums_small) OR ∃ y in (selected ∪ {x}) with z+y in (sums ∪ new_sums_small)
                blocked = False
                if (2 * z) in sums or (2 * z) in new_sums_small:
                    blocked = True
                else:
                    # check z + a for a in selected
                    for a in selected:
                        s = z + a
                        if s in sums or s in new_sums_small:
                            blocked = True
                            break
                    if not blocked:
                        # check z + x
                        s = z + x
                        if s in sums or s in new_sums_small:
                            blocked = True
                if blocked:
                    score += 1

            # random tie-break
            if score > best_score or (score == best_score and (best_tie is None or (rng and rng.random() < 0.5))):
                best_score = score
                best_x = x
                best_tie = True

        if best_x is None:
            # No compatible element left; set is maximal
            break

        # Add best_x
        selected.append(best_x)
        selected_set.add(best_x)
        unblocked.remove(best_x)

        # Update sums
        sums.add(2 * best_x)
        for a in selected[:-1]:
            sums.add(a + best_x)

    # If we stopped due to time, ensure maximality by a final linear scan
    if time.time() < deadline:
        # Rebuild sums just in case
        sums = _get_sums(selected)
        selected_set = set(selected)
        for x in range(1, N + 1):
            if x in selected_set:
                continue
            if _can_add(sums, selected, x):
                selected.append(x)
                selected_set.add(x)
                sums.add(2 * x)
                for a in selected[:-1]:
                    sums.add(a + x)

    return selected


# ---------- Local search ----------

def _local_search_minimize(selected, deadline):
    """
    Local search: try removing one element and re-saturate (max-blocking).
    Keep best improvement.
    """
    selected = list(selected)
    best = list(selected)
    best_size = len(selected)

    # We'll do a few passes; each pass tries all removals.
    while time.time() < deadline:
        improved = False
        n = len(selected)
        for i in range(n):
            if time.time() >= deadline:
                break
            trial = selected[:i] + selected[i + 1:]
            if not trial:
                continue
            # trial is Sidon if selected was Sidon
            # Saturate trial
            saturated = _max_blocking_saturate(seed_elements=trial, rng=None, deadline=deadline)
            if len(saturated) < best_size:
                best = saturated
                best_size = len(saturated)
                selected = list(saturated)
                improved = True
                break
        if not improved:
            break

    return best


def _iterated_local_search(rng, deadline, initial_solution):
    best = list(initial_solution)
    best_size = len(best)

    while time.time() < deadline:
        current = list(best)
        if not current:
            break

        # Perturb: remove 1-2 random elements
        n_remove = rng.randint(1, min(2, len(current)))
        indices = rng.sample(range(len(current)), n_remove)
        for idx in sorted(indices, reverse=True):
            current.pop(idx)

        # Saturate
        current = _max_blocking_saturate(seed_elements=current, rng=rng, deadline=deadline)

        # Local search with a short budget
        step_deadline = min(deadline, time.time() + 0.4)
        current = _local_search_minimize(current, step_deadline)

        if len(current) < best_size:
            best = current
            best_size = len(current)

    return best


# ---------- Backtracking for exact size k ----------

def _backtracking_search(k, deadline):
    """
    Try to find a maximal Sidon set of exactly size k via backtracking.
    Returns the set if found, else None.
    """
    def backtrack(current, current_sums, start_idx):
        if len(current) == k:
            if _is_maximal_fast(current, current_sums):
                return list(current)
            return None

        if time.time() >= deadline:
            return None

        remaining_needed = k - len(current)
        if N - start_idx + 1 < remaining_needed:
            return None

        for x in range(start_idx, N + 1):
            if not _can_add(current_sums, current, x):
                continue

            new_current = current + [x]
            new_sums = set(current_sums)
            for a in current:
                new_sums.add(a + x)
            new_sums.add(2 * x)

            result = backtrack(new_current, new_sums, x + 1)
            if result is not None:
                return result

        return None

    return backtrack([], set(), 1)


# ---------- Main ----------

def construct():
    deadline = time.time() + 27.0  # safety margin before 30s kill
    rng = random.Random(12345)

    best_maximal = None
    best_size = float('inf')

    # 1. Initial deterministic max-blocking saturation (no random tie-break)
    cand = _max_blocking_saturate(seed_elements=None, rng=None, deadline=deadline)
    if len(cand) < best_size:
        best_size = len(cand)
        best_maximal = list(cand)

    # 2. A few randomized max-blocking saturations (tie-break random)
    for _ in range(12):
        if time.time() >= deadline - 10:
            break
        cand = _max_blocking_saturate(seed_elements=None, rng=rng, deadline=deadline)
        if len(cand) < best_size:
            best_size = len(cand)
            best_maximal = list(cand)

    # 3. Local search on the best
    if best_maximal is not None:
        local_best = _local_search_minimize(best_maximal, min(deadline, time.time() + 5))
        if len(local_best) < best_size:
            best_size = len(local_best)
            best_maximal = local_best

    # 4. ILS
    if best_maximal is not None and time.time() < deadline - 8:
        ils_best = _iterated_local_search(rng, deadline - 8, best_maximal)
        if len(ils_best) < best_size:
            best_size = len(ils_best)
            best_maximal = ils_best

    # 5. Backtracking for small k (polish)
    min_k_to_try = max(5, best_size - 12)
    max_k_to_try = best_size - 1
    for k in range(max_k_to_try, min_k_to_try - 1, -1):
        if time.time() >= deadline - 3:
            break
        result = _backtracking_search(k, deadline - 3)
        if result is not None:
            if len(result) < best_size:
                best_size = len(result)
                best_maximal = result
            break

    # 6. Use remaining time for more randomized saturations + local search
    while time.time() < deadline - 1:
        cand = _max_blocking_saturate(seed_elements=None, rng=rng, deadline=deadline)
        if len(cand) < best_size:
            best_size = len(cand)
            best_maximal = list(cand)

        local_best = _local_search_minimize(cand, min(deadline - 1, time.time() + 1.5))
        if len(local_best) < best_size:
            best_size = len(local_best)
            best_maximal = local_best

    # Final validation
    if best_maximal is not None:
        if _is_sidon(best_maximal) and _is_maximal_fast(best_maximal, _get_sums(best_maximal)):
            return sorted(best_maximal)

    # Fallback: deterministic max-blocking saturation
    return sorted(_max_blocking_saturate(seed_elements=None, rng=None, deadline=deadline))
