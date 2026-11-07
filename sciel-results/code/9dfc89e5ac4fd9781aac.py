import time
import random

# EVOLVE-BLOCK-START

DEADLINE_SECONDS = 84.0  # safety margin under 90s wall-clock


def _check_antichain(sums):
    """
    Check if the set of sums is an antichain under divisibility:
    no two distinct elements divide one another.
    """
    if not sums:
        return True
    sorted_sums = sorted(sums)
    max_sum = sorted_sums[-1]
    is_sum = [False] * (max_sum + 1)
    for s in sorted_sums:
        is_sum[s] = True

    for s in sorted_sums:
        multiple = s * 2
        while multiple <= max_sum:
            if is_sum[multiple]:
                return False
            multiple += s
    return True


def _can_add(current_sums, candidate):
    """
    Check if adding 'candidate' to the set (with existing sums 'current_sums')
    keeps the antichain property. Returns the new sum set if valid, else None.
    """
    new_sums = set(current_sums)
    new_sums.add(candidate)
    for s in current_sums:
        new_sums.add(s + candidate)
    if not _check_antichain(new_sums):
        return None
    return new_sums


def _rebuild_sums(elements):
    """Recompute the full nonempty subset-sum set from a list of elements."""
    n = len(elements)
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for j in range(n):
            if mask & (1 << j):
                s += elements[j]
        sums.add(s)
    return sums


def _greedy_build(candidates, target_size, deadline, rng):
    """Greedy: add candidates in given order until valid set reaches target_size."""
    current_set = []
    current_sums = set()
    for c in candidates:
        if time.time() > deadline:
            break
        if len(current_set) >= target_size:
            break
        new_sums = _can_add(current_sums, c)
        if new_sums is not None:
            current_set.append(c)
            current_sums = new_sums
    return current_set, current_sums


def _anneal(current_set, current_sums, all_candidates, deadline, rng,
            t0=4.0, t_min=0.05, cooling=0.995):
    """
    Simulated annealing polish: try insert and swap moves to grow the set.
    Returns the best (set, sums) found.
    """
    best_set = list(current_set)
    best_sums = set(current_sums)
    best_size = len(best_set)
    t = t0
    in_set = set(current_set)

    while time.time() < deadline:
        if t < t_min:
            break
        move = rng.random()
        if move < 0.4 and len(current_set) < 30:
            # Insert a random candidate not in the set
            pool = [c for c in all_candidates if c not in in_set]
            if not pool:
                t *= cooling
                continue
            cand = rng.choice(pool)
            new_sums = _can_add(current_sums, cand)
            if new_sums is not None:
                # Accept (size increases)
                current_set.append(cand)
                current_sums = new_sums
                in_set.add(cand)
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)
                    best_sums = set(current_sums)
            else:
                # Reject: no state change
                pass
        else:
            # Swap: remove one element, try to insert a different one
            if len(current_set) < 2:
                t *= cooling
                continue
            remove_idx = rng.randrange(len(current_set))
            removed = current_set[remove_idx]
            pool = [c for c in all_candidates if c != removed and c not in in_set]
            if not pool:
                t *= cooling
                continue
            cand = rng.choice(pool)
            # Remove 'removed'
            temp_set = current_set[:remove_idx] + current_set[remove_idx + 1:]
            temp_sums = _rebuild_sums(temp_set)
            in_set.discard(removed)
            # Try to add cand
            new_sums = _can_add(temp_sums, cand)
            if new_sums is not None:
                # Accept swap
                temp_set.append(cand)
                current_set = temp_set
                current_sums = new_sums
                in_set.add(cand)
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)
                    best_sums = set(current_sums)
            else:
                # Reject: restore original state
                current_set = temp_set + [removed]
                current_sums = _rebuild_sums(current_set)
                in_set.add(removed)
        t *= cooling

    return best_set, best_sums


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    rng = random.Random(12345)

    all_candidates = list(range(1, 257))
    # Prefer larger numbers to reduce divisibility collisions
    large_pool = list(range(256, 127, -1))
    mid_pool = list(range(128, 63, -1))
    small_pool = list(range(64, 0, -1))

    best_set = []
    best_size = 0
    target_size = 10

    # Phase 1: Greedy construction with multiple orderings
    for restart in range(300):
        if time.time() > deadline:
            break

        # Build candidate ordering: mix large numbers first with some randomness
        candidates = list(large_pool)
        rng.shuffle(candidates)
        # Sprinkle in some mid numbers
        mid_sample = list(mid_pool)
        rng.shuffle(mid_sample)
        candidates = candidates[:128] + mid_sample[:32]

        current_set, current_sums = _greedy_build(
            candidates, target_size, deadline, rng
        )

        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)

        # Phase 2: Annealing polish on the greedy result
        polished_set, polished_sums = _anneal(
            current_set, current_sums, all_candidates, deadline, rng
        )
        if len(polished_set) > best_size:
            best_size = len(polished_set)
            best_set = list(polished_set)

        if best_size >= target_size:
            break

    # Final verification of best_set
    if best_set:
        verify_sums = _rebuild_sums(best_set)
        if not _check_antichain(verify_sums):
            # Fallback to a known small valid set
            best_set = [31, 32, 34]
            best_size = 3

    # Ensure all are Python ints and return a list
    return [int(x) for x in best_set]


# EVOLVE-BLOCK-END
