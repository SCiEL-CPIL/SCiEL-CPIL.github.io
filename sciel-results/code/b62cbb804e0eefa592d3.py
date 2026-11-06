"""
4-AP-free subset of [1,512]: maximize cardinality.

Combined design:
- Fast conflict machinery: precomputed 3-element conflict bitmasks per element
  (O(degree) add/validity checks) from the current program.
- Position-based blocker finder (precomputed d-ranges per AP position) from the
  inspiration, which is cleaner and faster than decoding masks back to elements.
- Diversified seeds: Cantor set, forward greedy, reverse greedy, density-weighted
  greedy, random-order greedy.
- Aggressive local search: add-any -> 1-element swap -> 2-element swap, each
  followed by a greedy refill; plus randomized restarts.
- Explicit wall-clock deadline guard with safety margin.
"""

import time
import random

N = 512

# ---------------------------------------------------------------------------
# Precompute position-based d-ranges (inspiration) for fast blocker finding.
# ---------------------------------------------------------------------------
_pos_last   = [tuple(range(1, (x - 1) // 3 + 1)) for x in range(N + 1)]
_pos_first  = [tuple(range(1, (N - x) // 3 + 1)) for x in range(N + 1)]
_pos_second = [tuple(range(1, min(x - 1, (N - x) // 2) + 1)) for x in range(N + 1)]
_pos_third  = [tuple(range(1, min((x - 1) // 2, N - x) + 1)) for x in range(N + 1)]

# ---------------------------------------------------------------------------
# Precompute conflict bitmasks (current program) for fast add/validity checks.
# ---------------------------------------------------------------------------
_all_4aps = []
_ap_by_element = [[] for _ in range(N + 1)]
for a in range(1, N - 3):
    for d in range(1, (N - a) // 3 + 1):
        e = a + 3 * d
        if e > N:
            break
        ap = (a, a + d, a + 2 * d, e)
        idx = len(_all_4aps)
        _all_4aps.append(ap)
        for elem in ap:
            _ap_by_element[elem].append(idx)

_conflict_bitmasks = [0] * (N + 1)
_conflict_mask_list = [[] for _ in range(N + 1)]
for x in range(1, N + 1):
    for idx in _ap_by_element[x]:
        a, b, c, e = _all_4aps[idx]
        others = tuple(t for t in (a, b, c, e) if t != x)
        mask = (1 << others[0]) | (1 << others[1]) | (1 << others[2])
        _conflict_mask_list[x].append(mask)
        _conflict_bitmasks[x] |= mask


def _can_add(x, bits):
    """Check if x can be added to set represented by bits without creating a 4-AP."""
    if (bits & _conflict_bitmasks[x]) == 0:
        return True
    for mask in _conflict_mask_list[x]:
        if (bits & mask) == mask:
            return False
    return True


def _is_valid(bits):
    """Check if the set is 4-AP-free."""
    for x in range(1, N + 1):
        if not ((bits >> x) & 1):
            continue
        if (bits & _conflict_bitmasks[x]) == 0:
            continue
        for mask in _conflict_mask_list[x]:
            if (bits & mask) == mask:
                return False
    return True


def _find_blockers(x, bits):
    """Find all elements that block x from being added (position-based)."""
    blockers = set()
    for d in _pos_last[x]:
        a, b, c = x - d, x - 2 * d, x - 3 * d
        if ((bits >> a) & 1) and ((bits >> b) & 1) and ((bits >> c) & 1):
            blockers.add(a); blockers.add(b); blockers.add(c)
    for d in _pos_first[x]:
        a, b, c = x + d, x + 2 * d, x + 3 * d
        if ((bits >> a) & 1) and ((bits >> b) & 1) and ((bits >> c) & 1):
            blockers.add(a); blockers.add(b); blockers.add(c)
    for d in _pos_second[x]:
        a, b, c = x - d, x + d, x + 2 * d
        if ((bits >> a) & 1) and ((bits >> b) & 1) and ((bits >> c) & 1):
            blockers.add(a); blockers.add(b); blockers.add(c)
    for d in _pos_third[x]:
        a, b, c = x - 2 * d, x - d, x + d
        if ((bits >> a) & 1) and ((bits >> b) & 1) and ((bits >> c) & 1):
            blockers.add(a); blockers.add(b); blockers.add(c)
    return blockers


def _bits_to_set(bits):
    return [i for i in range(1, N + 1) if (bits >> i) & 1]


def _cantor_bits():
    """Base-3 Cantor set: numbers with no digit 2 in base 3."""
    bits = 0
    for x in range(1, N + 1):
        t = x
        valid = True
        while t > 0:
            if t % 3 == 2:
                valid = False
                break
            t //= 3
        if valid:
            bits |= (1 << x)
    return bits


def _greedy_forward(bits_start=0):
    """Forward greedy: add elements in order 1..N if possible."""
    bits = bits_start
    for x in range(1, N + 1):
        if not ((bits >> x) & 1) and _can_add(x, bits):
            bits |= (1 << x)
    return bits


def _greedy_reverse(bits_start=0):
    """Reverse-order greedy."""
    bits = bits_start
    for x in range(N, 0, -1):
        if not ((bits >> x) & 1) and _can_add(x, bits):
            bits |= (1 << x)
    return bits


def _greedy_density(bits_start=0):
    """Greedy that prioritizes elements with fewer conflicts."""
    scores = [0] * (N + 1)
    for x in range(1, N + 1):
        scores[x] = len(_pos_last[x]) + len(_pos_first[x]) + len(_pos_second[x]) + len(_pos_third[x])
    order = sorted(range(1, N + 1), key=lambda x: (scores[x], x))
    bits = bits_start
    for x in order:
        if not ((bits >> x) & 1) and _can_add(x, bits):
            bits |= (1 << x)
    return bits


def _greedy_random(bits_start=0, rng=None):
    """Random-order greedy."""
    if rng is None:
        rng = random.Random()
    order = list(range(1, N + 1))
    rng.shuffle(order)
    bits = bits_start
    for x in order:
        if not ((bits >> x) & 1) and _can_add(x, bits):
            bits |= (1 << x)
    return bits


def _greedy_refill(bits, deadline):
    """Add any element that can be added (one sweep, forward order)."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, N + 1):
            if time.time() > deadline:
                return bits
            if not ((bits >> x) & 1) and _can_add(x, bits):
                bits |= (1 << x)
                changed = True
    return bits


def _local_search(bits, deadline, max_passes=20):
    """
    Local search: add-any, then 1-swap, then 2-swap, each followed by greedy refill.
    Uses bitmask conflict checks for speed and position-based blocker finding.
    """
    current = bits
    current_size = bin(current).count('1')

    for _pass in range(max_passes):
        if time.time() > deadline - 2:
            break
        improved = False

        # Pass 1: add any available element
        for x in range(1, N + 1):
            if time.time() > deadline - 2:
                return current
            if not ((current >> x) & 1) and _can_add(x, current):
                current |= (1 << x)
                current_size += 1
                improved = True

        # Pass 2: swaps
        if not improved:
            for x in range(1, N + 1):
                if time.time() > deadline - 2:
                    return current
                if (current >> x) & 1:
                    continue

                blockers = _find_blockers(x, current)
                if not blockers:
                    if _can_add(x, current):
                        current |= (1 << x)
                        current_size += 1
                        improved = True
                    continue

                candidates = list(blockers)
                found_swap = False

                # 1-element removals
                for rem in candidates:
                    if time.time() > deadline - 2:
                        return current
                    trial = current & ~(1 << rem)
                    if _can_add(x, trial):
                        trial |= (1 << x)
                        trial = _greedy_refill(trial, deadline - 2)
                        trial_size = bin(trial).count('1')
                        if trial_size > current_size:
                            current = trial
                            current_size = trial_size
                            improved = True
                            found_swap = True
                            break

                # 2-element removals (only if few blockers)
                if not found_swap and len(candidates) <= 12 and time.time() < deadline - 3:
                    for i in range(len(candidates)):
                        if time.time() > deadline - 2:
                            return current
                        for j in range(i + 1, len(candidates)):
                            trial = current & ~(1 << candidates[i]) & ~(1 << candidates[j])
                            if _can_add(x, trial):
                                trial |= (1 << x)
                                trial = _greedy_refill(trial, deadline - 2)
                                trial_size = bin(trial).count('1')
                                if trial_size > current_size:
                                    current = trial
                                    current_size = trial_size
                                    improved = True
                                    found_swap = True
                                    break
                        if found_swap:
                            break

            if not improved:
                break

    return current


def _randomized_local_search(bits, deadline, rng):
    """
    Randomized local search: random-order additions, then random swaps.
    """
    current = bits
    current_size = bin(current).count('1')

    for _pass in range(10):
        if time.time() > deadline - 2:
            break
        improved = False

        order = list(range(1, N + 1))
        rng.shuffle(order)

        for x in order:
            if time.time() > deadline - 2:
                return current
            if not ((current >> x) & 1) and _can_add(x, current):
                current |= (1 << x)
                current_size += 1
                improved = True

        if not improved:
            for _attempt in range(50):
                if time.time() > deadline - 2:
                    return current
                x = rng.randint(1, N)
                if (current >> x) & 1:
                    continue

                blockers = _find_blockers(x, current)
                if not blockers:
                    continue

                rem = rng.choice(list(blockers))
                trial = current & ~(1 << rem)
                if _can_add(x, trial):
                    trial |= (1 << x)
                    trial = _greedy_refill(trial, deadline - 2)
                    trial_size = bin(trial).count('1')
                    if trial_size > current_size:
                        current = trial
                        current_size = trial_size
                        improved = True

    return current


def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit

    # 1. Generate diverse seeds
    cantor = _cantor_bits()
    best = cantor
    best_size = bin(cantor).count('1')

    seeds = []
    for fn in (_greedy_forward, _greedy_reverse, _greedy_density):
        s = fn()
        seeds.append(s)
    rng_seed = random.Random(999)
    for _ in range(3):
        seeds.append(_greedy_random(0, rng_seed))

    for s in seeds:
        sz = bin(s).count('1')
        if sz > best_size:
            best = s
            best_size = sz

    # 2. Local search on the best seed
    if time.time() < deadline - 30:
        improved = _local_search(best, deadline - 10)
        imp_size = bin(improved).count('1')
        if imp_size > best_size:
            best = improved
            best_size = imp_size

    # 3. Randomized restarts
    rng = random.Random(12345)
    restart_count = 0
    while time.time() < deadline - 15 and restart_count < 60:
        restart_count += 1

        r = rng.random()
        if r < 0.20:
            bits = _greedy_random(cantor, rng)
        elif r < 0.40:
            bits = _greedy_random(0, rng)
        elif r < 0.55:
            bits = _greedy_random(best, rng)
        elif r < 0.70:
            # Perturb best: remove 1 random element, re-greedy
            bits = best
            if best_size > 1:
                elems = _bits_to_set(bits)
                if elems:
                    rem = rng.choice(elems)
                    bits = bits & ~(1 << rem)
                    bits = _greedy_forward(bits)
        elif r < 0.85:
            # Perturb best: remove 2 random elements, re-greedy
            bits = best
            if best_size > 2:
                elems = _bits_to_set(bits)
                if len(elems) >= 2:
                    rem1 = rng.choice(elems)
                    elems.remove(rem1)
                    rem2 = rng.choice(elems)
                    bits = bits & ~(1 << rem1) & ~(1 << rem2)
                    bits = _greedy_forward(bits)
        else:
            # Density-weighted from a perturbed start
            bits = best
            if best_size > 2:
                elems = _bits_to_set(bits)
                if len(elems) >= 2:
                    rem1 = rng.choice(elems)
                    elems.remove(rem1)
                    rem2 = rng.choice(elems)
                    bits = bits & ~(1 << rem1) & ~(1 << rem2)
                    bits = _greedy_density(bits)

        size = bin(bits).count('1')
        if size > best_size:
            best = bits
            best_size = size

        if time.time() < deadline - 10:
            improved = _local_search(bits, deadline - 8)
            imp_size = bin(improved).count('1')
            if imp_size > best_size:
                best = improved
                best_size = imp_size

        # Occasionally run randomized local search for diversity
        if time.time() < deadline - 8 and rng.random() < 0.3:
            improved = _randomized_local_search(best, deadline - 6, rng)
            imp_size = bin(improved).count('1')
            if imp_size > best_size:
                best = improved
                best_size = imp_size

    # 4. Final local search on best
    if time.time() < deadline - 5:
        improved = _local_search(best, deadline)
        imp_size = bin(improved).count('1')
        if imp_size > best_size:
            best = improved
            best_size = imp_size

    # 5. Verify
    if not _is_valid(best):
        best = cantor

    return sorted(_bits_to_set(best))


if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
