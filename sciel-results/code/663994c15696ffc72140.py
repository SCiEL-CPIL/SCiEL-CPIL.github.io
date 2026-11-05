"""Combined program: structural conflict-gcd characterization + greedy/local-search/SA refinement."""
import time
import random
import math

_DEADLINE = 85.0  # safety margin below 90s wall-clock
_N = 256


def _conflict(a, b):
    """True if (a*b) % (a+b) == 0, i.e., a and b conflict."""
    return (a * b) % (a + b) == 0


def _build_conflict_sets():
    """conflicts[x] = set of y != x such that (x*y) % (x+y) == 0."""
    conflicts = [set() for _ in range(_N + 1)]
    for a in range(1, _N + 1):
        for b in range(a + 1, _N + 1):
            if (a * b) % (a + b) == 0:
                conflicts[a].add(b)
                conflicts[b].add(a)
    return conflicts


def _validate(S):
    s = set(S)
    if len(s) != len(S):
        return False
    if any(x < 1 or x > _N for x in s):
        return False
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            if _conflict(S[i], S[j]):
                return False
    return True


def _can_add(x, S_set, conflicts):
    return not (S_set & conflicts[x])


def _greedy_add_all(S, S_set, remaining, conflicts, time_limit):
    """Repeatedly add any remaining element that fits, until none fit or time up."""
    S = list(S)
    S_set = set(S)
    remaining = set(remaining)
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        for x in sorted(remaining):
            if time.time() >= time_limit:
                break
            if _can_add(x, S_set, conflicts):
                S.append(x)
                S_set.add(x)
                remaining.discard(x)
                improved = True
    return S, S_set, remaining


def _swap_search(S, S_set, remaining, conflicts, time_limit):
    """Try 1-for-2 and 2-for-3 swaps to grow the set."""
    S = list(S)
    S_set = set(S)
    remaining = set(remaining)
    improved = True
    while improved and time.time() < time_limit:
        improved = False

        # 1-for-2: remove one element r, try to add two
        for r in list(S):
            if time.time() >= time_limit:
                break
            S_set.discard(r)
            candidates = [x for x in remaining if _can_add(x, S_set, conflicts)]
            added = 0
            added_elems = []
            for x in candidates:
                if time.time() >= time_limit:
                    break
                if added >= 2:
                    break
                if _can_add(x, S_set, conflicts):
                    S.append(x)
                    S_set.add(x)
                    remaining.discard(x)
                    added_elems.append(x)
                    added += 1
            if added >= 2:
                # net gain = added - 1 >= 1
                S.remove(r)
                improved = True
                break
            else:
                # revert
                for x in added_elems:
                    S.remove(x)
                    S_set.discard(x)
                    remaining.add(x)
                S_set.add(r)

        # 2-for-3: remove two elements, try to add three
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            if time.time() >= time_limit:
                break
            for j in range(i + 1, n):
                if time.time() >= time_limit:
                    break
                r1, r2 = S_list[i], S_list[j]
                S_set.discard(r1)
                S_set.discard(r2)
                candidates = [x for x in remaining if _can_add(x, S_set, conflicts)]
                added = 0
                added_elems = []
                for x in candidates:
                    if time.time() >= time_limit:
                        break
                    if added >= 3:
                        break
                    if _can_add(x, S_set, conflicts):
                        S.append(x)
                        S_set.add(x)
                        remaining.discard(x)
                        added_elems.append(x)
                        added += 1
                if added >= 3:
                    S.remove(r1)
                    S.remove(r2)
                    improved = True
                    break
                else:
                    for x in added_elems:
                        S.remove(x)
                        S_set.discard(x)
                        remaining.add(x)
                    S_set.add(r1)
                    S_set.add(r2)
            if improved:
                break
    return S, S_set, remaining


def _simulated_annealing(S, S_set, remaining, conflicts, time_limit, rng):
    """SA: randomly remove an element, greedily refill, accept with cooling."""
    S = list(S)
    S_set = set(S)
    remaining = set(remaining)
    best_S = list(S)
    best_size = len(S)
    temperature = 1.0
    cooling = 0.99

    while time.time() < time_limit and temperature > 0.01:
        if len(S) == 0:
            break
        idx = rng.randint(0, len(S) - 1)
        removed = S.pop(idx)
        S_set.discard(removed)
        remaining.add(removed)

        new_S, new_S_set, new_remaining = _greedy_add_all(
            S, S_set, remaining, conflicts, time.time() + 0.02
        )
        new_size = len(new_S)

        if new_size >= best_size:
            best_S = list(new_S)
            best_size = new_size
            S = new_S
            S_set = new_S_set
            remaining = new_remaining
        else:
            if rng.random() < temperature:
                S = new_S
                S_set = new_S_set
                remaining = new_remaining
            else:
                S = list(best_S)
                S_set = set(S)
                remaining = set(range(1, _N + 1)) - S_set
        temperature *= cooling

    return best_S, set(best_S), set(range(1, _N + 1)) - set(best_S)


def construct():
    t0 = time.time()
    rng = random.Random(42)

    def _deadline_ok():
        return time.time() - t0 < _DEADLINE

    conflicts = _build_conflict_sets()

    # Start with all odds as a strong baseline
    best = list(range(1, _N + 1, 2))
    best_size = len(best)

    # 1) Greedy least-conflicting build
    if _deadline_ok():
        S = []
        S_set = set()
        remaining = set(range(1, _N + 1))
        while remaining:
            best_x = None
            best_c = None
            for x in remaining:
                c = len(conflicts[x] & S_set)
                if best_c is None or c < best_c or (c == best_c and best_x is not None and x < best_x):
                    best_x = x
                    best_c = c
                if c == 0:
                    break
            if best_c == 0:
                S.append(best_x)
                S_set.add(best_x)
                remaining.discard(best_x)
            else:
                break
        if _validate(S) and len(S) > best_size:
            best = S
            best_size = len(S)

    # 2) Multiple random greedy orderings
    if _deadline_ok():
        for trial in range(300):
            if time.time() - t0 > _DEADLINE - 15:
                break
            order = list(range(1, _N + 1))
            rng.shuffle(order)
            S = []
            S_set = set()
            for x in order:
                if _can_add(x, S_set, conflicts):
                    S.append(x)
                    S_set.add(x)
            if len(S) > best_size:
                best = S
                best_size = len(S)

    # 3) Local search: add-only
    if _deadline_ok():
        S_set = set(best)
        remaining = set(range(1, _N + 1)) - S_set
        S, S_set, remaining = _greedy_add_all(best, S_set, remaining, conflicts, time.time() + 5)
        if _validate(S) and len(S) > best_size:
            best = S
            best_size = len(S)

    # 4) Swap search
    if _deadline_ok():
        S_set = set(best)
        remaining = set(range(1, _N + 1)) - S_set
        S, S_set, remaining = _swap_search(best, S_set, remaining, conflicts, time.time() + 10)
        if _validate(S) and len(S) > best_size:
            best = S
            best_size = len(S)

    # 5) Simulated annealing
    if _deadline_ok():
        S_set = set(best)
        remaining = set(range(1, _N + 1)) - S_set
        S, S_set, remaining = _simulated_annealing(
            best, S_set, remaining, conflicts, time.time() + (_DEADLINE - (time.time() - t0) - 2), rng
        )
        if _validate(S) and len(S) > best_size:
            best = S
            best_size = len(S)

    # Final validation
    if not _validate(best):
        # Fallback to odds
        best = list(range(1, _N + 1, 2))

    return best
