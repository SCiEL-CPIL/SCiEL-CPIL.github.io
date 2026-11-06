"""Extended Sidon set construction in [1,284] containing {1,2,5,11}.

Maximizes |S| subject to all pairwise sums a+b (a<=b) being distinct.
Strategy: fast greedy construction with randomized restarts, local
insert/swap improvement, and a bounded backtracking push to escape local
optima. A hard wall-clock deadline guard returns the best valid set found.
"""
import time
import random
import itertools

DEADLINE = 110.0          # safety margin under the 150s kill
START = time.time()

PRESCRIBED = (1, 2, 5, 11)
LO, HI = 1, 284


def _elapsed():
    return time.time() - START


def _sums_of(S):
    """Return the set of all pairwise sums a+b for a<=b in S."""
    S = sorted(S)
    out = set()
    n = len(S)
    for i in range(n):
        out.add(2 * S[i])
        for j in range(i + 1, n):
            out.add(S[i] + S[j])
    return out


def _is_sidon(S):
    """True iff all pairwise sums a+b (a<=b) are distinct."""
    S = sorted(S)
    n = len(S)
    seen = set()
    for i in range(n):
        v = 2 * S[i]
        if v in seen:
            return False
        seen.add(v)
        for j in range(i + 1, n):
            v = S[i] + S[j]
            if v in seen:
                return False
            seen.add(v)
    return True


def _can_add(S, sums, x):
    """Check if x can be added to S (with sum-set `sums`) keeping Sidon."""
    new = {x + y for y in S} | {2 * x}
    return len(new) == len(S) + 1 and not (new & sums)


def _greedy_build(order, start_set):
    """Greedily add elements in `order` that keep the set Sidon."""
    S = list(start_set)
    S_set = set(S)
    sums = _sums_of(S)
    for x in order:
        if x in S_set:
            continue
        if _can_add(S, sums, x):
            S.append(x)
            S_set.add(x)
            new = {x + y for y in S} | {2 * x}
            sums |= new
    return S


def _local_improve(S, rng, budget):
    """Try to grow S via insertions, 1-for-1 swaps, and 2-for-2 swaps.

    Returns a (possibly larger) valid Sidon set.
    """
    S = sorted(set(S))
    best = S
    best_len = len(S)
    t0 = time.time()

    improved = True
    while improved and (time.time() - t0) < budget:
        improved = False
        cur = list(best)
        cur_set = set(cur)
        cur_sums = _sums_of(cur)
        pool = [x for x in range(LO, HI + 1) if x not in cur_set]

        # Pass 1: pure insertions.
        for x in pool:
            if _can_add(cur, cur_sums, x):
                cur.append(x)
                cur_set.add(x)
                new = {x + y for y in cur} | {2 * x}
                cur_sums |= new
                improved = True

        # Pass 2: 1-for-1 swaps that are neutral but enable a subsequent insertion.
        if (time.time() - t0) < budget:
            cur_snapshot = list(cur)
            cur_set_snapshot = set(cur)
            for a in list(cur_snapshot):
                if a in PRESCRIBED:
                    continue
                cur_no_a = [v for v in cur_snapshot if v != a]
                for x in pool:
                    if x in cur_set_snapshot:
                        continue
                    trial = cur_no_a + [x]
                    if not _is_sidon(trial):
                        continue
                    trial_sums = _sums_of(trial)
                    for y in pool:
                        if y in cur_set_snapshot or y == x:
                            continue
                        if _can_add(trial, trial_sums, y):
                            cur = trial + [y]
                            cur_set = set(cur)
                            cur_sums = _sums_of(cur)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

        # Pass 3: 2-for-2 swaps (remove two non-prescribed, add two new) to escape local optima.
        if (time.time() - t0) < budget and not improved:
            cur_snapshot = list(cur)
            cur_set_snapshot = set(cur)
            non_prescribed = [v for v in cur_snapshot if v not in PRESCRIBED]
            # Sample a few pairs to keep it fast
            if len(non_prescribed) >= 2:
                pairs_to_try = []
                for i in range(min(len(non_prescribed), 15)):
                    for j in range(i + 1, min(len(non_prescribed), 15)):
                        pairs_to_try.append((non_prescribed[i], non_prescribed[j]))
                rng.shuffle(pairs_to_try)
                for a, b in pairs_to_try[:20]:
                    if (time.time() - t0) > budget:
                        break
                    cur_no_ab = [v for v in cur_snapshot if v != a and v != b]
                    cur_no_ab_set = set(cur_no_ab)
                    sums_no_ab = _sums_of(cur_no_ab)
                    # Find two elements x, y that can be added
                    candidates = [x for x in pool if x not in cur_set_snapshot and x != a and x != b]
                    for x in candidates[:50]:
                        if not _can_add(cur_no_ab, sums_no_ab, x):
                            continue
                        trial1 = cur_no_ab + [x]
                        trial1_sums = _sums_of(trial1)
                        for y in candidates[:50]:
                            if y == x or y in cur_set_snapshot or y == a or y == b:
                                continue
                            if _can_add(trial1, trial1_sums, y):
                                new_set = trial1 + [y]
                                if _is_sidon(new_set):
                                    cur = new_set
                                    cur_set = set(cur)
                                    cur_sums = _sums_of(cur)
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break

        if improved:
            best = sorted(set(cur))
            best_len = len(best)
    return best


def _backtrack_push(S, rng, budget):
    """Bounded DFS to add a few more elements, escaping local optima."""
    S = sorted(set(S))
    best = S
    best_len = len(S)
    t0 = time.time()

    def conflicts(x, sset, sums):
        new = {x + y for y in sset} | {2 * x}
        return len(new & sums)

    def dfs(cur, cur_sums, depth):
        nonlocal best, best_len
        if (time.time() - t0) > budget:
            return
        if len(cur) > best_len:
            best = list(cur)
            best_len = len(cur)
        if depth == 0:
            return
        cur_set = set(cur)
        cands = [x for x in range(LO, HI + 1) if x not in cur_set]
        # Sort by number of conflicts (fewer conflicts = better candidate)
        cands.sort(key=lambda x: conflicts(x, cur, cur_sums))
        for x in cands[:12]:  # wider branching
            if _can_add(cur, cur_sums, x):
                new = {x + y for y in cur} | {2 * x}
                dfs(cur + [x], cur_sums | new, depth - 1)
                if (time.time() - t0) > budget:
                    return

    dfs(list(S), _sums_of(S), 6)  # increased depth
    return best


def construct():
    global DEADLINE
    rng = random.Random(12345)

    # Baseline: deterministic greedy from the prescribed set.
    order = list(range(12, HI + 1))
    base = _greedy_build(order, list(PRESCRIBED))
    best = base
    best_len = len(base)

    deadline = _elapsed() + (DEADLINE - _elapsed())
    restart_count = 0
    # Randomized restarts with varied orders.
    while _elapsed() < deadline:
        restart_count += 1
        # Shuffle candidate order with a bias toward spread-out values.
        order = list(range(12, HI + 1))
        rng.shuffle(order)
        S = _greedy_build(order, list(PRESCRIBED))
        # Local improvement: scale budget with remaining time, cap at 12s.
        rem = max(1.0, deadline - _elapsed())
        S = _local_improve(S, rng, min(rem * 0.4, 12.0))
        # Backtracking push to escape local optima.
        if len(S) >= best_len - 1 and (deadline - _elapsed()) > 3.0:
            S = _backtrack_push(S, rng, min(3.0, deadline - _elapsed() - 0.5))
        if len(S) > best_len:
            best = S
            best_len = len(S)
            # If we reached a very large set, we can stop early.
            if best_len >= 22:
                break
        # After several restarts without improvement, try a longer local search.
        if restart_count % 8 == 0 and (deadline - _elapsed()) > 5.0:
            S = _local_improve(best, rng, min(5.0, deadline - _elapsed() - 1.0))
            if len(S) > best_len:
                best = S
                best_len = len(S)

    # Final polish: try to add one more element to the best set found.
    if (deadline - _elapsed()) > 1.0:
        best_cur = list(best)
        best_cur_set = set(best_cur)
        best_cur_sums = _sums_of(best_cur)
        for x in range(LO, HI + 1):
            if x not in best_cur_set and _can_add(best_cur, best_cur_sums, x):
                best_cur.append(x)
                best_cur_sums |= {x + y for y in best_cur} | {2 * x}
                best = best_cur
                best_len = len(best)
                break

    # Final validation guard.
    if not _is_sidon(best):
        # Fallback to a known-valid greedy set.
        best = _greedy_build(list(range(12, HI + 1)), list(PRESCRIBED))
    # Ensure prescribed elements present.
    best = sorted(set(best) | set(PRESCRIBED))
    # Re-validate after merge (merge only adds prescribed which are
    # already compatible by construction).
    if not _is_sidon(best):
        best = [1, 2, 5, 11]
    return best
