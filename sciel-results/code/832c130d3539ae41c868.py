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


def _greedy_build(order, start_set):
    """Greedily add elements in `order` that keep the set Sidon."""
    S = list(start_set)
    sums = _sums_of(S)
    for x in order:
        if x in S:
            continue
        new = {x + y for y in S} | {2 * x}
        if len(new) == len(S) + 1 and not (new & sums):
            S.append(x)
            sums |= new
    return S


def _local_improve(S, rng, budget):
    """Try to grow S via insertions and 1-for-1 / 1-for-2 swaps.

    Returns a (possibly larger) valid Sidon set.
    """
    S = sorted(set(S))
    best = S
    best_len = len(S)
    t0 = time.time()
    # Candidate pool: all numbers not in S.
    pool = [x for x in range(LO, HI + 1) if x not in set(S)]

    def try_insert(cur, x):
        if x in cur:
            return None
        sums = _sums_of(cur)
        new = {x + y for y in cur} | {2 * x}
        if len(new) == len(cur) + 1 and not (new & sums):
            return cur + [x]
        return None

    def count_insertable(cur, pool_list):
        """Count how many elements from pool can be inserted into cur."""
        cur_set = set(cur)
        sums = _sums_of(cur)
        cnt = 0
        for x in pool_list:
            if x in cur_set:
                continue
            new = {x + y for y in cur} | {2 * x}
            if len(new) == len(cur) + 1 and not (new & sums):
                cnt += 1
        return cnt

    improved = True
    while improved and (time.time() - t0) < budget:
        improved = False
        cur = list(best)
        cur_set = set(cur)
        
        # Pass 1: pure insertions (any order).
        for x in pool:
            cand = try_insert(cur, x)
            if cand is not None:
                cur = cand
                improved = True
        
        # Pass 2: 1-for-1 swaps that enable subsequent insertions.
        # Only accept a swap if it increases the number of insertable elements.
        if not improved and (time.time() - t0) < budget:
            baseline_insertable = count_insertable(cur, pool)
            cur_set = set(cur)
            for a in list(cur):
                if (time.time() - t0) > budget:
                    break
                for x in pool:
                    if x in cur_set:
                        continue
                    trial = [v for v in cur if v != a] + [x]
                    if _is_sidon(trial):
                        # Check if this swap enables more insertions
                        trial_set = set(trial)
                        # Update pool for trial: remove x (now in set), add a (now out)
                        trial_pool = [p for p in pool if p != x] + [a]
                        trial_insertable = count_insertable(trial, trial_pool)
                        if trial_insertable > baseline_insertable:
                            # Accept the swap and try to insert
                            cur = trial
                            cur_set = trial_set
                            improved = True
                            # Now try to insert the newly enabled elements
                            for y in trial_pool:
                                if (time.time() - t0) > budget:
                                    break
                                cand = try_insert(cur, y)
                                if cand is not None:
                                    cur = cand
                                    cur_set = set(cur)
                            break
                if improved:
                    break
        
        if improved:
            best = sorted(set(cur))
            best_len = len(best)
            pool = [x for x in range(LO, HI + 1) if x not in set(best)]
    return best


def _backtrack_push(S, rng, budget):
    """Bounded DFS to add a few more elements, escaping local optima."""
    S = sorted(set(S))
    best = S
    best_len = len(S)
    t0 = time.time()

    # Order candidates by a heuristic: prefer numbers that create few
    # conflicts with the current sum set.
    sums = _sums_of(S)
    cands = [x for x in range(LO, HI + 1) if x not in set(S)]

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
        # MRV-ish: pick candidate with fewest conflicts.
        cands = [x for x in range(LO, HI + 1) if x not in set(cur)]
        cands.sort(key=lambda x: conflicts(x, cur, cur_sums))
        for x in cands[:8]:  # limited branching
            new = {x + y for y in cur} | {2 * x}
            if len(new) == len(cur) + 1 and not (new & cur_sums):
                dfs(cur + [x], cur_sums | new, depth - 1)

    dfs(list(S), _sums_of(S), 4)
    return best

def _simulated_annealing_kick(S, rng, budget):
    """Use simulated annealing to kick out of local optima by accepting
    temporarily worse states (removing elements) that may enable larger
    reconstructions."""
    S = sorted(set(S))
    best = S
    best_len = len(S)
    t0 = time.time()
    
    # Start with current set
    cur = list(S)
    cur_sums = _sums_of(cur)
    
    # Temperature schedule
    T_start = 1.0
    T_end = 0.01
    n_steps = max(1, int(budget * 10))  # ~10 steps per second
    
    for step in range(n_steps):
        if (time.time() - t0) > budget:
            break
        
        # Temperature decay
        progress = step / n_steps
        T = T_start * (T_end / T_start) ** progress
        
        # Random move: either remove a random element or swap
        move_type = rng.random()
        
        if move_type < 0.5:
            # Remove a random element (kick)
            if len(cur) <= len(PRESCRIBED):
                continue
            # Don't remove prescribed elements
            removable = [x for x in cur if x not in PRESCRIBED]
            if not removable:
                continue
            a = rng.choice(removable)
            trial = [x for x in cur if x != a]
            if _is_sidon(trial):
                # Always accept removal (it's a kick to explore)
                cur = trial
                cur_sums = _sums_of(cur)
        else:
            # Swap: remove one non-prescribed, add one new
            removable = [x for x in cur if x not in PRESCRIBED]
            if not removable:
                continue
            a = rng.choice(removable)
            pool = [x for x in range(LO, HI + 1) if x not in set(cur)]
            if not pool:
                continue
            b = rng.choice(pool)
            trial = [x for x in cur if x != a] + [b]
            if _is_sidon(trial):
                # Accept with probability based on energy change (size)
                # Since size is same, always accept neutral swaps
                cur = trial
                cur_sums = _sums_of(cur)
        
        # Try to greedily expand from current state
        cur_set = set(cur)
        pool = [x for x in range(LO, HI + 1) if x not in cur_set]
        for x in pool:
            if (time.time() - t0) > budget:
                break
            new = {x + y for y in cur} | {2 * x}
            if len(new) == len(cur) + 1 and not (new & cur_sums):
                cur.append(x)
                cur_sums |= new
        
        # Update best
        if len(cur) > best_len:
            best = list(cur)
            best_len = len(cur)
    
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
    # Randomized restarts with varied orders.
    while _elapsed() < deadline:
        # Shuffle candidate order with a bias toward spread-out values.
        order = list(range(12, HI + 1))
        rng.shuffle(order)
        S = _greedy_build(order, list(PRESCRIBED))
        # Local improvement.
        rem = max(1.0, deadline - _elapsed())
        S = _local_improve(S, rng, min(rem * 0.4, 8.0))
        # Simulated annealing kick to escape local optima.
        if len(S) >= best_len - 1 and (deadline - _elapsed()) > 3.0:
            S = _simulated_annealing_kick(S, rng, min(3.0, deadline - _elapsed() - 1.0))
        # Backtracking push to escape local optima.
        if len(S) >= best_len - 1 and (deadline - _elapsed()) > 2.0:
            S = _backtrack_push(S, rng, min(2.0, deadline - _elapsed() - 0.5))
        if len(S) > best_len:
            best = S
            best_len = len(S)
            # If we reached a very large set, we can stop early.
            if best_len >= 22:
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
