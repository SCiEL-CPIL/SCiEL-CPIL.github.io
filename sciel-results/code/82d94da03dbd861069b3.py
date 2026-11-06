"""Rewritten program: conflict-graph-aware local search for the 3-sum divisibility task."""
import time
import random
from itertools import combinations

N = 192
MAX_K = 48
DEADLINE_FRACTION = 0.90  # safety margin against 90s wall-clock


def _valid_triple_sums_mod(a, chosen):
    """Return True iff there is NO triple of distinct elements of `chosen`
    (excluding a) whose sum is divisible by a.
    chosen is a list; a may or may not be in it (we exclude it explicitly).
    """
    others = [v for v in chosen if v != a]
    m = len(others)
    if m < 3:
        return True
    # Precompute prefix to allow O(1) pair-sum check via hashing
    # We need: exists i<j<k with (others[i]+others[j]+others[k]) % a == 0
    # Equiv: exists pair (i<j) and k>j with others[k] % a == -(others[i]+others[j]) % a
    rem = [v % a for v in others]
    # map remainder -> list of indices (or just count)
    from collections import defaultdict
    rem_to_idx = defaultdict(list)
    for idx, r in enumerate(rem):
        rem_to_idx[r].append(idx)
    seen_pairs = {}
    # For each pair (i<j), check if there exists k>j with rem[k] == target
    # To be efficient, iterate k from high to low and maintain available remainders
    # Simpler: for each k, for each pair among indices < k, check.
    # But O(m^3) is too slow. Use: for each pair (i<j), target = (-rem[i]-rem[j])%a,
    # then check if any index k>j has rem[k]==target.
    # Precompute suffix presence of each remainder: suffix_has[k][r] = any index >=k with rem==r
    # Build suffix counts.
    suffix = [dict() for _ in range(m + 1)]
    # suffix[i] = multiset of remainders at indices i..m-1
    cur = {}
    for i in range(m - 1, -1, -1):
        cur = dict(cur)
        cur[rem[i]] = cur.get(rem[i], 0) + 1
        suffix[i] = cur
    for i in range(m):
        for j in range(i + 1, m):
            target = (-rem[i] - rem[j]) % a
            # need some k > j with rem[k] == target
            if j + 1 < m and suffix[j + 1].get(target, 0) > 0:
                return False
    return True


def _is_valid_set(chosen):
    """Check that for every a in chosen, no triple of distinct others sums to a multiple of a."""
    for a in chosen:
        if not _valid_triple_sums_mod(a, chosen):
            return False
    return True


def _greedy_seed():
    """Descending greedy: add x if the resulting set is valid."""
    selected = []
    for x in range(N, 0, -1):
        candidate = selected + [x]
        # Only need to check the new element x against existing, AND existing against the set
        # but adding x can only create a conflict for x (as divisor) or for existing a (as summand).
        # For existing a: a divides sum of 3 others; the new set includes x, so a new triple
        # could include x. So we must re-check all. But to be fast, check:
        #  (1) x as divisor: triple among selected
        #  (2) each a in selected: triple among (selected without a) + x  -> i.e., among selected
        #      plus possibly x. Equivalent to checking a against candidate.
        ok = _valid_triple_sums_mod(x, candidate)
        if ok:
            for a in selected:
                if not _valid_triple_sums_mod(a, candidate):
                    ok = False
                    break
        if ok:
            selected.append(x)
    return selected


def _local_search(initial, time_budget):
    """Try to grow the set by adding elements, and swap to improve.
    Returns the best valid set found within time_budget seconds.
    """
    best = list(initial)
    best_len = len(best)
    start = time.time()
    rng = random.Random(12345)

    # Build a quick candidate pool: numbers not in best
    def try_add(x, s):
        # s is list; check if adding x keeps validity
        cand = s + [x]
        if not _valid_triple_sums_mod(x, cand):
            return None
        for a in s:
            if not _valid_triple_sums_mod(a, cand):
                return None
        return cand

    def try_swap(x, remove_idx, s):
        # remove s[remove_idx], add x
        new_s = [v for i, v in enumerate(s) if i != remove_idx] + [x]
        # check all elements
        for a in new_s:
            if not _valid_triple_sums_mod(a, new_s):
                return None
        return new_s

    # Phase 1: greedy additions in random order
    pool = [x for x in range(1, N + 1) if x not in best]
    rng.shuffle(pool)
    for x in pool:
        if time.time() - start > time_budget:
            break
        cand = try_add(x, best)
        if cand is not None:
            best = cand
            best_len = len(best)
            pool.remove(x)
            if best_len >= MAX_K:
                break

    # Phase 2: random swaps to try to reach MAX_K
    for _ in range(200000):
        if time.time() - start > time_budget:
            break
        if best_len >= MAX_K:
            break
        x = rng.choice(pool) if pool else rng.randint(1, N)
        if x in best:
            continue
        # pick a random element to remove
        ri = rng.randrange(len(best))
        cand = try_swap(x, ri, best)
        if cand is not None:
            best = cand
            best_len = len(best)
            # update pool
            removed = None
            for i, v in enumerate(cand):
                pass
            # rebuild pool cheaply
            in_best = set(best)
            pool = [v for v in range(1, N + 1) if v not in in_best]
            if best_len >= MAX_K:
                break

    # Phase 3: local improvement - try to add remaining pool in best-first
    in_best = set(best)
    pool = [v for v in range(1, N + 1) if v not in in_best]
    # sort by how "safe" they are: larger numbers first
    pool.sort(reverse=True)
    for x in pool:
        if time.time() - start > time_budget:
            break
        if best_len >= MAX_K:
            break
        cand = try_add(x, best)
        if cand is not None:
            best = cand
            best_len = len(best)

    return best


def construct():
    start = time.time()
    # Greedy seed
    seed = _greedy_seed()
    # Local search with a generous but bounded budget
    time_budget = 75.0  # leaves 15s margin
    result = _local_search(seed, time_budget)

    # Final safety: ensure validity and cap at MAX_K
    if len(result) > MAX_K:
        result = result[:MAX_K]
    # Re-validate; if invalid, fall back to greedy seed
    if not _is_valid_set(result):
        result = seed
        if len(result) > MAX_K:
            result = result[:MAX_K]
    return result


if __name__ == "__main__":
    print(construct())
