"""Linear triple systems on 16 points; minimize independence number."""
import time
import random
from itertools import combinations

_N = 16
_DEADLINE = 150.0
_MARGIN = 2.0  # safety margin seconds


def _all_triples():
    return list(combinations(range(_N), 3))


def _independence_number(triples):
    """Exact alpha: largest subset of points containing no triple.
    Use bitmask: for each point subset, check if any triple is contained.
    Precompute triple masks; a subset S is independent iff no triple mask T has T & S == T.
    We iterate subsets by size descending and return first independent one's size.
    """
    t_masks = [0] * len(triples)
    for i, (a, b, c) in enumerate(triples):
        t_masks[i] = (1 << a) | (1 << b) | (1 << c)
    n = _N
    total = 1 << n
    # Precompute for each subset whether it contains any triple.
    # A subset S contains triple T iff (S & T) == T.
    # We want max |S| such that for all T, (S & T) != T.
    # Enumerate subsets by descending popcount.
    # For speed, precompute popcount via int.bit_count().
    # Build list of subsets sorted by popcount descending - too big to sort 65536 each call.
    # Instead, iterate by size from n down to 1, and within a size, check subsets.
    # Precompute subsets_by_size once globally.
    if not hasattr(_independence_number, "_by_size"):
        by_size = [[] for _ in range(n + 1)]
        for s in range(1 << n):
            by_size[s.bit_count()].append(s)
        _independence_number._by_size = by_size
    by_size = _independence_number._by_size
    for size in range(n, 0, -1):
        for s in by_size[size]:
            ok = True
            for tm in t_masks:
                if (s & tm) == tm:
                    ok = False
                    break
            if ok:
                return size
    return 0


def _pair_conflict_count(triples):
    """Count pairs of points that appear together in more than one triple (should be 0 for linear)."""
    from collections import defaultdict
    pair_count = defaultdict(int)
    for a, b, c in triples:
        pair_count[(a, b)] += 1
        pair_count[(a, c)] += 1
        pair_count[(b, c)] += 1
    bad = 0
    for v in pair_count.values():
        if v > 1:
            bad += 1
    return bad


def _is_linear(triples):
    from collections import defaultdict
    pair_count = defaultdict(int)
    for a, b, c in triples:
        pair_count[(a, b)] += 1
        pair_count[(a, c)] += 1
        pair_count[(b, c)] += 1
    for v in pair_count.values():
        if v > 1:
            return False
    return True


def _random_maximal_packing(rng):
    """Greedy: randomly add triples until no more can be added (maximal linear system)."""
    all_t = _all_triples()
    rng.shuffle(all_t)
    from collections import defaultdict
    pair_count = defaultdict(int)
    triples = []
    for t in all_t:
        a, b, c = t
        if pair_count[(a, b)] + pair_count[(a, c)] + pair_count[(b, c)] == 0:
            triples.append(list(t))
            pair_count[(a, b)] += 1
            pair_count[(a, c)] += 1
            pair_count[(b, c)] += 1
    return triples


def _try_add(triples, pair_count, t):
    """Check if triple t can be added without breaking linearity."""
    a, b, c = t
    if pair_count[(a, b)] + pair_count[(a, c)] + pair_count[(b, c)] == 0:
        return True
    return False


def _add_triple(triples, pair_count, t):
    a, b, c = t
    triples.append(list(t))
    pair_count[(a, b)] += 1
    pair_count[(a, c)] += 1
    pair_count[(b, c)] += 1


def _remove_triple(triples, pair_count, idx):
    a, b, c = triples[idx]
    pair_count[(a, b)] -= 1
    pair_count[(a, c)] -= 1
    pair_count[(b, c)] -= 1
    triples[idx] = triples[-1]
    triples.pop()


def _local_search(triples, rng, deadline):
    """Improve by adding valid triples and swapping."""
    from collections import defaultdict
    pair_count = defaultdict(int)
    for a, b, c in triples:
        pair_count[(a, b)] += 1
        pair_count[(a, c)] += 1
        pair_count[(b, c)] += 1

    all_t = _all_triples()
    cur_alpha = _independence_number(triples)
    best_alpha = cur_alpha
    best_triples = [list(t) for t in triples]

    while time.time() < deadline:
        improved = False
        # Try adding any valid triple
        for t in all_t:
            if _try_add(triples, pair_count, t):
                new_alpha = _independence_number(triples + [list(t)])
                if new_alpha < cur_alpha:
                    _add_triple(triples, pair_count, t)
                    cur_alpha = new_alpha
                    if cur_alpha < best_alpha:
                        best_alpha = cur_alpha
                        best_triples = [list(x) for x in triples]
                    improved = True
                    break
                else:
                    # Adding doesn't help alpha, but might enable future swaps; skip for now
                    pass
        if improved:
            continue

        # Try swaps: remove one triple, add one or two
        if len(triples) > 0:
            idx = rng.randrange(len(triples))
            old_t = triples[idx]
            _remove_triple(triples, pair_count, idx)
            # Try adding a single new triple
            candidates = [t for t in all_t if t != tuple(old_t) and _try_add(triples, pair_count, t)]
            rng.shuffle(candidates)
            found = False
            for t in candidates[:20]:
                new_alpha = _independence_number(triples + [list(t)])
                if new_alpha < cur_alpha:
                    _add_triple(triples, pair_count, t)
                    cur_alpha = new_alpha
                    if cur_alpha < best_alpha:
                        best_alpha = cur_alpha
                        best_triples = [list(x) for x in triples]
                    found = True
                    break
            if not found:
                # Revert
                _add_triple(triples, pair_count, old_t)
        else:
            break

        if time.time() > deadline:
            break

    return best_triples, best_alpha


def construct():
    start = time.time()
    deadline = start + _DEADLINE - _MARGIN
    rng = random.Random(42)

    all_t = _all_triples()

    best_triples = None
    best_alpha = _N + 1

    # Phase 1: multiple random maximal packings, take best alpha
    while time.time() < deadline:
        triples = _random_maximal_packing(rng)
        alpha = _independence_number(triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = [list(t) for t in triples]
        if best_alpha <= 4:  # can't do much better
            break

    if best_triples is None:
        return [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]

    # Phase 2: local search on the best
    if time.time() < deadline:
        improved_triples, improved_alpha = _local_search(
            [list(t) for t in best_triples], rng, deadline
        )
        if improved_alpha < best_alpha:
            best_alpha = improved_alpha
            best_triples = improved_triples

    # Final validation: ensure linearity
    if not _is_linear(best_triples):
        # Fallback: rebuild a valid one
        from collections import defaultdict
        pair_count = defaultdict(int)
        valid = []
        for t in all_t:
            a, b, c = t
            if pair_count[(a, b)] + pair_count[(a, c)] + pair_count[(b, c)] == 0:
                valid.append(list(t))
                pair_count[(a, b)] += 1
                pair_count[(a, c)] += 1
                pair_count[(b, c)] += 1
        best_triples = valid
        best_alpha = _independence_number(best_triples)

    return best_triples
