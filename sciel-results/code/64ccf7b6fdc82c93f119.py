"""Linear triple systems on 16 points - minimize independence number.

Structural redesign:
  * Build a strong seed via a deterministic Steiner-style 2-(16,3,1) packing
    (greedy over balanced point degrees, with a fixed tie-break order).
  * Optimize with a fast branch-and-bound maximum-triple-free-set solver
    (bitsets + pruning) used as the independence oracle.
  * Local search = simulated annealing on triple swaps/add/remove with the
    fast oracle; deterministic seeds + a small random budget.
"""
import itertools
import random
import time

N = 16
ALL_TRIPLES = list(itertools.combinations(range(N), 3))
# Precompute pair -> whether it's used is dynamic; precompute triple bitmask
TRIPLE_MASK = [0] * len(ALL_TRIPLES)
TRIPLE_SET = [frozenset(t) for t in ALL_TRIPLES]
for i, t in enumerate(ALL_TRIPLES):
    m = 0
    for p in t:
        m |= (1 << p)
    TRIPLE_MASK[i] = m


def _triple_free_size(triples):
    """Largest subset of points containing no complete triple (bitmask B&B)."""
    # A set S is triple-free iff for every triple t, t is not a subset of S.
    # We want max |S|. Use branch & bound over points 0..15.
    # Represent candidate triples that are still "alive" (not yet broken).
    # A triple is broken if one of its points is excluded.
    # We build S by choosing to include/exclude points.
    # Upper bound: current size + number of remaining points.
    # Prune: if a triple is fully contained in (chosen + remaining), it's not
    #       broken yet; we must ensure we don't complete it.
    #
    # Simpler robust B&B: state = (chosen_mask, remaining_mask, alive_triples_mask)
    # A triple is alive if all its points are in (chosen_mask | remaining_mask)
    # and not all in chosen_mask (i.e., not already broken by an exclusion).
    # Actually a triple is "safe to complete" only if all three are chosen.
    # We break a triple by excluding at least one of its points.
    #
    # Let me use: chosen_mask, and for each triple track how many of its
    # points are still available (in remaining or chosen). A triple becomes
    # "forced-bad" if all 3 are in chosen. We prevent that.
    #
    # Efficient approach: recursive on point index.
    #   include point p: new_chosen = chosen | (1<<p). Check no triple subset.
    #   exclude point p: just drop it.
    # Upper bound = popcount(chosen) + (N - idx).
    # Prune if upper bound <= best.
    # Additionally, if including p would complete some triple, we must exclude p
    # (or one of the other two). We can detect via precomputed: for each point p,
    # the set of triples containing p.

    triples_by_point = [[] for _ in range(N)]
    for i, t in enumerate(triples):
        for p in t:
            triples_by_point[p].append(i)

    # For fast "does adding p complete a triple", precompute for the current
    # chosen set the count per triple of chosen points. We maintain an array
    # cnt[i] = number of points of triple i that are in chosen_mask.
    # A triple is completed when cnt[i] reaches 3.
    # We maintain cnt as we recurse (increment on include, decrement on backtrack).

    best = [0]
    cnt = [0] * len(triples)

    # Precompute for each point the list of triple indices containing it.
    def rec(idx, chosen_count):
        if idx == N:
            if chosen_count > best[0]:
                best[0] = chosen_count
            return
        # Upper bound
        if chosen_count + (N - idx) <= best[0]:
            return
        p = idx
        # Option 1: include p (if it doesn't complete any triple)
        ok = True
        affected = triples_by_point[p]
        for ti in affected:
            if cnt[ti] == 2:
                ok = False
                break
        if ok:
            for ti in affected:
                cnt[ti] += 1
            rec(idx + 1, chosen_count + 1)
            for ti in affected:
                cnt[ti] -= 1
        # Option 2: exclude p
        rec(idx + 1, chosen_count)

    rec(0, 0)
    return best[0]


def _is_linear(triples):
    used = set()
    for t in triples:
        a, b, c = t
        for pair in ((a, b), (a, c), (b, c)):
            if pair in used:
                return False
            used.add(pair)
    return True


def _deterministic_packing():
    """Greedy Steiner-style 2-(16,3,1) packing, balanced by degree."""
    triples = []
    used_pairs = set()
    deg = [0] * N
    # Candidate order: all triples; at each step pick the one maximizing
    # (min degree of its points) then a stable tie-break.
    # To keep it fast and deterministic, iterate a fixed shuffled-by-value order
    # but always re-pick the best available.
    remaining = list(ALL_TRIPLES)
    while remaining:
        best_t = None
        best_key = None
        for t in remaining:
            a, b, c = t
            pairs = {(a, b), (a, c), (b, c)}
            if any(p in used_pairs for p in pairs):
                continue
            # key: prefer triples covering the currently least-used points
            key = (min(deg[a], deg[b], deg[c]), deg[a] + deg[b] + deg[c], t)
            if best_key is None or key < best_key:
                best_key = key
                best_t = t
        if best_t is None:
            break
        a, b, c = best_t
        triples.append(best_t)
        for pair in ((a, b), (a, c), (b, c)):
            used_pairs.add(pair)
        deg[a] += 1
        deg[b] += 1
        deg[c] += 1
        remaining.remove(best_t)
    return triples


def _local_search(triples, max_iters, deadline, rng):
    """Simulated annealing on triple set."""
    triples = list(triples)
    cur = _triple_free_size(triples)
    best = cur
    best_triples = list(triples)
    T = 1.0
    cool = 0.999

    for it in range(max_iters):
        if time.time() > deadline:
            break
        # Pick a move
        move = rng.random()
        if move < 0.5 and triples:
            # swap
            i = rng.randrange(len(triples))
            old = triples[i]
            # find a valid replacement
            cand = None
            for _ in range(60):
                t = ALL_TRIPLES[rng.randrange(len(ALL_TRIPLES))]
                if t == old:
                    continue
                a, b, c = t
                ok = True
                for ot in triples:
                    if ot is old:
                        continue
                    shared = len(set((a, b, c)) & set(ot))
                    if shared > 1:
                        ok = False
                        break
                if ok:
                    cand = t
                    break
            if cand is None:
                continue
            new_triples = triples[:i] + [cand] + triples[i + 1:]
        elif move < 0.8 and triples:
            # remove
            i = rng.randrange(len(triples))
            new_triples = triples[:i] + triples[i + 1:]
        else:
            # add
            cand = None
            for _ in range(60):
                t = ALL_TRIPLES[rng.randrange(len(ALL_TRIPLES))]
                if t in triples:
                    continue
                a, b, c = t
                ok = True
                for ot in triples:
                    if len(set((a, b, c)) & set(ot)) > 1:
                        ok = False
                        break
                if ok:
                    cand = t
                    break
            if cand is None:
                continue
            new_triples = triples + [cand]

        new = _triple_free_size(new_triples)
        delta = new - cur
        if delta <= 0 or rng.random() < pow(2.718281828459045, -delta / max(T, 1e-9)):
            triples = new_triples
            cur = new
            if cur < best:
                best = cur
                best_triples = list(triples)
        T *= cool
    return best_triples, best


def construct():
    start = time.time()
    deadline = start + 140  # safety margin before 150s kill

    # Deterministic seed
    seed_triples = _deterministic_packing()
    best = _triple_free_size(seed_triples)
    best_triples = list(seed_triples)

    # Local search with several deterministic seeds
    rng = random.Random(12345)
    iters_per_run = 400
    for run in range(60):
        if time.time() > deadline - 3:
            break
        cur_triples = list(seed_triples)
        # perturb
        for _ in range(rng.randint(0, 5)):
            if cur_triples and rng.random() < 0.5:
                i = rng.randrange(len(cur_triples))
                cur_triples.pop(i)
        improved_triples, improved = _local_search(
            cur_triples, iters_per_run, deadline - 3, rng
        )
        if improved < best:
            best = improved
            best_triples = improved_triples
            seed_triples = list(improved_triples)

    if not best_triples:
        best_triples = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    return best_triples
