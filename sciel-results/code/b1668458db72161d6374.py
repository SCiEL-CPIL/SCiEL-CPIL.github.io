"""Fresh design: greedy + local search for 3-uniform hypergraph on 16 pts
avoiding all 8 transversals of any three disjoint pairs (K(2,2,2))."""
import time
import random
import sys

N = 16
DEADLINE = 118.0  # safety margin before 150s kill

def construct():
    start = time.time()
    deadline = start + DEADLINE
    best_triples = []
    best_count = 0

    # All possible triples
    all_triples = [[a, b, c] for a in range(N) for b in range(a + 1, N) for c in range(b + 1, N)]
    nT = len(all_triples)  # 560

    # Precompute for each triple the 21 "other-pairs" (pairs not sharing a point)
    triple_other_pairs = []
    for t in all_triples:
        s = set(t)
        o = []
        for i in range(N):
            for j in range(i + 1, N):
                if i not in s and j not in s:
                    o.append((i, j))
        triple_other_pairs.append(o)

    # Precompute for each pair the 6 transversals
    pair_transversals = {}
    for i in range(N):
        for j in range(i + 1, N):
            s = {i, j}
            tv = []
            for a in range(N):
                for b in range(N):
                    for c in range(N):
                        if a != b and b != c and a != c and a not in s and b not in s and c not in s:
                            tv.append((a, b, c))
            pair_transversals[(i, j)] = tv
    # sanity: each has 6
    for k, v in pair_transversals.items():
        assert len(v) == 6, (k, len(v))

    def elapsed():
        return time.time() - start

    def try_local_search(triples, budget):
        nonlocal best_triples, best_count
        triples = list(triples)
        # Build data structures
        triple_set = set()
        for t in triples:
            triple_set.add(tuple(t))
        present = [False] * nT
        for t in triples:
            present[all_triples.index(t)] = True  # index lookup slow; use dict
        # Use dict for present by tuple
        present_map = {tuple(t): True for t in triples}
        # danger count per pair
        danger = {}
        for (i, j), tv in pair_transversals.items():
            cnt = 0
            for tv_t in tv:
                if tv_t in present_map:
                    cnt += 1
            danger[(i, j)] = cnt
        # danger per triple
        triple_danger = {}
        for t in triples:
            d = 0
            for (i, j) in triple_other_pairs[all_triples.index(t)]:
                d += danger[(i, j)]
            triple_danger[tuple(t)] = d
        # danger per (triple, pair)
        tp_danger = {}
        for t in triples:
            tt = tuple(t)
            for (i, j) in triple_other_pairs[all_triples.index(t)]:
                tp_danger[(tt, (i, j))] = danger[(i, j)]

        count = len(triples)

        def add_triple(t):
            nonlocal count
            tt = tuple(t)
            triples.append(t)
            present_map[tt] = True
            for (i, j) in triple_other_pairs[all_triples.index(t)]:
                danger[(i, j)] += 1
                tp_danger[(tt, (i, j))] = danger[(i, j)]
            d = 0
            for (i, j) in triple_other_pairs[all_triples.index(t)]:
                d += danger[(i, j)]
            triple_danger[tt] = d
            count += 1

        def remove_triple(t):
            nonlocal count
            tt = tuple(t)
            triples.remove(t)
            del present_map[tt]
            for (i, j) in triple_other_pairs[all_triples.index(t)]:
                danger[(i, j)] -= 1
                tp_danger[(tt, (i, j))] = danger[(i, j)]
            d = 0
            for (i, j) in triple_other_pairs[all_triples.index(t)]:
                d += danger[(i, j)]
            triple_danger[tt] = d
            count -= 1

        def add_valid(t):
            tt = tuple(t)
            if tt in present_map:
                return False
            for (i, j) in triple_other_pairs[all_triples.index(t)]:
                if danger[(i, j)] >= 7:
                    return False
            add_triple(t)
            return True

        def remove_valid(t):
            tt = tuple(t)
            if tt not in present_map:
                return False
            remove_triple(t)
            return True

        # Local search: try to add a random absent triple; if blocked, remove a
        # triple that is a transversal of one of the blocking pairs, then retry.
        rng = random.Random(12345)
        absent = [t for t in all_triples if tuple(t) not in present_map]
        # iterate
        no_progress = 0
        while elapsed() < budget and no_progress < 2000:
            # pick a random absent triple
            if not absent:
                break
            t = rng.choice(absent)
            tt = tuple(t)
            # find blocking pairs
            blockers = []
            for (i, j) in triple_other_pairs[all_triples.index(t)]:
                if danger[(i, j)] >= 7:
                    blockers.append((i, j))
            if not blockers:
                if add_valid(t):
                    no_progress = 0
                    # remove from absent
                    try:
                        absent.remove(t)
                    except ValueError:
                        pass
                    if count > best_count:
                        best_count = count
                        best_triples = list(triples)
                else:
                    no_progress += 1
            else:
                # pick a blocker, find a transversal of it that is present, remove it
                bp = rng.choice(blockers)
                tv = pair_transversals[bp]
                candidates = [tv_t for tv_t in tv if tv_t in present_map]
                if not candidates:
                    no_progress += 1
                    continue
                # choose the candidate with highest danger (most impactful removal)
                # removal reduces danger of its other-pairs; we want to unblock t.
                # Pick one that shares the most other-pairs with t's blockers.
                best_cand = None
                best_score = -1
                for tv_t in candidates:
                    # after removing tv_t, which of t's blockers get unblocked?
                    # tv_t is a transversal of bp; removing it reduces danger[bp] by 1.
                    # We need danger[bp] to drop below 7, i.e. it was 7.
                    # Also it may help other blockers if tv_t is also a transversal of them.
                    # A triple is a transversal of at most... it's disjoint from bp,
                    # so it's a transversal of bp only. But it could be a transversal
                    # of another blocker bp2 if tv_t disjoint from bp2.
                    score = 0
                    for (i2, j2) in blockers:
                        if (i2, j2) == bp:
                            if danger[bp] == 7:
                                score += 2
                        else:
                            # is tv_t a transversal of (i2,j2)?
                            s2 = {i2, j2}
                            if all(x not in s2 for x in tv_t):
                                if danger[(i2, j2)] == 7:
                                    score += 1
                    if score > best_score:
                        best_score = score
                        best_cand = tv_t
                if best_cand is None or best_score <= 0:
                    no_progress += 1
                    continue
                if remove_valid(best_cand):
                    # add best_cand back to absent
                    absent.append(list(best_cand))
                    # now try to add t again
                    if add_valid(t):
                        no_progress = 0
                        try:
                            absent.remove(t)
                        except ValueError:
                            pass
                        if count > best_count:
                            best_count = count
                            best_triples = list(triples)
                    else:
                        no_progress += 1
                else:
                    no_progress += 1
        return triples

    # Phase 1: Greedy construction with randomization over multiple seeds
    for seed in range(0, 40):
        if elapsed() > deadline - 5:
            break
        rng = random.Random(seed)
        # random order of all triples
        order = list(range(nT))
        rng.shuffle(order)
        triples = []
        present_map = set()
        danger = {}
        for (i, j), tv in pair_transversals.items():
            danger[(i, j)] = 0
        # greedy
        for idx in order:
            t = all_triples[idx]
            tt = tuple(t)
            if tt in present_map:
                continue
            ok = True
            for (i, j) in triple_other_pairs[idx]:
                if danger[(i, j)] >= 7:
                    ok = False
                    break
            if ok:
                triples.append(t)
                present_map.add(tt)
                for (i, j) in triple_other_pairs[idx]:
                    danger[(i, j)] += 1
        count = len(triples)
        if count > best_count:
            best_count = count
            best_triples = list(triples)

        # local search on this greedy result
        if elapsed() < deadline - 10:
            ls_triples = try_local_search(triples, deadline - 12)
            if len(ls_triples) > best_count:
                best_count = len(ls_triples)
                best_triples = list(ls_triples)

    # Return best
    # ensure distinct and valid format
    result = [list(t) for t in best_triples]
    return result
