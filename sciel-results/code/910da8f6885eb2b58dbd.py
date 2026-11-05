import itertools
import random
import time


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under 120s

    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)  # 560

    # Build the forbidden transversals as bitmask masks.
    # A transversal = 3 disjoint pairs on 6 points -> 8 transversal triples.
    transversals = []
    for six_pts in itertools.combinations(points, 6):
        sp = list(six_pts)
        matchings = set()
        for perm in itertools.permutations(range(6)):
            p1 = tuple(sorted((sp[perm[0]], sp[perm[1]])))
            p2 = tuple(sorted((sp[perm[2]], sp[perm[3]])))
            p3 = tuple(sorted((sp[perm[4]], sp[perm[5]])))
            matchings.add(tuple(sorted((p1, p2, p3))))
        for m in matchings:
            (p1, p2, p3) = m
            a, b = p1
            c, d = p2
            e, f = p3
            mask = 0
            for (x, y, z) in ((a, c, e), (a, c, f), (a, d, e), (a, d, f),
                              (b, c, e), (b, c, f), (b, d, e), (b, d, f)):
                mask |= 1 << triple_to_idx[tuple(sorted((x, y, z)))]
            transversals.append(mask)
    num_trans = len(transversals)

    # For local search: for each triple, the list of transversals it participates in.
    triple_trans = [[] for _ in range(num_triples)]
    for ti, tm in enumerate(transversals):
        m = tm
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            triple_trans[idx].append(ti)
            m ^= lsb

    # Conflict count for a given state (int mask), using a local list for speed.
    def conflict_count(state):
        c = 0
        for tm in transversals:
            if (state & tm) == tm:
                c += 1
        return c

    # ---- Phase 1: greedy removal from the full set ----
    state = (1 << num_triples) - 1  # all triples present
    random.shuffle(transversals)
    while True:
        if time.time() > deadline:
            break
        conflict = False
        for tm in transversals:
            if (state & tm) == tm:
                conflict = True
                # Remove a random triple from this conflict.
                bits = state & tm
                # pick a random set bit
                r = random.getrandbits(64)
                # rotate bits to pick randomly
                chosen = bits
                # simple: iterate
                m = bits
                cnt = 0
                # count bits in conflict (all 8 present)
                pick = random.randrange(8)
                mm = bits
                for _ in range(pick):
                    mm &= (mm - 1)
                lsb = mm & (-mm)
                state &= ~lsb
                break
        if not conflict:
            break

    best_state = state
    best_conf = conflict_count(best_state)
    best_count = best_state.bit_count()

    # ---- Phase 2: local search (add / remove) to minimize conflicts ----
    def try_local_search(state, conf, max_iters, time_limit):
        nonlocal best_state, best_conf, best_count
        for _ in range(max_iters):
            if time.time() > time_limit:
                break
            # Build candidate list: try adding each absent triple, removing each present.
            # Only consider moves that reduce conflict count.
            candidates = []
            # Adding a triple: only useful if it's in some conflict? No—adding increases
            # or keeps conflicts. Adding can only increase conflicts, never reduce. So
            # adding is only good if it lets us later add more... but for pure reduction
            # we remove. However removing a triple that's NOT in any conflict just reduces
            # count with no conflict benefit. So removals in conflicts are the play.
            # Let's evaluate removals of triples that are in at least one conflict.
            # Determine which triples are in conflicts.
            # For speed, compute conflict transversals.
            conflict_trans = []
            for ti, tm in enumerate(transversals):
                if (state & tm) == tm:
                    conflict_trans.append(ti)
            if not conflict_trans:
                break
            # For each triple in a conflict, compute delta if removed.
            # Removing triple i: conflicts that lose it become non-conflicting.
            # A conflict loses it iff i is present (it is) and i in that transversal.
            # So delta = -(number of conflict transversals containing i).
            # We want to maximize the number broken (most negative delta).
            best_break = 0
            best_i = -1
            for ti in conflict_trans:
                tm = transversals[ti]
                m = tm
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    # count how many conflict transversals contain idx
                    # we can just accumulate
                    m ^= lsb
                    # tally via a dict
                    # use a list
                    # We'll tally counts in a local dict
                    break
            # tally counts
            tally = {}
            for ti in conflict_trans:
                m = transversals[ti]
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    tally[idx] = tally.get(idx, 0) + 1
                    m ^= lsb
            for idx, cnt in tally.items():
                if cnt > best_break:
                    best_break = cnt
                    best_i = idx
            if best_i < 0 or best_break <= 0:
                break
            # remove best_i
            state &= ~(1 << best_i)
            conf -= best_break
            if conf < best_conf or (conf == best_conf and state.bit_count() > best_count):
                best_state = state
                best_conf = conf
                best_count = state.bit_count()
            if conf == 0:
                break
        return state, conf

    # Run local search with a few restarts to escape local minima.
    for restart in range(6):
        if time.time() > deadline - 1.0:
            break
        # For restart>0, start from current best with small random perturbation.
        if restart == 0:
            cur_state = state
            cur_conf = best_conf
        else:
            cur_state = best_state
            cur_conf = best_conf
            # perturb: randomly remove a few then re-run greedy
            for _ in range(random.randint(5, 20)):
                # remove a random present triple
                m = cur_state
                if not m:
                    break
                r = random.getrandbits(64)
                # pick random present triple
                # use bit trick: rotate
                # simpler: gather indices
                idxs = []
                mm = cur_state
                while mm:
                    lsb = mm & (-mm)
                    idxs.append(lsb.bit_length() - 1)
                    mm ^= lsb
                if not idxs:
                    break
                cur_state &= ~(1 << random.choice(idxs))
            cur_conf = conflict_count(cur_state)
        # local search
        cur_state, cur_conf = try_local_search(cur_state, cur_conf, 20000, deadline - 0.5)
        if cur_conf < best_conf or (cur_conf == best_conf and cur_state.bit_count() > best_count):
            best_state = cur_state
            best_conf = cur_conf
            best_count = cur_state.bit_count()

    # ---- Phase 3: if conflicts remain, do a final greedy clean to ensure validity ----
    # The evaluator requires a VALID result (no full transversal). Ensure best_state is valid.
    # If best_state still has conflicts, greedily remove until valid.
    final_state = best_state
    random.shuffle(transversals)
    while True:
        conflict = False
        for tm in transversals:
            if (final_state & tm) == tm:
                conflict = True
                m = final_state & tm
                pick = random.randrange(8)
                mm = m
                for _ in range(pick):
                    mm &= (mm - 1)
                lsb = mm & (-mm)
                final_state &= ~lsb
                break
        if not conflict:
            break

    result = [list(all_triples[i]) for i in range(num_triples) if (final_state >> i) & 1]
    return result
