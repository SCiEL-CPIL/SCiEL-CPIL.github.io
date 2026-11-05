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
    seen_masks = set()
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
            if mask not in seen_masks:
                seen_masks.add(mask)
                transversals.append(mask)
    num_trans = len(transversals)

    # Precompute the list of triple indices for each transversal
    trans_triples = []
    for tm in transversals:
        indices = []
        m = tm
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            indices.append(idx)
            m ^= lsb
        trans_triples.append(indices)

    # Build triple_to_trans_list: for each triple, which transversals it belongs to
    triple_to_trans = [[] for _ in range(num_triples)]
    for ti in range(num_trans):
        for t_idx in trans_triples[ti]:
            triple_to_trans[t_idx].append(ti)

    best_state = None
    best_count = 0

    def optimize_v2(start_state, start_counts, start_conf, start_count):
        nonlocal best_state, best_count

        cur_state = start_state.copy()
        cur_counts = start_counts.copy()
        cur_conf = start_conf
        cur_count = start_count

        # Phase 1: Resolve conflicts
        while cur_conf > 0:
            if time.time() > deadline:
                break
            # Find a conflict
            conflict_ti = -1
            for ti in range(num_trans):
                if cur_counts[ti] == 8:
                    conflict_ti = ti
                    break

            if conflict_ti == -1:
                break

            # Pick a triple to remove that breaks the most conflicts
            best_t = -1
            max_breaks = 0
            for t_idx in trans_triples[conflict_ti]:
                if cur_state[t_idx]:
                    breaks = 0
                    for ti in triple_to_trans[t_idx]:
                        if cur_counts[ti] == 8:
                            breaks += 1
                    if breaks > max_breaks:
                        max_breaks = breaks
                        best_t = t_idx

            if best_t == -1:
                best_t = random.choice(trans_triples[conflict_ti])

            # Remove best_t
            cur_state[best_t] = False
            cur_count -= 1
            for ti in triple_to_trans[best_t]:
                cur_counts[ti] -= 1
                if cur_counts[ti] == 7:
                    cur_conf -= 1

            if cur_conf == 0:
                if cur_count > best_count:
                    best_count = cur_count
                    best_state = cur_state.copy()

        # Phase 2: Add triples greedily
        progress = True
        while progress:
            if time.time() > deadline:
                break
            progress = False
            indices = list(range(num_triples))
            random.shuffle(indices)
            for i in indices:
                if not cur_state[i]:
                    # Check if safe to add (doesn't complete any transversal to 8)
                    safe = True
                    for ti in triple_to_trans[i]:
                        if cur_counts[ti] == 7:
                            safe = False
                            break
                    if safe:
                        cur_state[i] = True
                        cur_count += 1
                        for ti in triple_to_trans[i]:
                            cur_counts[ti] += 1
                        progress = True
                        if cur_count > best_count:
                            best_count = cur_count
                            best_state = cur_state.copy()

        # Phase 3: Local search - try swapping triples to improve
        if time.time() < deadline - 5.0:
            for _ in range(200):
                if time.time() > deadline:
                    break
                # Try to find a triple to remove and a different triple to add
                # such that the count stays the same or increases
                if not cur_state:
                    break
                present = [i for i in range(num_triples) if cur_state[i]]
                absent = [i for i in range(num_triples) if not cur_state[i]]
                if not present or not absent:
                    break
                random.shuffle(present)
                random.shuffle(absent)

                improved = False
                for rem in present[:50]:
                    # Check what conflicts we'd resolve by removing rem
                    resolved = 0
                    for ti in triple_to_trans[rem]:
                        if cur_counts[ti] == 8:
                            resolved += 1
                    if resolved == 0:
                        continue

                    # Try adding some absent triple
                    for add in absent[:50]:
                        # Check if adding add creates any new conflict
                        creates = 0
                        for ti in triple_to_trans[add]:
                            if cur_counts[ti] == 7:
                                creates += 1
                        if creates == 0:
                            # Safe to add, and we resolved conflicts
                            cur_state[rem] = False
                            cur_state[add] = True
                            for ti in triple_to_trans[rem]:
                                cur_counts[ti] -= 1
                                if cur_counts[ti] == 7:
                                    cur_conf -= 1
                            for ti in triple_to_trans[add]:
                                cur_counts[ti] += 1
                            if cur_count > best_count:
                                best_count = cur_count
                                best_state = cur_state.copy()
                            improved = True
                            break
                    if improved:
                        break

                if not improved:
                    break

        return cur_state, cur_conf, cur_count

    # Initialization: Start with all triples present
    init_state = [True] * num_triples
    init_counts = [8] * num_trans
    init_conf = num_trans
    init_count = num_triples

    # Run optimization from full set
    optimize_v2(init_state, init_counts, init_conf, init_count)

    # Restart with perturbations
    for restart in range(20):
        if time.time() > deadline - 5.0:
            break

        if best_state is None:
            break

        perturbed_state = best_state.copy()
        perturbed_counts = [0] * num_trans
        perturbed_conf = 0
        perturbed_count = 0

        # Recalculate counts for perturbed state
        for i in range(num_triples):
            if perturbed_state[i]:
                perturbed_count += 1
                for ti in triple_to_trans[i]:
                    perturbed_counts[ti] += 1

        for ti in range(num_trans):
            if perturbed_counts[ti] == 8:
                perturbed_conf += 1

        # Remove some random triples to create space
        num_removals = random.randint(15, 60)
        present_indices = [i for i in range(num_triples) if perturbed_state[i]]
        random.shuffle(present_indices)
        for i in present_indices[:num_removals]:
            perturbed_state[i] = False
            perturbed_count -= 1
            for ti in triple_to_trans[i]:
                perturbed_counts[ti] -= 1

        # Recalculate conflicts after removal
        perturbed_conf = 0
        for ti in range(num_trans):
            if perturbed_counts[ti] == 8:
                perturbed_conf += 1

        # Optimize
        optimize_v2(perturbed_state, perturbed_counts, perturbed_conf, perturbed_count)

    if best_state is None:
        return []

    result = [list(all_triples[i]) for i in range(num_triples) if best_state[i]]
    return result
