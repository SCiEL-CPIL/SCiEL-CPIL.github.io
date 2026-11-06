"""
Maximize number of triples in a 3-uniform hypergraph on 16 vertices
such that no three disjoint pairs have all 8 transversal triples present.

Strategy (Max-SAT / WalkSAT-style with conflict-driven moves):
1. Precompute all K(2,2,2) configurations and for each triple, the list of
   configs it participates in.
2. Greedy random initialization.
3. WalkSAT-style local search with:
   - Conflict-driven move selection: identify which present triples block
     each absent triple, prefer removing those.
   - Targeted 1-out-k-in swaps: after removal, specifically try to add
     the previously blocked triples.
   - Fluctuation phases to escape local optima.
4. Multiple restarts within a 115s deadline.
"""

import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115

    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_total = len(all_triples)  # 560

    # Precompute all K(2,2,2) configs
    k222_configs = []
    triple_to_k222 = [[] for _ in range(num_total)]

    for six_vertices in itertools.combinations(range(N), 6):
        a, b, c, d, e, f = six_vertices
        partitions = [
            ((a,b),(c,d),(e,f)),
            ((a,b),(c,e),(d,f)),
            ((a,b),(c,f),(d,e)),
            ((a,c),(b,d),(e,f)),
            ((a,c),(b,e),(d,f)),
            ((a,c),(b,f),(d,e)),
            ((a,d),(b,c),(e,f)),
            ((a,d),(b,e),(c,f)),
            ((a,d),(b,f),(c,e)),
            ((a,e),(b,c),(d,f)),
            ((a,e),(b,d),(c,f)),
            ((a,e),(b,f),(c,d)),
            ((a,f),(b,c),(d,e)),
            ((a,f),(b,d),(c,e)),
            ((a,f),(b,e),(c,d)),
        ]
        for p1, p2, p3 in partitions:
            transversals = []
            for x in p1:
                for y in p2:
                    for z in p3:
                        t = tuple(sorted((x, y, z)))
                        transversals.append(t)
            trans_indices = frozenset(triple_to_idx[t] for t in transversals)
            k222_configs.append(trans_indices)
            config_idx = len(k222_configs) - 1
            for ti in trans_indices:
                triple_to_k222[ti].append(config_idx)

    num_k222 = len(k222_configs)

    # Convert to numpy arrays for faster operations
    triple_to_k222_np = [np.array(lst, dtype=np.int32) for lst in triple_to_k222]

    def time_check():
        return time.time() >= deadline

    best_result = []
    best_count = 0

    def can_add_fast(ti, k222_count):
        """Check if triple ti can be added given current k222_count."""
        for ci in triple_to_k222[ti]:
            if k222_count[ci] >= 7:
                return False
        return True

    def get_blockers(ti, k222_count, present):
        """Get list of present triples that block ti from being added."""
        blockers = []
        for ci in triple_to_k222[ti]:
            if k222_count[ci] >= 7:
                # This config is tight; find which present triples in it block us
                for other_ti in k222_configs[ci]:
                    if other_ti != ti and present[other_ti]:
                        blockers.append(other_ti)
        return blockers

    def do_add(ti, present, k222_count):
        present[ti] = True
        for ci in triple_to_k222[ti]:
            k222_count[ci] += 1

    def do_remove(ti, present, k222_count):
        present[ti] = False
        for ci in triple_to_k222[ti]:
            k222_count[ci] -= 1

    def greedy_init(rng):
        present = [False] * num_total
        k222_count = [0] * num_k222
        count = 0
        order = list(range(num_total))
        rng.shuffle(order)
        for ti in order:
            if can_add_fast(ti, k222_count):
                do_add(ti, present, k222_count)
                count += 1
        return present, k222_count, count

    def local_search(present, k222_count, count, rng, max_steps=300000):
        steps = 0
        no_improve = 0
        max_no_improve = 15000
        fluctuate_every = 5000

        # Maintain list of absent triples for faster sampling
        absent_list = [i for i in range(num_total) if not present[i]]
        present_list = [i for i in range(num_total) if present[i]]

        while steps < max_steps and no_improve < max_no_improve:
            if time_check():
                break

            # Periodic fluctuation
            if steps > 0 and steps % fluctuate_every == 0:
                num_remove = rng.randint(3, 8)
                if len(present_list) > num_remove:
                    to_remove = rng.sample(present_list, num_remove)
                    for ti in to_remove:
                        do_remove(ti, present, k222_count)
                        count -= 1
                        present_list.remove(ti)
                        absent_list.append(ti)
                no_improve = 0
                steps += 1
                continue

            moved = False

            # 1. Try to add a triple from absent list
            if absent_list:
                for _ in range(40):
                    idx = rng.randrange(len(absent_list))
                    ti = absent_list[idx]
                    if present[ti]:
                        continue
                    if can_add_fast(ti, k222_count):
                        do_add(ti, present, k222_count)
                        count += 1
                        absent_list.pop(idx)
                        present_list.append(ti)
                        moved = True
                        no_improve = 0
                        break

            if moved:
                steps += 1
                continue

            # 2. Conflict-driven swap: pick an absent triple, find its blockers,
            #    remove one blocker, then try to add the original triple + others
            swapped = False
            if absent_list and present_list:
                for _ in range(50):
                    # Pick a random absent triple
                    idx = rng.randrange(len(absent_list))
                    ti_target = absent_list[idx]
                    if present[ti_target]:
                        continue

                    # Find blockers
                    blockers = []
                    for ci in triple_to_k222[ti_target]:
                        if k222_count[ci] >= 7:
                            for other_ti in k222_configs[ci]:
                                if other_ti != ti_target and present[other_ti]:
                                    blockers.append(other_ti)

                    if not blockers:
                        continue

                    # Pick a blocker to remove - prefer ones in many tight configs
                    rng.shuffle(blockers)
                    best_blocker = None
                    best_score = -1
                    for b in blockers[:10]:  # limit candidates
                        score = sum(1 for ci in triple_to_k222[b] if k222_count[ci] == 7)
                        if score > best_score:
                            best_score = score
                            best_blocker = b

                    if best_blocker is None:
                        continue

                    # Remove the blocker
                    do_remove(best_blocker, present, k222_count)
                    count -= 1
                    try:
                        present_list.remove(best_blocker)
                    except ValueError:
                        pass
                    absent_list.append(best_blocker)

                    # Try to add the target triple first
                    added_count = 0
                    if can_add_fast(ti_target, k222_count):
                        do_add(ti_target, present, k222_count)
                        count += 1
                        added_count += 1
                        try:
                            absent_list.remove(ti_target)
                        except ValueError:
                            pass
                        present_list.append(ti_target)

                    # Try to add more triples
                    for _ in range(100):
                        if not absent_list:
                            break
                        idx2 = rng.randrange(len(absent_list))
                        ti_add = absent_list[idx2]
                        if present[ti_add]:
                            continue
                        if can_add_fast(ti_add, k222_count):
                            do_add(ti_add, present, k222_count)
                            count += 1
                            added_count += 1
                            absent_list.pop(idx2)
                            present_list.append(ti_add)
                            if added_count >= 2:
                                break

                    if added_count >= 1:
                        swapped = True
                        no_improve = 0
                    else:
                        # Revert
                        do_add(best_blocker, present, k222_count)
                        count += 1
                        try:
                            present_list.remove(best_blocker)
                        except ValueError:
                            pass
                        present_list.append(best_blocker)
                        absent_list.remove(best_blocker)

            if swapped:
                steps += 1
                continue

            no_improve += 1
            steps += 1

        return present, count

    rng = random.Random(42)

    for restart in range(500):
        if time_check():
            break

        present, k222_count, count = greedy_init(rng)

        if count > best_count:
            best_count = count
            best_result = [list(all_triples[i]) for i in range(num_total) if present[i]]

        present, count = local_search(present, k222_count, count, rng)

        if count > best_count:
            best_count = count
            best_result = [list(all_triples[i]) for i in range(num_total) if present[i]]

        if time_check():
            break

    if not best_result:
        best_result = [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]

    return best_result
