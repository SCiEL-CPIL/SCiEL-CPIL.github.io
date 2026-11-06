"""
Maximize number of triples in a 3-uniform hypergraph on 16 vertices
such that no three disjoint pairs have all 8 transversal triples present (K(2,2,2)-free).

Combined strategy:
1. Precompute all K(2,2,2) configurations and their constituent triples.
2. Greedy initialization with randomization.
3. Local search with:
   - Fast add attempts (try to add a random absent triple)
   - 1-out-k-in swaps (remove one, try to add up to k others)
   - 2-out-3-in swaps for deeper local optima
   - Periodic fluctuation (randomly remove a few triples) to escape local optima
4. Multiple restarts within time budget.
"""

import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill

    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_total = len(all_triples)  # 560

    # Precompute all K(2,2,2) configurations
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

    def time_check():
        return time.time() >= deadline

    def can_add_fast(ti, k222_count):
        for ci in triple_to_k222[ti]:
            if k222_count[ci] >= 7:
                return False
        return True

    def do_add(ti, present, k222_count):
        present[ti] = True
        for ci in triple_to_k222[ti]:
            k222_count[ci] += 1

    def do_remove(ti, present, k222_count):
        present[ti] = False
        for ci in triple_to_k222[ti]:
            k222_count[ci] -= 1

    best_result = []
    best_count = 0

    # Greedy initialization
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

    # Local search with efficient moves and fluctuation
    def local_search(present, k222_count, count, rng, max_steps=200000):
        steps = 0
        no_improve = 0
        max_no_improve = 3000
        fluctuate_every = 5000

        while steps < max_steps and no_improve < max_no_improve:
            if time_check():
                break

            # Periodic fluctuation: randomly remove a few triples to escape local optima
            if steps > 0 and steps % fluctuate_every == 0:
                num_remove = rng.randint(3, 8)
                present_indices = [i for i in range(num_total) if present[i]]
                if len(present_indices) > num_remove:
                    to_remove = rng.sample(present_indices, num_remove)
                    for ti in to_remove:
                        do_remove(ti, present, k222_count)
                        count -= 1
                no_improve = 0
                steps += 1
                continue

            moved = False

            # Phase 1: Try to add triples
            for attempt in range(40):
                ti = rng.randint(0, num_total - 1)
                if present[ti]:
                    continue
                if can_add_fast(ti, k222_count):
                    do_add(ti, present, k222_count)
                    count += 1
                    moved = True
                    no_improve = 0
                    break

            if moved:
                steps += 1
                continue

            # Phase 2: Try 1-out-k-in swaps
            swap_done = False
            for attempt in range(20):
                ti_remove = rng.randint(0, num_total - 1)
                if not present[ti_remove]:
                    continue

                # Temporarily remove
                do_remove(ti_remove, present, k222_count)
                count -= 1

                # Try to add up to 2 triples
                added = 0
                for add_attempt in range(300):
                    ti_candidate = rng.randint(0, num_total - 1)
                    if present[ti_candidate]:
                        continue
                    if can_add_fast(ti_candidate, k222_count):
                        do_add(ti_candidate, present, k222_count)
                        count += 1
                        added += 1
                        if added >= 2:
                            break

                if added >= 1:
                    swap_done = True
                    no_improve = 0
                else:
                    # Revert removal
                    do_add(ti_remove, present, k222_count)
                    count += 1

            if swap_done:
                steps += 1
                continue

            # Phase 3: Try 2-out-3-in swaps (deeper)
            deep_swap_done = False
            if no_improve > 500:
                for attempt in range(10):
                    # Pick two random present triples to remove
                    r1, r2 = rng.sample(range(num_total), 2)
                    if not present[r1] or not present[r2]:
                        continue

                    # Remove both
                    do_remove(r1, present, k222_count)
                    count -= 1
                    do_remove(r2, present, k222_count)
                    count -= 1

                    # Try to add up to 3 triples
                    added = 0
                    for add_attempt in range(500):
                        ti_candidate = rng.randint(0, num_total - 1)
                        if present[ti_candidate]:
                            continue
                        if can_add_fast(ti_candidate, k222_count):
                            do_add(ti_candidate, present, k222_count)
                            count += 1
                            added += 1
                            if added >= 3:
                                break

                    if added >= 2:
                        deep_swap_done = True
                        no_improve = 0
                    else:
                        # Revert
                        do_add(r1, present, k222_count)
                        count += 1
                        do_add(r2, present, k222_count)
                        count += 1

            if deep_swap_done:
                steps += 1
                continue

            # Stuck
            no_improve += 1
            steps += 1

        return present, count

    # Main loop with multiple restarts
    rng = random.Random(42)

    num_restarts = 0
    while not time_check() and num_restarts < 500:
        num_restarts += 1

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
