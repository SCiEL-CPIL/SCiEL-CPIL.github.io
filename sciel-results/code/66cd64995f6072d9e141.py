import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill

    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)  # 560
    triple_index = {t: i for i, t in enumerate(all_triples)}

    # Enumerate all K(2,2,2) forbidden configurations
    forbidden_configs = []

    for six_points in itertools.combinations(range(n), 6):
        p0, p1, p2, p3, p4, p5 = six_points
        pairings = [
            [(p0,p1),(p2,p3),(p4,p5)],
            [(p0,p1),(p2,p4),(p3,p5)],
            [(p0,p1),(p2,p5),(p3,p4)],
            [(p0,p2),(p1,p3),(p4,p5)],
            [(p0,p2),(p1,p4),(p3,p5)],
            [(p0,p2),(p1,p5),(p3,p4)],
            [(p0,p3),(p1,p2),(p4,p5)],
            [(p0,p3),(p1,p4),(p2,p5)],
            [(p0,p3),(p1,p5),(p2,p4)],
            [(p0,p4),(p1,p2),(p3,p5)],
            [(p0,p4),(p1,p3),(p2,p5)],
            [(p0,p4),(p1,p5),(p2,p3)],
            [(p0,p5),(p1,p2),(p3,p4)],
            [(p0,p5),(p1,p3),(p2,p4)],
            [(p0,p5),(p1,p4),(p2,p3)],
        ]

        for pairs in pairings:
            (a1,b1), (a2,b2), (a3,b3) = pairs
            transversals = []
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        t = tuple(sorted([(a1,b1)[i], (a2,b2)[j], (a3,b3)[k]]))
                        if t in triple_index:
                            transversals.append(triple_index[t])

            if len(transversals) == 8:
                forbidden_configs.append(frozenset(transversals))

    forbidden_configs = list(set(forbidden_configs))
    num_configs = len(forbidden_configs)

    # Build incidence structure
    triple_to_configs = [[] for _ in range(num_triples)]
    config_to_triples = [list(c) for c in forbidden_configs]
    for ci, config in enumerate(forbidden_configs):
        for t in config:
            triple_to_configs[t].append(ci)

    def greedy_delete():
        """Start with all triples, remove those hitting most active configs."""
        present = [True] * num_triples
        active = [True] * num_configs
        triple_active_count = [0] * num_triples
        for t in range(num_triples):
            for ci in triple_to_configs[t]:
                if active[ci]:
                    triple_active_count[t] += 1

        num_active = num_configs
        while num_active > 0:
            if time.time() > deadline - 10:
                break
            max_count = -1
            candidates = []
            for t in range(num_triples):
                if present[t]:
                    if triple_active_count[t] > max_count:
                        max_count = triple_active_count[t]
                        candidates = [t]
                    elif triple_active_count[t] == max_count:
                        candidates.append(t)
            if not candidates:
                break
            best_t = random.choice(candidates)
            present[best_t] = False
            for ci in triple_to_configs[best_t]:
                if active[ci]:
                    active[ci] = False
                    num_active -= 1
                    for t_other in config_to_triples[ci]:
                        if t_other != best_t and present[t_other]:
                            triple_active_count[t_other] -= 1
        return present

    def greedy_add_random():
        """Greedy add in random order."""
        present = [False] * num_triples
        config_count = [0] * num_configs
        order = list(range(num_triples))
        random.shuffle(order)
        for t in order:
            if time.time() > deadline - 10:
                break
            can_add = True
            for ci in triple_to_configs[t]:
                if config_count[ci] >= 7:
                    can_add = False
                    break
            if can_add:
                present[t] = True
                for ci in triple_to_configs[t]:
                    config_count[ci] += 1
        return present

    def local_search(present, max_iters=200):
        """Local search with add, 1-swap, and 2-swap moves using unsafe_count."""
        config_count = [0] * num_configs
        for t in range(num_triples):
            if present[t]:
                for ci in triple_to_configs[t]:
                    config_count[ci] += 1

        unsafe_count = [0] * num_triples
        for t in range(num_triples):
            if not present[t]:
                for ci in triple_to_configs[t]:
                    if config_count[ci] >= 7:
                        unsafe_count[t] += 1

        improved = True
        stall = 0
        it = 0
        while improved and it < max_iters and time.time() < deadline - 10:
            improved = False
            it += 1

            # Try add: pick absent triple with unsafe_count 0, prefer fewer blocked configs
            added = False
            best_t = -1
            best_score = -1
            for _ in range(200):
                t = random.randint(0, num_triples - 1)
                if present[t] or unsafe_count[t] != 0:
                    continue
                # score: fewer configs at count 6 (i.e., less likely to block others)
                score = 0
                for ci in triple_to_configs[t]:
                    if config_count[ci] == 6:
                        score -= 1
                if score > best_score:
                    best_score = score
                    best_t = t
            if best_t >= 0:
                t = best_t
                present[t] = True
                for ci in triple_to_configs[t]:
                    old = config_count[ci]
                    config_count[ci] = old + 1
                    if old == 6:
                        for t_other in config_to_triples[ci]:
                            if t_other != t and not present[t_other]:
                                unsafe_count[t_other] += 1
                improved = True
                added = True

            if added:
                stall = 0
                continue

            # Try 1-swap
            swapped = False
            present_list = [t for t in range(num_triples) if present[t]]
            absent_safe = [t for t in range(num_triples) if not present[t] and unsafe_count[t] == 0]
            if present_list and absent_safe:
                for _ in range(100):
                    t_rem = random.choice(present_list)
                    t_add = random.choice(absent_safe)
                    # Check if t_add is safe after removing t_rem
                    safe = True
                    for ci in triple_to_configs[t_add]:
                        cnt = config_count[ci]
                        if t_rem in config_to_triples[ci] and ci in triple_to_configs[t_rem]:
                            cnt -= 1
                        if cnt >= 7:
                            safe = False
                            break
                    if safe:
                        # Perform swap
                        present[t_rem] = False
                        present[t_add] = True
                        for ci in triple_to_configs[t_rem]:
                            old = config_count[ci]
                            config_count[ci] = old - 1
                            if old == 7:
                                for t_other in config_to_triples[ci]:
                                    if t_other != t_rem and not present[t_other]:
                                        unsafe_count[t_other] -= 1
                        for ci in triple_to_configs[t_add]:
                            old = config_count[ci]
                            config_count[ci] = old + 1
                            if old == 6:
                                for t_other in config_to_triples[ci]:
                                    if t_other != t_add and not present[t_other]:
                                        unsafe_count[t_other] += 1
                        improved = True
                        swapped = True
                        break

            if swapped:
                stall = 0
                continue

            # Try 2-swap: remove two present, add two absent
            two_swapped = False
            if len(present_list) >= 2 and len(absent_safe) >= 2:
                for _ in range(50):
                    t_rem1 = random.choice(present_list)
                    t_rem2 = random.choice(present_list)
                    if t_rem1 == t_rem2:
                        continue
                    t_add1 = random.choice(absent_safe)
                    t_add2 = random.choice(absent_safe)
                    if t_add1 == t_add2:
                        continue
                    if t_add1 in (t_rem1, t_rem2) or t_add2 in (t_rem1, t_rem2):
                        continue
                    # Simulate config counts after removing both
                    temp_cc = config_count[:]
                    for ci in triple_to_configs[t_rem1]:
                        temp_cc[ci] -= 1
                    for ci in triple_to_configs[t_rem2]:
                        temp_cc[ci] -= 1
                    # Check if both adds are safe
                    safe = True
                    for t_add in (t_add1, t_add2):
                        for ci in triple_to_configs[t_add]:
                            if temp_cc[ci] >= 7:
                                safe = False
                                break
                        if not safe:
                            break
                    if safe:
                        # Perform 2-swap
                        for t_rem in (t_rem1, t_rem2):
                            present[t_rem] = False
                            for ci in triple_to_configs[t_rem]:
                                old = config_count[ci]
                                config_count[ci] = old - 1
                                if old == 7:
                                    for t_other in config_to_triples[ci]:
                                        if t_other != t_rem and not present[t_other]:
                                            unsafe_count[t_other] -= 1
                        for t_add in (t_add1, t_add2):
                            present[t_add] = True
                            for ci in triple_to_configs[t_add]:
                                old = config_count[ci]
                                config_count[ci] = old + 1
                                if old == 6:
                                    for t_other in config_to_triples[ci]:
                                        if t_other != t_add and not present[t_other]:
                                            unsafe_count[t_other] += 1
                        improved = True
                        two_swapped = True
                        break

            if two_swapped:
                stall = 0
                continue

            stall += 1
            if stall > 40:
                break

        return present

    def polish(present):
        """Systematic add of all safe triples."""
        config_count = [0] * num_configs
        for t in range(num_triples):
            if present[t]:
                for ci in triple_to_configs[t]:
                    config_count[ci] += 1
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for t in range(num_triples):
                if present[t]:
                    continue
                can_add = True
                for ci in triple_to_configs[t]:
                    if config_count[ci] >= 7:
                        can_add = False
                        break
                if can_add:
                    present[t] = True
                    for ci in triple_to_configs[t]:
                        config_count[ci] += 1
                    improved = True
        return present

    best_present = None
    best_count = 0

    # Deterministic baseline: greedy delete
    det_present = greedy_delete()
    det_present = local_search(det_present, max_iters=150)
    det_present = polish(det_present)
    det_count = sum(det_present)
    if det_count > best_count:
        best_count = det_count
        best_present = det_present[:]

    # Randomized restarts
    num_restarts = 0
    while time.time() < deadline - 20:
        # Alternate between greedy-delete and greedy-add-random seeds
        if num_restarts % 2 == 0:
            present = greedy_delete()
        else:
            present = greedy_add_random()

        present = local_search(present, max_iters=150)
        present = polish(present)

        count = sum(present)
        if count > best_count:
            best_count = count
            best_present = present[:]

        num_restarts += 1
        if num_restarts > 200:
            break

    # Final polish on best
    if best_present is not None:
        best_present = polish(best_present)
        best_count = sum(best_present)

    if best_present is None:
        return []

    return [list(all_triples[i]) for i in range(num_triples) if best_present[i]]
