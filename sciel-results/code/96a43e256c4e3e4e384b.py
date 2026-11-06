"""Max K(2,2,2)-free triple system on 16 points via simulated annealing + greedy repair."""
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    num_triples = len(all_triples)  # 560
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}

    # Precompute all configurations (sets of 6 points partitioned into 3 pairs)
    # and which triples belong to which configurations.
    # A configuration is "satisfied" (violating) if all 8 transversal triples are present.
    # We maintain config_count[c] = number of transversals present for config c.
    # A triple can be added only if for all configs containing it, config_count[c] < 7.

    # Hardcoded 15 pairings of 6 indices into 3 pairs
    pairings = [
        [(0,1),(2,3),(4,5)],
        [(0,1),(2,4),(3,5)],
        [(0,1),(2,5),(3,4)],
        [(0,2),(1,3),(4,5)],
        [(0,2),(1,4),(3,5)],
        [(0,2),(1,5),(3,4)],
        [(0,3),(1,2),(4,5)],
        [(0,3),(1,4),(2,5)],
        [(0,3),(1,5),(2,4)],
        [(0,4),(1,2),(3,5)],
        [(0,4),(1,3),(2,5)],
        [(0,4),(1,5),(2,3)],
        [(0,5),(1,2),(3,4)],
        [(0,5),(1,3),(2,4)],
        [(0,5),(1,4),(2,3)]
    ]

    # triple_configs[i] = list of config indices that contain triple i
    triple_configs = [[] for _ in range(num_triples)]
    config_triples = []  # For each config, list of the 8 triple indices

    for vertices in itertools.combinations(range(N), 6):
        for p in pairings:
            a1, a2 = vertices[p[0][0]], vertices[p[0][1]]
            b1, b2 = vertices[p[1][0]], vertices[p[1][1]]
            c1, c2 = vertices[p[2][0]], vertices[p[2][1]]

            t_indices = []
            for a in (a1, a2):
                for b in (b1, b2):
                    for c in (c1, c2):
                        t = tuple(sorted((a, b, c)))
                        idx = triple_to_idx[t]
                        t_indices.append(idx)

            config_idx = len(config_triples)
            config_triples.append(t_indices)
            for idx in t_indices:
                triple_configs[idx].append(config_idx)

    num_configs = len(config_triples)

    def can_add(i, config_counts):
        for c in triple_configs[i]:
            if config_counts[c] == 7:
                return False
        return True

    def do_add(i, in_set, config_counts):
        in_set[i] = True
        for c in triple_configs[i]:
            config_counts[c] += 1

    def do_remove(i, in_set, config_counts):
        in_set[i] = False
        for c in triple_configs[i]:
            config_counts[c] -= 1

    def greedy_init(seed):
        random.seed(seed)
        in_set = [False] * num_triples
        config_counts = [0] * num_configs
        indices = list(range(num_triples))
        random.shuffle(indices)
        for i in indices:
            if can_add(i, config_counts):
                do_add(i, in_set, config_counts)
        return in_set, config_counts

    def simulated_annealing(in_set, config_counts, time_limit):
        current_size = sum(in_set)
        best_size = current_size
        best_state = in_set[:]
        best_counts = config_counts[:]

        end_time = time.time() + time_limit
        temp = 5.0
        cooling = 0.9995

        while time.time() < end_time:
            # Try a random move: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.4:
                # Try to add a random missing triple
                missing = [i for i in range(num_triples) if not in_set[i]]
                if not missing:
                    continue
                i = random.choice(missing)
                if can_add(i, config_counts):
                    do_add(i, in_set, config_counts)
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_state = in_set[:]
                        best_counts = config_counts[:]
                else:
                    # Can't add, so try removing a random present triple
                    present = [i for i in range(num_triples) if in_set[i]]
                    if present:
                        i_out = random.choice(present)
                        do_remove(i_out, in_set, config_counts)
                        current_size -= 1
                        # Try to add back or add new
                        missing2 = [i for i in range(num_triples) if not in_set[i]]
                        if missing2:
                            i_in = random.choice(missing2)
                            if can_add(i_in, config_counts):
                                do_add(i_in, in_set, config_counts)
                                current_size += 1
                                if current_size > best_size:
                                    best_size = current_size
                                    best_state = in_set[:]
                                    best_counts = config_counts[:]
            elif move_type < 0.7:
                # Try to remove a random present triple and add a random missing one
                present = [i for i in range(num_triples) if in_set[i]]
                missing = [i for i in range(num_triples) if not in_set[i]]
                if not present or not missing:
                    continue
                i_out = random.choice(present)
                i_in = random.choice(missing)
                do_remove(i_out, in_set, config_counts)
                current_size -= 1
                if can_add(i_in, config_counts):
                    do_add(i_in, in_set, config_counts)
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_state = in_set[:]
                        best_counts = config_counts[:]
                else:
                    # Revert
                    do_add(i_out, in_set, config_counts)
                    current_size += 1
            else:
                # Try a 2-for-1 or 1-for-2 swap
                present = [i for i in range(num_triples) if in_set[i]]
                missing = [i for i in range(num_triples) if not in_set[i]]
                if len(present) < 2 or not missing:
                    continue
                
                # Remove two random present triples
                i_out1, i_out2 = random.sample(present, 2)
                do_remove(i_out1, in_set, config_counts)
                do_remove(i_out2, in_set, config_counts)
                current_size -= 2
                
                # Try to add up to 3 missing triples
                added = []
                for _ in range(3):
                    if not missing:
                        break
                    i_in = random.choice(missing)
                    if can_add(i_in, config_counts):
                        do_add(i_in, in_set, config_counts)
                        added.append(i_in)
                        current_size += 1
                        missing.remove(i_in)
                
                if current_size > best_size:
                    best_size = current_size
                    best_state = in_set[:]
                    best_counts = config_counts[:]
                else:
                    # Revert
                    for i_in in added:
                        do_remove(i_in, in_set, config_counts)
                        current_size -= 1
                    do_add(i_out1, in_set, config_counts)
                    do_add(i_out2, in_set, config_counts)
                    current_size += 2

            temp *= cooling
            if temp < 0.01:
                temp = 0.01

        return best_state, best_counts, best_size

    def local_search(in_set, config_counts, time_limit):
        current_size = sum(in_set)
        best_size = current_size
        best_state = in_set[:]
        best_counts = config_counts[:]

        end_time = time.time() + time_limit

        while time.time() < end_time:
            improved = False

            # Phase 1: Try to add any single missing triple
            missing = [i for i in range(num_triples) if not in_set[i]]
            if not missing:
                break
            random.shuffle(missing)

            for i in missing:
                if can_add(i, config_counts):
                    do_add(i, in_set, config_counts)
                    current_size += 1
                    improved = True
                    break

            if improved:
                if current_size > best_size:
                    best_size = current_size
                    best_state = in_set[:]
                    best_counts = config_counts[:]
                continue

            # Phase 2: Try 1-out, k-in swaps
            present = [i for i in range(num_triples) if in_set[i]]
            if not present:
                break

            i_out = random.choice(present)
            do_remove(i_out, in_set, config_counts)
            current_size -= 1

            added_count = 0
            added_indices = []

            missing = [i for i in range(num_triples) if not in_set[i]]
            random.shuffle(missing)

            for i in missing:
                if can_add(i, config_counts):
                    do_add(i, in_set, config_counts)
                    added_count += 1
                    added_indices.append(i)
                    current_size += 1

            if added_count > 1:
                improved = True
                if current_size > best_size:
                    best_size = current_size
                    best_state = in_set[:]
                    best_counts = config_counts[:]
            else:
                # Revert
                for i in added_indices:
                    do_remove(i, in_set, config_counts)
                    current_size -= 1
                do_add(i_out, in_set, config_counts)
                current_size += 1

            # Phase 3: Try 2-out, k-in swaps (occasionally)
            if not improved and random.random() < 0.1:
                if len(present) >= 2:
                    i_out1, i_out2 = random.sample(present, 2)
                    do_remove(i_out1, in_set, config_counts)
                    do_remove(i_out2, in_set, config_counts)
                    current_size -= 2

                    added_count = 0
                    added_indices = []
                    missing = [i for i in range(num_triples) if not in_set[i]]
                    random.shuffle(missing)

                    for i in missing:
                        if can_add(i, config_counts):
                            do_add(i, in_set, config_counts)
                            added_count += 1
                            added_indices.append(i)
                            current_size += 1

                    if added_count > 2:
                        improved = True
                        if current_size > best_size:
                            best_size = current_size
                            best_state = in_set[:]
                            best_counts = config_counts[:]
                    else:
                        for i in added_indices:
                            do_remove(i, in_set, config_counts)
                            current_size -= 1
                        do_add(i_out1, in_set, config_counts)
                        do_add(i_out2, in_set, config_counts)
                        current_size += 2

        return best_state, best_counts, best_size

    # Main loop: Random Restarts with simulated annealing and local search
    global_best_size = 0
    global_best_state = None
    global_best_counts = None

    num_restarts = 8
    time_per_restart = (deadline - time.time()) / num_restarts

    for r in range(num_restarts):
        if time.time() > deadline:
            break

        seed = 42 + r
        in_set, config_counts = greedy_init(seed)

        # Adjust time for local search to not exceed global deadline
        remaining = deadline - time.time()
        # Leave a small buffer
        search_time = min(time_per_restart, remaining - 1)

        if search_time <= 0:
            break

        # Alternate between simulated annealing and local search
        if r % 2 == 0:
            best_state, best_counts, best_size = simulated_annealing(in_set, config_counts, search_time)
        else:
            best_state, best_counts, best_size = local_search(in_set, config_counts, search_time)

        if best_size > global_best_size:
            global_best_size = best_size
            global_best_state = best_state
            global_best_counts = best_counts

    if global_best_state is None:
        # Fallback if something went wrong
        in_set, config_counts = greedy_init(42)
        global_best_state = in_set
        global_best_counts = config_counts
        global_best_size = sum(in_set)

    return [list(all_triples[i]) for i in range(num_triples) if global_best_state[i]]
