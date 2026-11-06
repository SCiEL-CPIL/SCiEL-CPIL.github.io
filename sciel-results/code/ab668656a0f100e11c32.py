import time
import random
import itertools
import numpy as np


def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 120s limit

    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    num_triples = len(all_triples)

    triple_to_idx = {t: i for i, t in enumerate(all_triples)}

    # Precompute all K(2,2,2) configurations
    configs = []
    config_id_map = {}
    triples_in_config = []

    points = range(N)
    for pts in itertools.combinations(points, 6):
        p0, p1, p2, p3, p4, p5 = pts
        pairs = [
            [(p0, p1), (p2, p3), (p4, p5)],
            [(p0, p1), (p2, p4), (p3, p5)],
            [(p0, p1), (p2, p5), (p3, p4)],
            [(p0, p2), (p1, p3), (p4, p5)],
            [(p0, p2), (p1, p4), (p3, p5)],
            [(p0, p2), (p1, p5), (p3, p4)],
            [(p0, p3), (p1, p2), (p4, p5)],
            [(p0, p3), (p1, p4), (p2, p5)],
            [(p0, p3), (p1, p5), (p2, p4)],
            [(p0, p4), (p1, p2), (p3, p5)],
            [(p0, p4), (p1, p3), (p2, p5)],
            [(p0, p4), (p1, p5), (p2, p3)],
            [(p0, p5), (p1, p2), (p3, p4)],
            [(p0, p5), (p1, p3), (p2, p4)],
            [(p0, p5), (p1, p4), (p2, p3)]
        ]
        for pair_set in pairs:
            norm_pairs = tuple(sorted(tuple(sorted(p)) for p in pair_set))
            if norm_pairs not in config_id_map:
                cid = len(configs)
                config_id_map[norm_pairs] = cid
                configs.append(norm_pairs)

                (a, b), (c, d), (e, f) = norm_pairs
                tris = []
                for x in (a, b):
                    for y in (c, d):
                        for z in (e, f):
                            t = tuple(sorted((x, y, z)))
                            tris.append(triple_to_idx[t])
                triples_in_config.append(tris)

    num_configs = len(configs)

    triple_to_configs = [[] for _ in range(num_triples)]
    for cid, tris in enumerate(triples_in_config):
        for t_idx in tris:
            triple_to_configs[t_idx].append(cid)

    present = np.zeros(num_triples, dtype=bool)
    config_counts = np.zeros(num_configs, dtype=np.int8)

    def add_triple(t_idx):
        present[t_idx] = True
        for cid in triple_to_configs[t_idx]:
            config_counts[cid] += 1
            if config_counts[cid] == 8:
                return False
        return True

    def remove_triple(t_idx):
        present[t_idx] = False
        for cid in triple_to_configs[t_idx]:
            config_counts[cid] -= 1
        return True

    def is_valid_addition(t_idx):
        if present[t_idx]:
            return False
        for cid in triple_to_configs[t_idx]:
            if config_counts[cid] == 7:
                return False
        return True

    def greedy_construct():
        """Greedy construction with random ordering."""
        present[:] = False
        config_counts[:] = 0
        current_set = set()
        order = list(range(num_triples))
        random.shuffle(order)
        for t_idx in order:
            if is_valid_addition(t_idx):
                add_triple(t_idx)
                current_set.add(t_idx)
        return len(current_set), set(current_set)

    def local_search(initial_set, max_iters=50000):
        """
        Improved local search with multiple move types:
        - Single addition
        - Single removal + addition (swap)
        - Double removal + addition (2-out 1-in)
        - Double removal + double addition (2-out 2-in)
        - Random restarts
        """
        present[:] = False
        config_counts[:] = 0
        current_set = set(initial_set)

        count = len(current_set)
        best_count = count
        best_set_snapshot = set(current_set)

        # Precompute: for each triple, the list of configs it's in
        triple_config_list = triple_to_configs

        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break

            if count > best_count:
                best_count = count
                best_set_snapshot = set(current_set)

            move_type = random.random()

            if move_type < 0.4:
                # Try to add a random missing triple
                missing = [i for i in range(num_triples) if not present[i]]
                if not missing:
                    continue
                t_idx = random.choice(missing)
                if is_valid_addition(t_idx):
                    add_triple(t_idx)
                    current_set.add(t_idx)
                    count += 1
                else:
                    # Blocked - find conflicting configs
                    conflicts = [cid for cid in triple_config_list[t_idx]
                                 if config_counts[cid] == 7]
                    if conflicts:
                        # Try to remove one triple from a conflicting config
                        cid = random.choice(conflicts)
                        present_in_cid = [t for t in triples_in_config[cid]
                                          if present[t]]
                        if present_in_cid:
                            # Remove the most "expensive" triple (appears in most configs)
                            u_idx = max(present_in_cid,
                                        key=lambda x: len(triple_config_list[x]))
                            remove_triple(u_idx)
                            current_set.remove(u_idx)
                            count -= 1
                            if is_valid_addition(t_idx):
                                add_triple(t_idx)
                                current_set.add(t_idx)
                                count += 1

            elif move_type < 0.6:
                # 2-out 1-in move
                if len(current_set) >= 2:
                    to_remove = random.sample(list(current_set), 2)
                    for t in to_remove:
                        remove_triple(t)
                        current_set.remove(t)
                        count -= 1

                    missing = [i for i in range(num_triples) if not present[i]]
                    if missing:
                        # Try to add up to 2 triples to make it worth it
                        candidates = random.sample(missing, min(10, len(missing)))
                        added = []
                        for t_idx in candidates:
                            if is_valid_addition(t_idx):
                                add_triple(t_idx)
                                current_set.add(t_idx)
                                count += 1
                                added.append(t_idx)
                                if len(added) >= 2:
                                    break
                        # If we didn't add enough, restore
                        if len(added) < 2:
                            for t in to_remove:
                                add_triple(t)
                                current_set.add(t)
                                count += 1

            elif move_type < 0.7:
                # Random kick: remove 1-3 random triples, then greedy add
                kick_size = random.randint(1, 3)
                to_remove = random.sample(list(current_set),
                                          min(kick_size, len(current_set)))
                for t in to_remove:
                    remove_triple(t)
                    current_set.remove(t)
                    count -= 1

                # Greedy add back
                missing = [i for i in range(num_triples) if not present[i]]
                random.shuffle(missing)
                for t_idx in missing[:20]:
                    if is_valid_addition(t_idx):
                        add_triple(t_idx)
                        current_set.add(t_idx)
                        count += 1

            else:
                # Simulated annealing style: accept worse moves with probability
                if len(current_set) >= 1:
                    u_idx = random.choice(list(current_set))
                    remove_triple(u_idx)
                    current_set.remove(u_idx)
                    count -= 1

                    missing = [i for i in range(num_triples) if not present[i]]
                    if missing:
                        t_idx = random.choice(missing)
                        if is_valid_addition(t_idx):
                            add_triple(t_idx)
                            current_set.add(t_idx)
                            count += 1
                        else:
                            # Restore
                            add_triple(u_idx)
                            current_set.add(u_idx)
                            count += 1

        return best_count, list(best_set_snapshot)

    def get_initial_lines():
        """Affine lines in F2^4."""
        lines = []
        seen = set()
        for x in range(N):
            for y in range(N):
                if x == y:
                    continue
                z = x ^ y
                if z == x or z == y:
                    continue
                t = tuple(sorted((x, y, z)))
                if t not in seen:
                    seen.add(t)
                    lines.append(t)
        return lines

    # Get initial lines from affine geometry
    initial_lines = get_initial_lines()

    best_count = 0
    best_set = []

    # First, do a greedy construction
    count, current_set = greedy_construct()
    if count > best_count:
        best_count = count
        best_set = set(current_set)

    # Local search from greedy result
    count, s = local_search(list(current_set), max_iters=30000)
    if count > best_count:
        best_count = count
        best_set = set(s)

    # Local search from affine lines
    count, s = local_search(initial_lines, max_iters=30000)
    if count > best_count:
        best_count = count
        best_set = set(s)

    # Try more iterations with different random seeds
    num_extra_tries = 0
    while time.time() < deadline - 10 and num_extra_tries < 5:
        # Random greedy start
        present[:] = False
        config_counts[:] = 0
        current_set = set()
        order = list(range(num_triples))
        random.shuffle(order)
        for t_idx in order:
            if is_valid_addition(t_idx):
                add_triple(t_idx)
                current_set.add(t_idx)

        init_triples = [all_triples[i] for i in current_set]

        count2, s2 = local_search(init_triples, max_iters=20000)
        if count2 > best_count:
            best_count = count2
            best_set = set(s2)

        num_extra_tries += 1

    result = [all_triples[i] for i in best_set]

    return result
