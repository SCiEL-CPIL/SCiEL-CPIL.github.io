import itertools
import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s kill

    n = 16
    vertices = list(range(n))
    all_edges = list(itertools.combinations(vertices, 3))
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    num_edges = len(all_edges)  # 560

    # Precompute matchings of 6 vertices into 3 pairs
    matchings = [
        ((0,1),(2,3),(4,5)),
        ((0,1),(2,4),(3,5)),
        ((0,1),(2,5),(3,4)),
        ((0,2),(1,3),(4,5)),
        ((0,2),(1,4),(3,5)),
        ((0,2),(1,5),(3,4)),
        ((0,3),(1,2),(4,5)),
        ((0,3),(1,4),(2,5)),
        ((0,3),(1,5),(2,4)),
        ((0,4),(1,2),(3,5)),
        ((0,4),(1,3),(2,5)),
        ((0,4),(1,5),(2,3)),
        ((0,5),(1,2),(3,4)),
        ((0,5),(1,3),(2,4)),
        ((0,5),(1,4),(2,3)),
    ]

    # Build all configurations (sets of 8 transversal triples for 3 disjoint pairs)
    configs = []
    for subset in itertools.combinations(vertices, 6):
        v = list(subset)
        for m in matchings:
            p1 = (v[m[0][0]], v[m[0][1]])
            p2 = (v[m[1][0]], v[m[1][1]])
            p3 = (v[m[2][0]], v[m[2][1]])
            edges_in_config = []
            for x1 in p1:
                for x2 in p2:
                    for x3 in p3:
                        edge = tuple(sorted((x1, x2, x3)))
                        edges_in_config.append(edge_to_idx[edge])
            configs.append(tuple(edges_in_config))

    num_configs = len(configs)  # 120120

    # Build configs_in_edge: for each edge, list of config indices containing it
    configs_in_edge = [[] for _ in range(num_edges)]
    for c_idx, c in enumerate(configs):
        for e_idx in c:
            configs_in_edge[e_idx].append(c_idx)

    # Convert to numpy arrays for fast access
    configs_in_edge_np = [np.array(lst, dtype=np.int32) for lst in configs_in_edge]
    config_to_edges = [np.array(list(c), dtype=np.int32) for c in configs]

    # Precompute edge degree (number of configs each edge belongs to) for weighted greedy
    edge_degrees = np.array([len(lst) for lst in configs_in_edge], dtype=np.int32)

    def run_greedy(seed, time_budget):
        """
        Random greedy with multiple strategies: random order, degree-weighted, and mixed.
        """
        random.seed(seed)
        conf_count = np.zeros(num_configs, dtype=np.int32)
        selected = set()
        t_end = start_time + time_budget

        strategy = seed % 3
        if strategy == 0:
            order = list(range(num_edges))
            random.shuffle(order)
        elif strategy == 1:
            # Weight by inverse degree: prefer edges in fewer configs
            weights = 1.0 / (edge_degrees + 1.0)
            order = list(range(num_edges))
            random.shuffle(order)
            # Sort by weight with randomness
            order.sort(key=lambda e: weights[e] + random.random() * 0.1)
        else:
            # Mixed: half random, half degree-ordered
            order1 = list(range(num_edges))
            random.shuffle(order1)
            order2 = list(range(num_edges))
            order2.sort(key=lambda e: edge_degrees[e])
            order = order1[:280] + order2[280:]

        for e_idx in order:
            if time.time() > t_end:
                break

            c_indices = configs_in_edge_np[e_idx]
            if len(c_indices) == 0:
                selected.add(e_idx)
                continue

            if np.max(conf_count[c_indices]) < 7:
                selected.add(e_idx)
                conf_count[c_indices] += 1

        return selected

    def run_local_search(selected, seed, time_budget):
        """
        Local search with swap moves using numpy membership for speed.
        """
        random.seed(seed)
        conf_count = np.zeros(num_configs, dtype=np.int32)
        for e in selected:
            c_indices = configs_in_edge_np[e]
            if len(c_indices) > 0:
                conf_count[c_indices] += 1

        best_selected = set(selected)
        best_size = len(selected)
        no_improve_count = 0
        local_deadline = start_time + time_budget

        # Precompute numpy membership array
        selected_np = np.zeros(num_edges, dtype=bool)
        for e in best_selected:
            selected_np[e] = True

        while time.time() < local_deadline and no_improve_count < 50:
            no_improve_count += 1
            improved = False

            # Try random swaps - more iterations
            for _ in range(500):
                if time.time() > local_deadline:
                    break

                # Pick a random unselected edge
                unselected_mask = ~selected_np
                unselected_indices = np.where(unselected_mask)[0]
                if len(unselected_indices) == 0:
                    break

                add_e = int(random.choice(unselected_indices))
                c_indices_add = configs_in_edge_np[add_e]

                # Check if adding is possible directly
                if len(c_indices_add) == 0 or np.max(conf_count[c_indices_add]) < 7:
                    best_selected.add(add_e)
                    selected_np[add_e] = True
                    if len(c_indices_add) > 0:
                        conf_count[c_indices_add] += 1
                    if len(best_selected) > best_size:
                        best_size = len(best_selected)
                        no_improve_count = 0
                        improved = True
                else:
                    # Need to remove something
                    violating = np.where(conf_count[c_indices_add] >= 7)[0]
                    if len(violating) == 0:
                        continue

                    # Pick a random violating config
                    v_c = int(random.choice(violating))
                    edges_in_vc = config_to_edges[v_c]
                    # Find selected edges in this config
                    selected_in_vc = edges_in_vc[selected_np[edges_in_vc]]
                    if len(selected_in_vc) == 0:
                        continue

                    rem_e = int(random.choice(selected_in_vc))
                    c_indices_rem = configs_in_edge_np[rem_e]
                    rem_set = set(c_indices_rem.tolist()) if len(c_indices_rem) > 0 else set()

                    # Check if swap is valid
                    can_swap = True
                    for c in c_indices_add:
                        new_count = conf_count[c] - (1 if c in rem_set else 0)
                        if new_count >= 7:
                            can_swap = False
                            break

                    if not can_swap:
                        continue

                    # Perform the swap
                    best_selected.remove(rem_e)
                    selected_np[rem_e] = False
                    for c in c_indices_rem:
                        conf_count[c] -= 1
                    best_selected.add(add_e)
                    selected_np[add_e] = True
                    for c in c_indices_add:
                        conf_count[c] += 1

                    if len(best_selected) > best_size:
                        best_size = len(best_selected)
                        no_improve_count = 0
                        improved = True

            if not improved:
                no_improve_count += 5

        return best_selected

    def run_sa(selected, seed, time_budget):
        """
        Simulated annealing with swap moves using numpy membership.
        """
        random.seed(seed)
        conf_count = np.zeros(num_configs, dtype=np.int32)
        for e in selected:
            c_indices = configs_in_edge_np[e]
            if len(c_indices) > 0:
                conf_count[c_indices] += 1

        current_selected = set(selected)
        current_size = len(selected)
        best_selected = set(selected)
        best_size = len(selected)

        # Numpy membership for fast unselected lookup
        selected_np = np.zeros(num_edges, dtype=bool)
        for e in current_selected:
            selected_np[e] = True

        T = 1.0
        T_min = 0.01
        cooling_rate = 0.999
        local_deadline = start_time + time_budget
        steps_since_improve = 0

        while time.time() < local_deadline and T > T_min:
            T *= cooling_rate

            # Pick a random move
            move_type = random.random()
            if move_type < 0.4:
                # Try to add an unselected edge
                unselected_mask = ~selected_np
                unselected_indices = np.where(unselected_mask)[0]
                if len(unselected_indices) == 0:
                    continue
                add_e = int(random.choice(unselected_indices))
                c_indices_add = configs_in_edge_np[add_e]

                if len(c_indices_add) > 0 and np.max(conf_count[c_indices_add]) >= 7:
                    continue

                # Tentatively add
                for c in c_indices_add:
                    conf_count[c] += 1
                current_selected.add(add_e)
                selected_np[add_e] = True
                current_size += 1

                if current_size > best_size:
                    best_size = current_size
                    best_selected = set(current_selected)
                    steps_since_improve = 0
                else:
                    steps_since_improve += 1
            else:
                # Try a swap
                selected_indices = np.where(selected_np)[0]
                if len(selected_indices) == 0:
                    continue
                rem_e = int(random.choice(selected_indices))
                unselected_mask = ~selected_np
                unselected_indices = np.where(unselected_mask)[0]
                if len(unselected_indices) == 0:
                    continue
                add_e = int(random.choice(unselected_indices))

                c_indices_rem = configs_in_edge_np[rem_e]
                c_indices_add = configs_in_edge_np[add_e]
                rem_set = set(c_indices_rem.tolist()) if len(c_indices_rem) > 0 else set()

                # Check if swap is valid
                can_swap = True
                for c in c_indices_add:
                    new_count = conf_count[c] - (1 if c in rem_set else 0)
                    if new_count >= 7:
                        can_swap = False
                        break

                if not can_swap:
                    continue

                # Perform swap
                current_selected.remove(rem_e)
                selected_np[rem_e] = False
                for c in c_indices_rem:
                    conf_count[c] -= 1
                current_selected.add(add_e)
                selected_np[add_e] = True
                for c in c_indices_add:
                    conf_count[c] += 1

                # Accept swap with probability based on temperature
                if random.random() < 0.2 + 0.5 * T:
                    if len(current_selected) > best_size:
                        best_size = len(current_selected)
                        best_selected = set(current_selected)
                        steps_since_improve = 0
                    else:
                        steps_since_improve += 1
                else:
                    # Revert
                    current_selected.remove(add_e)
                    selected_np[add_e] = False
                    for c in c_indices_add:
                        conf_count[c] -= 1
                    current_selected.add(rem_e)
                    selected_np[rem_e] = True
                    for c in c_indices_rem:
                        conf_count[c] += 1

        return best_selected

    # Main search strategy
    best_selected = set()
    best_count = 0

    # Phase 1: Multiple greedy restarts with diverse strategies
    greedy_deadline = start_time + 20
    num_runs = 0
    greedy_results = []
    while time.time() < greedy_deadline:
        seed = num_runs
        result = run_greedy(seed, 0)
        greedy_results.append(result)
        if len(result) > best_count:
            best_count = len(result)
            best_selected = result
        num_runs += 1

    # Phase 2: Local search from top greedy results
    if time.time() < deadline - 15:
        # Sort greedy results by size, take top few
        greedy_results.sort(key=len, reverse=True)
        top_results = greedy_results[:min(3, len(greedy_results))]

        ls_deadline = deadline - 10
        for i, gr in enumerate(top_results):
            remaining = ls_deadline - time.time()
            if remaining < 3:
                break
            seed = 100 + i
            result = run_local_search(gr, seed, remaining - 2)
            if len(result) > best_count:
                best_count = len(result)
                best_selected = result

    # Phase 3: Simulated annealing
    if time.time() < deadline - 5:
        sa_deadline = deadline - 2
        sa_time = sa_deadline - time.time()
        if sa_time > 3:
            result = run_sa(best_selected, 200, sa_time)
            if len(result) > best_count:
                best_count = len(result)
                best_selected = result

    # Phase 4: Final local search polish
    if time.time() < deadline - 2:
        polish_time = deadline - 1.5 - time.time()
        if polish_time > 1:
            result = run_local_search(best_selected, 300, polish_time)
            if len(result) > best_count:
                best_count = len(result)
                best_selected = result

    # Return the result
    return [list(all_edges[i]) for i in sorted(best_selected)]
