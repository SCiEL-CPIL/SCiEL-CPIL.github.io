import random
import time
import math
import itertools

def construct():
    start_time = time.time()
    time_limit = 115.0  # safety margin for 120s limit

    n = 16
    full_mask = (1 << 28) - 1

    # 1. Precompute Q3 masks in K8
    q3_edges_local = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]

    bit_index_map = {}
    idx = 0
    for i in range(8):
        for j in range(i + 1, 8):
            bit_index_map[(i, j)] = idx
            idx += 1

    # Generate all 840 Q3 masks
    q3_masks = set()
    for perm in itertools.permutations(range(8)):
        mask = 0
        for u, v in q3_edges_local:
            pu = perm[u]
            pv = perm[v]
            if pu > pv:
                pu, pv = pv, pu
            mask |= (1 << bit_index_map[(pu, pv)])
        q3_masks.add(mask)

    q3_masks_list = list(q3_masks)

    # 2. Precompute Lookup Table
    lookup = bytearray(1 << 28)
    for qm in q3_masks_list:
        Z = full_mask ^ qm
        s = 0
        while True:
            lookup[qm | s] = 1
            if s == Z:
                break
            s = (s - Z) & Z

    if time.time() - start_time > time_limit:
        return []

    # 3. Precompute Subsets and Edge Positions (fast data structures)
    subsets = list(itertools.combinations(range(n), 8))
    num_subsets = len(subsets)

    # For each edge (u,v), store list of (subset_idx, bit_value)
    edge_to_subsets = [[[] for _ in range(n)] for _ in range(n)]

    # For each subset, store {(u,v): bit_value}
    subset_bits = []

    for s_idx, combo in enumerate(subsets):
        sb = {}
        for i in range(8):
            for j in range(i + 1, 8):
                u = combo[i]
                v = combo[j]
                bit_val = 1 << bit_index_map[(i, j)]
                sb[(u, v)] = bit_val
                edge_to_subsets[u][v].append((s_idx, bit_val))
                edge_to_subsets[v][u].append((s_idx, bit_val))
        subset_bits.append(sb)

    if time.time() - start_time > time_limit:
        return []

    def build_initial_adj(density, rng):
        """Build initial adjacency from random edge density."""
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def build_induced_masks(adj):
        """Compute induced masks for all subsets."""
        masks = [0] * num_subsets
        for s_idx in range(num_subsets):
            mask = 0
            for (u, v), bit_val in subset_bits[s_idx].items():
                if (adj[u] >> v) & 1:
                    mask |= bit_val
            masks[s_idx] = mask
        return masks

    def count_violations(masks):
        c = 0
        for m in masks:
            if lookup[m]:
                c += 1
        return c

    def greedy_fix(adj, masks, max_iters=100):
        """Greedy edge removal to eliminate Q3 violations."""
        vc = count_violations(masks)
        for _ in range(max_iters):
            if vc == 0:
                break
            if time.time() - start_time > time_limit - 15:
                break

            # Find a violating subset
            viol_subset = -1
            for s_idx in range(num_subsets):
                if lookup[masks[s_idx]]:
                    viol_subset = s_idx
                    break
            if viol_subset == -1:
                break

            combo = subsets[viol_subset]
            sb = subset_bits[viol_subset]
            best_edge = None
            best_delta = 0

            for (u, v), bit_val in sb.items():
                if not ((adj[u] >> v) & 1):
                    continue
                old_mask = masks[viol_subset]
                new_mask = old_mask ^ bit_val
                delta = -lookup[old_mask] + lookup[new_mask]
                for (s_idx2, bit_val2) in edge_to_subsets[u][v]:
                    if s_idx2 == viol_subset:
                        continue
                    old_m2 = masks[s_idx2]
                    new_m2 = old_m2 ^ bit_val2
                    delta += -lookup[old_m2] + lookup[new_m2]
                if delta < best_delta:
                    best_delta = delta
                    best_edge = (u, v, bit_val)

            if best_edge is None:
                u = combo[random.randint(0, 7)]
                v = combo[random.randint(0, 7)]
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                if (adj[u] >> v) & 1:
                    bit_val = sb.get((u, v))
                    if bit_val is not None:
                        for (s_idx2, bit_val2) in edge_to_subsets[u][v]:
                            old_m = masks[s_idx2]
                            new_m = old_m ^ bit_val2
                            vc += -lookup[old_m] + lookup[new_m]
                            masks[s_idx2] = new_m
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
            else:
                u, v, bit_val = best_edge
                for (s_idx2, bit_val2) in edge_to_subsets[u][v]:
                    old_m = masks[s_idx2]
                    new_m = old_m ^ bit_val2
                    vc += -lookup[old_m] + lookup[new_m]
                    masks[s_idx2] = new_m
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
        return vc

    def edge_count(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def sa_polish(adj, masks, vc, ec, time_budget):
        """Simulated annealing polish. Returns (best_adj, best_ec)."""
        best_violation = vc
        best_adj = adj[:]
        best_edge_count = ec
        T = 2.0
        cooling = 0.9997
        min_T = 0.01
        no_improve_count = 0
        restart_threshold = 8000
        cur_ec = ec

        while time.time() - start_time < time_budget:
            if T < min_T:
                T = min_T

            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u

            affected = edge_to_subsets[u][v]
            delta_violation = 0
            changes = []

            for (s_idx, bit_val) in affected:
                old_mask = masks[s_idx]
                new_mask = old_mask ^ bit_val
                changes.append((s_idx, new_mask))
                delta_violation += lookup[new_mask] - lookup[old_mask]

            edge_exists = (adj[u] >> v) & 1
            delta_edges = -1 if edge_exists else 1

            # Acceptance criteria
            if delta_violation < 0:
                accept = True
            elif delta_violation == 0:
                if delta_edges > 0:
                    accept = True
                else:
                    accept = random.random() < 0.02
            else:
                cost = delta_violation * 50 - delta_edges
                if cost <= 0:
                    accept = True
                else:
                    accept = random.random() < math.exp(-cost / T)

            if accept:
                for (s_idx, new_mask) in changes:
                    masks[s_idx] = new_mask
                vc += delta_violation
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                cur_ec += delta_edges

                if vc < best_violation or (vc == best_violation and cur_ec > best_edge_count):
                    best_violation = vc
                    best_adj = adj[:]
                    best_edge_count = cur_ec
                    no_improve_count = 0
                else:
                    no_improve_count += 1

            T *= cooling

            if no_improve_count > restart_threshold:
                adj = best_adj[:]
                masks = build_induced_masks(adj)
                vc = best_violation
                cur_ec = best_edge_count
                T = 2.0
                no_improve_count = 0
                restart_threshold = max(2000, int(restart_threshold * 0.85))

        return best_adj, best_edge_count

    def final_greedy_add(adj, masks):
        """Try to add back edges that don't create Q3."""
        ec = edge_count(adj)
        vc = count_violations(masks)
        if vc > 0:
            return adj, ec

        # Try each absent edge in random order
        edges_to_try = []
        for i in range(n):
            for j in range(i + 1, n):
                if not ((adj[i] >> j) & 1):
                    edges_to_try.append((i, j))
        random.shuffle(edges_to_try)

        for (u, v) in edges_to_try:
            if time.time() - start_time > time_limit - 2:
                break
            affected = edge_to_subsets[u][v]
            ok = True
            for (s_idx, bit_val) in affected:
                new_mask = masks[s_idx] | bit_val
                if lookup[new_mask]:
                    ok = False
                    break
            if ok:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                ec += 1
                for (s_idx, bit_val) in affected:
                    masks[s_idx] |= bit_val

        return adj, ec

    # 4. Multi-restart strategy
    rng = random.Random(42)
    best_overall_adj = None
    best_overall_ec = 0

    # Restart configurations: (density, description)
    # Dense start + random starts with varying densities
    restart_configs = [
        (1.0, "dense"),
        (0.9, "r09"),
        (0.8, "r08"),
        (0.7, "r07"),
        (0.6, "r06"),
        (0.85, "r085"),
        (0.75, "r075"),
        (0.65, "r065"),
    ]

    # Allocate time: ~10s per restart for 8 restarts = 80s, leaving 35s for final
    per_restart_time = 10.0

    for density, desc in restart_configs:
        if time.time() - start_time > time_limit - 20:
            break

        if density == 1.0:
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        else:
            adj = build_initial_adj(density, rng)

        masks = build_induced_masks(adj)
        vc = count_violations(masks)

        # Greedy fix
        if vc > 0:
            vc = greedy_fix(adj, masks)
            if vc > 0:
                # Still has violations; try harder or skip
                continue

        ec = edge_count(adj)
        if ec > best_overall_ec:
            best_overall_ec = ec
            best_overall_adj = adj[:]

        # SA polish with time budget
        restart_end = time.time() + per_restart_time
        adj, ec = sa_polish(adj, masks, vc, ec, restart_end)

        # Final greedy add
        masks = build_induced_masks(adj)
        vc = count_violations(masks)
        if vc == 0:
            adj, ec = final_greedy_add(adj, masks)

        if ec > best_overall_ec:
            best_overall_ec = ec
            best_overall_adj = adj[:]

    # If no valid graph found, fall back to dense start
    if best_overall_adj is None:
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        masks = build_induced_masks(adj)
        vc = greedy_fix(adj, masks, max_iters=200)
        if vc == 0:
            best_overall_adj = adj
            best_overall_ec = edge_count(adj)
        else:
            # Last resort: return empty
            return []

    # 5. Final global SA polish with remaining time
    if time.time() - start_time < time_limit - 3:
        adj = best_overall_adj[:]
        masks = build_induced_masks(adj)
        vc = count_violations(masks)
        ec = best_overall_ec

        if vc == 0:
            adj, ec = sa_polish(adj, masks, vc, ec, time_limit - 1)
            masks = build_induced_masks(adj)
            vc = count_violations(masks)
            if vc == 0:
                adj, ec = final_greedy_add(adj, masks)

            if ec > best_overall_ec:
                best_overall_ec = ec
                best_overall_adj = adj

    # Return best graph
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_overall_adj[i] >> j) & 1:
                edges.append([i, j])

    return edges
