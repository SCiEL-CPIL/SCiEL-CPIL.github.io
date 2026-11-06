import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin before 150s kill

    n = 16
    full_mask_28 = (1 << 28) - 1

    # Q3 edges in canonical 8-vertex labeling
    q3_edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]

    # Map each pair (i,j) in [0,8) to a bit position in a 28-bit mask
    bit_pos = {}
    idx = 0
    for i in range(8):
        for j in range(i+1, 8):
            bit_pos[(i,j)] = idx
            idx += 1

    # Generate all distinct Q3 masks (under all 8! labelings, deduplicated)
    q3_mask_set = set()
    for perm in itertools.permutations(range(8)):
        mask = 0
        for u, v in q3_edges:
            pu, pv = perm[u], perm[v]
            if pu > pv:
                pu, pv = pv, pu
            mask |= (1 << bit_pos[(pu, pv)])
        q3_mask_set.add(mask)

    # Build lookup table: lookup[m] = 1 if mask m contains any Q3 as subgraph
    # i.e., m is a supergraph of some Q3 mask
    lookup = bytearray(1 << 28)
    for qm in q3_mask_set:
        Z = full_mask_28 ^ qm
        s = 0
        while True:
            lookup[qm | s] = 1
            if s == Z:
                break
            s = (s - Z) & Z

    if time.time() > deadline - 80:
        return []

    # Precompute all C(16,8) = 12870 subsets of 8 vertices
    subsets = list(itertools.combinations(range(n), 8))
    num_subsets = len(subsets)

    # For each subset, map local pair (i,j) to global edge (u,v) and bit position
    subset_edge_info = []
    for combo in subsets:
        info = {}
        for i in range(8):
            for j in range(i+1, 8):
                info[(i,j)] = (combo[i], combo[j], bit_pos[(i,j)])
        subset_edge_info.append(info)

    # For each global edge (u,v), list of (subset_idx, bit_pos) it appears in
    edge_to_subsets = [[[] for _ in range(n)] for _ in range(n)]
    for s_idx, combo in enumerate(subsets):
        info = subset_edge_info[s_idx]
        for (i,j), (u, v, bp) in info.items():
            edge_to_subsets[u][v].append((s_idx, bp))
            edge_to_subsets[v][u].append((s_idx, bp))

    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]

    def compute_masks(adj):
        """Compute induced masks for all 8-vertex subsets."""
        masks = [0] * num_subsets
        for s_idx in range(num_subsets):
            mask = 0
            info = subset_edge_info[s_idx]
            for (i,j), (u, v, bp) in info.items():
                if (adj[u] >> v) & 1:
                    mask |= (1 << bp)
            masks[s_idx] = mask
        return masks

    def count_violations(masks):
        return sum(lookup[m] for m in masks)

    def edge_count(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def try_add_edge(adj, masks, u, v):
        """Try to add edge (u,v). Returns (success, new_adj, new_masks, delta_viol)."""
        if (adj[u] >> v) & 1:
            return False, adj, masks, 0
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        delta_viol = 0
        for (s_idx, bp) in edge_to_subsets[u][v]:
            bit = 1 << bp
            old_mask = masks[s_idx]
            new_mask = old_mask | bit
            if new_mask != old_mask:
                masks[s_idx] = new_mask
                delta_viol += lookup[new_mask] - lookup[old_mask]
        return True, adj, masks, delta_viol

    def try_remove_edge(adj, masks, u, v):
        """Remove edge (u,v). Returns (success, new_adj, new_masks, delta_viol)."""
        if not ((adj[u] >> v) & 1):
            return False, adj, masks, 0
        adj[u] ^= (1 << v)
        adj[v] ^= (1 << u)
        delta_viol = 0
        for (s_idx, bp) in edge_to_subsets[u][v]:
            bit = 1 << bp
            old_mask = masks[s_idx]
            new_mask = old_mask & ~bit
            if new_mask != old_mask:
                masks[s_idx] = new_mask
                delta_viol += lookup[new_mask] - lookup[old_mask]
        return True, adj, masks, delta_viol

    def greedy_add(adj, masks, viol_count, edge_order, dl):
        """Add edges in given order, skipping those that create Q3."""
        for u, v in edge_order:
            if time.time() > dl:
                break
            if (adj[u] >> v) & 1:
                continue
            # Check if adding would create violation
            would_violate = False
            for (s_idx, bp) in edge_to_subsets[u][v]:
                bit = 1 << bp
                new_mask = masks[s_idx] | bit
                if lookup[new_mask]:
                    would_violate = True
                    break
            if not would_violate:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                for (s_idx, bp) in edge_to_subsets[u][v]:
                    bit = 1 << bp
                    old_mask = masks[s_idx]
                    new_mask = old_mask | bit
                    if new_mask != old_mask:
                        masks[s_idx] = new_mask
                        viol_count += lookup[new_mask] - lookup[old_mask]
        return adj, masks, viol_count

    def local_search_add(adj, masks, viol_count, dl):
        """Try to add all possible missing edges that don't create Q3."""
        improved = True
        while improved and time.time() < dl:
            improved = False
            for u, v in all_edges:
                if time.time() > dl:
                    break
                if (adj[u] >> v) & 1:
                    continue
                # Check if adding would create violation
                would_violate = False
                for (s_idx, bp) in edge_to_subsets[u][v]:
                    bit = 1 << bp
                    new_mask = masks[s_idx] | bit
                    if lookup[new_mask]:
                        would_violate = True
                        break
                if not would_violate:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    for (s_idx, bp) in edge_to_subsets[u][v]:
                        bit = 1 << bp
                        old_mask = masks[s_idx]
                        new_mask = old_mask | bit
                        if new_mask != old_mask:
                            masks[s_idx] = new_mask
                            viol_count += lookup[new_mask] - lookup[old_mask]
                    improved = True
        return adj, masks, viol_count

    def repair(adj, masks, viol_count, dl):
        """Remove edges to eliminate all Q3 violations, choosing edges that minimize new violations."""
        for _ in range(200):
            if time.time() > dl:
                break
            if viol_count == 0:
                break

            # Find a violating subset
            viol_subset = -1
            for s_idx in range(num_subsets):
                if lookup[masks[s_idx]]:
                    viol_subset = s_idx
                    break
            if viol_subset == -1:
                break

            info = subset_edge_info[viol_subset]
            best_edge = None
            best_delta = 999999

            # For each existing edge in this subset, compute delta if removed
            for (i,j), (u, v, bp) in info.items():
                if not ((adj[u] >> v) & 1):
                    continue
                bit = 1 << bp
                old_mask = masks[viol_subset]
                new_mask = old_mask & ~bit
                delta = -lookup[old_mask] + lookup[new_mask]
                # Check other subsets affected by this edge
                for (s_idx2, bp2) in edge_to_subsets[u][v]:
                    if s_idx2 == viol_subset:
                        continue
                    old_m2 = masks[s_idx2]
                    new_m2 = old_m2 & ~(1 << bp2)
                    delta += -lookup[old_m2] + lookup[new_m2]
                if delta < best_delta:
                    best_delta = delta
                    best_edge = (u, v, bp, viol_subset)

            if best_edge is None:
                # Fallback: remove a random existing edge from violating subset
                existing = [(u,v) for (i,j),(u,v,bp) in info.items() if (adj[u]>>v)&1]
                if not existing:
                    break
                u, v = random.choice(existing)
                for (s_idx2, bp2) in edge_to_subsets[u][v]:
                    old_m = masks[s_idx2]
                    new_m = old_m & ~(1 << bp2)
                    viol_count += -lookup[old_m] + lookup[new_m]
                    masks[s_idx2] = new_m
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
            else:
                u, v, bp, s_idx = best_edge
                for (s_idx2, bp2) in edge_to_subsets[u][v]:
                    old_m = masks[s_idx2]
                    new_m = old_m & ~(1 << bp2)
                    viol_count += -lookup[old_m] + lookup[new_m]
                    masks[s_idx2] = new_m
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)

        return adj, masks, viol_count

    def simulated_annealing(adj, masks, viol_count, dl):
        """SA to maximize edges while keeping Q3-free (viol_count == 0)."""
        current_ec = edge_count(adj)
        best_ec = current_ec
        best_adj = adj[:]
        T = 5.0
        cooling = 0.9997
        min_T = 0.05
        no_improve = 0
        max_no_improve = 3000

        while time.time() < dl and no_improve < max_no_improve:
            if T < min_T:
                T = min_T

            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            if u > v:
                u, v = v, u

            edge_exists = (adj[u] >> v) & 1
            delta_ec = -1 if edge_exists else 1

            # Compute delta violation
            delta_viol = 0
            changes = []
            for (s_idx, bp) in edge_to_subsets[u][v]:
                bit = 1 << bp
                old_mask = masks[s_idx]
                new_mask = old_mask ^ bit
                changes.append((s_idx, new_mask))
                delta_viol += lookup[new_mask] - lookup[old_mask]

            # Acceptance: if result is valid (viol==0), always accept if improves
            # If result is invalid, only accept with probability based on T
            if not edge_exists:
                new_viol = viol_count + delta_viol
                if new_viol == 0:
                    # Valid improvement, always accept
                    for (s_idx, new_mask) in changes:
                        masks[s_idx] = new_mask
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    viol_count += delta_viol
                    current_ec += delta_ec
                    no_improve = 0
                    if current_ec > best_ec:
                        best_ec = current_ec
                        best_adj = adj[:]
                else:
                    # Invalid, accept with probability
                    if random.random() < min(1.0, T / max(1, new_viol)):
                        for (s_idx, new_mask) in changes:
                            masks[s_idx] = new_mask
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        viol_count += delta_viol
                        current_ec += delta_ec
                        no_improve += 1
                    else:
                        no_improve += 1
            else:
                new_viol = viol_count + delta_viol
                if new_viol == 0:
                    for (s_idx, new_mask) in changes:
                        masks[s_idx] = new_mask
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    viol_count += delta_viol
                    current_ec += delta_ec
                    no_improve += 1
                else:
                    if random.random() < min(1.0, T / max(1, new_viol)):
                        for (s_idx, new_mask) in changes:
                            masks[s_idx] = new_mask
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                        viol_count += delta_viol
                        current_ec += delta_ec
                        no_improve += 1
                    else:
                        no_improve += 1

            T *= cooling

        return best_adj, best_ec

    best_ec = 0
    best_adj = None

    # Phase 1: Multiple greedy restarts
    trials = 0
    while time.time() < deadline - 30:
        trials += 1
        random.seed(trials * 7919 + 104729)

        edge_order = all_edges[:]
        random.shuffle(edge_order)

        adj = [0] * n
        masks = [0] * num_subsets
        viol_count = 0

        adj, masks, viol_count = greedy_add(adj, masks, viol_count, edge_order, deadline - 25)
        adj, masks, viol_count = local_search_add(adj, masks, viol_count, deadline - 20)

        ec = edge_count(adj)
        if viol_count == 0 and ec > best_ec:
            best_ec = ec
            best_adj = adj[:]

        if trials > 50:
            break

    # Phase 2: Dense random start + repair
    if time.time() < deadline - 25:
        for trial in range(10):
            if time.time() > deadline - 25:
                break
            random.seed(54321 + trial * 31337)
            adj = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.75:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)

            masks = compute_masks(adj)
            viol_count = count_violations(masks)

            adj, masks, viol_count = repair(adj, masks, viol_count, deadline - 20)
            adj, masks, viol_count = local_search_add(adj, masks, viol_count, deadline - 15)

            ec = edge_count(adj)
            if viol_count == 0 and ec > best_ec:
                best_ec = ec
                best_adj = adj[:]

    # Phase 3: Simulated annealing on best solution
    if best_adj is not None and time.time() < deadline - 10:
        adj = best_adj[:]
        masks = compute_masks(adj)
        viol_count = count_violations(masks)
        if viol_count == 0:
            sa_adj, sa_ec = simulated_annealing(adj, masks, viol_count, deadline - 3)
            if sa_ec > best_ec:
                best_ec = sa_ec
                best_adj = sa_adj

    # Final local search
    if best_adj is not None and time.time() < deadline - 2:
        adj = best_adj[:]
        masks = compute_masks(adj)
        viol_count = count_violations(masks)
        if viol_count == 0:
            adj, masks, viol_count = local_search_add(adj, masks, viol_count, deadline - 1)
            ec = edge_count(adj)
            if ec > best_ec:
                best_ec = ec
                best_adj = adj[:]

    if best_adj is None:
        return []

    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])

    return edges

construct()
