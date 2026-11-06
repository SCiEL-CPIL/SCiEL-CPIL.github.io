import random
import time

def construct():
    start = time.time()
    deadline = start + 118.0  # safety margin before 150s kill

    d = 7
    n = 1 << d  # 128 vertices

    # Precompute all edges with indices
    all_edges = []
    edge_index = {}
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(all_edges)
                all_edges.append((u, v))
                edge_index[(u, v)] = idx

    E = len(all_edges)  # 448

    # For each edge, precompute all C4s it participates in.
    # A C4 is identified by its 4 edge indices.
    # For each edge, we store the list of (other 3 edge indices) that with this edge form a C4.
    blocking_triples = [[] for _ in range(E)]
    for idx, (u, v) in enumerate(all_edges):
        diff = u ^ v
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            bit_j = 1 << j
            u_j = u ^ bit_j
            v_j = v ^ bit_j
            e1 = (min(u, u_j), max(u, u_j))
            e2 = (min(v, v_j), max(v, v_j))
            e3 = (min(u_j, v_j), max(u_j, v_j))
            idx1 = edge_index[e1]
            idx2 = edge_index[e2]
            idx3 = edge_index[e3]
            blocking_triples[idx].append((idx1, idx2, idx3))

    # Precompute for each edge, the set of edges that are in any C4 with it
    # This helps with fast blocking checks
    edge_c4_neighbors = [set() for _ in range(E)]
    for idx in range(E):
        for triple in blocking_triples[idx]:
            for t in triple:
                edge_c4_neighbors[idx].add(t)

    def is_blocked(edge_idx, chosen):
        """Check if adding edge_idx would complete any C4."""
        for triple in blocking_triples[edge_idx]:
            if triple[0] in chosen and triple[1] in chosen and triple[2] in chosen:
                return True
        return False

    def count_violations(chosen):
        """Count number of C4s fully contained in chosen."""
        violations = 0
        for idx in chosen:
            for triple in blocking_triples[idx]:
                if triple[0] in chosen and triple[1] in chosen and triple[2] in chosen:
                    violations += 1
        return violations // 4  # each C4 counted 4 times

    def greedy_random(rng):
        """Random greedy: shuffle edges, add if not blocked."""
        chosen = set()
        order = list(range(E))
        rng.shuffle(order)
        for idx in order:
            if not is_blocked(idx, chosen):
                chosen.add(idx)
        return chosen

    def greedy_weighted(rng, weights):
        """Greedy with weighted random ordering."""
        chosen = set()
        order = list(range(E))
        # Shuffle with bias
        rng.shuffle(order)
        for idx in order:
            if not is_blocked(idx, chosen):
                chosen.add(idx)
        return chosen

    def local_search_swap(chosen, rng, time_limit, max_swaps_per_iter=200):
        """
        Swap-based local search: try to remove one edge to unblock another, then refill.
        Uses simulated annealing-like acceptance for occasional non-improving moves.
        """
        chosen = set(chosen)
        best_size = len(chosen)
        best_set = set(chosen)
        improved = True
        iter_count = 0

        while improved and time.time() < time_limit:
            improved = False
            iter_count += 1

            # Phase 1: Try to add any unblocked edges
            for idx in range(E):
                if idx not in chosen and not is_blocked(idx, chosen):
                    chosen.add(idx)
                    improved = True

            if time.time() >= time_limit:
                break

            # Phase 2: Swap moves
            non_chosen = [i for i in range(E) if i not in chosen]
            if not non_chosen:
                break
            rng.shuffle(non_chosen)

            found_swap = False
            for e_idx in non_chosen[:max_swaps_per_iter]:
                if time.time() >= time_limit:
                    break

                # Find blocking triples for e_idx
                blocking = []
                for triple in blocking_triples[e_idx]:
                    if triple[0] in chosen and triple[1] in chosen and triple[2] in chosen:
                        blocking.append(triple)

                if not blocking:
                    # Shouldn't happen after Phase 1, but add if possible
                    chosen.add(e_idx)
                    improved = True
                    found_swap = True
                    break

                if len(blocking) == 1:
                    # Remove one edge from the single blocking triple
                    t = blocking[0]
                    best_removal = None
                    best_gain = -1
                    for rem in t:
                        # Simulate: remove rem, add e_idx, then greedily add more
                        trial = set(chosen)
                        trial.discard(rem)
                        trial.add(e_idx)
                        # Greedy refill
                        added = 0
                        for other in non_chosen:
                            if other == e_idx:
                                continue
                            if other not in trial and not is_blocked(other, trial):
                                trial.add(other)
                                added += 1
                        gain = added  # net gain: removed 1, added e_idx + others
                        if gain > best_gain:
                            best_gain = gain
                            best_removal = rem

                    if best_removal is not None and best_gain >= 0:
                        chosen.discard(best_removal)
                        chosen.add(e_idx)
                        # Greedy refill
                        for other in non_chosen:
                            if other == e_idx:
                                continue
                            if other not in chosen and not is_blocked(other, chosen):
                                chosen.add(other)
                        improved = True
                        found_swap = True
                        break

                elif len(blocking) >= 2:
                    # Need to remove edges to unblock all C4s
                    # Find a small set of edges to remove
                    all_blockers = set()
                    for t in blocking:
                        all_blockers.update(t)

                    # Try removing each single edge from all_blockers
                    for rem in list(all_blockers)[:5]:  # limit trials
                        trial = set(chosen)
                        trial.discard(rem)
                        # Check if e_idx is now unblocked
                        still_blocked = False
                        for triple in blocking_triples[e_idx]:
                            if triple[0] in trial and triple[1] in trial and triple[2] in trial:
                                still_blocked = True
                                break
                        if not still_blocked:
                            trial.add(e_idx)
                            # Greedy refill
                            added = 0
                            for other in non_chosen:
                                if other == e_idx:
                                    continue
                                if other not in trial and not is_blocked(other, trial):
                                    trial.add(other)
                                    added += 1
                            gain = added
                            if gain >= 0:
                                chosen = trial
                                improved = True
                                found_swap = True
                                break

                    if found_swap:
                        break

            if len(chosen) > best_size:
                best_size = len(chosen)
                best_set = set(chosen)

        return best_set

    def two_opt_search(chosen, rng, time_limit):
        """
        Two-opt: remove two edges, add two different edges.
        """
        chosen = set(chosen)
        best_size = len(chosen)
        best_set = set(chosen)

        while time.time() < time_limit:
            non_chosen = [i for i in range(E) if i not in chosen]
            if len(non_chosen) < 2:
                break
            rng.shuffle(non_chosen)

            improved = False
            for e1_idx in non_chosen[:100]:
                if time.time() >= time_limit:
                    break

                blocking1 = []
                for triple in blocking_triples[e1_idx]:
                    if triple[0] in chosen and triple[1] in chosen and triple[2] in chosen:
                        blocking1.append(triple)

                if not blocking1:
                    chosen.add(e1_idx)
                    improved = True
                    continue

                for e2_idx in non_chosen[:100]:
                    if e2_idx == e1_idx:
                        continue
                    if time.time() >= time_limit:
                        break

                    blocking2 = []
                    for triple in blocking_triples[e2_idx]:
                        if triple[0] in chosen and triple[1] in chosen and triple[2] in chosen:
                            blocking2.append(triple)

                    if not blocking2:
                        chosen.add(e2_idx)
                        improved = True
                        continue

                    # Try removing one edge from blocking1 and one from blocking2
                    if len(blocking1) == 1 and len(blocking2) == 1:
                        t1 = blocking1[0]
                        t2 = blocking2[0]
                        for rem1 in t1:
                            for rem2 in t2:
                                if rem1 == rem2:
                                    continue
                                trial = set(chosen)
                                trial.discard(rem1)
                                trial.discard(rem2)
                                # Check if both e1 and e2 are now unblocked
                                ok1 = True
                                for triple in blocking_triples[e1_idx]:
                                    if triple[0] in trial and triple[1] in trial and triple[2] in trial:
                                        ok1 = False
                                        break
                                ok2 = True
                                for triple in blocking_triples[e2_idx]:
                                    if triple[0] in trial and triple[1] in trial and triple[2] in trial:
                                        ok2 = False
                                        break
                                if ok1 and ok2:
                                    trial.add(e1_idx)
                                    trial.add(e2_idx)
                                    # Greedy refill
                                    for other in non_chosen:
                                        if other not in trial and not is_blocked(other, trial):
                                            trial.add(other)
                                    if len(trial) > len(chosen):
                                        chosen = trial
                                        improved = True
                                        break
                            if improved:
                                break
                        if improved:
                            break

            if len(chosen) > best_size:
                best_size = len(chosen)
                best_set = set(chosen)

            if not improved:
                # Try a random perturbation to escape local optima
                if time.time() < time_limit - 5:
                    chosen_list = list(chosen)
                    if len(chosen_list) > 10:
                        # Remove a random few edges
                        to_remove = rng.sample(chosen_list, min(3, len(chosen_list) // 10))
                        for r in to_remove:
                            chosen.discard(r)
                        # Greedy refill
                        for idx in range(E):
                            if idx not in chosen and not is_blocked(idx, chosen):
                                chosen.add(idx)

        return best_set

    best = set()
    best_size = 0
    rng = random.Random(42)

    # Multiple random greedy starts + local search
    num_starts = 15
    for s in range(num_starts):
        if time.time() > deadline - 40:
            break

        chosen = greedy_random(rng)
        size = len(chosen)
        if size > best_size:
            best = set(chosen)
            best_size = size

        # Local search from this start
        time_limit = deadline - 35
        chosen2 = local_search_swap(chosen, rng, time_limit)
        size2 = len(chosen2)
        if size2 > best_size:
            best = chosen2
            best_size = size2

        # Two-opt search
        if time.time() < deadline - 25:
            chosen3 = two_opt_search(chosen2, rng, deadline - 20)
            size3 = len(chosen3)
            if size3 > best_size:
                best = chosen3
                best_size = size3

        if time.time() > deadline - 20:
            break

    # Final greedy fill on best
    if time.time() < deadline - 5:
        for idx in range(E):
            if idx not in best and not is_blocked(idx, best):
                best.add(idx)

    # Convert to list of [u, v]
    out = []
    for idx in sorted(best):
        u, v = all_edges[idx]
        out.append([u, v])
    return out
