import random
import time

def construct():
    start = time.time()
    deadline = start + 115.0  # safety margin before 150s kill

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

    # For each edge index, precompute blocking triples (indices of 3 other edges
    # that form a C4 with this edge)
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

    def is_blocked(edge_idx, chosen):
        """Check if adding edge_idx would complete any C4."""
        for triple in blocking_triples[edge_idx]:
            if triple[0] in chosen and triple[1] in chosen and triple[2] in chosen:
                return True
        return False

    def greedy_random(rng):
        """Random greedy: shuffle edges, add if not blocked."""
        chosen = set()
        order = list(range(E))
        rng.shuffle(order)
        for idx in order:
            if not is_blocked(idx, chosen):
                chosen.add(idx)
        return chosen

    def local_search(chosen, rng, time_limit):
        """Swap-based local search: try to remove one edge to unblock another, then refill."""
        chosen = set(chosen)
        improved = True

        while improved and time.time() < time_limit:
            improved = False

            # Phase 1: Try to add any unblocked edges (shouldn't exist in valid set, but check)
            for idx in range(E):
                if idx not in chosen and not is_blocked(idx, chosen):
                    chosen.add(idx)
                    improved = True

            if time.time() >= time_limit:
                break

            # Phase 2: Swap moves - remove one edge from chosen to unblock a non-chosen edge
            non_chosen = [i for i in range(E) if i not in chosen]
            rng.shuffle(non_chosen)

            for e_idx in non_chosen[:300]:  # limit trials per iteration
                if time.time() >= time_limit:
                    break

                # Find blocking triples for e_idx
                blocking = []
                for triple in blocking_triples[e_idx]:
                    if triple[0] in chosen and triple[1] in chosen and triple[2] in chosen:
                        blocking.append(triple)

                if not blocking:
                    # Shouldn't happen, but add if possible
                    chosen.add(e_idx)
                    improved = True
                    break

                if len(blocking) == 1:
                    # Remove one edge from the single blocking triple
                    t = blocking[0]
                    # Try each of the 3 edges to remove, pick one that leads to most additions
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
                        gain = added - 1  # removed 1, added e_idx + others
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
                        break

                elif len(blocking) == 2:
                    # Need to remove an edge common to both triples
                    t1 = set(blocking[0])
                    t2 = set(blocking[1])
                    common = t1 & t2
                    if common:
                        rem = next(iter(common))
                        chosen.discard(rem)
                        chosen.add(e_idx)
                        # Greedy refill
                        for other in non_chosen:
                            if other == e_idx:
                                continue
                            if other not in chosen and not is_blocked(other, chosen):
                                chosen.add(other)
                        improved = True
                        break

        return chosen

    best = set()
    best_size = 0
    rng = random.Random(42)

    # Multiple random greedy starts + local search
    num_starts = 10
    for s in range(num_starts):
        if time.time() > deadline - 40:
            break

        chosen = greedy_random(rng)
        size = len(chosen)
        if size > best_size:
            best = set(chosen)
            best_size = size

        # Local search from this start
        time_limit = deadline - 30
        chosen2 = local_search(chosen, rng, time_limit)
        size2 = len(chosen2)
        if size2 > best_size:
            best = chosen2
            best_size = size2

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
