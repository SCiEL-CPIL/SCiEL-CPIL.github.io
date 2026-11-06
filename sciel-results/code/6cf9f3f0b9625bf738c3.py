import random
import time

def construct():
    d = 7
    n_vertices = 2 ** d
    total_edges = d * (2 ** (d - 1))

    # Generate all edges with canonical representation
    edges = []
    edge_index = {}
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))
                edge_index[(x, y)] = len(edges) - 1
                edge_index[(y, x)] = len(edges) - 1

    # Precompute conflicts: for each edge, list of (i, j, k) tuples of other edges
    # such that if all three are selected, this edge cannot be added
    conflicts = [[] for _ in range(total_edges)]

    for idx, (u, v) in enumerate(edges):
        diff = u ^ v
        i_bit = diff.bit_length() - 1

        for j in range(d):
            if j == i_bit:
                continue
            bit_j = 1 << j
            u_prime = u ^ bit_j
            v_prime = v ^ bit_j

            e1 = (min(u, u_prime), max(u, u_prime))
            e2 = (min(v, v_prime), max(v, v_prime))
            e3 = (min(u_prime, v_prime), max(u_prime, v_prime))

            idx1 = edge_index[e1]
            idx2 = edge_index[e2]
            idx3 = edge_index[e3]
            conflicts[idx].append((idx1, idx2, idx3))

    def can_add(idx, selected):
        for conflict in conflicts[idx]:
            if conflict[0] in selected and conflict[1] in selected and conflict[2] in selected:
                return False
        return True

    best_count = 0
    best_edges = []

    start_time = time.time()
    time_limit = 115

    rng = random.Random(42)

    # Strategy: multiple random orderings + occasional local search improvement
    while time.time() - start_time < time_limit:
        current_edges = list(range(total_edges))
        rng.shuffle(current_edges)

        selected = set()
        count = 0

        for idx in current_edges:
            if can_add(idx, selected):
                selected.add(idx)
                count += 1

        if count > best_count:
            best_count = count
            best_edges = [list(edges[i]) for i in selected]

        if best_count >= 308:
            break

    # Local search: try to add edges that were rejected, and remove conflicting ones
    # Do a few rounds of improvement
    if best_edges:
        for iteration in range(20):
            if time.time() - start_time > time_limit:
                break

            # Build selected set from best_edges
            selected = set()
            for e in best_edges:
                idx = edge_index[(min(e[0], e[1]), max(e[0], e[1]))]
                selected.add(idx)

            # Try to add each non-selected edge by removing minimal conflicts
            unselected = [i for i in range(total_edges) if i not in selected]
            rng.shuffle(unselected)

            improved = False
            for idx in unselected:
                if time.time() - start_time > time_limit:
                    break

                # Check if we can add idx by removing some edges
                # Find all conflicts where idx would complete a C4
                conflicting_edges = set()
                for conflict in conflicts[idx]:
                    if conflict[0] in selected and conflict[1] in selected and conflict[2] in selected:
                        conflicting_edges.add(conflict[0])
                        conflicting_edges.add(conflict[1])
                        conflicting_edges.add(conflict[2])

                # If removing fewer than 3 edges lets us add one, it's a net gain only if we remove <= 0
                # Actually adding 1 edge requires removing k edges, net = 1 - k
                # Only worth it if k == 0 (already handled by greedy) or we do a 1-for-1 swap
                # Let's try: for each edge in conflicting_edges, see if removing it allows adding idx
                for rem in conflicting_edges:
                    new_selected = selected - {rem}
                    new_selected.add(idx)
                    # Verify no new C4 is formed by adding idx (we already checked conflicts)
                    # But removing rem might not break all conflicts
                    # Check if idx can be added to new_selected
                    can = True
                    for conflict in conflicts[idx]:
                        if all(k in new_selected for k in conflict):
                            can = False
                            break
                    if can:
                        # Check that removing rem doesn't create issues (it can't, since we're removing)
                        # Update
                        selected = new_selected
                        improved = True
                        break

                if improved:
                    # Rebuild best_edges
                    best_edges = [list(edges[i]) for i in selected]
                    break

            if not improved:
                break

    return best_edges
