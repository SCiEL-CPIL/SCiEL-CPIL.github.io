import time
import random
import sys

N = 40
DEADLINE = 140.0  # safety margin before 150s kill

def construct():
    start = time.time()
    random.seed(42)

    # adjacency as list of ints (bitmasks)
    adj = [0] * N

    def add_edge(u, v):
        adj[u] |= 1 << v
        adj[v] |= 1 << u

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def edge_count():
        return sum(bin(a).count('1') for a in adj) // 2

    def path5(u, v):
        """Check if there's a simple path of length 5 from u to v (u-a-b-c-d-v)."""
        a_mask = adj[u] & ~(1 << v)
        while a_mask:
            ab = a_mask & (-a_mask)
            a = ab.bit_length() - 1
            a_mask ^= ab
            b_mask = adj[a] & ~(1 << u) & ~(1 << v)
            while b_mask:
                bb = b_mask & (-b_mask)
                b = bb.bit_length() - 1
                b_mask ^= bb
                c_mask = adj[b] & ~(1 << u) & ~(1 << a) & ~(1 << v)
                while c_mask:
                    cb = c_mask & (-c_mask)
                    c = cb.bit_length() - 1
                    c_mask ^= cb
                    d_mask = adj[c] & ~(1 << u) & ~(1 << a) & ~(1 << b) & ~(1 << v)
                    d_mask &= adj[v]
                    if d_mask:
                        return True
        return False

    def greedy_max_independent_set(conflict, candidates, current, best):
        """Greedy: pick vertices with lowest degree in the induced subgraph."""
        if not candidates:
            return current, best
        # Order by degree within candidates
        cand_list = list(candidates)
        degs = []
        for v in cand_list:
            d = 0
            cm = conflict[v]
            for w in cand_list:
                if w != v and (cm >> w) & 1:
                    d += 1
            degs.append((d, v))
        degs.sort()
        chosen = []
        remaining = set(cand_list)
        for d, v in degs:
            if v in remaining:
                chosen.append(v)
                remaining.discard(v)
                # Remove neighbors of v from remaining
                cm = conflict[v]
                to_remove = []
                for w in remaining:
                    if (cm >> w) & 1:
                        to_remove.append(w)
                for w in to_remove:
                    remaining.discard(w)
        return current + len(chosen), best

    def branch_and_bound(conflict, candidates, current_size, best_size, time_limit):
        """Branch-and-bound for maximum independent set.
        conflict: list of bitmasks (conflict[v] has bit w set if v and w conflict)
        candidates: set of vertex indices still available
        Returns best independent set size found.
        """
        if time.time() > time_limit:
            return best_size
        if not candidates:
            return best_size

        # Upper bound: current + len(candidates)
        if current_size + len(candidates) <= best_size:
            return best_size

        # Greedy coloring bound for tighter pruning
        # Color the candidate subgraph greedily (conflict graph restricted to candidates)
        # The number of colors is an upper bound on the independence number
        cand_list = sorted(candidates)
        # Greedy coloring: assign smallest available color
        color_of = {}
        colors_used = 0
        for v in cand_list:
            used = set()
            cm = conflict[v]
            for w in cand_list:
                if w != v and (cm >> w) & 1 and w in color_of:
                    used.add(color_of[w])
            c = 0
            while c in used:
                c += 1
            color_of[v] = c
            colors_used = max(colors_used, c + 1)

        if current_size + colors_used <= best_size:
            return best_size

        # Choose pivot vertex: highest degree in candidate subgraph
        pivot = None
        max_deg = -1
        for v in cand_list:
            d = 0
            cm = conflict[v]
            for w in cand_list:
                if w != v and (cm >> w) & 1:
                    d += 1
            if d > max_deg:
                max_deg = d
                pivot = v

        # Branch: include pivot or exclude pivot
        # Include pivot
        new_candidates = set(cand_list)
        new_candidates.discard(pivot)
        cm = conflict[pivot]
        to_remove = []
        for w in new_candidates:
            if (cm >> w) & 1:
                to_remove.append(w)
        for w in to_remove:
            new_candidates.discard(w)

        best_size = branch_and_bound(conflict, new_candidates, current_size + 1, best_size, time_limit)

        # Exclude pivot
        new_candidates = set(cand_list)
        new_candidates.discard(pivot)
        best_size = branch_and_bound(conflict, new_candidates, current_size, best_size, time_limit)

        return best_size

    # Build initial graph: 8 K5's
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                add_edge(base + a, base + b)

    # Greedy: add all edges that don't create C6
    improved = True
    while improved:
        improved = False
        for u in range(N):
            for v in range(u + 1, N):
                if time.time() > DEADLINE - 30:
                    break
                if not ((adj[u] >> v) & 1):
                    if not path5(u, v):
                        add_edge(u, v)
                        improved = True
        if time.time() > DEADLINE - 30:
            break

    best_edges = edge_count()

    # Conflict-graph based improvement:
    # For each missing edge, determine which other missing edges conflict with it.
    # Then find a maximum independent set of missing edges to add simultaneously.

    # Repeatedly try to add a batch of non-conflicting missing edges
    for iteration in range(50):
        if time.time() > DEADLINE - 5:
            break

        # Collect all missing edges
        missing = []
        for u in range(N):
            for v in range(u + 1, N):
                if not ((adj[u] >> v) & 1):
                    missing.append((u, v))

        if not missing:
            break

        m = len(missing)

        # For each missing edge, check if it can be added (no C6)
        # and which other missing edges conflict with it
        # Two missing edges (u1,v1) and (u2,v2) conflict if adding both creates a C6
        # We check: after adding (u1,v1), does (u2,v2) create a C6?
        # Simpler: just check each edge individually first, then check pairwise conflicts

        # First pass: which missing edges are individually addable?
        addable = []
        for i, (u, v) in enumerate(missing):
            if not path5(u, v):
                addable.append(i)

        if not addable:
            # No single edge can be added; try swap-based improvement
            # Remove one edge, then add up to 2
            found_swap = False
            for u in range(N):
                for v in range(u + 1, N):
                    if time.time() > DEADLINE - 5:
                        break
                    if not ((adj[u] >> v) & 1):
                        continue
                    remove_edge(u, v)
                    added = []
                    for u2 in range(N):
                        for v2 in range(u2 + 1, N):
                            if time.time() > DEADLINE - 5:
                                break
                            if not ((adj[u2] >> v2) & 1):
                                if not path5(u2, v2):
                                    add_edge(u2, v2)
                                    added.append((u2, v2))
                                    if len(added) == 2:
                                        break
                        if len(added) == 2:
                            break
                    if len(added) > 1:
                        found_swap = True
                        best_edges = edge_count()
                    else:
                        for (a, b) in added:
                            remove_edge(a, b)
                        add_edge(u, v)
                    if found_swap:
                        break
                if found_swap:
                    break
            if not found_swap:
                break
            continue

        # Build conflict graph among addable missing edges
        # Two addable edges conflict if adding the first makes the second non-addable
        # (or vice versa). We check: after tentatively adding edge i, is edge j still addable?
        # For efficiency, we only check pairs where at least one endpoint overlaps or is close

        n_addable = len(addable)
        conflict = [0] * n_addable

        for i_idx in range(n_addable):
            u1, v1 = missing[addable[i_idx]]
            for j_idx in range(i_idx + 1, n_addable):
                u2, v2 = missing[addable[j_idx]]
                # Quick check: do they share a vertex? If not, they can't directly conflict
                # but they could still conflict via a C6 that uses both edges
                # Actually, two disjoint edges can conflict if together they complete a C6
                # Let's just check: add edge i, then see if edge j creates C6
                # But this is expensive. Let's use a heuristic: only check if they share a vertex
                # or if the distance between them is small
                shares = (u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2)
                if not shares:
                    # Check if they're within distance 2 (could form part of C6)
                    # u1-v1 and u2-v2: C6 would need path u1-...-u2 of length 4 without using v1,v2
                    # This is expensive. Skip for now (conservative: assume no conflict)
                    continue

                # Add edge i temporarily
                add_edge(u1, v1)
                # Check if edge j creates C6
                if path5(u2, v2):
                    conflict[i_idx] |= 1 << j_idx
                    conflict[j_idx] |= 1 << i_idx
                # Remove edge i
                remove_edge(u1, v1)

        # Now find maximum independent set in conflict graph
        # Use branch and bound
        candidates = set(range(n_addable))
        best_iset_size = 0

        # Simple greedy first
        greedy_set = []
        remaining = set(range(n_addable))
        while remaining:
            # Pick vertex with minimum degree in remaining
            min_deg = float('inf')
            pick = -1
            for v in remaining:
                d = 0
                cm = conflict[v]
                for w in remaining:
                    if w != v and (cm >> w) & 1:
                        d += 1
                if d < min_deg:
                    min_deg = d
                    pick = v
            greedy_set.append(pick)
            remaining.discard(pick)
            cm = conflict[pick]
            to_remove = [w for w in remaining if (cm >> w) & 1]
            for w in to_remove:
                remaining.discard(w)

        best_iset_size = len(greedy_set)

        # Try branch and bound if not too many candidates
        if n_addable <= 30 and time.time() < DEADLINE - 10:
            # Limit the B&B time
            bb_time_limit = min(time.time() + 5, DEADLINE - 10)
            best_iset_size = max(best_iset_size, branch_and_bound(
                conflict, candidates, 0, best_iset_size, bb_time_limit))

        # Add the best independent set of edges
        # Reconstruct which edges to add
        # For greedy, we have greedy_set. For B&B, we need to re-run to get the set.
        # Let's just use the greedy result for now (it's fast)
        # Actually, let's just add all edges from the greedy independent set
        for idx in greedy_set:
            u, v = missing[addable[idx]]
            if not ((adj[u] >> v) & 1):
                add_edge(u, v)

        new_edges = edge_count()
        if new_edges > best_edges:
            best_edges = new_edges
        else:
            # Revert
            for idx in greedy_set:
                u, v = missing[addable[idx]]
                remove_edge(u, v)
            break

    # Final greedy pass
    improved = True
    while improved:
        improved = False
        for u in range(N):
            for v in range(u + 1, N):
                if time.time() > DEADLINE - 2:
                    break
                if not ((adj[u] >> v) & 1):
                    if not path5(u, v):
                        add_edge(u, v)
                        improved = True
        if time.time() > DEADLINE - 2:
            break

    # Build edge list
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if (adj[u] >> v) & 1:
                edges.append([u, v])
    return edges
