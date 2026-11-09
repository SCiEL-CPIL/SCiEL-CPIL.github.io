import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)

        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        C = list(range(k))
        I = list(range(k, n))

        edge_mask = [0] * k
        for u in range(k):
            for v in range(u + 1, k):
                edge_mask[u] |= (1 << v)
                edge_mask[v] |= (1 << u)

        partition = []

        # Precompute neighborhood masks for independent vertices
        I_neigh_masks = {}
        for v in I:
            mask = 0
            for u in adj[v]:
                if u < k:
                    mask |= (1 << u)
            I_neigh_masks[v] = mask

        # Group by neighborhood mask
        groups = {}
        for v in I:
            m = I_neigh_masks[v]
            if m == 0:
                continue
            groups.setdefault(m, []).append(v)

        # Sort groups by mask size (popcount) ascending
        def popcount(x):
            return bin(x).count('1')

        sorted_groups = sorted(groups.items(), key=lambda item: popcount(item[0]))

        for mask, vertices in sorted_groups:
            # For each vertex in this group, we need to cover edges to C
            # The neighborhood is exactly 'mask'
            # We want to find a maximal clique in G[C] induced by 'mask'
            # and use it to cover as many edges as possible

            # But we need to be careful: after using a clique S, the edges (v, s) for s in S
            # are covered. The remaining edges from v are to mask \ S.
            # We need to iterate until all edges from v to mask are covered.

            for v in vertices:
                remaining_mask = mask
                while remaining_mask:
                    # Find maximal clique in remaining_mask
                    # Start with vertex of max degree in remaining_mask
                    max_deg_vertex = None
                    max_deg = -1
                    temp = remaining_mask
                    while temp:
                        lsb = temp & (-temp)
                        u = lsb.bit_length() - 1
                        temp ^= lsb
                        deg = bin(edge_mask[u] & remaining_mask).count('1')
                        if deg > max_deg:
                            max_deg = deg
                            max_deg_vertex = u

                    if max_deg_vertex is None:
                        break

                    current_clique = {max_deg_vertex}
                    common_mask = edge_mask[max_deg_vertex] & remaining_mask

                    # Greedily extend
                    while True:
                        best_candidate = None
                        best_score = -1
                        temp = common_mask
                        while temp:
                            lsb = temp & (-temp)
                            w = lsb.bit_length() - 1
                            temp ^= lsb
                            score = bin(edge_mask[w] & common_mask).count('1')
                            if score > best_score:
                                best_score = score
                                best_candidate = w
                        if best_candidate is None:
                            break
                        current_clique.add(best_candidate)
                        common_mask &= edge_mask[best_candidate]

                    clique_list = sorted(list(current_clique | {v}))
                    partition.append(clique_list)

                    # Remove edges
                    clique_sorted = sorted(current_clique)
                    for i in range(len(clique_sorted)):
                        u = clique_sorted[i]
                        for j in range(i + 1, len(clique_sorted)):
                            w = clique_sorted[j]
                            edge_mask[u] &= ~(1 << w)
                            edge_mask[w] &= ~(1 << u)

                    # Remove from remaining_mask
                    clique_mask = 0
                    for u in current_clique:
                        clique_mask |= (1 << u)
                    remaining_mask &= ~clique_mask

        # Remaining edges in C
        remaining_edges = []
        for u in range(k):
            for v in range(u + 1, k):
                if edge_mask[u] & (1 << v):
                    remaining_edges.append((u, v))

        while remaining_edges:
            # Pick edge with maximum common neighborhood
            best_edge = None
            best_common = -1
            for e in remaining_edges:
                u, v = e
                common_mask = edge_mask[u] & edge_mask[v]
                common_count = bin(common_mask).count('1')
                if common_count > best_common:
                    best_common = common_count
                    best_edge = e

            if best_edge is None:
                best_edge = remaining_edges[0]

            u, v = best_edge
            current_clique = {u, v}
            common_mask = edge_mask[u] & edge_mask[v]

            while True:
                best_candidate = None
                best_score = -1
                temp = common_mask
                while temp:
                    lsb = temp & (-temp)
                    w = lsb.bit_length() - 1
                    temp ^= lsb
                    score = bin(edge_mask[w]).count('1')
                    if score > best_score:
                        best_score = score
                        best_candidate = w
                if best_candidate is None:
                    break
                current_clique.add(best_candidate)
                common_mask &= edge_mask[best_candidate]

            clique_list = sorted(list(current_clique))
            if len(clique_list) >= 2:
                partition.append(clique_list)

            clique_sorted = sorted(current_clique)
            for i in range(len(clique_sorted)):
                u2 = clique_sorted[i]
                for j in range(i + 1, len(clique_sorted)):
                    v2 = clique_sorted[j]
                    edge_mask[u2] &= ~(1 << v2)
                    edge_mask[v2] &= ~(1 << u2)

            clique_set = set(current_clique)
            new_remaining = []
            for e in remaining_edges:
                if e[0] in clique_set and e[1] in clique_set:
                    continue
                new_remaining.append(e)
            remaining_edges = new_remaining

        return partition

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > deadline:
            while len(results) < 3:
                results.append([])
            return results
        res = solve_instance(*spec)
        cleaned = []
        for clique in res:
            cleaned.append([int(x) for x in clique])
        results.append(cleaned)

    return results
