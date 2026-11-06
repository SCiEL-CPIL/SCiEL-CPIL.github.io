import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 115.0  # Safety margin before 150s kill

    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append((u, v))
        return adj, edges

    def solve_instance(n, p, seed):
        adj, edges = generate_graph(n, p, seed)
        cliques = []

        # remaining_adj[i] = bitmask of neighbors j > i that still have edge (i,j) remaining
        remaining_adj = [0] * n
        for u, v in edges:
            if u < v:
                remaining_adj[u] |= (1 << v)
            else:
                remaining_adj[v] |= (1 << u)

        def count_remaining():
            cnt = 0
            for i in range(n):
                mask = remaining_adj[i]
                while mask:
                    cnt += 1
                    mask &= mask - 1
            return cnt

        def edge_exists(u, v):
            if u > v:
                u, v = v, u
            return bool(remaining_adj[u] & (1 << v))

        def remove_edge(u, v):
            if u > v:
                u, v = v, u
            remaining_adj[u] &= ~(1 << v)

        def bron_kerbosch_maximal_cliques(R, P, X, adj_mask):
            """Yield all maximal cliques using Bron-Kerbosch with pivot."""
            if not P and not X:
                yield R
                return
            # Choose pivot u from P ∪ X that maximizes |N(u) ∩ P|
            pivot_set = P | X
            best_u = -1
            best_count = -1
            while pivot_set:
                u_bit = pivot_set & (-pivot_set)
                u = u_bit.bit_length() - 1
                pivot_set ^= u_bit
                count = bin(P & adj_mask[u]).count('1')
                if count > best_count:
                    best_count = count
                    best_u = u
            if best_u >= 0:
                candidates = P & ~adj_mask[best_u]
            else:
                candidates = P
            while candidates:
                v_bit = candidates & (-candidates)
                v = v_bit.bit_length() - 1
                candidates ^= v_bit
                R_new = R | v_bit
                P_new = P & adj_mask[v]
                X_new = X & adj_mask[v]
                yield from bron_kerbosch_maximal_cliques(R_new, P_new, X_new, adj_mask)
                P ^= v_bit
                X |= v_bit

        def find_best_clique():
            """Find the best clique to remove using multiple strategies."""
            if count_remaining() == 0:
                return []

            best_clique = []
            best_edges_covered = 0

            # Build adj_mask from remaining edges for Bron-Kerbosch
            adj_mask = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if remaining_adj[i] & (1 << j):
                        adj_mask[i] |= (1 << j)
                        adj_mask[j] |= (1 << i)

            # Strategy 1: Enumerate maximal cliques and pick the one covering most edges
            # To limit time, we'll do a bounded search
            time_limit = min(0.5, max(0.1, (timeout - (time.time() - start_time)) / 10))
            deadline = time.time() + time_limit

            # Start from high-degree vertices
            deg = [0] * n
            for i in range(n):
                deg[i] = bin(adj_mask[i]).count('1')

            verts_by_deg = sorted(range(n), key=lambda x: -deg[x])

            # Try starting from top vertices
            tried_starts = set()
            for start in verts_by_deg[:8]:
                if deg[start] < 1:
                    continue
                if start in tried_starts:
                    continue
                tried_starts.add(start)

                # Get neighbors of start in remaining graph
                neighbors_mask = adj_mask[start]
                if not neighbors_mask:
                    continue

                # Enumerate maximal cliques containing start
                R = 1 << start
                P = neighbors_mask
                X = 0

                found_count = 0
                for clique in bron_kerbosch_maximal_cliques(R, P, X, adj_mask):
                    found_count += 1
                    if found_count > 5000:  # Limit enumeration
                        break
                    if time.time() > deadline:
                        break

                    # Convert bitmask to list
                    clique_list = []
                    m = clique
                    while m:
                        bit = m & (-m)
                        v = bit.bit_length() - 1
                        clique_list.append(v)
                        m ^= bit

                    if len(clique_list) < 2:
                        continue

                    # Count edges covered
                    edges_covered = 0
                    for a, b in combinations(clique_list, 2):
                        if a < b:
                            if remaining_adj[a] & (1 << b):
                                edges_covered += 1
                        else:
                            if remaining_adj[b] & (1 << a):
                                edges_covered += 1

                    if edges_covered > best_edges_covered:
                        best_edges_covered = edges_covered
                        best_clique = clique_list

            # Strategy 2: If no good clique found, try greedy from random edges
            if not best_clique or best_edges_covered < 2:
                all_edges = []
                for i in range(n):
                    mask = remaining_adj[i]
                    while mask:
                        v = (mask & -mask).bit_length() - 1
                        all_edges.append((i, v))
                        mask &= mask - 1

                if all_edges:
                    random.shuffle(all_edges)
                    sample_size = min(30, len(all_edges))
                    for u, v in all_edges[:sample_size]:
                        # Greedy clique building
                        C = [u, v]
                        candidates = []
                        for w in range(n):
                            if w != u and w != v and edge_exists(w, u) and edge_exists(w, v):
                                candidates.append(w)

                        while candidates:
                            best_c = None
                            best_score = -1
                            for c in candidates:
                                score = 0
                                for c2 in candidates:
                                    if c != c2 and edge_exists(c, c2):
                                        score += 1
                                if score > best_score:
                                    best_score = score
                                    best_c = c
                            if best_c is None:
                                break
                            C.append(best_c)
                            new_cands = []
                            for c in candidates:
                                if c != best_c and edge_exists(c, best_c):
                                    new_cands.append(c)
                            candidates = new_cands

                        if len(C) > len(best_clique):
                            best_clique = C

            return best_clique

        while count_remaining() > 0:
            if time.time() - start_time > timeout:
                # Fallback: add all remaining edges as 2-cliques
                for i in range(n):
                    mask = remaining_adj[i]
                    while mask:
                        v = (mask & -mask).bit_length() - 1
                        cliques.append([i, v])
                        mask &= mask - 1
                break

            C = find_best_clique()

            if not C or len(C) < 2:
                # Find any remaining edge
                for i in range(n):
                    mask = remaining_adj[i]
                    if mask:
                        v = (mask & -mask).bit_length() - 1
                        C = [i, v]
                        break
                if not C or len(C) < 2:
                    break

            # Remove all edges in C from remaining
            for a, b in combinations(C, 2):
                if a < b:
                    remaining_adj[a] &= ~(1 << b)
                else:
                    remaining_adj[b] &= ~(1 << a)

            cliques.append(C)

        return cliques

    c1 = solve_instance(32, 0.6, 601)
    c2 = solve_instance(64, 0.55, 602)

    return [c1, c2]
