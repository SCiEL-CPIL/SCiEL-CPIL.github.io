import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s kill

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

        def bron_kerbosch_maximal_cliques(R, P, X, callback):
            """Bron-Kerbosch with pivot to enumerate maximal cliques in the remaining graph."""
            if not P and not X:
                callback(R)
                return

            # Choose pivot u from P ∪ X that maximizes |P ∩ N(u)|
            best_pivot = -1
            best_count = -1
            for u in P:
                count = 0
                for v in P:
                    if u != v and edge_exists(u, v):
                        count += 1
                if count > best_count:
                    best_count = count
                    best_pivot = u
            if best_pivot == -1:
                for u in X:
                    count = 0
                    for v in P:
                        if edge_exists(u, v):
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_pivot = u

            if best_pivot != -1:
                candidates = [v for v in P if not edge_exists(v, best_pivot)]
            else:
                candidates = list(P)

            for v in candidates:
                new_R = R + [v]
                new_P = [w for w in P if w != v and edge_exists(v, w)]
                new_X = [w for w in X if edge_exists(v, w)]
                bron_kerbosch_maximal_cliques(new_R, new_P, new_X, callback)
                P.remove(v)
                X.append(v)

        def find_best_clique(n, adj_orig):
            """Find the best clique using Bron-Kerbosch on the remaining graph."""
            if count_remaining() == 0:
                return []

            best_clique = []

            # Get all vertices that have remaining edges
            active_vertices = set()
            for i in range(n):
                if remaining_adj[i]:
                    active_vertices.add(i)
                    mask = remaining_adj[i]
                    while mask:
                        v = (mask & -mask).bit_length() - 1
                        active_vertices.add(v)
                        mask &= mask - 1
                for j in range(i):
                    if remaining_adj[j] & (1 << i):
                        active_vertices.add(i)
                        active_vertices.add(j)

            if not active_vertices:
                return []

            active_list = list(active_vertices)

            # Use Bron-Kerbosch to find all maximal cliques, then pick the largest
            maximal_cliques = []

            def callback(clique):
                if len(clique) > len(best_clique):
                    best_clique.clear()
                    best_clique.extend(clique)
                maximal_cliques.append(list(clique))

            # To avoid recursion depth issues, limit the search
            # Start with a small P and expand
            # For efficiency, we'll run BK on the full active set but with a time limit
            P = list(active_list)
            X = []
            R = []

            # Run Bron-Kerbosch with a time check
            def bk_with_time_limit(R, P, X):
                if time.time() - start_time > timeout - 5:
                    return False

                if not P and not X:
                    callback(R)
                    return True

                # Choose pivot
                best_pivot = -1
                best_count = -1
                for u in P:
                    count = 0
                    for v in P:
                        if u != v and edge_exists(u, v):
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_pivot = u
                if best_pivot == -1:
                    for u in X:
                        count = 0
                        for v in P:
                            if edge_exists(u, v):
                                count += 1
                        if count > best_count:
                            best_count = count
                            best_pivot = u

                if best_pivot != -1:
                    candidates = [v for v in P if not edge_exists(v, best_pivot)]
                else:
                    candidates = list(P)

                for v in candidates:
                    if time.time() - start_time > timeout - 5:
                        return False
                    new_R = R + [v]
                    new_P = [w for w in P if w != v and edge_exists(v, w)]
                    new_X = [w for w in X if edge_exists(v, w)]
                    if not bk_with_time_limit(new_R, new_P, new_X):
                        return False
                    P.remove(v)
                    X.append(v)

                return True

            # Try to run BK, but if it takes too long, fall back to greedy
            bk_success = bk_with_time_limit(R, P, X)

            if not bk_success and len(best_clique) < 2:
                # Fallback to greedy if BK timed out without finding anything
                def find_clique_from_start(start_vertices):
                    C = list(start_vertices)
                    if len(C) < 2:
                        return C

                    def get_candidates(current_clique):
                        cands = []
                        for w in active_list:
                            if w in current_clique:
                                continue
                            ok = True
                            for x in current_clique:
                                if not edge_exists(w, x):
                                    ok = False
                                    break
                            if ok:
                                cands.append(w)
                        return cands

                    candidates = get_candidates(C)

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

                    return C

                # Try a few starting points
                for i in range(min(5, len(active_list))):
                    v = active_list[i]
                    neighbors = [w for w in active_list if w != v and edge_exists(v, w)]
                    if neighbors:
                        for nb in neighbors[:3]:
                            C = find_clique_from_start([v, nb])
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

            C = find_best_clique(n, adj)

            if not C or len(C) < 2:
                # Shouldn't happen if edges remain, but safety check
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
