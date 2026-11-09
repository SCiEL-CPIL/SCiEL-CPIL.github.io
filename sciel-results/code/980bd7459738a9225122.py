"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = P['n']
    edges = P['edges']

    # Build adjacency sets and edge set
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        a, b = (u, v) if u < v else (v, u)
        adj[a].add(b)
        adj[b].add(a)
        edge_set.add((a, b))

    total_edges = len(edge_set)

    def make_edge(u, v):
        return (u, v) if u < v else (v, u)

    def time_left():
        return deadline - time.time()

    def expand_biclique(L, R, remaining):
        L = set(L)
        R = set(R)
        changed = True
        while changed:
            if time_left() < 0.5:
                break
            changed = False
            # Expand L: vertices connected to all in R
            new_L = set()
            for w in range(n):
                if w in L or w in R:
                    continue
                ok = True
                for r in R:
                    if make_edge(w, r) not in remaining:
                        ok = False
                        break
                if ok:
                    new_L.add(w)
            if new_L:
                L |= new_L
                changed = True
            # Expand R: vertices connected to all in L
            new_R = set()
            for w in range(n):
                if w in L or w in R:
                    continue
                ok = True
                for l in L:
                    if make_edge(w, l) not in remaining:
                        ok = False
                        break
                if ok:
                    new_R.add(w)
            if new_R:
                R |= new_R
                changed = True
        return L, R

    def verify_biclique(L, R, remaining):
        if not L or not R:
            return False
        if L & R:
            return False
        for u in L:
            for v in R:
                if make_edge(u, v) not in remaining:
                    return False
        return True

    def biclique_edges(L, R):
        es = set()
        for u in L:
            for v in R:
                if u != v:
                    es.add(make_edge(u, v))
        return es

    def greedy_partition(remaining, adj_rem, seed_order):
        bicliques = []
        rem = set(remaining)
        ar = [set(s) for s in adj_rem]

        while rem:
            if time_left() < 0.5:
                break
            best_biclique = None
            best_size = 0
            best_edges = None

            # Sort candidates by degree heuristic
            candidates = list(rem)
            candidates.sort(key=lambda e: (-(len(ar[e[0]]) + len(ar[e[1]])), e[0], e[1]))

            tried = 0
            for (u, v) in candidates:
                if time_left() < 0.3:
                    break
                tried += 1
                if tried > 60:
                    break

                # Try u left, v right
                L1, R1 = expand_biclique({u}, {v}, rem)
                if L1 and R1 and verify_biclique(L1, R1, rem):
                    sz = len(L1) * len(R1)
                    if sz > best_size:
                        best_size = sz
                        best_biclique = (L1, R1)
                        best_edges = biclique_edges(L1, R1)

                # Try v left, u right
                L2, R2 = expand_biclique({v}, {u}, rem)
                if L2 and R2 and verify_biclique(L2, R2, rem):
                    sz = len(L2) * len(R2)
                    if sz > best_size:
                        best_size = sz
                        best_biclique = (L2, R2)
                        best_edges = biclique_edges(L2, R2)

            if best_biclique is None:
                e = next(iter(rem))
                best_biclique = ({e[0]}, {e[1]})
                best_edges = {e}

            L, R = best_biclique
            bicliques.append((L, R))

            for e in best_edges:
                rem.discard(e)
                u, v = e
                ar[u].discard(v)
                ar[v].discard(u)

        for e in rem:
            bicliques.append(({e[0]}, {e[1]}))

        return bicliques

    def try_merge(bicliques, edge_set):
        if time_left() < 1.0:
            return bicliques
        improved = True
        while improved:
            if time_left() < 0.5:
                break
            improved = False
            new_bicliques = list(bicliques)
            i = 0
            while i < len(new_bicliques):
                if time_left() < 0.3:
                    break
                L1, R1 = new_bicliques[i]
                j = i + 1
                while j < len(new_bicliques):
                    if time_left() < 0.2:
                        break
                    L2, R2 = new_bicliques[j]
                    Lnew = set(L1) | set(L2)
                    Rnew = set(R1) | set(R2)
                    if Lnew and Rnew and not (Lnew & Rnew):
                        ok = True
                        for u in Lnew:
                            for v in Rnew:
                                if make_edge(u, v) not in edge_set:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if ok:
                            ei = biclique_edges(L1, R1)
                            ej = biclique_edges(L2, R2)
                            en = biclique_edges(Lnew, Rnew)
                            if ei | ej == en and not (ei & ej):
                                new_bicliques.pop(j)
                                new_bicliques[i] = (Lnew, Rnew)
                                improved = True
                                j = i + 1
                                continue
                    j += 1
                i += 1
            bicliques = new_bicliques
        return bicliques

    def try_split(bicliques, edge_set):
        """Try to split a biclique into two that cover the same edges with fewer total bicliques.
        Actually this increases count, so skip. Instead try to redistribute edges."""
        return bicliques

    # Multiple strategies for seed ordering
    def run_with_strategy(strategy):
        adj_rem = [set() for _ in range(n)]
        for u, v in edge_set:
            adj_rem[u].add(v)
            adj_rem[v].add(u)

        bicliques = greedy_partition(edge_set, adj_rem, strategy)
        bicliques = try_merge(bicliques, edge_set)
        return bicliques

    best_result = None
    best_count = float('inf')

    # Strategy 1: default (degree-sorted)
    if time_left() > 1.0:
        bicliques = run_with_strategy('default')
        count = len(bicliques)
        if count < best_count:
            best_count = count
            best_result = bicliques

    # Strategy 2: try different seed limits
    if time_left() > 2.0:
        adj_rem = [set() for _ in range(n)]
        for u, v in edge_set:
            adj_rem[u].add(v)
            adj_rem[v].add(u)

        # Try with more seed attempts
        rem = set(edge_set)
        ar = [set(s) for s in adj_rem]
        bicliques2 = []
        while rem:
            if time_left() < 1.0:
                break
            best_biclique = None
            best_size = 0
            best_edges = None
            candidates = list(rem)
            candidates.sort(key=lambda e: (-(len(ar[e[0]]) + len(ar[e[1]])), e[0], e[1]))
            tried = 0
            for (u, v) in candidates:
                if time_left() < 0.5:
                    break
                tried += 1
                if tried > 100:
                    break
                L1, R1 = expand_biclique({u}, {v}, rem)
                if L1 and R1 and verify_biclique(L1, R1, rem):
                    sz = len(L1) * len(R1)
                    if sz > best_size:
                        best_size = sz
                        best_biclique = (L1, R1)
                        best_edges = biclique_edges(L1, R1)
                L2, R2 = expand_biclique({v}, {u}, rem)
                if L2 and R2 and verify_biclique(L2, R2, rem):
                    sz = len(L2) * len(R2)
                    if sz > best_size:
                        best_size = sz
                        best_biclique = (L2, R2)
                        best_edges = biclique_edges(L2, R2)
            if best_biclique is None:
                e = next(iter(rem))
                best_biclique = ({e[0]}, {e[1]})
                best_edges = {e}
            L, R = best_biclique
            bicliques2.append((L, R))
            for e in best_edges:
                rem.discard(e)
                u, v = e
                ar[u].discard(v)
                ar[v].discard(u)
        for e in rem:
            bicliques2.append(({e[0]}, {e[1]}))
        bicliques2 = try_merge(bicliques2, edge_set)
        if len(bicliques2) < best_count:
            best_count = len(bicliques2)
            best_result = bicliques2

    # Convert best_result to proper format
    if best_result is None:
        # Fallback: star decomposition
        result = []
        for u in range(n):
            neighbors = [v for v in adj[u]]
            if neighbors:
                result.append([[int(u)], [int(v) for v in sorted(neighbors)]])
        return result

    result = []
    for L, R in best_result:
        L_list = sorted([int(x) for x in L])
        R_list = sorted([int(x) for x in R])
        result.append([L_list, R_list])

    # Verify
    covered = set()
    valid = True
    for L, R in result:
        for u in L:
            for v in R:
                if u == v:
                    valid = False
                    break
                key = make_edge(u, v)
                if key in covered:
                    valid = False
                    break
                if key not in edge_set:
                    valid = False
                    break
                covered.add(key)
            if not valid:
                break

    if not valid or covered != edge_set:
        # Fallback: star decomposition
        result = []
        for u in range(n):
            neighbors = [v for v in adj[u]]
            if neighbors:
                result.append([[int(u)], [int(v) for v in sorted(neighbors)]])

    return result
# EVOLVE-BLOCK-END
