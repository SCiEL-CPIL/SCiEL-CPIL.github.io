import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    start_time = time.time()
    deadline = start_time + 85

    n = P['n']
    edges = P['edges']

    def make_edge(u, v):
        return (min(u, v), max(u, v))

    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        e = make_edge(u, v)
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add(e)

    total_edges = len(edge_set)

    def biclique_edges(L, R):
        out = set()
        for l in L:
            for r in R:
                if l != r:
                    out.add(make_edge(l, r))
        return out

    def is_valid_biclique(L, R, edge_set_check):
        for l in L:
            for r in R:
                if l == r:
                    continue
                if make_edge(l, r) not in edge_set_check:
                    return False
        return True

    def expand_biclique(L, R, remaining_edges, n):
        L = set(L)
        R = set(R)
        changed = True
        while changed:
            changed = False
            new_L = set()
            for w in range(n):
                if w in L or w in R:
                    continue
                valid = True
                for r in R:
                    if make_edge(w, r) not in remaining_edges:
                        valid = False
                        break
                if valid:
                    new_L.add(w)
            if new_L:
                L |= new_L
                changed = True
            new_R = set()
            for w in range(n):
                if w in L or w in R:
                    continue
                valid = True
                for l in L:
                    if make_edge(w, l) not in remaining_edges:
                        valid = False
                        break
                if valid:
                    new_R.add(w)
            if new_R:
                R |= new_R
                changed = True
        return L, R

    def find_max_biclique(remaining, n, strategy='expand'):
        if not remaining:
            return None

        best_biclique = None
        best_size = 0

        def try_biclique(L, R):
            nonlocal best_biclique, best_size
            if not L or not R:
                return
            L_exp, R_exp = expand_biclique(L, R, remaining, n)
            if is_valid_biclique(L_exp, R_exp, remaining):
                size = len(L_exp) * len(R_exp)
                if size > best_size:
                    best_size = size
                    best_biclique = (L_exp, R_exp)

        if strategy == 'expand':
            for (u, v) in remaining:
                try_biclique({u}, {v})
        elif strategy == 'common_neighbors':
            for u in range(n):
                neighbors = set()
                for v in range(n):
                    if v != u and make_edge(u, v) in remaining:
                        neighbors.add(v)
                if neighbors:
                    try_biclique({u}, neighbors)
        elif strategy == 'degree_heuristic':
            deg = {}
            for v in range(n):
                deg[v] = 0
            for (u, v) in remaining:
                deg[u] += 1
                deg[v] += 1
            for v in sorted(range(n), key=lambda x: -deg[x]):
                if deg[v] == 0:
                    continue
                neighbors = set()
                for w in range(n):
                    if w != v and make_edge(v, w) in remaining:
                        neighbors.add(w)
                if neighbors:
                    try_biclique({v}, neighbors)
        elif strategy == 'mixed':
            for (u, v) in list(remaining)[:50]:
                try_biclique({u}, {v})
            for u in range(n):
                neighbors = set()
                for v in range(n):
                    if v != u and make_edge(u, v) in remaining:
                        neighbors.add(v)
                if neighbors:
                    try_biclique({u}, neighbors)
        return best_biclique

    def greedy_partition(remaining_edges, n, strategy='expand'):
        bicliques = []
        remaining = set(remaining_edges)
        while remaining:
            if time.time() > deadline:
                break
            bc = find_max_biclique(remaining, n, strategy)
            if bc is None:
                for (u, v) in remaining:
                    bicliques.append((frozenset([u]), frozenset([v])))
                remaining = set()
                break
            L, R = bc
            bicliques.append((frozenset(L), frozenset(R)))
            to_remove = biclique_edges(L, R)
            remaining -= to_remove
        return bicliques, remaining

    def try_merge_all(bicliques):
        improved = True
        current = list(bicliques)
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(current)):
                    L1, R1 = current[i]
                    L2, R2 = current[j]
                    newL = L1 | L2
                    newR = R1 | R2
                    if newL & newR:
                        continue
                    valid = True
                    for l in newL:
                        for r in newR:
                            if l == r:
                                continue
                            if make_edge(l, r) not in edge_set:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        current[i] = (newL, newR)
                        current.pop(j)
                        improved = True
                        break
                if improved:
                    break
        return current

    def set_cover_greedy_penalized(remaining_edges, n):
        if not remaining_edges:
            return [], set()

        candidates = []
        seen = set()

        for (u, v) in remaining_edges:
            L, R = expand_biclique({u}, {v}, remaining_edges, n)
            key = (frozenset(L), frozenset(R))
            if key not in seen:
                seen.add(key)
                candidates.append((L, R))

        for u in range(n):
            neighbors = set()
            for v in range(n):
                if v != u and make_edge(u, v) in remaining_edges:
                    neighbors.add(v)
            if neighbors:
                L, R = expand_biclique({u}, neighbors, remaining_edges, n)
                key = (frozenset(L), frozenset(R))
                if key not in seen:
                    seen.add(key)
                    candidates.append((L, R))

        candidates.sort(key=lambda x: len(x[0]) * len(x[1]), reverse=True)

        selected = []
        uncovered = set(remaining_edges)
        selected_edges = set()

        for L, R in candidates:
            if not uncovered:
                break
            if time.time() > deadline:
                break
            bc_edges = biclique_edges(L, R)
            new_cover = bc_edges & uncovered
            overlap = bc_edges & selected_edges
            effective_gain = len(new_cover) - 0.5 * len(overlap)
            if effective_gain > 0:
                selected.append((frozenset(L), frozenset(R)))
                selected_edges |= bc_edges
                uncovered -= new_cover

        for (u, v) in uncovered:
            selected.append((frozenset([u]), frozenset([v])))

        return selected, set()

    def local_search(bicliques, n):
        improved = True
        cur = list(bicliques)
        while improved and time.time() < deadline:
            improved = False
            merged = try_merge_all(cur)
            if len(merged) < len(cur):
                cur = merged
                improved = True
            if not improved:
                for i in range(len(cur)):
                    if time.time() > deadline:
                        break
                    Li, Ri = cur[i]
                    if len(Li) >= 2:
                        Li_list = list(Li)
                        for a in range(len(Li_list)):
                            for b in range(a + 1, len(Li_list)):
                                L1 = frozenset([Li_list[a]])
                                L2 = frozenset(Li_list) - L1
                                if is_valid_biclique(L1, Ri, edge_set) and is_valid_biclique(L2, Ri, edge_set):
                                    new_cur = list(cur)
                                    new_cur.pop(i)
                                    new_cur.append((L1, frozenset(Ri)))
                                    new_cur.append((L2, frozenset(Ri)))
                                    merged_new = try_merge_all(new_cur)
                                    if len(merged_new) < len(cur):
                                        cur = merged_new
                                        improved = True
                                        break
                            if improved:
                                break
                    if improved:
                        break
        return cur

    def vertex_split_search(bicliques, n):
        cur = list(bicliques)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(cur)):
                if time.time() > deadline:
                    break
                Li, Ri = cur[i]
                for j in range(len(cur)):
                    if i == j or time.time() > deadline:
                        continue
                    Lj, Rj = cur[j]
                    for v in Li:
                        if v in Lj or v in Rj:
                            continue
                        valid = True
                        for r in Rj:
                            if make_edge(v, r) not in edge_set:
                                valid = False
                                break
                        if not valid:
                            continue
                        new_Li = Li - {v}
                        if not new_Li:
                            continue
                        if not is_valid_biclique(new_Li, Ri, edge_set):
                            continue
                        new_Lj = Lj | {v}
                        if not is_valid_biclique(new_Lj, Rj, edge_set):
                            continue
                        new_cur = list(cur)
                        new_cur[i] = (frozenset(new_Li), frozenset(Ri))
                        new_cur[j] = (frozenset(new_Lj), frozenset(Rj))
                        merged = try_merge_all(new_cur)
                        if len(merged) < len(cur):
                            cur = merged
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return cur

    def simulated_annealing(bicliques, n):
        """Simulated annealing to escape local optima."""
        cur = list(bicliques)
        cur_count = len(cur)
        best = list(cur)
        best_count = cur_count
        T = 5.0
        cooling = 0.95
        iterations = 0
        max_iterations = 2000

        while iterations < max_iterations and time.time() < deadline:
            iterations += 1
            # Random move: pick a biclique and try to split or merge with another
            move_type = random.random()
            if len(cur) > 1 and move_type < 0.4:
                # Try to split a biclique
                i = random.randint(0, len(cur) - 1)
                Li, Ri = cur[i]
                if len(Li) >= 2:
                    Li_list = list(Li)
                    a, b = random.sample(range(len(Li_list)), 2)
                    L1 = frozenset([Li_list[a]])
                    L2 = frozenset(Li_list) - L1
                    if is_valid_biclique(L1, Ri, edge_set) and is_valid_biclique(L2, Ri, edge_set):
                        new_cur = list(cur)
                        new_cur.pop(i)
                        new_cur.append((L1, frozenset(Ri)))
                        new_cur.append((L2, frozenset(Ri)))
                        new_count = len(new_cur)
                        delta = new_count - cur_count
                        if delta <= 0 or random.random() < pow(2.718, -delta / T):
                            cur = new_cur
                            cur_count = new_count
                            if cur_count < best_count:
                                best = list(cur)
                                best_count = cur_count
                elif len(Ri) >= 2:
                    Ri_list = list(Ri)
                    a, b = random.sample(range(len(Ri_list)), 2)
                    R1 = frozenset([Ri_list[a]])
                    R2 = frozenset(Ri_list) - R1
                    if is_valid_biclique(Li, R1, edge_set) and is_valid_biclique(Li, R2, edge_set):
                        new_cur = list(cur)
                        new_cur.pop(i)
                        new_cur.append((frozenset(Li), R1))
                        new_cur.append((frozenset(Li), R2))
                        new_count = len(new_cur)
                        delta = new_count - cur_count
                        if delta <= 0 or random.random() < pow(2.718, -delta / T):
                            cur = new_cur
                            cur_count = new_count
                            if cur_count < best_count:
                                best = list(cur)
                                best_count = cur_count
            elif len(cur) > 1 and move_type < 0.8:
                # Try to merge two bicliques
                i, j = random.sample(range(len(cur)), 2)
                L1, R1 = cur[i]
                L2, R2 = cur[j]
                newL = L1 | L2
                newR = R1 | R2
                if not (newL & newR):
                    valid = True
                    for l in newL:
                        for r in newR:
                            if l == r:
                                continue
                            if make_edge(l, r) not in edge_set:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        new_cur = list(cur)
                        new_cur.pop(j)
                        new_cur.pop(i)
                        new_cur.append((newL, newR))
                        new_count = len(new_cur)
                        delta = new_count - cur_count
                        if delta <= 0 or random.random() < pow(2.718, -delta / T):
                            cur = new_cur
                            cur_count = new_count
                            if cur_count < best_count:
                                best = list(cur)
                                best_count = cur_count
            else:
                # Random perturbation: move a vertex between bicliques
                if len(cur) >= 2:
                    i, j = random.sample(range(len(cur)), 2)
                    Li, Ri = cur[i]
                    Lj, Rj = cur[j]
                    if Li:
                        v = random.choice(list(Li))
                        if v not in Lj and v not in Rj:
                            new_Li = Li - {v}
                            new_Lj = Lj | {v}
                            if new_Li and is_valid_biclique(new_Li, Ri, edge_set) and is_valid_biclique(new_Lj, Rj, edge_set):
                                new_cur = list(cur)
                                new_cur[i] = (frozenset(new_Li), frozenset(Ri))
                                new_cur[j] = (frozenset(new_Lj), frozenset(Rj))
                                new_count = len(new_cur)
                                delta = new_count - cur_count
                                if delta <= 0 or random.random() < pow(2.718, -delta / T):
                                    cur = new_cur
                                    cur_count = new_count
                                    if cur_count < best_count:
                                        best = list(cur)
                                        best_count = cur_count
            T *= cooling

        return best

    # Try multiple strategies and keep the best
    best_result = None
    best_count = total_edges

    strategies = ['expand', 'common_neighbors', 'degree_heuristic', 'mixed']
    for strat in strategies:
        if time.time() > deadline:
            break
        bicliques, leftover = greedy_partition(edge_set, n, strat)
        if leftover:
            for (u, v) in leftover:
                bicliques.append((frozenset([u]), frozenset([v])))
        merged = try_merge_all(bicliques)
        if len(merged) < best_count:
            best_count = len(merged)
            best_result = merged

    if time.time() < deadline:
        sc_bicliques, _ = set_cover_greedy_penalized(edge_set, n)
        if sc_bicliques:
            merged_sc = try_merge_all(sc_bicliques)
            if len(merged_sc) < best_count:
                best_count = len(merged_sc)
                best_result = merged_sc

    if best_result is None:
        bicliques, leftover = greedy_partition(edge_set, n, 'expand')
        for (u, v) in leftover:
            bicliques.append((frozenset([u]), frozenset([v])))
        best_result = try_merge_all(bicliques)

    if time.time() < deadline:
        current = local_search(best_result, n)
        if len(current) < best_count:
            best_count = len(current)
            best_result = current

    if time.time() < deadline:
        current = vertex_split_search(best_result, n)
        if len(current) < best_count:
            best_count = len(current)
            best_result = current

    # Simulated annealing for additional improvement
    if time.time() < deadline:
        current = simulated_annealing(best_result, n)
        if len(current) < best_count:
            best_count = len(current)
            best_result = current

    best_result = try_merge_all(best_result)

    result = []
    for L, R in best_result:
        result.append([list(L), list(R)])

    return result
