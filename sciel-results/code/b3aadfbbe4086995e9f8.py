"""Combined approach: bitmask set cover + local search for hitting all maximal cliques."""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj):
        """Enumerate all maximal cliques of size >= 2."""
        n = len(adj)
        cliques = []

        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            # Pivot: choose vertex in P∪X with max neighbors in P
            candidates_pool = P | X
            if candidates_pool:
                pivot = max(candidates_pool, key=lambda v: len(adj[v] & P))
                candidates = [v for v in P if v not in adj[pivot]]
            else:
                return
            for v in candidates:
                bk(R | {v}, P & adj[v], X & adj[v])
                P.discard(v)
                X.add(v)

        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        cliques = bron_kerbosch(adj)
        if not cliques:
            return []

        num_cliques = len(cliques)
        clique_id = {c: i for i, c in enumerate(cliques)}

        # Bitmask: mask[v] has bit i set if vertex v is in clique i
        masks = [0] * n
        for c in cliques:
            idx = clique_id[c]
            bit = 1 << idx
            for v in c:
                masks[v] |= bit

        all_mask = (1 << num_cliques) - 1

        # Precompute: for each vertex, which cliques it covers (as bitmask)
        # and count of cliques per vertex
        vertex_clique_count = [masks[v].bit_count() for v in range(n)]

        # Precompute clique membership as frozensets for fast intersection checks
        # Also build: for each clique, the set of vertices in it
        clique_vertex_sets = [set(c) for c in cliques]

        best_solution = None
        best_size = float('inf')

        def compute_cover(selected_set):
            """Return cover_count array: cover_count[i] = number of selected vertices in clique i."""
            cover_count = [0] * num_cliques
            for v in selected_set:
                m = masks[v]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cover_count[idx] += 1
                    m ^= lsb
            return cover_count

        def is_valid(selected_set):
            """Check if selected_set hits all cliques using bitmask."""
            covered = 0
            for v in selected_set:
                covered |= masks[v]
                if covered == all_mask:
                    return True
            return covered == all_mask

        def minimize(selected_set):
            """Remove redundant vertices using bitmask cover counts."""
            if not selected_set:
                return set(selected_set)
            cover_count = compute_cover(selected_set)
            result = set(selected_set)
            # Try removing vertices; repeat until no more removals possible
            changed = True
            while changed:
                changed = False
                for v in list(result):
                    m = masks[v]
                    critical = False
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if cover_count[idx] == 1:
                            critical = True
                            break
                        m ^= lsb
                    if not critical:
                        result.discard(v)
                        m = masks[v]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cover_count[idx] -= 1
                            m ^= lsb
                        changed = True
            return result

        def local_search(selected_set, max_rounds=200):
            """1-removal and 1-for-1 swap local search with bitmasks."""
            if not selected_set:
                return selected_set
            result = set(selected_set)
            cover_count = compute_cover(result)

            for rnd in range(max_rounds):
                if time.time() > deadline:
                    break
                improved = False

                # Phase 1: Try 1-removal using cover_count
                for v in list(result):
                    if time.time() > deadline:
                        break
                    m = masks[v]
                    critical = False
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if cover_count[idx] == 1:
                            critical = True
                            break
                        m ^= lsb
                    if not critical:
                        result.discard(v)
                        m = masks[v]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cover_count[idx] -= 1
                            m ^= lsb
                        improved = True
                        break

                if improved:
                    continue

                # Phase 2: Try 1-for-1 swap
                found_swap = False
                for v in list(result):
                    if time.time() > deadline:
                        break
                    # Find cliques that only v covers (cover_count == 1 and v is in that clique)
                    critical_cliques = []
                    m = masks[v]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if cover_count[idx] == 1:
                            critical_cliques.append(idx)
                        m ^= lsb

                    if not critical_cliques:
                        # Can remove v
                        result.discard(v)
                        m = masks[v]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cover_count[idx] -= 1
                            m ^= lsb
                        improved = True
                        found_swap = True
                        break

                    # Candidates: vertices that are in ALL critical cliques and not in result
                    # Compute intersection of critical cliques as bitmask of vertices
                    # For each critical clique idx, we need vertices in clique_vertex_sets[idx]
                    inter_vertices = set(clique_vertex_sets[critical_cliques[0]])
                    for idx in critical_cliques[1:]:
                        inter_vertices &= clique_vertex_sets[idx]
                    candidates = [w for w in inter_vertices if w not in result]
                    if not candidates:
                        continue

                    # Pick best candidate: maximize number of additional cliques covered
                    # (cliques that would become covered after adding w, considering v is removed)
                    best_w = None
                    best_gain = -1
                    for w in candidates:
                        # After removing v, which cliques are uncovered?
                        # A clique is uncovered if cover_count[idx] == 1 and the only covering vertex was v
                        # i.e., critical_cliques
                        # Adding w covers cliques in masks[w]
                        # Gain = number of critical_cliques that w covers
                        gain = 0
                        mw = masks[w]
                        for idx in critical_cliques:
                            if mw & (1 << idx):
                                gain += 1
                        # Also consider: does w cover any non-critical cliques that become critical?
                        # No - removing v only affects critical cliques (those with count==1)
                        # But adding w might cover extra cliques beyond critical ones
                        # For swap, we need all critical cliques to be covered by w
                        if gain == len(critical_cliques):
                            # w can replace v for critical cliques
                            # Score by total additional coverage
                            total_extra = 0
                            mm = mw
                            while mm:
                                lsb = mm & -mm
                                idx = lsb.bit_length() - 1
                                if cover_count[idx] == 0:
                                    total_extra += 1
                                mm ^= lsb
                            if total_extra > best_gain:
                                best_gain = total_extra
                                best_w = w

                    if best_w is not None:
                        # Remove v, add best_w
                        result.discard(v)
                        m = masks[v]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cover_count[idx] -= 1
                            m ^= lsb
                        result.add(best_w)
                        m = masks[best_w]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cover_count[idx] += 1
                            m ^= lsb
                        improved = True
                        found_swap = True
                        break

                if not improved:
                    break

            return result

        # Multiple greedy strategies
        def greedy_randomized(rng, strategy='noise'):
            """Greedy set cover with randomization."""
            uncovered = all_mask
            selected = set()
            available = list(range(n))

            while uncovered:
                if time.time() > deadline:
                    break
                best_v = -1
                best_score = -1
                for v in available:
                    if not (masks[v] & uncovered):
                        continue
                    hits = (masks[v] & uncovered).bit_count()
                    if strategy == 'noise':
                        score = hits + rng.random() * 0.5
                    elif strategy == 'degree_bias':
                        score = hits * 1.0 + vertex_clique_count[v] * 0.1 + rng.random() * 0.3
                    else:
                        score = hits + rng.random() * 0.2
                    if score > best_score:
                        best_score = score
                        best_v = v
                if best_v == -1:
                    break
                selected.add(best_v)
                available.remove(best_v)
                uncovered &= ~masks[best_v]
            return selected

        def greedy_weighted(rng):
            """Greedy with dynamic weights: prioritize hard-to-cover cliques."""
            uncovered = all_mask
            selected = set()
            # Weight each clique by 1/(number of vertices covering it)
            clique_weights = [1.0] * num_cliques
            for c in cliques:
                cnt = len(c)
                for v in c:
                    clique_weights[clique_id[c]] = 1.0 / cnt

            available = list(range(n))
            while uncovered:
                if time.time() > deadline:
                    break
                best_v = -1
                best_score = -1
                for v in available:
                    m = masks[v] & uncovered
                    if not m:
                        continue
                    # Weighted score
                    score = 0.0
                    mm = m
                    while mm:
                        lsb = mm & -mm
                        idx = lsb.bit_length() - 1
                        score += clique_weights[idx]
                        mm ^= lsb
                    score += rng.random() * 0.3
                    if score > best_score:
                        best_score = score
                        best_v = v
                if best_v == -1:
                    break
                selected.add(best_v)
                available.remove(best_v)
                uncovered &= ~masks[best_v]
            return selected

        def greedy_las(rng):
            """Largest Average Set greedy: pick vertex covering most uncovered cliques per clique size."""
            uncovered = all_mask
            selected = set()
            available = list(range(n))

            # Precompute for each vertex: list of cliques it belongs to
            vertex_cliques = [[] for _ in range(n)]
            for c in cliques:
                idx = clique_id[c]
                for v in c:
                    vertex_cliques[v].append(idx)

            while uncovered:
                if time.time() > deadline:
                    break
                best_v = -1
                best_score = -1
                for v in available:
                    m = masks[v] & uncovered
                    if not m:
                        continue
                    # Count uncovered cliques covered by v
                    hits = m.bit_count()
                    # Penalty: prefer vertices in smaller cliques (harder to cover)
                    # Average clique size among covered cliques
                    total_size = 0
                    mm = m
                    while mm:
                        lsb = mm & -mm
                        idx = lsb.bit_length() - 1
                        total_size += len(clique_vertex_sets[idx])
                        mm ^= lsb
                    avg_size = total_size / hits if hits > 0 else n
                    score = hits / (avg_size + 1.0) + rng.random() * 0.3
                    if score > best_score:
                        best_score = score
                        best_v = v
                if best_v == -1:
                    break
                selected.add(best_v)
                available.remove(best_v)
                uncovered &= ~masks[best_v]
            return selected

        # Main search loop
        strategies = ['noise', 'degree_bias', 'weighted', 'las']
        num_restarts = 300

        for restart in range(num_restarts):
            if time.time() > deadline - 5.0:
                break

            rng = random.Random(seed * 1000 + restart * 7 + 13)
            strat = strategies[restart % len(strategies)]

            if strat == 'weighted':
                selected = greedy_weighted(rng)
            elif strat == 'las':
                selected = greedy_las(rng)
            else:
                selected = greedy_randomized(rng, strategy=strat)

            # Minimize
            selected = minimize(selected)

            # Local search
            selected = local_search(selected, max_rounds=100)

            # Final minimize
            selected = minimize(selected)

            if len(selected) < best_size:
                best_size = len(selected)
                best_solution = set(selected)

            if best_size == 0:
                break

            # If solution is small, do more intensive local search
            if best_size <= 15 and time.time() < deadline - 15:
                selected = local_search(selected, max_rounds=300)
                selected = minimize(selected)
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_solution = set(selected)

        if best_solution is None:
            # Fallback: greedy without randomization
            selected = greedy_randomized(random.Random(0), strategy='noise')
            best_solution = minimize(selected)

        return sorted(best_solution)

    # Solve each instance
    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 10:
            # Emergency: return all vertices in any edge
            adj = random_graph(n, p, seed)
            touched = set()
            for u in range(n):
                for v in adj[u]:
                    touched.add(u)
                    touched.add(v)
            results.append(sorted(touched))
            continue

        sol = solve_instance(n, p, seed)
        results.append(sol)

    return results
