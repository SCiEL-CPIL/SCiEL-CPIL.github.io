import time
import math
import sys
import random
from collections import Counter

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin for 60s limit (evaluator kills at 90s)

    N = 128
    full_mask = (1 << (N + 1)) - 2  # bits 1..N set

    # Precompute LCMs
    def lcm(a, b):
        return a * b // math.gcd(a, b)

    # Precompute bad triples and edge weights
    conflict = [[0] * (N + 1) for _ in range(N + 1)]
    edge_weight = [[0.0] * (N + 1) for _ in range(N + 1)]

    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm(a, b)
            for c in range(b + 1, N + 1):
                if lab == lcm(a, c) == lcm(b, c):
                    conflict[a][b] |= (1 << c)
                    conflict[b][a] |= (1 << c)
                    conflict[a][c] |= (1 << b)
                    conflict[c][a] |= (1 << b)
                    conflict[b][c] |= (1 << a)
                    conflict[c][b] |= (1 << a)
                    edge_weight[a][b] += 1.0
                    edge_weight[b][a] += 1.0
                    edge_weight[a][c] += 1.0
                    edge_weight[c][a] += 1.0
                    edge_weight[b][c] += 1.0
                    edge_weight[c][b] += 1.0

    # Build adjacency for conflict graph
    adj = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if edge_weight[a][b] > 0:
                adj[a].append((b, edge_weight[a][b]))
                adj[b].append((a, edge_weight[b][a]))

    # Compute degree-weighted conflict score
    conflict_score = [0.0] * (N + 1)
    for a in range(1, N + 1):
        for b, w in adj[a]:
            conflict_score[a] += w

    # Identify elements with zero conflicts - these can always be included
    zero_conflict = set()
    for a in range(1, N + 1):
        if conflict_score[a] == 0:
            zero_conflict.add(a)

    # Maximum independent set upper bound using greedy coloring
    # In the conflict graph, we want max independent set
    # Upper bound: greedy coloring of the available subgraph
    def max_independent_set_bound(available):
        """Upper bound on independent set size in the conflict graph restricted to available nodes."""
        avail_list = []
        temp = available
        while temp:
            low = temp & (-temp)
            x = low.bit_length() - 1
            avail_list.append(x)
            temp ^= low

        if not avail_list:
            return 0

        # For independent set in G, we can use clique bound on complement
        # But simpler: greedy coloring gives upper bound on clique, not independent set
        # For independent set, we need a different approach
        # Use: partition into cliques (each clique contributes at most 1 to independent set)
        # Greedy: repeatedly find a maximal clique, remove it, count
        # This gives an upper bound on independent set size

        remaining = set(avail_list)
        bound = 0
        while remaining:
            # Find a maximal clique greedily
            # Pick vertex with highest degree in remaining
            if not remaining:
                break
            best_v = max(remaining, key=lambda v: sum(1 for u in remaining if u != v and edge_weight[v][u] > 0))
            clique = {best_v}
            candidates = set(remaining - {best_v})
            for v in list(candidates):
                if all(edge_weight[v][u] > 0 for u in clique):
                    clique.add(v)
            bound += 1
            remaining -= clique

        return bound

    # Better upper bound: use the fact that in the conflict graph,
    # if we color the graph, each color class is an independent set
    # So the chromatic number is a lower bound on... no, that's wrong
    # For maximum independent set, a good upper bound is:
    # - Trivial: |V|
    # - Greedy clique cover: partition into cliques, each contributes at most 1
    # - Or use: alpha(G) <= n - tau(G) where tau is vertex cover (LP relaxation)

    # Let's use a simpler but effective bound:
    # The maximum independent set in G is at most the size of the largest
    # set where no two are adjacent. We can bound this by:
    # For each vertex v, alpha(G) <= alpha(G - v) + 1
    # But that's recursive. Instead, use greedy:
    # Order vertices, greedily assign to "bins" where each bin is a clique
    # The number of bins is an upper bound on alpha

    def greedy_clique_cover_bound(available):
        """Upper bound on independent set size via greedy clique cover."""
        avail_list = []
        temp = available
        while temp:
            low = temp & (-temp)
            x = low.bit_length() - 1
            avail_list.append(x)
            temp ^= low

        if not avail_list:
            return 0

        remaining = set(avail_list)
        cover_count = 0
        while remaining:
            # Greedily build a maximal clique
            # Pick the vertex with highest degree in remaining
            best_v = max(remaining, key=lambda v: sum(1 for u in remaining if u != v and edge_weight[v][u] > 0))
            clique = {best_v}
            for v in list(remaining - {best_v}):
                if all(edge_weight[v][u] > 0 for u in clique):
                    clique.add(v)
            cover_count += 1
            remaining -= clique

        return cover_count

    # Weighted greedy
    def weighted_greedy(order, base_set=None):
        S = list(base_set) if base_set else []
        S_mask = 0
        for s in S:
            S_mask |= (1 << s)
        for x in order:
            if not (full_mask >> x) & 1:
                continue
            # Check if x is compatible with S
            compatible = True
            for s in S:
                if conflict[s][x] & S_mask:
                    compatible = False
                    break
            if compatible:
                S.append(x)
                S_mask |= (1 << x)
        return S

    # Try several greedy orderings
    best_set = []
    best_size = 0

    # Always include zero-conflict elements first
    base_set = list(zero_conflict)
    base_mask = 0
    for x in base_set:
        base_mask |= (1 << x)

    remaining = [x for x in range(1, N + 1) if x not in zero_conflict]

    # Natural order (with zero-conflict elements first)
    S = base_set + weighted_greedy(remaining)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S

    # Reverse order
    S = base_set + weighted_greedy(list(reversed(remaining)))
    if len(S) > best_size:
        best_size = len(S)
        best_set = S

    # Sort by conflict score (ascending - prefer low-conflict elements)
    order = sorted(remaining, key=lambda x: conflict_score[x])
    S = base_set + weighted_greedy(order)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S

    # Sort by conflict score (descending)
    order = sorted(remaining, key=lambda x: -conflict_score[x])
    if len(S) > best_size:
        best_size = len(S)
        best_set = S

    # Sort by number of prime factors
    def num_prime_factors(x):
        count = 0
        d = 2
        n = x
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    order = sorted(remaining, key=num_prime_factors)
    S = base_set + weighted_greedy(order)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S

    # Multiple random orderings
    for seed in range(20):
        if time.time() - start_time > deadline - 10:
            break
        rng = random.Random(seed)
        order = remaining[:]
        rng.shuffle(order)
        S = base_set + weighted_greedy(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    # Spectral ordering: use power iteration on (D - W) to get ranking
    random.seed(42)
    v = [0.0] * (N + 1)
    for i in range(1, N + 1):
        v[i] = random.random()

    norm = math.sqrt(sum(v[i]**2 for i in range(1, N + 1)))
    if norm > 0:
        for i in range(1, N + 1):
            v[i] /= norm

    for _ in range(100):
        if time.time() - start_time > deadline - 5:
            break
        new_v = [0.0] * (N + 1)
        for i in range(1, N + 1):
            new_v[i] = conflict_score[i] * v[i]
            for j, w in adj[i]:
                new_v[i] -= w * v[j]

        norm = math.sqrt(sum(new_v[i]**2 for i in range(1, N + 1)))
        if norm > 1e-10:
            for i in range(1, N + 1):
                new_v[i] /= norm
        v = new_v

    spectral_order = sorted(remaining, key=lambda x: -v[x])
    S = base_set + weighted_greedy(spectral_order)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S

    spectral_order_asc = sorted(remaining, key=lambda x: v[x])
    S = base_set + weighted_greedy(spectral_order_asc)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S

    # Branch and bound with improved upper bound
    def solve(selected, available):
        nonlocal best_size, best_set, start_time

        if time.time() - start_time > deadline:
            return

        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]

        if not available:
            return

        # Upper bound: use maximum independent set bound on the conflict graph
        avail_count = available.bit_count()
        if len(selected) + avail_count <= best_size:
            return

        # Better upper bound using greedy clique cover
        if time.time() - start_time < deadline - 2:
            ub = greedy_clique_cover_bound(available)
            if len(selected) + ub <= best_size:
                return

        # Find all available elements
        avail_list = []
        temp = available
        while temp:
            low = temp & (-temp)
            x = low.bit_length() - 1
            avail_list.append(x)
            temp ^= low

        if not avail_list:
            return

        # Pick element with highest conflict degree in available
        best_x = avail_list[0]
        best_conflict = -1
        for x in avail_list:
            c = 0.0
            for y, w in adj[x]:
                if (available >> y) & 1:
                    c += w
            if c > best_conflict:
                best_conflict = c
                best_x = x
        x = best_x

        # Branch 1: Include x
        new_selected = selected + [x]
        new_avail = available & ~(1 << x)
        for s in selected:
            new_avail &= ~conflict[s][x]

        solve(new_selected, new_avail)

        if time.time() - start_time > deadline:
            return

        # Branch 2: Exclude x
        solve(selected, available & ~(1 << x))

    solve([], full_mask)

    # Local search polish: try to swap elements to increase size
    def local_search_polish(S, max_iterations=200):
        """Try to improve the set by swapping elements."""
        S_set = set(S)
        S_mask = 0
        for s in S_set:
            S_mask |= (1 << s)

        improved = True
        iterations = 0
        while improved and iterations < max_iterations and time.time() - start_time < deadline - 1:
            improved = False
            iterations += 1

            # Try adding an element
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                # Check if x is compatible
                compatible = True
                for s in S_set:
                    if conflict[s][x] & S_mask:
                        compatible = False
                        break
                if compatible:
                    S_set.add(x)
                    S_mask |= (1 << x)
                    improved = True
                    break

            # Try swapping: remove one element, add another
            if not improved:
                for remove in list(S_set):
                    # Check what becomes available
                    temp_S_set = S_set - {remove}
                    temp_S_mask = S_mask & ~(1 << remove)

                    for add in range(1, N + 1):
                        if add in S_set:
                            continue
                        # Check if add is compatible with temp_S_set
                        compatible = True
                        for s in temp_S_set:
                            if conflict[s][add] & temp_S_mask:
                                compatible = False
                                break
                        if compatible:
                            S_set.remove(remove)
                            S_set.add(add)
                            S_mask = temp_S_mask | (1 << add)
                            improved = True
                            break
                    if improved:
                        break

            # Try 2-swap: remove two elements, add two others
            if not improved and time.time() - start_time < deadline - 2:
                S_list = list(S_set)
                for i in range(len(S_list)):
                    for j in range(i + 1, len(S_list)):
                        remove1, remove2 = S_list[i], S_list[j]
                        temp_S_set = S_set - {remove1, remove2}
                        temp_S_mask = S_mask & ~(1 << remove1) & ~(1 << remove2)

                        # Find all compatible additions
                        additions = []
                        for add in range(1, N + 1):
                            if add in S_set:
                                continue
                            compatible = True
                            for s in temp_S_set:
                                if conflict[s][add] & temp_S_mask:
                                    compatible = False
                                    break
                            if compatible:
                                additions.append(add)

                        # Try adding two from additions
                        for k in range(len(additions)):
                            for l in range(k + 1, len(additions)):
                                add1, add2 = additions[k], additions[l]
                                # Check if add1 and add2 are compatible with each other
                                if conflict[add1][add2] & (1 << add2) or conflict[add2][add1] & (1 << add1):
                                    continue
                                # Check if add1 is compatible with add2 in context
                                new_mask = temp_S_mask | (1 << add1)
                                if conflict[add2][add1] & new_mask:
                                    continue
                                # Check full compatibility
                                final_set = temp_S_set | {add1, add2}
                                final_mask = temp_S_mask | (1 << add1) | (1 << add2)
                                fully_compatible = True
                                for s1 in final_set:
                                    for s2 in final_set:
                                        if s1 >= s2:
                                            continue
                                        if conflict[s1][s2] & final_mask:
                                            fully_compatible = False
                                            break
                                    if not fully_compatible:
                                        break
                                if fully_compatible:
                                    S_set = final_set
                                    S_mask = final_mask
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break

        return list(S_set)

    # Apply local search polish
    if best_set:
        polished = local_search_polish(best_set)
        if len(polished) > best_size:
            best_size = len(polished)
            best_set = polished

    return best_set
