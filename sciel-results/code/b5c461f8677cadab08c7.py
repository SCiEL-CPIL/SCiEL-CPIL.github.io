"""Clique partition of split graphs using optimized partitioning strategy."""
import time
from random import Random
from itertools import combinations

def construct():
    _T0 = time.time()
    _DEADLINE = 145.0  # safety margin before 150s kill

    def _time_up():
        return (time.time() - _T0) > _DEADLINE

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []

    for spec in specs:
        if _time_up():
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            # Fallback: 1 clique for C, 1 per cross edge
            cliques = [list(range(k))]
            for (u, v) in edges:
                if u >= k or v >= k:
                    cliques.append([u, v])
            results.append(cliques)
            continue

        n, k, p, seed = spec
        edges = instance(n, k, p, seed)

        # Build adjacency
        edge_set = set()
        for (a, b) in edges:
            edge_set.add((a, b) if a < b else (b, a))

        # For each I-vertex, find its neighbors in C
        N_C = {}
        for v in range(k, n):
            neigh = frozenset(u for u in range(k) if (u, v) in edge_set)
            N_C[v] = neigh

        all_cc_edges = set((u, v) for u in range(k) for v in range(u + 1, k))
        total_cc = len(all_cc_edges)

        def _cover_remaining_cc(used_cc, cliques_out):
            """Cover remaining C-C edges with C-only cliques using greedy max-clique selection."""
            remaining = all_cc_edges - used_cc
            if not remaining:
                return 0
            if len(remaining) == total_cc:
                cliques_out.append(list(range(k)))
                return 1
            # Build adjacency for remaining C-C edges
            rem_adj = [set() for _ in range(k)]
            for (a, b) in remaining:
                rem_adj[a].add(b)
                rem_adj[b].add(a)
            covered = set()
            count = 0
            while remaining - covered:
                # Greedy: find the clique covering the most uncovered edges
                best_clique = None
                best_edges = -1
                for start in range(k):
                    if not rem_adj[start]:
                        continue
                    clique = [start]
                    for w in range(start + 1, k):
                        if all(w in rem_adj[x] for x in clique):
                            clique.append(w)
                    cl_edges = set()
                    for i in range(len(clique)):
                        for j in range(i + 1, len(clique)):
                            a, b = clique[i], clique[j]
                            e = (a, b) if a < b else (b, a)
                            cl_edges.add(e)
                    uncov = len(cl_edges - covered)
                    if uncov > best_edges:
                        best_edges = uncov
                        best_clique = clique
                if best_clique is None or best_edges <= 0:
                    for (a, b) in remaining - covered:
                        cliques_out.append([a, b])
                        covered.add((a, b))
                    count += len(remaining - covered)
                    break
                cliques_out.append(best_clique)
                count += 1
                for i in range(len(best_clique)):
                    for j in range(i + 1, len(best_clique)):
                        a, b = best_clique[i], best_clique[j]
                        e = (a, b) if a < b else (b, a)
                        covered.add(e)
            return count

        def _greedy_set_cover():
            """
            Greedy set cover on bipartite cross-edges.
            Each I-vertex v defines a 'set' = all cross-edges incident to v.
            We want to cover all cross-edges with minimum number of cliques.
            
            Key insight: For each I-vertex v, we can form cliques {v} ∪ S where S ⊆ N_C[v].
            The cross-edges covered are {(v, u) : u ∈ S}.
            We need to partition all cross-edges into such cliques.
            
            Strategy: Process I-vertices in order of decreasing degree.
            For each v, greedily select the largest subset S ⊆ N_C[v] such that
            no C-C edge within S conflicts with already-used C-C edges.
            """
            used_cc = set()
            cliques = []
            
            # Sort I-vertices by degree descending
            I_verts = sorted(range(k, n), key=lambda v: len(N_C[v]), reverse=True)
            
            for v in I_verts:
                neigh = list(N_C[v])
                d = len(neigh)
                if d == 0:
                    continue
                if d == 1:
                    cliques.append([v, neigh[0]])
                    continue
                
                # Greedily build the largest conflict-free subset
                # Start with all neighbors, try to add them one by one
                # Sort neighbors to prioritize those with fewer conflicts
                remaining_neigh = set(neigh)
                group = []
                
                while remaining_neigh:
                    # Find the neighbor that can be added with least additional conflict
                    best_cand = None
                    best_conflicts = float('inf')
                    for cand in remaining_neigh:
                        # Count C-C edges between cand and existing group that are in used_cc
                        conflicts = 0
                        for g in group:
                            e = (cand, g) if cand < g else (g, cand)
                            if e in used_cc:
                                conflicts += 1
                        if conflicts < best_conflicts:
                            best_conflicts = conflicts
                            best_cand = cand
                        if conflicts == 0:
                            break
                    if best_cand is None:
                        break
                    # Only add if no conflicts
                    if best_conflicts == 0:
                        group.append(best_cand)
                        remaining_neigh.discard(best_cand)
                    else:
                        break
                
                if not group:
                    # Can't add any, use individual edges
                    for u in neigh:
                        cliques.append([v, u])
                else:
                    # Use the group as one clique if size >= 2
                    if len(group) >= 2:
                        cliques.append([v] + group)
                        for i in range(len(group)):
                            for j in range(i + 1, len(group)):
                                a, b = group[i], group[j]
                                e = (a, b) if a < b else (b, a)
                                used_cc.add(e)
                    else:
                        cliques.append([v, group[0]])
                    
                    # Handle remaining neighbors individually
                    for u in remaining_neigh:
                        cliques.append([v, u])
            
            # Cover remaining C-C edges
            ccount = _cover_remaining_cc(used_cc, cliques)
            return cliques, len(cliques)

        def _big_strategy():
            """Try to use one big clique per I-vertex with all neighbors."""
            used_cc = set()
            cliques = []
            I_verts = sorted(range(k, n), key=lambda v: len(N_C[v]), reverse=True)
            for v in I_verts:
                neigh = list(N_C[v])
                d = len(neigh)
                if d == 0:
                    continue
                elif d == 1:
                    cliques.append([v, neigh[0]])
                else:
                    cc_in_neigh = set()
                    for i in range(d):
                        for j in range(i + 1, d):
                            a, b = neigh[i], neigh[j]
                            e = (a, b) if a < b else (b, a)
                            cc_in_neigh.add(e)
                    if cc_in_neigh & used_cc:
                        for u in neigh:
                            cliques.append([v, u])
                    else:
                        cliques.append([v] + neigh)
                        used_cc |= cc_in_neigh
            ccount = _cover_remaining_cc(used_cc, cliques)
            return cliques, len(cliques)

        def _individual_strategy():
            """One clique per cross-edge."""
            used_cc = set()
            cliques = []
            for v in range(k, n):
                for u in N_C[v]:
                    cliques.append([v, u])
            ccount = _cover_remaining_cc(used_cc, cliques)
            return cliques, len(cliques)

        def _smart_partition():
            """
            Improved strategy: For each I-vertex, try multiple partition strategies
            and pick the one that minimizes total clique count.
            """
            used_cc = set()
            cliques = []
            
            # Sort I-vertices by degree descending
            I_verts = sorted(range(k, n), key=lambda v: len(N_C[v]), reverse=True)
            
            for v in I_verts:
                neigh = list(N_C[v])
                d = len(neigh)
                if d == 0:
                    continue
                if d == 1:
                    cliques.append([v, neigh[0]])
                    continue
                
                # Try to find the largest conflict-free subset
                # Multiple strategies for building groups
                best_group = []
                
                # Strategy 1: Greedy by minimum conflicts
                remaining_neigh = set(neigh)
                group = []
                while remaining_neigh:
                    best_cand = None
                    best_conflicts = float('inf')
                    for cand in remaining_neigh:
                        conflicts = 0
                        for g in group:
                            e = (cand, g) if cand < g else (g, cand)
                            if e in used_cc:
                                conflicts += 1
                        if conflicts < best_conflicts:
                            best_conflicts = conflicts
                            best_cand = cand
                        if conflicts == 0:
                            break
                    if best_cand is None:
                        break
                    if best_conflicts == 0:
                        group.append(best_cand)
                        remaining_neigh.discard(best_cand)
                    else:
                        break
                
                if len(group) > len(best_group):
                    best_group = group
                
                # Strategy 2: Try all subsets of size 2,3,4 and find max conflict-free
                if d <= 8:
                    for size in range(min(4, d), 1, -1):
                        found = False
                        for subset in combinations(neigh, size):
                            conflict_free = True
                            for i in range(len(subset)):
                                for j in range(i + 1, len(subset)):
                                    a, b = subset[i], subset[j]
                                    e = (a, b) if a < b else (b, a)
                                    if e in used_cc:
                                        conflict_free = False
                                        break
                                if not conflict_free:
                                    break
                            if conflict_free and len(subset) > len(best_group):
                                best_group = list(subset)
                                found = True
                                break
                        if found:
                            break
                
                if not best_group:
                    best_group = [neigh[0]]
                
                if len(best_group) >= 2:
                    cliques.append([v] + best_group)
                    for i in range(len(best_group)):
                        for j in range(i + 1, len(best_group)):
                            a, b = best_group[i], best_group[j]
                            e = (a, b) if a < b else (b, a)
                            used_cc.add(e)
                else:
                    cliques.append([v, best_group[0]])
                
                # Handle remaining neighbors individually
                for u in neigh:
                    if u not in best_group:
                        cliques.append([v, u])
            
            # Cover remaining C-C edges
            ccount = _cover_remaining_cc(used_cc, cliques)
            return cliques, len(cliques)

        best_cliques = None
        best_count = float('inf')

        for strategy_name, strategy_fn in [('greedy_set_cover', _greedy_set_cover),
                                           ('big', _big_strategy),
                                           ('individual', _individual_strategy),
                                           ('smart_partition', _smart_partition)]:
            if _time_up():
                break
            result = strategy_fn()
            if result is not None:
                cliques, cnt = result
                if cnt < best_count:
                    best_count = cnt
                    best_cliques = cliques

        if best_cliques is None:
            best_cliques = [list(range(k))]
            for (u, v) in edge_set:
                if u >= k or v >= k:
                    best_cliques.append([u, v])
            best_count = len(best_cliques)

        results.append(best_cliques)

    return results
