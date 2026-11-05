"""Clique transversal via greedy + local search with 2-for-1 moves."""
import time
import random as _random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = t0 + 140.0  # safety margin before 150s kill

    def build_graph(n, p, seed):
        rng = _random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj, n):
        """Enumerate all maximal cliques using Bron-Kerbosch with pivot."""
        cliques = []
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            # Choose pivot u from P ∪ X maximizing |P ∩ N(u)|
            PcupX = P | X
            if not PcupX:
                return
            # Find pivot
            best_u = None
            best_count = -1
            for u in PcupX:
                c = len(P & adj[u])
                if c > best_count:
                    best_count = c
                    best_u = u
            # Iterate over P \ N(pivot)
            candidates = P - adj[best_u]
            for v in list(candidates):
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                bk(new_R, new_P, new_X)
                P.remove(v)
                X.add(v)
        bk(frozenset(), frozenset(range(n)), frozenset())
        return cliques

    def greedy_cover(cliques, adj, n):
        """Greedy set cover: pick vertex covering most uncovered cliques."""
        # Map each vertex to the list of cliques it covers
        vertex_cliques = [[] for _ in range(n)]
        for i, c in enumerate(cliques):
            for v in c:
                vertex_cliques[v].append(i)
        
        uncovered = set(range(len(cliques)))
        selected = set()
        
        while uncovered:
            best_v = None
            best_count = -1
            for v in range(n):
                if v in selected:
                    continue
                cnt = sum(1 for ci in vertex_cliques[v] if ci in uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_v = v
            if best_v is None or best_count <= 0:
                break
            selected.add(best_v)
            for ci in vertex_cliques[best_v]:
                if ci in uncovered:
                    uncovered.discard(ci)
        
        return selected

    def is_valid(selected, cliques):
        for c in cliques:
            if not c & selected:
                return False
        return True

    def local_search(selected, cliques, adj, n, time_limit):
        """Local search with 1-for-0, 1-for-1, 2-for-1 moves."""
        selected = set(selected)
        if not is_valid(selected, cliques):
            return selected
        
        # Precompute: for each clique, which vertices in selected cover it
        # For efficiency, track coverage counts
        def coverage_map(sel):
            cov = {}
            for c in cliques:
                covering = c & sel
                cov[frozenset(c)] = covering
            return cov
        
        # Map clique -> set of vertices in selected that cover it
        clique_to_covers = {}
        for c in cliques:
            clique_to_covers[frozenset(c)] = c & selected
        
        # vertex -> list of cliques it covers (within selected)
        vertex_to_cliques = {v: [] for v in range(n)}
        for c in cliques:
            for v in (c & selected):
                vertex_to_cliques[v].append(frozenset(c))
        
        improved = True
        t_start = time.time()
        while improved and (time.time() - t_start) < time_limit:
            improved = False
            
            # Pass 1: 1-for-0 removal
            for v in list(selected):
                # Check if removing v leaves all cliques covered
                ok = True
                for c in vertex_to_cliques[v]:
                    # c is a clique that v covers; check if another selected vertex covers it
                    if len(clique_to_covers[c]) <= 1:
                        ok = False
                        break
                if ok:
                    # Remove v
                    for c in vertex_to_cliques[v]:
                        clique_to_covers[c].discard(v)
                    selected.discard(v)
                    del vertex_to_cliques[v]
                    improved = True
            
            # Pass 2: 2-for-1 replacement
            if not improved:
                sel_list = list(selected)
                for i in range(len(sel_list)):
                    for j in range(i + 1, len(sel_list)):
                        a, b = sel_list[i], sel_list[j]
                        if a not in selected or b not in selected:
                            continue
                        # Find cliques uniquely covered by a or b
                        unique_to_ab = set()
                        for c in vertex_to_cliques[a]:
                            if len(clique_to_covers[c]) <= 1:
                                unique_to_ab.add(c)
                        for c in vertex_to_cliques[b]:
                            if len(clique_to_covers[c]) <= 1:
                                unique_to_ab.add(c)
                        
                        # Find a candidate vertex w that covers all unique_to_ab
                        # and is not already in selected
                        candidate = None
                        for w in range(n):
                            if w in selected:
                                continue
                            covers_all = True
                            for c in unique_to_ab:
                                if w not in c:
                                    covers_all = False
                                    break
                            if covers_all:
                                candidate = w
                                break
                        
                        if candidate is not None:
                            # Perform the swap: remove a, b; add candidate
                            # Remove a
                            for c in vertex_to_cliques[a]:
                                clique_to_covers[c].discard(a)
                            selected.discard(a)
                            del vertex_to_cliques[a]
                            # Remove b
                            for c in vertex_to_cliques[b]:
                                clique_to_covers[c].discard(b)
                            selected.discard(b)
                            del vertex_to_cliques[b]
                            # Add candidate
                            selected.add(candidate)
                            vertex_to_cliques[candidate] = []
                            for c in cliques:
                                if candidate in c:
                                    clique_to_covers[frozenset(c)].add(candidate)
                                    vertex_to_cliques[candidate].append(frozenset(c))
                            improved = True
                            break
                    if improved:
                        break
            
            # Pass 3: 1-for-1 swap (try to replace a vertex with one that covers more)
            if not improved:
                for v in list(selected):
                    # Find cliques uniquely covered by v
                    unique_to_v = set()
                    for c in vertex_to_cliques[v]:
                        if len(clique_to_covers[c]) <= 1:
                            unique_to_v.add(c)
                    
                    # Find w not in selected that covers all unique_to_v
                    # and covers at least as many total cliques
                    best_w = None
                    best_gain = 0
                    for w in range(n):
                        if w in selected:
                            continue
                        covers_all = True
                        for c in unique_to_v:
                            if w not in c:
                                covers_all = False
                                break
                        if not covers_all:
                            continue
                        # Count how many cliques w covers that v doesn't
                        gain = 0
                        for c in cliques:
                            if w in c and v not in c:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_w = w
                    
                    if best_w is not None and best_gain > 0:
                        # Perform swap
                        for c in vertex_to_cliques[v]:
                            clique_to_covers[c].discard(v)
                        selected.discard(v)
                        del vertex_to_cliques[v]
                        selected.add(best_w)
                        vertex_to_cliques[best_w] = []
                        for c in cliques:
                            if best_w in c:
                                clique_to_covers[frozenset(c)].add(best_w)
                                vertex_to_cliques[best_w].append(frozenset(c))
                        improved = True
                        break
        
        return selected

    def solve_instance(n, p, seed, time_budget):
        adj = build_graph(n, p, seed)
        cliques = bron_kerbosch(adj, n)
        
        if not cliques:
            return []
        
        # Greedy initial solution
        selected = greedy_cover(cliques, adj, n)
        
        # Local search with time budget
        selected = local_search(selected, cliques, adj, n, time_budget * 0.6)
        
        # Try a few randomized restarts if time permits
        best = selected
        t_start = time.time()
        while time.time() - t_start < time_budget * 0.4:
            # Perturb: remove 1-2 random vertices, re-greedy
            perturbed = set(best)
            k = min(len(perturbed), _random.randint(1, 2))
            for _ in range(k):
                if perturbed:
                    perturbed.discard(_random.choice(list(perturbed)))
            # Re-greedy to fill gaps
            uncovered = set(range(len(cliques)))
            for c in cliques:
                if c & perturbed:
                    uncovered.discard(frozenset(c))
            
            # Greedy fill
            vertex_cliques = [[] for _ in range(n)]
            for i, c in enumerate(cliques):
                for v in c:
                    vertex_cliques[v].append(i)
            
            while uncovered:
                best_v = None
                best_count = -1
                for v in range(n):
                    if v in perturbed:
                        continue
                    cnt = sum(1 for ci in vertex_cliques[v] if ci in uncovered)
                    if cnt > best_count:
                        best_count = cnt
                        best_v = v
                if best_v is None or best_count <= 0:
                    break
                perturbed.add(best_v)
                for ci in vertex_cliques[best_v]:
                    if ci in uncovered:
                        uncovered.discard(ci)
            
            if not is_valid(perturbed, cliques):
                continue
            
            # Local search on perturbed
            perturbed = local_search(perturbed, cliques, adj, n, time_budget * 0.15)
            
            if len(perturbed) < len(best):
                best = perturbed
        
        return sorted(best)

    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    # Allocate time proportional to n^2
    weights = [n * n for n, _, _ in instances]
    total_w = sum(weights)
    time_budget = 130.0  # leave 10s for overhead
    
    for (n, p, seed), w in zip(instances, weights):
        if time.time() > DEADLINE - 5.0:
            # Emergency: return minimal valid (all vertices in edges)
            adj = build_graph(n, p, seed)
            touched = set()
            for u in range(n):
                for v in adj[u]:
                    touched.add(u)
                    touched.add(v)
            results.append(sorted(touched))
            continue
        tb = time_budget * w / total_w
        result = solve_instance(n, p, seed, tb)
        results.append(result)
    
    return results
# EVOLVE-BLOCK-END
