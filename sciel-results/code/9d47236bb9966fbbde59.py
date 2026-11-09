"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import Counter, defaultdict
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s out of 150s budget for safety

    def instance(n, p, seed):
        rng = random.Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

    def build_adjacency(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def bron_kerbosch_pivot(adj, n):
        """Enumerate all maximal cliques using Bron-Kerbosch with pivot."""
        cliques = []
        
        def expand(R, P, X):
            if not P and not X:
                if len(R) >= 2:  # Only cliques with at least 2 vertices
                    cliques.append(frozenset(R))
                return
            
            # Choose pivot u to minimize branching
            # Pivot should be from P ∪ X with maximum degree in P
            if P:
                pivot_candidates = P | X
                # Pick pivot with max |P ∩ N(u)|
                best_pivot = None
                best_count = -1
                for u in pivot_candidates:
                    count = len(P & adj[u])
                    if count > best_count:
                        best_count = count
                        best_pivot = u
                    if best_count == len(P):
                        break
                # Only recurse on vertices in P \ N(pivot)
                candidates = [v for v in P if v not in adj[best_pivot]]
            else:
                candidates = list(X)
            
            for v in candidates:
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                expand(new_R, new_P, new_X)
                P.remove(v)
                X.add(v)
        
        expand(set(), set(range(n)), set())
        return cliques

    def solve_hitting_set(cliques, n, adj, time_limit):
        """
        Find a minimum hitting set for the family of cliques.
        Each clique must be hit by at least one vertex in the hitting set.
        Uses greedy + local search.
        """
        if not cliques:
            return set()
        
        # Convert cliques to list of vertex sets
        clique_list = [set(c) for c in cliques]
        
        # Count how many cliques each vertex hits
        vertex_clique_count = Counter()
        for c in clique_list:
            for v in c:
                vertex_clique_count[v] += 1
        
        # Greedy set cover: repeatedly pick vertex hitting most uncovered cliques
        uncovered = set(range(len(clique_list)))
        chosen = set()
        
        while uncovered and time.time() < time_limit:
            best_v = None
            best_count = -1
            for v in range(n):
                if v in chosen:
                    continue
                count = sum(1 for i in uncovered if v in clique_list[i])
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v is None or best_count <= 0:
                break
            
            chosen.add(best_v)
            # Remove all cliques hit by best_v
            uncovered = {i for i in uncovered if best_v not in clique_list[i]}
        
        # Local search: try to remove vertices from chosen that are not essential
        # and try to swap for smaller set
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try removing each vertex
            for v in list(chosen):
                if len(chosen) <= 1:
                    continue
                trial = chosen - {v}
                # Check if trial still hits all cliques
                ok = True
                for c in clique_list:
                    if not (c & trial):
                        ok = False
                        break
                if ok:
                    chosen = trial
                    improved = True
            
            # Try swapping: remove v, add w where w is in some cliques v was hitting
            if not improved and time.time() < time_limit:
                for v in list(chosen):
                    if len(chosen) <= 1:
                        continue
                    # Find cliques that only v hits in current solution
                    # Actually, let's try: remove v, see which cliques become uncovered
                    trial = chosen - {v}
                    uncovered_after = [c for c in clique_list if not (c & trial)]
                    if not uncovered_after:
                        chosen = trial
                        improved = True
                        break
                    # Try to cover uncovered_after with a single new vertex w
                    best_w = None
                    best_cover = 0
                    for w in range(n):
                        if w in chosen:
                            continue
                        cover = sum(1 for c in uncovered_after if w in c)
                        if cover > best_cover:
                            best_cover = cover
                            best_w = w
                    if best_w is not None and best_cover == len(uncovered_after):
                        chosen = trial | {best_w}
                        # Same size, but might enable future improvements
                        # Only accept if it leads to same size (no gain in size)
                        # Skip for now to maintain size reduction focus
                        pass
        
        return chosen

    def improve_with_local_search(chosen, clique_list, n, time_limit):
        """Further improve the hitting set using local search."""
        if not clique_list:
            return chosen
        
        current = set(chosen)
        
        while time.time() < time_limit:
            improved = False
            
            # Try removing each vertex
            for v in list(current):
                if len(current) <= 1:
                    continue
                trial = current - {v}
                ok = all(any(x in trial for x in c) for c in clique_list)
                if ok:
                    current = trial
                    improved = True
            
            # Try 1-for-1 swaps
            if not improved:
                for v in list(current):
                    trial = current - {v}
                    # Find uncovered cliques
                    uncovered = [c for c in clique_list if not (c & trial)]
                    if not uncovered:
                        current = trial
                        improved = True
                        break
                    # Try to find w that covers all uncovered
                    for w in range(n):
                        if w in current:
                            continue
                        if all(w in c for c in uncovered):
                            current = trial | {w}
                            improved = True
                            break
                    if improved:
                        break
            
            # Try 2-for-1 swaps: remove two vertices, add one
            if not improved and len(current) > 2:
                for v1, v2 in itertools.combinations(current, 2):
                    trial = current - {v1, v2}
                    uncovered = [c for c in clique_list if not (c & trial)]
                    if not uncovered:
                        current = trial
                        improved = True
                        break
                    # Try to find w that covers all uncovered
                    for w in range(n):
                        if w in current:
                            continue
                        if all(w in c for c in uncovered):
                            current = trial | {w}
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return current

    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for idx, (n, p, seed) in enumerate(instances):
        if time.time() > deadline - 20:
            # Not enough time left, use greedy only
            time_limit = deadline - 1
        else:
            time_limit = deadline - 10 - idx * 5
        
        edges = instance(n, p, seed)
        adj = build_adjacency(n, edges)
        
        # Enumerate maximal cliques
        cliques = bron_kerbosch_pivot(adj, n)
        clique_list = [set(c) for c in cliques]
        
        if not clique_list:
            results.append([])
            continue
        
        # Solve hitting set
        chosen = solve_hitting_set(cliques, n, adj, time_limit)
        
        # Improve with local search
        if time.time() < deadline - 5:
            chosen = improve_with_local_search(chosen, clique_list, n, deadline - 5)
        
        results.append(sorted(chosen))
    
    return results

# EVOLVE-BLOCK-END
