import time
from collections import defaultdict
import random as _random

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def random_graph(n, p, seed):
        """Generate edge list using the same RNG as the seed program."""
        import random
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch(adj, n):
        """
        Enumerate all maximal cliques of size >= 2 using Bron-Kerbosch with pivoting.
        adj: list of sets, adj[i] = set of neighbors of i
        Returns list of frozensets (each a maximal clique of size >= 2)
        """
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Choose pivot u in P ∪ X to minimize branching
            # Pick vertex with max |P ∩ adj[u]|
            best_u = None
            best_count = -1
            for u in P:
                cnt = len(P & adj[u])
                if cnt > best_count:
                    best_count = cnt
                    best_u = u
            for u in X:
                cnt = len(P & adj[u])
                if cnt > best_count:
                    best_count = cnt
                    best_u = u
            
            if best_u is None:
                candidates = list(P)
            else:
                candidates = list(P - adj[best_u])
            
            for v in candidates:
                if time.time() - start_time > DEADLINE:
                    return
                R.add(v)
                P_new = P & adj[v]
                X_new = X & adj[v]
                bk(R, P_new, X_new)
                R.discard(v)
                P.discard(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return cliques

    def build_clique_membership(cliques, n):
        """Precompute for each vertex, the set of clique indices it belongs to."""
        vert_cliques = [set() for _ in range(n)]
        for ci, c in enumerate(cliques):
            for v in c:
                vert_cliques[v].add(ci)
        return vert_cliques

    def is_valid(solution, cliques):
        """Check if solution hits all cliques."""
        sol_set = set(solution)
        for c in cliques:
            if not (c & sol_set):
                return False
        return True

    def greedy_hit(cliques, n, vert_cliques, order_seed=None):
        """
        Greedy set cover with optional randomized ordering.
        Returns a set of vertices
        """
        if not cliques:
            return set()
        
        m = len(cliques)
        uncovered = set(range(m))
        result = set()
        
        # Precompute for each vertex which cliques it covers
        vert_cover = [set() for _ in range(n)]
        for ci, c in enumerate(cliques):
            for v in c:
                vert_cover[v].add(ci)
        
        # Create ordering
        if order_seed is not None:
            rng = _random.Random(order_seed)
            order = list(range(n))
            rng.shuffle(order)
        else:
            # Default: sort by total coverage descending
            order = sorted(range(n), key=lambda v: -len(vert_cover[v]))
        
        while uncovered:
            if time.time() - start_time > DEADLINE:
                break
            best_v = -1
            best_covered = 0
            for v in order:
                if v in result:
                    continue
                covered = len(uncovered & vert_cover[v])
                if covered > best_covered:
                    best_covered = covered
                    best_v = v
            if best_v == -1 or best_covered == 0:
                break
            result.add(best_v)
            uncovered -= vert_cover[best_v]
        
        return result

    def local_search_reduce(result, cliques, vert_cliques):
        """
        Remove vertices that are not needed.
        Returns improved result set.
        """
        result = set(result)
        m = len(cliques)
        
        # Build coverage count: for each clique, how many solution vertices cover it
        cover_count = [0] * m
        for v in result:
            for ci in vert_cliques[v]:
                cover_count[ci] += 1
        
        improved = True
        while improved:
            improved = False
            for v in list(result):
                # Check if removing v leaves any clique uncovered
                can_remove = True
                for ci in vert_cliques[v]:
                    if cover_count[ci] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove v
                    for ci in vert_cliques[v]:
                        cover_count[ci] -= 1
                    result.discard(v)
                    improved = True
        
        return result

    def local_search_swap(result, cliques, vert_cliques, n):
        """
        Try swapping one vertex out for another to reduce size or enable further reductions.
        """
        result = set(result)
        m = len(cliques)
        
        # Build coverage count
        cover_count = [0] * m
        for v in result:
            for ci in vert_cliques[v]:
                cover_count[ci] += 1
        
        improved = True
        iters = 0
        max_iters = 100
        while improved and iters < max_iters and time.time() - start_time < DEADLINE:
            improved = False
            iters += 1
            
            # First try pure removals
            for v in list(result):
                can_remove = True
                for ci in vert_cliques[v]:
                    if cover_count[ci] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for ci in vert_cliques[v]:
                        cover_count[ci] -= 1
                    result.discard(v)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swaps: remove v, add w where w covers all cliques that v uniquely covers
            best_swap = None
            for v in result:
                # Find cliques that would become uncovered if v is removed
                unique_cliques = set()
                for ci in vert_cliques[v]:
                    if cover_count[ci] <= 1:
                        unique_cliques.add(ci)
                
                if not unique_cliques:
                    continue
                
                # Try each candidate w not in result
                for w in range(n):
                    if w in result:
                        continue
                    # Check if w covers all unique_cliques
                    w_cliques = vert_cliques[w]
                    if all(ci in w_cliques for ci in unique_cliques):
                        # This swap is valid; it doesn't reduce size but may enable future reductions
                        # Only accept if it could lead to improvement (heuristic: w covers more total)
                        if len(w_cliques) > len(vert_cliques[v]):
                            best_swap = (v, w)
                            break
                if best_swap:
                    break
            
            if best_swap:
                v, w = best_swap
                # Remove v
                for ci in vert_cliques[v]:
                    cover_count[ci] -= 1
                result.discard(v)
                # Add w
                for ci in vert_cliques[w]:
                    cover_count[ci] += 1
                result.add(w)
                improved = True
        
        return result

    def iterated_local_search(cliques, n, vert_cliques, max_iters=50):
        """
        Run multiple greedy initializations with different orderings, then local search.
        Keep the best result.
        """
        if not cliques:
            return set()
        
        best = None
        best_size = float('inf')
        
        # Try several different greedy orderings
        for seed in [None, 42, 123, 456, 789, 1000, 2000, 3000]:
            if time.time() - start_time > DEADLINE:
                break
            init = greedy_hit(cliques, n, vert_cliques, order_seed=seed)
            sol = local_search_reduce(init, cliques, vert_cliques)
            sol = local_search_swap(sol, cliques, vert_cliques, n)
            if len(sol) < best_size:
                best_size = len(sol)
                best = set(sol)
        
        # Iterated local search: perturb best solution
        if best is not None:
            for _ in range(max_iters):
                if time.time() - start_time > DEADLINE:
                    break
                
                # Perturb: randomly remove 1-2 vertices and re-add to fix
                perturbed = set(best)
                if len(perturbed) <= 1:
                    break
                
                # Remove 1-2 random vertices
                to_remove = list(perturbed)
                _random.shuffle(to_remove)
                num_remove = min(2, len(to_remove))
                for v in to_remove[:num_remove]:
                    perturbed.discard(v)
                
                # Fix: add vertices to cover uncovered cliques
                m = len(cliques)
                cover_count = [0] * m
                for v in perturbed:
                    for ci in vert_cliques[v]:
                        cover_count[ci] += 1
                
                uncovered = [ci for ci in range(m) if cover_count[ci] == 0]
                for ci in uncovered:
                    # Find a vertex in clique ci not already in perturbed
                    c = cliques[ci]
                    candidates = [v for v in c if v not in perturbed]
                    if candidates:
                        # Pick the one with highest coverage
                        best_v = max(candidates, key=lambda v: len(vert_cliques[v]))
                        perturbed.add(best_v)
                        for c2 in vert_cliques[best_v]:
                            cover_count[c2] += 1
                
                # Local search
                sol = local_search_reduce(perturbed, cliques, vert_cliques)
                sol = local_search_swap(sol, cliques, vert_cliques, n)
                
                if len(sol) < best_size:
                    best_size = len(sol)
                    best = set(sol)
        
        return best if best is not None else set()

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges = random_graph(n, p, seed)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Enumerate maximal cliques of size >= 2
        cliques = bron_kerbosch(adj, n)
        
        if not cliques:
            results.append([])
            continue
        
        # Build membership structure
        vert_cliques = build_clique_membership(cliques, n)
        
        # Run improved search
        hit = iterated_local_search(cliques, n, vert_cliques)
        
        results.append(sorted(hit))
    
    return results
