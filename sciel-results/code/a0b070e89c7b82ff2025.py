import time
import random as _random

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

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

    def eliminate_dominated(vert_cliques, n):
        """
        Remove vertices whose clique coverage is a strict subset of another vertex's.
        Returns a dict: dominated_vertex -> dominating_vertex, or None if not dominated.
        """
        dominated_by = {}
        for v in range(n):
            if v in dominated_by:
                continue
            for w in range(n):
                if v == w or w in dominated_by:
                    continue
                if vert_cliques[v] and vert_cliques[v] < vert_cliques[w]:
                    dominated_by[v] = w
                    break
        return dominated_by

    def greedy_hit(cliques, n, vert_cliques, order_seed=None, dominated_by=None):
        """
        Greedy set cover with optional randomized ordering and dominance pruning.
        Returns a set of vertices
        """
        if not cliques:
            return set()
        
        m = len(cliques)
        uncovered = set(range(m))
        result = set()
        
        # Create ordering
        if order_seed is not None:
            rng = _random.Random(order_seed)
            order = list(range(n))
            rng.shuffle(order)
        else:
            # Default: sort by total coverage descending
            order = sorted(range(n), key=lambda v: -len(vert_cliques[v]))
        
        # If dominance info provided, skip dominated vertices in ordering
        if dominated_by:
            order = [v for v in order if v not in dominated_by]
        
        while uncovered:
            if time.time() - start_time > DEADLINE:
                break
            best_v = -1
            best_covered = 0
            for v in order:
                if v in result:
                    continue
                covered = len(uncovered & vert_cliques[v])
                if covered > best_covered:
                    best_covered = covered
                    best_v = v
            if best_v == -1 or best_covered == 0:
                break
            result.add(best_v)
            uncovered -= vert_cliques[best_v]
        
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
        max_iters = 200
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
                        # Accept if w covers at least as many cliques, or if it enables reduction
                        if len(w_cliques) >= len(vert_cliques[v]):
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

    def iterated_local_search(cliques, n, vert_cliques, dominated_by=None, max_iters=80):
        """
        Run multiple greedy initializations with different orderings, then local search.
        Keep the best result.
        """
        if not cliques:
            return set()
        
        best = None
        best_size = float('inf')
        
        # Try several different greedy orderings
        seeds = [None, 42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000]
        for seed in seeds:
            if time.time() - start_time > DEADLINE:
                break
            init = greedy_hit(cliques, n, vert_cliques, order_seed=seed, dominated_by=dominated_by)
            sol = local_search_reduce(init, cliques, vert_cliques)
            sol = local_search_swap(sol, cliques, vert_cliques, n)
            if len(sol) < best_size:
                best_size = len(sol)
                best = set(sol)
        
        # Iterated local search: perturb best solution
        if best is not None:
            for it in range(max_iters):
                if time.time() - start_time > DEADLINE:
                    break
                
                # Perturb: randomly remove 1 vertex and re-add to fix
                perturbed = set(best)
                if len(perturbed) <= 1:
                    break
                
                # Remove 1 random vertex
                to_remove = list(perturbed)
                _random.shuffle(to_remove)
                v_remove = to_remove[0]
                perturbed.discard(v_remove)
                
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
        
        # Eliminate dominated vertices
        dominated_by = eliminate_dominated(vert_cliques, n)
        
        # Run improved search
        hit = iterated_local_search(cliques, n, vert_cliques, dominated_by=dominated_by)
        
        results.append(sorted(hit))
    
    return results
