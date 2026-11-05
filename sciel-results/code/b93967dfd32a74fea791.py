import time
import random
from collections import defaultdict, deque
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

    def bron_kerbosch(R, P, X, adj, n):
        if not P and not X:
            if len(R) >= 2:
                yield frozenset(R)
            return
        
        if P:
            pivot = max(P, key=lambda p: len(P & adj[p]))
        elif X:
            pivot = max(X, key=lambda p: len(P & adj[p]))
        else:
            return
            
        for v in list(P - adj[pivot]):
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            yield from bron_kerbosch(new_R, new_P, new_X, adj, n)
            P.remove(v)
            X.add(v)

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        all_cliques = list(bron_kerbosch(frozenset(), set(range(n)), set(), adj, n))
        valid_cliques = [c for c in all_cliques if len(c) >= 2]
        
        if not valid_cliques:
            return []
            
        C = len(valid_cliques)
        
        # Precompute clique membership for vertices
        # vertex_cliques[v] = list of indices of cliques containing v
        vertex_cliques = [[] for _ in range(n)]
        clique_vertices = [list(c) for c in valid_cliques]
        for i, c in enumerate(valid_cliques):
            for v in c:
                vertex_cliques[v].append(i)
                
        best_set = set()
        best_size = float('inf')
        
        rng_solver = random.Random(seed + 123)
        
        def greedy_set_cover():
            hit_cliques = [0] * C
            current_set = set()
            
            while any(c == 0 for c in hit_cliques):
                best_v = -1
                max_new_hits = -1
                candidates = []
                
                for v in range(n):
                    if v in current_set:
                        continue
                    new_hits = 0
                    for ci in vertex_cliques[v]:
                        if hit_cliques[ci] == 0:
                            new_hits += 1
                            
                    if new_hits > max_new_hits:
                        max_new_hits = new_hits
                        best_v = v
                        candidates = [v]
                    elif new_hits == max_new_hits and new_hits > 0:
                        candidates.append(v)
                        
                if max_new_hits == 0:
                    break
                    
                v = rng_solver.choice(candidates)
                current_set.add(v)
                for ci in vertex_cliques[v]:
                    hit_cliques[ci] += 1
                    
            return current_set, hit_cliques

        def local_search_reduce(current_set, hit_cliques):
            """
            Fast local search using hit counts.
            Returns (new_set, new_hit_cliques)
            """
            current_set = set(current_set)
            hit_cliques = list(hit_cliques)
            improved = True
            
            while improved:
                improved = False
                vs = list(current_set)
                rng_solver.shuffle(vs)
                
                for v in vs:
                    if time.time() > deadline:
                        return current_set, hit_cliques
                    
                    # Tentatively remove v
                    for ci in vertex_cliques[v]:
                        hit_cliques[ci] -= 1
                        if hit_cliques[ci] < 0:
                            # Should not happen if logic is correct, but safety
                            hit_cliques[ci] = 0
                    
                    # Check if any clique became uncovered
                    valid = True
                    for ci in vertex_cliques[v]:
                        if hit_cliques[ci] == 0:
                            valid = False
                            break
                            
                    if valid:
                        current_set.remove(v)
                        improved = True
                    else:
                        # Revert
                        for ci in vertex_cliques[v]:
                            hit_cliques[ci] += 1
                            
            return current_set, hit_cliques

        def local_search_swap(current_set, hit_cliques):
            """
            Try 1-swap: remove v, add w.
            Only valid if size doesn't increase (we want min size, so swap keeps size same, but might enable further reductions).
            Actually, standard local search for minimization tries to reduce size. 
            If we can't reduce size, we might try to rearrange to allow future reductions.
            But for simplicity and speed, let's stick to reduction first.
            If reduction stalls, we can try swaps that maintain size but change composition.
            """
            current_set = set(current_set)
            hit_cliques = list(hit_cliques)
            improved = True
            
            while improved:
                improved = False
                if time.time() > deadline:
                    return current_set, hit_cliques
                    
                vs = list(current_set)
                rng_solver.shuffle(vs)
                
                for v in vs:
                    if time.time() > deadline:
                        return current_set, hit_cliques
                        
                    # Try to remove v
                    for ci in vertex_cliques[v]:
                        hit_cliques[ci] -= 1
                    
                    uncovered = [ci for ci in vertex_cliques[v] if hit_cliques[ci] == 0]
                    
                    if not uncovered:
                        # Can remove
                        current_set.remove(v)
                        improved = True
                        continue
                    else:
                        # Must swap v with some w that covers at least one uncovered clique
                        # And ideally covers more or is "better"
                        # For speed, just try to find any w that fixes the uncovered cliques
                        candidates = []
                        for w in range(n):
                            if w in current_set:
                                continue
                            # Check if w covers all uncovered cliques
                            covers_all = True
                            for ci in uncovered:
                                if ci not in vertex_cliques[w]:
                                    covers_all = False
                                    break
                            if covers_all:
                                candidates.append(w)
                                
                        if candidates:
                            w = rng_solver.choice(candidates)
                            # Perform swap: remove v, add w
                            current_set.remove(v)
                            current_set.add(w)
                            
                            # Update hit counts
                            # v was already decremented above for its cliques.
                            # We need to increment for w's cliques.
                            for ci in vertex_cliques[w]:
                                hit_cliques[ci] += 1
                                
                            improved = True
                        else:
                            # Revert v removal
                            for ci in vertex_cliques[v]:
                                hit_cliques[ci] += 1

            return current_set, hit_cliques

        # Strategy 1: Greedy + Local Search
        current_set, hit_cliques = greedy_set_cover()
        current_set, hit_cliques = local_search_reduce(current_set, hit_cliques)
        
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = set(current_set)

        # Strategy 2: Multiple Randomized Greedy Restarts
        num_restarts = 50 # Reduced to ensure time safety
        
        for run in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            # Randomized Greedy
            hit_cliques = [0] * C
            current_set = set()
            
            # Shuffle vertex order for randomization
            vertices = list(range(n))
            rng_temp = random.Random(seed + 123 + run)
            rng_temp.shuffle(vertices)
            
            # We need a random tie-breaking greedy
            # Standard greedy picks best. Randomized greedy picks from top-k or random valid.
            # Let's just use the standard greedy but with a different random seed for tie-breaking
            # The existing greedy_set_cover uses rng_solver which is fixed.
            # Let's create a local rng for this restart.
            
            # Re-implementing a quick randomized greedy inline to avoid function overhead
            while any(c == 0 for c in hit_cliques):
                best_v = -1
                max_new_hits = -1
                candidates = []
                
                for v in vertices:
                    if v in current_set:
                        continue
                    new_hits = 0
                    for ci in vertex_cliques[v]:
                        if hit_cliques[ci] == 0:
                            new_hits += 1
                            
                    if new_hits > max_new_hits:
                        max_new_hits = new_hits
                        best_v = v
                        candidates = [v]
                    elif new_hits == max_new_hits and new_hits > 0:
                        candidates.append(v)
                        
                if max_new_hits == 0:
                    break
                    
                v = rng_temp.choice(candidates)
                current_set.add(v)
                for ci in vertex_cliques[v]:
                    hit_cliques[ci] += 1
                    
            # Local Search
            current_set, hit_cliques = local_search_reduce(current_set, hit_cliques)
            
            if len(current_set) < best_size:
                best_size = len(current_set)
                best_set = set(current_set)
                
            # Optional: Try Swap Search if time permits
            if time.time() < deadline - 10:
                current_set, hit_cliques = local_search_swap(current_set, hit_cliques)
                current_set, hit_cliques = local_search_reduce(current_set, hit_cliques)
                
                if len(current_set) < best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)

        return sorted(best_set) if best_set else []

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 20:  # Reserve 20s for remaining
            # Fallback: Return all vertices? No, that's huge.
            # Better fallback: Random subset?
            # Just return a valid but potentially large set quickly.
            # A vertex cover is valid. Let's find a maximal matching vertex cover.
            edges = instance(n, p, seed)
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            
            # Maximal Matching for Vertex Cover
            matched = [False] * n
            vc = set()
            for u, v in edges:
                if not matched[u] and not matched[v]:
                    matched[u] = True
                    matched[v] = True
                    vc.add(u)
                    vc.add(v)
            results.append(sorted(vc))
            continue
            
        res = solve_instance(n, p, seed)
        results.append(res)
        
    return results
