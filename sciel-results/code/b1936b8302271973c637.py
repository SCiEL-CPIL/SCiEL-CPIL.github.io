import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

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
        
        vertex_cliques = defaultdict(set)
        for i, c in enumerate(valid_cliques):
            for v in c:
                vertex_cliques[v].add(i)
                
        best_set = None
        best_size = float('inf')
        
        rng_solver = random.Random(seed + 123)
        
        # Pre-compute clique intersection graph for potential future use or analysis
        # clique_adj[ci] = set of cj such that valid_cliques[ci] & valid_cliques[cj] is non-empty
        # This helps identify structural properties, though the primary strategy remains greedy + local search
        # on the original vertex-clique incidence structure.
        
        # Strategy: Randomized Greedy with Local Search (Reduction)
        # This approach is robust for hitting set problems.
        
        # Increase restarts significantly since the greedy step is O(n * C) worst case but often much less
        # and local search is fast.
        num_restarts = 2000
        
        for run in range(num_restarts):
            if time.time() > deadline:
                break
                
            hit_cliques = set()
            current_set = set()
            
            # Greedy Phase: Max Coverage
            while len(hit_cliques) < C:
                best_v = -1
                max_hits = -1
                candidates = []
                
                for v in range(n):
                    if v in current_set:
                        continue
                    hits = 0
                    for ci in vertex_cliques[v]:
                        if ci not in hit_cliques:
                            hits += 1
                    if hits > max_hits:
                        max_hits = hits
                        best_v = v
                        candidates = [v]
                    elif hits == max_hits and hits > 0:
                        candidates.append(v)
                        
                if max_hits == 0:
                    break
                    
                v = rng_solver.choice(candidates)
                current_set.add(v)
                for ci in vertex_cliques[v]:
                    hit_cliques.add(ci)
                    
            # Local Search: Reduction
            # Try to remove vertices that are redundant.
            improved = True
            while improved:
                improved = False
                vs = list(current_set)
                rng_solver.shuffle(vs)
                for v in vs:
                    if time.time() > deadline:
                        break
                    can_remove = True
                    for ci in vertex_cliques[v]:
                        c = valid_cliques[ci]
                        found_other = False
                        for u in c:
                            if u != v and u in current_set:
                                found_other = True
                                break
                        if not found_other:
                            can_remove = False
                            break
                    if can_remove:
                        current_set.remove(v)
                        improved = True
                        
            if len(current_set) < best_size:
                best_size = len(current_set)
                best_set = set(current_set)
                
                # If we found a very small set, we can try to be even more aggressive with polishing
                # or break early if we hit a theoretical lower bound (not easily computed here)
                
        # Additional Polish: Try swap-based local search on the best set found
        if best_set:
            polish_set = set(best_set)
            # Try single swaps: remove v, add w if it improves (reduces size or keeps size but allows further reduction)
            # Since we are minimizing size, a swap is only useful if it allows removing more than one vertex later,
            # or if it allows removing one and adding none (which is just reduction).
            # A more effective polish is to try adding a vertex that covers many cliques currently covered by
            # multiple vertices, allowing those to be removed.
            
            for _ in range(10): # Polish iterations
                if time.time() > deadline:
                    break
                improved = False
                
                # Try to find a vertex w not in set that, if added, allows removal of k > 1 vertices
                # Or more simply, try to find a set of 2 vertices in current_set that can be replaced by 1.
                
                # Simpler approach: Just run reduction again with a different random order, 
                # and also try "add and reduce"
                
                # Add and Reduce:
                # Pick a random vertex w not in set
                # Add w
                # Run reduction
                # If size decreased, keep it
                candidates_to_add = [v for v in range(n) if v not in polish_set]
                if candidates_to_add:
                    w = rng_solver.choice(candidates_to_add)
                    trial_set = set(polish_set)
                    trial_set.add(w)
                    
                    # Run reduction on trial_set
                    improved_trial = True
                    while improved_trial:
                        improved_trial = False
                        vs = list(trial_set)
                        rng_solver.shuffle(vs)
                        for v in vs:
                            if time.time() > deadline:
                                break
                            can_remove = True
                            for ci in vertex_cliques[v]:
                                c = valid_cliques[ci]
                                found_other = False
                                for u in c:
                                    if u != v and u in trial_set:
                                        found_other = True
                                        break
                                if not found_other:
                                    can_remove = False
                                    break
                            if can_remove:
                                trial_set.remove(v)
                                improved_trial = True
                                
                    if len(trial_set) < len(polish_set):
                        polish_set = trial_set
                        best_size = len(polish_set)
                        best_set = set(polish_set)
                        improved = True
                else:
                    break
                    
                if not improved:
                    break

        return sorted(best_set) if best_set else []

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            edges = instance(n, p, seed)
            touched = {x for e in edges for x in e}
            results.append(sorted(touched))
            continue
            
        res = solve_instance(n, p, seed)
        results.append(res)
        
    return results
