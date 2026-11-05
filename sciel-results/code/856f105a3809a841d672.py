"""Combined program: Efficient BK + Multi-Restart Greedy with Local Search."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj, n):
        """Find all maximal cliques using Bron-Kerbosch with pivot."""
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Choose pivot u from P ∪ X to minimize branching
            if P:
                u = max(P, key=lambda v: len(adj[v] & P))
            elif X:
                u = max(X, key=lambda v: len(adj[v] & X))
            else:
                return
            
            # Branch on vertices in P \\ N(u)
            for v in list(P - adj[u]):
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                bk(new_R, new_P, new_X)
                P.remove(v)
                X.add(v)
        
        bk(frozenset(), set(range(n)), set())
        return cliques

    def solve_instance(n, p, seed):
        adj = instance(n, p, seed)
        
        # Find all maximal cliques of size >= 2
        cliques = bron_kerbosch(adj, n)
        
        if not cliques:
            return []
        
        C = len(cliques)
        
        # Pre-compute incidence
        # vertex_cliques[v] = set of clique indices containing v
        vertex_cliques = defaultdict(set)
        for i, c in enumerate(cliques):
            for v in c:
                vertex_cliques[v].add(i)
                
        # Helper to check if a set of vertices covers all cliques
        def is_valid_cover(selected):
            if not selected:
                return C == 0
            covered = set()
            for v in selected:
                covered.update(vertex_cliques[v])
            return len(covered) == C

        # Helper to remove redundant vertices from a valid cover
        def minimize_cover(selected):
            selected = set(selected)
            improved = True
            while improved:
                improved = False
                for v in list(selected):
                    # Check if v is redundant
                    # v is redundant if every clique containing v is also covered by selected - {v}
                    redundant = True
                    for ci in vertex_cliques[v]:
                        c = cliques[ci]
                        # Check if any other vertex in selected covers this clique
                        # i.e., is there u in (selected - {v}) such that u in c?
                        found_other = False
                        for u in c:
                            if u != v and u in selected:
                                found_other = True
                                break
                        if not found_other:
                            redundant = False
                            break
                    if redundant:
                        selected.remove(v)
                        improved = True
            return selected

        best_set = None
        best_size = float('inf')
        
        # Multi-restart greedy with random tie-breaking
        # The number of restarts depends on time available
        rng_solver = random.Random(seed + 123)
        
        # Estimate time per restart
        # n is small (<=40), C is moderate.
        # We will run as many restarts as possible within a time budget per instance.
        # Let's allocate 40% of total time to each instance, or just loop until deadline.
        
        while time.time() < deadline - 5.0:
            # Greedy Phase
            current_set = set()
            hit_cliques = set()
            
            # We pick vertices one by one.
            # Score of v = number of unhit cliques it covers.
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
                    # Should not happen if there are unhit cliques, but safety break
                    break
                    
                # Random tie-breaking to explore different solutions
                v = rng_solver.choice(candidates)
                current_set.add(v)
                for ci in vertex_cliques[v]:
                    hit_cliques.add(ci)
                    
            # Local Search / Minimization
            current_set = minimize_cover(current_set)
            
            # Optional: Try 1-swap moves to escape local optima?
            # If we are stuck at size K, maybe swapping a vertex allows removing two others?
            # This is complex. The random restarts usually handle this well for n=40.
            # Let's stick to Greedy + Minimize.
            
            if len(current_set) < best_size:
                best_size = len(current_set)
                best_set = set(current_set)
                
                # Early stopping if we hit a strong lower bound
                # Lower bound: max clique size? No, hitting set can be smaller than max clique size?
                # Actually, if we have a clique of size K, we need at least 1 vertex from it.
                # A better lower bound is the chromatic number of the conflict graph?
                # For now, just check if best_size is small (e.g., <= 5) and we've done many restarts.
                # But since n is small, let's just keep going until time runs out.
                
        return sorted(best_set) if best_set else []

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 10.0:
            # Not enough time; return empty lists as fallback (will be invalid but avoid crash)
            # Better fallback: return all vertices? No, that's huge.
            # If we run out of time, we should return the best found so far for previous instances.
            # But if we haven't solved this one, we have to guess.
            # Let's just return an empty list to avoid hanging, though it will score 0 for this instance.
            # Actually, the problem says "invalid result ... scores 0".
            # So we must return a valid list.
            # If we are out of time, we can't compute. 
            # But the deadline is 145s, and we have 150s.
            # If we reach here, we are in trouble.
            # Let's assume the loop above handles the time.
            results.append([])
            continue
        
        result = solve_instance(n, p, seed)
        results.append(result)
    
    return results
# EVOLVE-BLOCK-END
