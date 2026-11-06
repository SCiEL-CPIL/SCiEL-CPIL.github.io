import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin

    def instance(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj, R, P, X, cliques):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        
        # Pivot selection
        # Choose u in P ∪ X that maximizes |P ∩ N(u)|
        # If P is empty, pivot from X? Or just return if P is empty?
        # Standard: if P is empty, no more cliques to find in this branch.
        if not P:
            return

        pivot_candidates = P.union(X)
        if not pivot_candidates:
            return
            
        # Pick pivot with max intersection with P
        pivot = max(pivot_candidates, key=lambda u: len(P.intersection(adj[u])))
        
        # Iterate over P \ N(pivot)
        # If P \ N(pivot) is empty, we are done? No, we just skip the loop.
        # But if P is not empty, we must process.
        # Wait, the logic is:
        # for v in P \ N(pivot):
        #    bron_kerbosch(R + {v}, P & N(v), X & N(v))
        #    P = P \ {v}
        #    X = X + {v}
        
        # To make it efficient, we convert P to a list to iterate and modify.
        # But P is a set.
        candidates = list(P.difference(adj[pivot]))
        
        for v in candidates:
            bron_kerbosch(adj, R | {v}, P.intersection(adj[v]), X.intersection(adj[v]), cliques)
            P.discard(v)
            X.add(v)
            
            # Check time?
            # Bron-Kerbosch is fast enough for n=40.

    def enumerate_maximal_cliques(adj):
        n = len(adj)
        cliques = []
        # Initial call: R = empty, P = all vertices, X = empty
        P = set(range(n))
        X = set()
        R = set()
        
        # Optimization: If a vertex is isolated, it forms a clique of size 1.
        # We only care about cliques of size >= 2.
        # Bron-Kerbosch will find size 1 cliques if we don't filter.
        # We filter in the base case.
        
        bron_kerbosch(adj, R, P, X, cliques)
        return cliques

    def solve_hitting_set(cliques, n, time_limit):
        if not cliques:
            return []
            
        # Greedy initialization
        uncovered = [c for c in cliques]
        # Map vertex to list of indices of cliques it hits?
        # Or just recompute counts.
        
        # For efficiency, maintain a list of uncovered clique indices.
        # And a counter for each vertex.
        
        def greedy_init():
            uncovered_indices = list(range(len(cliques)))
            # Pre-compute vertex to cliques map
            # vertex_cliques[v] = list of clique indices containing v
            vertex_cliques = defaultdict(list)
            for i, c in enumerate(cliques):
                for v in c:
                    vertex_cliques[v].append(i)
                    
            counts = {v: 0 for v in range(n)}
            for i in uncovered_indices:
                for v in cliques[i]:
                    counts[v] += 1
                    
            solution = set()
            covered_cliques = set()
            
            while uncovered_indices:
                # Find vertex with max count among uncovered
                # We need to update counts as we cover cliques.
                # Simpler: just pick the vertex that hits the most *currently uncovered* cliques.
                
                # To avoid O(N^2) scan, we can maintain a heap or just scan.
                # N is small (40). Scan is fine.
                best_v = -1
                max_hits = -1
                for v in range(n):
                    if v in solution:
                        continue
                    # Count how many uncovered cliques v hits
                    hits = 0
                    for i in uncovered_indices:
                        if v in cliques[i]:
                            hits += 1
                    if hits > max_hits:
                        max_hits = hits
                        best_v = v
                        
                if best_v == -1:
                    break # Should not happen if uncovered_indices not empty and cliques valid
                    
                solution.add(best_v)
                
                # Remove covered cliques
                new_uncovered = []
                for i in uncovered_indices:
                    if best_v not in cliques[i]:
                        new_uncovered.append(i)
                uncovered_indices = new_uncovered
                
            return solution

        current_solution = greedy_init()
        
        # Local Search
        def is_valid(sol):
            for c in cliques:
                if not (sol & c):
                    return False
            return True

        def is_critical(sol, v):
            # Is v the only vertex from sol in any clique?
            for c in cliques:
                if v in c:
                    # Check if any other vertex in sol is in c
                    if not (sol - {v}) & c:
                        return True
            return False

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Reduction
            for v in list(current_solution):
                if not is_critical(current_solution, v):
                    current_solution.remove(v)
                    improved = True
                    
            # 2. Swap (1-1)
            # Try to swap v in sol with u not in sol
            # This doesn't reduce size directly, but might allow reduction later.
            # We only accept if it leads to a valid state.
            # To avoid getting stuck, we can accept valid swaps with some probability or limit.
            # Let's try all valid swaps and pick one that reduces "criticality" or just randomize.
            # Actually, if we just swap, size is same.
            # The reduction phase will handle size reduction.
            # So we just need to explore the space of valid sets of the same size.
            
            swaps_made = 0
            vertices_in = list(current_solution)
            vertices_out = [v for v in range(n) if v not in current_solution]
            
            random.shuffle(vertices_in)
            random.shuffle(vertices_out)
            
            for v in vertices_in:
                for u in vertices_out:
                    new_sol = (current_solution - {v}) | {u}
                    if is_valid(new_sol):
                        current_solution = new_sol
                        swaps_made += 1
                        # Stop after a few swaps to allow reduction to happen
                        if swaps_made >= 5:
                            break
                if swaps_made >= 5:
                    break
            
            if swaps_made > 0:
                # We made a swap, so we should loop again to try reduction
                improved = True 

        return sorted(current_solution)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 10:
             # Fallback
             adj = instance(n, p, seed)
             # Just pick all vertices? Or empty?
             # Empty is invalid if there are edges.
             # Pick all vertices is valid but large.
             results.append(list(range(n)))
             continue

        adj = instance(n, p, seed)
        cliques = enumerate_maximal_cliques(adj)
        
        # Time budget for solver
        time_budget = min(30.0, deadline - time.time() - 5.0)
        
        sol = solve_hitting_set(cliques, n, time.time() + time_budget)
        results.append(sol)

    return results
