import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def build_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj):
        n = len(adj)
        maximal_cliques = []
        
        def expand(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(frozenset(R))
                return
            
            # Pivot selection: choose vertex in P union X with max degree in P
            # This reduces branching
            if P or X:
                pivot = max(P | X, key=lambda v: len(adj[v] & P))
                candidates = P - adj[pivot]
            else:
                candidates = P

            for v in list(candidates):
                expand(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        expand(set(), set(range(n)), set())
        return maximal_cliques

    def solve_instance(n, p, seed):
        adj = build_graph(n, p, seed)
        cliques = bron_kerbosch(adj)
        
        if not cliques:
            return []
        
        # Filter cliques of size >= 2 (Bron-Kerbosch with len>=2 check already does this if we add it there, 
        # but let's be safe. The prompt says "meet every maximal clique with at least two vertices".
        # My bron_kerbosch above adds if len(R)>=2. So all returned cliques are size >= 2.)
        
        # Greedy Initial Solution
        # We want to hit all cliques.
        # active_cliques: list of cliques not yet hit
        active_cliques = [set(c) for c in cliques]
        
        def get_greedy_solution(active_cliques):
            if not active_cliques:
                return set()
            
            sol = set()
            # Map vertex to list of indices of cliques it is in? 
            # Or just iterate.
            # Since n is small, we can just count.
            
            # To speed up, let's keep a list of cliques and remove them as we go.
            # But removing from list is O(N).
            # Alternative: keep a count of how many times each vertex appears in active cliques.
            
            # Let's just do simple greedy on the list.
            # Sort cliques by size? No, just pick vertex with max coverage.
            
            current_cliques = active_cliques[:]
            
            while current_cliques:
                # Count occurrences
                counts = {}
                for c in current_cliques:
                    for v in c:
                        counts[v] = counts.get(v, 0) + 1
                
                if not counts:
                    break # Should not happen if cliques non-empty
                    
                # Pick vertex with max count
                best_v = max(counts, key=lambda v: counts[v])
                sol.add(best_v)
                
                # Remove all cliques containing best_v
                current_cliques = [c for c in current_cliques if best_v not in c]
                
            return sol

        def is_valid_solution(sol, cliques):
            for c in cliques:
                if not (sol & c):
                    return False
            return True

        def reduce_solution(sol, cliques):
            # Try to remove each vertex if possible
            changed = True
            while changed:
                changed = False
                for v in list(sol):
                    new_sol = sol - {v}
                    if is_valid_solution(new_sol, cliques):
                        sol = new_sol
                        changed = True
            return sol

        # Initial Greedy
        best_sol = get_greedy_solution(active_cliques)
        best_sol = reduce_solution(best_sol, cliques)
        best_size = len(best_sol)
        
        current_sol = set(best_sol)
        
        # Local Search
        # While time remains
        while time.time() < deadline:
            # Perturb: remove a random vertex from current_sol
            if not current_sol:
                break
                
            # Try to find a better solution by modifying current_sol
            # Strategy: Pick a random vertex v in current_sol, remove it.
            # Then greedily add vertices to cover any cliques that become uncovered.
            
            # To avoid getting stuck in a loop, we can try all single-vertex removals?
            # Or just random.
            
            # Let's try all single vertex removals first (Hill Climbing)
            improved = False
            for v in list(current_sol):
                trial_sol = current_sol - {v}
                # Check if valid
                if is_valid_solution(trial_sol, cliques):
                    # It is valid and smaller!
                    current_sol = trial_sol
                    improved = True
                    break # Restart loop with new smaller set
            
            if improved:
                if len(current_sol) < best_size:
                    best_sol = set(current_sol)
                    best_size = len(current_sol)
                continue # Re-evaluate with new current_sol
            
            # If no single removal works, try "swap" or "replace"
            # Pick a random v in current_sol, remove it.
            # Find cliques uncovered.
            # Add vertices to cover them.
            
            v_to_remove = random.choice(list(current_sol))
            trial_sol = current_sol - {v_to_remove}
            
            # Find uncovered cliques
            uncovered = [c for c in cliques if not (trial_sol & c)]
            
            if not uncovered:
                # Should have been caught above, but just in case
                current_sol = trial_sol
                if len(current_sol) < best_size:
                    best_sol = set(current_sol)
                    best_size = len(current_sol)
                continue

            # Greedily cover uncovered cliques
            # We need to add vertices to trial_sol such that all cliques in 'uncovered' are hit.
            # Note: Adding a vertex might also hit other cliques, but we only care about 'uncovered' for the immediate fix.
            
            active_unc = uncovered[:]
            new_vertices = set()
            
            while active_unc:
                # Count frequency in active_unc
                counts = {}
                for c in active_unc:
                    for u in c:
                        if u not in trial_sol: # Only consider adding vertices not already there
                            counts[u] = counts.get(u, 0) + 1
                
                if not counts:
                    # This means we cannot cover the remaining cliques with new vertices?
                    # Impossible, unless the cliques are empty or something.
                    # Break to avoid infinite loop
                    break
                
                best_v = max(counts, key=lambda u: counts[u])
                new_vertices.add(best_v)
                trial_sol.add(best_v)
                
                # Remove cliques hit by best_v
                active_unc = [c for c in active_unc if best_v not in c]
            
            # Now we have a valid solution: current_sol - {v_to_remove} + new_vertices
            # Reduce it
            candidate_sol = reduce_solution(trial_sol, cliques)
            
            if len(candidate_sol) < len(current_sol):
                current_sol = candidate_sol
                if len(current_sol) < best_size:
                    best_sol = set(current_sol)
                    best_size = len(current_sol)
            elif len(candidate_sol) == len(current_sol) and random.random() < 0.5:
                # Accept equal size to escape local minima?
                # Only if it's different
                if candidate_sol != current_sol:
                    current_sol = candidate_sol
            
        return sorted(best_sol)

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        res = solve_instance(n, p, seed)
        out.append(res)
        
    return out
