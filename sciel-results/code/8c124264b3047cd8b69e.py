import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [[False] * n for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u][v] = True
                    adj[v][u] = True
        return adj

    def bron_kerbosch(adj, n):
        """
        Enumerate all maximal cliques using Bron-Kerbosch with pivot.
        Uses bitmasks for speed.
        """
        neigh = [0] * n
        for u in range(n):
            mask = 0
            for v in range(n):
                if adj[u][v]:
                    mask |= (1 << v)
            neigh[u] = mask
        
        cliques = []
        all_mask = (1 << n) - 1
        
        def expand(R, P, X):
            if not P and not X:
                cliques.append(R)
                return
            
            union = P | X
            if union:
                best_u = -1
                best_count = -1
                u = union
                while u:
                    lsb = u & (-u)
                    v = lsb.bit_length() - 1
                    u ^= lsb
                    count = bin(P & neigh[v]).count('1')
                    if count > best_count:
                        best_count = count
                        best_u = v
                u = best_u
            else:
                u = None
                
            if u is not None:
                candidates = P & ~neigh[u]
            else:
                candidates = P
                
            while candidates:
                lsb = candidates & (-candidates)
                v = lsb.bit_length() - 1
                candidates ^= lsb
                
                new_R = R | lsb
                new_P = P & neigh[v]
                new_X = X & neigh[v]
                expand(new_R, new_P, new_X)
                
                P &= ~lsb
                X |= lsb
                
        expand(0, all_mask, 0)
        return cliques

    def is_valid_hit_set(sol_set, cliques):
        """Check if sol_set hits all cliques."""
        for c in cliques:
            if not sol_set.intersection(c):
                return False
        return True

    def local_search_reduce(sol_list, cliques):
        """
        Try to remove vertices from the solution one by one if the set remains valid.
        Returns a potentially smaller valid hitting set.
        """
        S = set(sol_list)
        improved = True
        while improved:
            improved = False
            # Iterate over a copy of the list to allow modification
            for v in list(S):
                S_test = S - {v}
                if is_valid_hit_set(S_test, cliques):
                    S = S_test
                    improved = True
                    break # Restart the loop as validity of other removals might change
        return list(S)

    def greedy_set_cover(cliques, n, rng):
        """
        Greedy set cover algorithm with randomization.
        cliques: list of sets of vertices.
        n: number of vertices.
        rng: random.Random instance for tie-breaking/noise.
        """
        # Map each clique to an index
        m = len(cliques)
        if m == 0:
            return []
        
        # uncovered_cliques: set of indices of cliques not yet hit
        uncovered = set(range(m))
        selected = []
        
        # Precompute for each vertex, which cliques it belongs to
        # vertex_to_cliques[v] = list of clique indices containing v
        vertex_to_cliques = [[] for _ in range(n)]
        for i, c in enumerate(cliques):
            for v in c:
                vertex_to_cliques[v].append(i)
        
        while uncovered:
            # Calculate coverage for each vertex
            counts = [0] * n
            for v in range(n):
                # Count how many uncovered cliques this vertex hits
                cnt = 0
                for idx in vertex_to_cliques[v]:
                    if idx in uncovered:
                        cnt += 1
                counts[v] = cnt
            
            # Find max count
            max_count = max(counts)
            
            if max_count == 0:
                # Should not happen if uncovered is not empty and cliques are non-empty
                break
            
            # Get candidates with max count
            candidates = [v for v in range(n) if counts[v] == max_count]
            
            # Randomized selection: pick from top candidates
            # To introduce diversity, we can pick the best with prob 0.8, 
            # or a random one from top 3 if available, etc.
            # Simple approach: pick random from candidates
            v = rng.choice(candidates)
            
            selected.append(v)
            
            # Update uncovered
            for idx in vertex_to_cliques[v]:
                if idx in uncovered:
                    uncovered.remove(idx)
            
            if not uncovered:
                break
                
        return selected

    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        all_cliques = bron_kerbosch(adj, n)
        
        # Filter cliques with size >= 2
        relevant_cliques_bitmasks = [c for c in all_cliques if bin(c).count('1') >= 2]
        
        if not relevant_cliques_bitmasks:
            return []
        
        # Convert bitmasks to sets of vertices
        cliques = []
        for c in relevant_cliques_bitmasks:
            s = set()
            temp = c
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                s.add(v)
                temp ^= lsb
            cliques.append(s)
        
        # Multi-start greedy with local search
        best_sol = None
        best_size = float('inf')
        
        # Determine number of restarts based on time
        elapsed = time.time() - start_time
        time_left = deadline - elapsed
        # Reserve 10s for other instances and overhead
        # We have 3 instances. This is one of them.
        # Let's allocate ~30s per instance for search.
        # If we have plenty of time, do more restarts.
        
        # Fixed number of restarts is safer to avoid running out of time.
        # 100 restarts of greedy on n=40 is very fast (<1s).
        num_restarts = 200
        
        # Use a deterministic seed for the solver's internal randomness to ensure reproducibility?
        # The task doesn't require reproducibility of the solver, only the graph generation.
        # But using a fixed seed for the solver makes debugging easier.
        solver_rng = random.Random(42)
        
        for i in range(num_restarts):
            # Check time
            if time.time() > deadline - 5:
                break
            
            sol = greedy_set_cover(cliques, n, solver_rng)
            sol = local_search_reduce(sol, cliques)
            
            # Verify validity (safety check)
            if not is_valid_hit_set(set(sol), cliques):
                # Fallback to a valid greedy solution if local search broke it (shouldn't happen)
                sol = greedy_set_cover(cliques, n, solver_rng)
                if not is_valid_hit_set(set(sol), cliques):
                    # Extremely unlikely, but ensure we return something valid
                    # Just pick vertices from the first clique? No, must hit ALL.
                    # Greedy always produces valid sets.
                    continue

            if len(sol) < best_size:
                best_size = len(sol)
                best_sol = sol
                
                # If we found a size 1 solution (only possible if one vertex hits all cliques), 
                # that's optimal.
                if best_size == 1:
                    break
        
        if best_sol is None:
            # Fallback: standard greedy
            best_sol = greedy_set_cover(cliques, n, solver_rng)
            best_sol = local_search_reduce(best_sol, cliques)
            
        return sorted(best_sol)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 10:
            # Not enough time for this instance, return empty (invalid) or best effort?
            # Better to return a quick greedy solution
            adj = random_graph(n, p, seed)
            all_cliques = bron_kerbosch(adj, n)
            relevant = [c for c in all_cliques if bin(c).count('1') >= 2]
            if not relevant:
                results.append([])
            else:
                cliques = []
                for c in relevant:
                    s = set()
                    temp = c
                    while temp:
                        lsb = temp & (-temp)
                        v = lsb.bit_length() - 1
                        s.add(v)
                        temp ^= lsb
                    cliques.append(s)
                # Quick greedy
                sol = greedy_set_cover(cliques, n, random.Random(0))
                sol = local_search_reduce(sol, cliques)
                results.append(sorted(sol))
            continue
            
        sol = solve_instance(n, p, seed)
        results.append(sol)
        
    return results
