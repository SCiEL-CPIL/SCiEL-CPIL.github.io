import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin from 150s
    
    def get_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch_maximal_cliques(adj, n):
        """Find all maximal cliques using Bron-Kerbosch with pivot."""
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Choose pivot u from P ∪ X that maximizes |P ∩ N(u)|
            P_union_X = P | X
            if P_union_X:
                pivot = max(P_union_X, key=lambda u: len(P & adj[u]))
                candidates = P - adj[pivot]
            else:
                candidates = P
            
            for v in list(candidates):
                R.add(v)
                P_new = P & adj[v]
                X_new = X & adj[v]
                bk(R, P_new, X_new)
                R.remove(v)
                P.remove(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return cliques

    def min_transversal_bnb(n, cliques, time_limit):
        """
        Find minimum vertex set that hits every clique (transversal)
        using branch and bound with bitmasks.
        """
        if not cliques:
            return []
        
        # Convert cliques to bitmasks
        clique_masks = []
        for clique in cliques:
            mask = 0
            for v in clique:
                mask |= (1 << v)
            clique_masks.append(mask)
        
        # Sort cliques by size (smaller first for better pruning)
        clique_masks.sort(key=lambda m: bin(m).count('1'))
        num_cliques = len(clique_masks)
        
        # Greedy initial solution: pick vertices that cover most uncovered cliques
        # Each vertex covers a set of cliques
        vertex_clique_cover = [0] * n  # bitmask of cliques covered by vertex i
        for i, cm in enumerate(clique_masks):
            for v in range(n):
                if (cm >> v) & 1:
                    vertex_clique_cover[v] |= (1 << i)
        
        # Greedy: repeatedly pick vertex covering most uncovered cliques
        uncovered = (1 << num_cliques) - 1
        greedy_cover = set()
        while uncovered:
            best_v = -1
            best_count = -1
            for v in range(n):
                if v in greedy_cover:
                    continue
                count = bin(uncovered & vertex_clique_cover[v]).count('1')
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v == -1 or best_count == 0:
                break
            greedy_cover.add(best_v)
            uncovered &= ~vertex_clique_cover[best_v]
        
        best_solution = set(greedy_cover)
        best_size = len(best_solution)
        
        # If greedy already covers everything, try to improve
        if uncovered == 0:
            # Try local search to shrink
            improved = True
            while improved and time.time() < time_limit:
                improved = False
                for u in list(best_solution):
                    new_sol = best_solution - {u}
                    # Check if new_sol still covers all cliques
                    ok = True
                    for cm in clique_masks:
                        covered = False
                        for v in new_sol:
                            if (cm >> v) & 1:
                                covered = True
                                break
                        if not covered:
                            ok = False
                            break
                    if ok:
                        best_solution = new_sol
                        best_size -= 1
                        improved = True
                        break
        
        # Branch and bound
        # State: (current_set_size, current_mask_of_covered_cliques, vertex_to_consider)
        # We'll use a recursive approach with pruning
        
        def bb_solve(covered_mask, current_size, start_vertex):
            """
            Try to find a solution of size < best_size.
            covered_mask: bitmask of cliques already covered
            current_size: number of vertices chosen so far
            start_vertex: minimum vertex index to consider (to avoid duplicates)
            Returns True if a better solution was found.
            """
            nonlocal best_solution, best_size
            
            if time.time() > time_limit:
                return False
            
            if covered_mask == (1 << num_cliques) - 1:
                # All cliques covered
                if current_size < best_size:
                    best_size = current_size
                    # We need to recover the actual solution - this is tricky in B&B
                    # For now, just update size; we'll reconstruct separately
                return True
            
            # Pruning: even if we pick all remaining vertices, can we beat best?
            remaining_vertices = n - start_vertex
            if current_size + 0 >= best_size:
                # Can't improve (we need strictly less, but let's be safe)
                pass
            
            # Upper bound on how many more cliques we can cover
            # Simple bound: remaining vertices each cover at most some cliques
            # Better bound: compute union of all vertex_clique_cover from start_vertex onwards
            potential_cover = 0
            for v in range(start_vertex, n):
                potential_cover |= vertex_clique_cover[v]
            
            if (covered_mask | potential_cover) != (1 << num_cliques) - 1:
                return False  # Can't cover all cliques
            
            # Try adding each vertex
            for v in range(start_vertex, n):
                if time.time() > time_limit:
                    return False
                
                new_covered = covered_mask | vertex_clique_cover[v]
                new_size = current_size + 1
                
                # Prune: if new_size >= best_size, no point
                if new_size >= best_size:
                    continue
                
                # Prune: if this vertex doesn't add any new coverage
                if new_covered == covered_mask:
                    continue
                
                if bb_solve(new_covered, new_size, v + 1):
                    # A better solution was found; we could stop here
                    # but continue to see if even better exists
                    pass
            
            return False
        
        # Run B&B from empty set
        bb_solve(0, 0, 0)
        
        # After B&B, best_size is optimal (or close). Now find a solution of that size.
        # Use a separate search to find the actual set
        if best_size < len(greedy_cover):
            # Find a set of size best_size that covers all cliques
            target_size = best_size
            
            def find_set_of_size_k(k, covered_mask, start_vertex, current_set):
                """Find a set of exactly k vertices that covers all cliques."""
                if time.time() > time_limit:
                    return None
                
                if k == 0:
                    if covered_mask == (1 << num_cliques) - 1:
                        return current_set
                    return None
                
                # Pruning: can we cover remaining cliques with k vertices?
                potential = 0
                for v in range(start_vertex, n):
                    potential |= vertex_clique_cover[v]
                if (covered_mask | potential) != (1 << num_cliques) - 1:
                    return None
                
                # Pruning: even picking best k vertices, can we cover?
                # Simple: just try
                for v in range(start_vertex, n):
                    if time.time() > time_limit:
                        return None
                    
                    new_covered = covered_mask | vertex_clique_cover[v]
                    result = find_set_of_size_k(k - 1, new_covered, v + 1, current_set | {v})
                    if result is not None:
                        return result
                
                return None
            
            sol = find_set_of_size_k(target_size, 0, 0, set())
            if sol is not None:
                best_solution = sol
        
        return sorted(best_solution)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > DEADLINE:
            # Fallback: return all non-isolated vertices
            adj = get_graph(n, p, seed)
            touched = set()
            for i in range(n):
                if adj[i]:
                    touched.add(i)
            results.append(sorted(touched))
            continue
            
        adj = get_graph(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(adj, n)
        
        # Allocate time based on remaining budget
        remaining_time = DEADLINE - time.time()
        # Give more time to earlier instances if they're harder
        time_limit = time.time() + remaining_time * 0.6  # Use 60% of remaining
        
        cover = min_transversal_bnb(n, cliques, time_limit)
        results.append(cover)

    return results
