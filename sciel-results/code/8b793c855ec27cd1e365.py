import time
from collections import defaultdict

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

    def calculate_lower_bound(uncovered_cliques):
        """
        Calculate a lower bound on the number of vertices needed to cover uncovered_cliques.
        We find a maximal set of pairwise disjoint cliques.
        """
        if not uncovered_cliques:
            return 0
        
        disjoint_count = 0
        remaining = list(uncovered_cliques)
        
        # Sort by size to pick small cliques first? Or just greedy?
        # Greedy: pick a clique, remove all overlapping ones.
        # To maximize the count of disjoint cliques, picking smaller cliques first might be better.
        remaining.sort(key=len)
        
        i = 0
        while i < len(remaining):
            c = remaining[i]
            disjoint_count += 1
            # Remove all cliques that intersect with c
            # We need to filter the rest of the list
            new_remaining = []
            for j in range(i + 1, len(remaining)):
                if not (c & remaining[j]):
                    new_remaining.append(remaining[j])
            remaining = new_remaining
            i = 0 # Reset index since we rebuilt the list
            
        return disjoint_count

    def branch_and_bound(cliques, n):
        """
        Exact Branch and Bound solver for Hitting Set.
        """
        if not cliques:
            return set()
        
        # Initial greedy solution to set an upper bound
        # Simple greedy: pick vertex in most cliques
        vertex_counts = [0] * n
        for c in cliques:
            for v in c:
                vertex_counts[v] += 1
        
        initial_solution = set()
        uncovered = set(range(len(cliques)))
        while uncovered:
            best_v = -1
            max_cover = -1
            for v in range(n):
                if v in initial_solution:
                    continue
                cover = 0
                for i in uncovered:
                    if v in cliques[i]:
                        cover += 1
                if cover > max_cover:
                    max_cover = cover
                    best_v = v
            if best_v == -1:
                break
            initial_solution.add(best_v)
            uncovered = {i for i in uncovered if best_v not in cliques[i]}
        
        best_solution = frozenset(initial_solution)
        best_size = len(best_solution)
        
        # Precompute for each vertex, which cliques it covers
        # vertex_cliques[v] = set of indices of cliques containing v
        vertex_cliques = [set() for _ in range(n)]
        for i, c in enumerate(cliques):
            for v in c:
                vertex_cliques[v].add(i)
        
        def search(uncovered_indices, current_solution, current_size):
            nonlocal best_solution, best_size
            
            if time.time() - start_time > DEADLINE:
                return
            
            if not uncovered_indices:
                if current_size < best_size:
                    best_size = current_size
                    best_solution = frozenset(current_solution)
                return
            
            # Lower bound
            lb = calculate_lower_bound([cliques[i] for i in uncovered_indices])
            if current_size + lb >= best_size:
                return
            
            # Branching: Pick a clique to branch on
            # Heuristic: Pick the clique with fewest vertices that are still "available" or just smallest clique?
            # Smallest clique reduces branching factor.
            # Also prefer cliques that are "hard" to cover?
            min_clique_idx = None
            min_clique_size = float('inf')
            for i in uncovered_indices:
                c = cliques[i]
                if len(c) < min_clique_size:
                    min_clique_size = len(c)
                    min_clique_idx = i
            
            if min_clique_idx is None:
                return
                
            target_clique = cliques[min_clique_idx]
            
            # Order vertices in target_clique by frequency in uncovered cliques (desc)
            # This might find a good solution faster
            v_scores = []
            for v in target_clique:
                score = sum(1 for i in uncovered_indices if v in cliques[i])
                v_scores.append((score, v))
            v_scores.sort(reverse=True)
            
            for score, v in v_scores:
                if time.time() - start_time > DEADLINE:
                    return
                
                new_solution = current_solution | {v}
                new_size = current_size + 1
                
                # Update uncovered
                # Remove all cliques that contain v
                new_uncovered = set()
                for i in uncovered_indices:
                    if v not in cliques[i]:
                        new_uncovered.add(i)
                
                search(new_uncovered, new_solution, new_size)
                
                # If we found a solution of size 1 (or 0), we can stop?
                # Since we are looking for minimum, if best_size == 1, we can't do better than 1 (unless 0, but cliques exist).
                # Actually if best_size == 1, we can return early from the whole thing? 
                # Not necessarily, but we can prune heavily.
                
        # Start search
        initial_uncovered = set(range(len(cliques)))
        search(initial_uncovered, set(), 0)
        
        return set(best_solution)

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
        
        # Use Branch and Bound
        hit = branch_and_bound(cliques, n)
        
        # Fallback to greedy if B&B didn't finish or returned empty (shouldn't happen)
        if not hit:
             # Fallback greedy
             vertex_clique_count = [0] * n
             for c in cliques:
                 for v in c:
                     vertex_clique_count[v] += 1
             uncovered = set(range(len(cliques)))
             result = set()
             while uncovered:
                 best_v = -1
                 best_covered = 0
                 for v in range(n):
                     if v in result:
                         continue
                     covered = 0
                     for i in uncovered:
                         if v in cliques[i]:
                             covered += 1
                     if covered > best_covered:
                         best_covered = covered
                         best_v = v
                 if best_v == -1 or best_covered == 0:
                     break
                 result.add(best_v)
                 uncovered = {i for i in uncovered if best_v not in cliques[i]}
             hit = result

        results.append(sorted(hit))
    
    return results
