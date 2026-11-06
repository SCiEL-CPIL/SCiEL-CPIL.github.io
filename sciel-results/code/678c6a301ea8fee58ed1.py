import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    def random_graph(n, p, seed):
        """Generate random graph, return adjacency as list of sets."""
        import random
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj
    
    def bron_kerbosch(adj, n):
        """Enumerate all maximal cliques using Bron-Kerbosch with pivot."""
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Pivot selection: pick vertex in P∪X with max |P ∩ N(u)|
            candidates = list(P | X)
            if candidates:
                pivot = max(candidates, key=lambda u: len(P & adj[u]))
                candidates_to_iter = P - adj[pivot]
            else:
                candidates_to_iter = P
            
            for v in list(candidates_to_iter):
                bk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return cliques
    
    def greedy_hit(cliques, n):
        """Greedy set cover: pick vertex in most uncovered cliques."""
        uncovered = list(cliques)
        if not uncovered:
            return set()
        
        # Count how many cliques each vertex is in
        from collections import defaultdict
        vertex_clique_count = defaultdict(int)
        for c in uncovered:
            for v in c:
                vertex_clique_count[v] += 1
        
        selected = set()
        while uncovered:
            # Find vertex in most uncovered cliques
            best_v = -1
            best_count = -1
            for v in range(n):
                count = sum(1 for c in uncovered if v in c)
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v == -1 or best_count <= 0:
                break
            
            selected.add(best_v)
            # Remove all cliques hit by best_v
            uncovered = [c for c in uncovered if best_v not in c]
        
        return selected
    
    def is_valid(solution, cliques):
        """Check if solution hits all cliques."""
        for c in cliques:
            if not (solution & c):
                return False
        return True
    
    def local_search(solution, cliques, n, time_limit):
        """Improve solution via removal and swap."""
        solution = set(solution)
        if not cliques:
            return solution
        
        # Precompute: for each vertex, which cliques it hits
        # And for each clique, which vertices hit it
        
        # Phase 1: Remove redundant vertices
        changed = True
        while changed:
            if time.time() > time_limit:
                break
            changed = False
            for v in list(solution):
                new_sol = solution - {v}
                if is_valid(new_sol, cliques):
                    solution = new_sol
                    changed = True
        
        # Phase 2: Try 1-swap improvements
        improved = True
        while improved:
            if time.time() > time_limit:
                break
            improved = False
            for v in list(solution):
                if time.time() > time_limit:
                    break
                new_sol = solution - {v}
                # Find cliques not hit by new_sol
                uncovered = [c for c in cliques if not (new_sol & c)]
                if not uncovered:
                    solution = new_sol
                    improved = True
                    break
                # Try adding each vertex not in solution
                for w in range(n):
                    if w in new_sol:
                        continue
                    # Check if w hits all uncovered cliques
                    if all(w in c for c in uncovered):
                        # This swap works: remove v, add w
                        if len(solution) > 0:  # only if we actually reduce or same size
                            new_sol2 = new_sol | {w}
                            if len(new_sol2) < len(solution):
                                solution = new_sol2
                                improved = True
                                break
                if improved:
                    break
        
        return solution
    
    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Fallback: use all vertices in edges
            adj = random_graph(n, p, seed)
            touched = set()
            for u in range(n):
                for v in adj[u]:
                    touched.add(u)
                    touched.add(v)
            results.append(sorted(touched))
            continue
        
        adj = random_graph(n, p, seed)
        cliques = bron_kerbosch(adj, n)
        
        if not cliques:
            results.append([])
            continue
        
        # Greedy initial solution
        sol = greedy_hit(cliques, n)
        
        # Local search improvement
        sol = local_search(sol, cliques, n, deadline - 5)
        
        results.append(sorted(sol))
    
    return results
