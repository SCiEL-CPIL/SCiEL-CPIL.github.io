"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch_maximal_cliques(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        max_cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    max_cliques.append(frozenset(R))
                return
            
            # Pivot selection: choose vertex in P ∪ X with max degree in P
            # This reduces recursion depth
            pivot_candidates = list(P) + list(X)
            if not pivot_candidates:
                return
            
            # Choose pivot with max |P ∩ N(u)|
            best_pivot = None
            max_common = -1
            for u in pivot_candidates:
                common = len(P & adj[u])
                if common > max_common:
                    max_common = common
                    best_pivot = u
            
            # Recurse on P \ N(pivot)
            # Optimization: if P is empty, handle directly
            candidates = list(P)
            # To reduce branching, we iterate over P \ N(pivot)
            # But standard BK iterates over P. The pivot optimization says:
            # We only need to consider vertices in P that are NOT adjacent to pivot.
            # If a vertex is adjacent to pivot, its clique is covered by pivot's branch.
            # Wait, the standard algorithm is:
            # for v in P \ N(pivot):
            #    BK(R + {v}, P ∩ N(v), X ∩ N(v))
            #    P.remove(v)
            #    X.add(v)
            
            # We need to be careful with sets.
            P_set = set(P)
            X_set = set(X)
            
            # Get candidates
            if best_pivot is not None:
                candidates = [v for v in P_set if v not in adj[best_pivot]]
            else:
                candidates = list(P_set)
                
            for v in candidates:
                new_R = R | {v}
                new_P = P_set & adj[v]
                new_X = X_set & adj[v]
                
                bk(new_R, new_P, new_X)
                
                P_set.remove(v)
                X_set.add(v)
                
                if time.time() > deadline - 5:
                    return

        bk(frozenset(), set(range(n)), set())
        return max_cliques

    def greedy_hitting_set(cliques, n, random_seed=0):
        """
        Find a small set of vertices hitting all cliques.
        Uses greedy set cover initialization + local search.
        """
        if not cliques:
            return set()
        
        # Map clique index to set of vertices
        # We want to select vertices such that every clique has at least one selected vertex.
        
        # Initial solution: Greedy Set Cover
        # For each vertex v, count how many cliques it hits.
        # Pick vertex hitting most uncovered cliques.
        
        cliques_list = list(cliques)
        # Convert frozensets to lists for faster intersection? Or keep as sets.
        
        # To speed up, let's pre-compute for each vertex which cliques it belongs to.
        vertex_to_cliques = {v: [] for v in range(n)}
        for i, c in enumerate(cliques_list):
            for v in c:
                vertex_to_cliques[v].append(i)
        
        uncovered = set(range(len(cliques_list)))
        solution = set()
        
        while uncovered:
            best_v = None
            max_hits = 0
            # Random tie-breaking for diversity? Or just deterministic for now.
            # We will run multiple restarts with different random orders.
            for v in range(n):
                if v in solution:
                    continue
                hits = 0
                for ci in vertex_to_cliques[v]:
                    if ci in uncovered:
                        hits += 1
                if hits > max_hits:
                    max_hits = hits
                    best_v = v
            
            if best_v is None or max_hits == 0:
                # Should not happen if cliques are non-empty
                break
                
            solution.add(best_v)
            for ci in vertex_to_cliques[best_v]:
                if ci in uncovered:
                    uncovered.discard(ci)
                    
            if time.time() > deadline - 5:
                break
        
        # Local Search: Try to swap vertices to reduce size
        # 1. Try to remove a vertex if all cliques are still covered.
        # 2. Try to swap a vertex with another that covers more.
        
        def is_valid(sol):
            # Check if sol hits all cliques
            for c in cliques_list:
                if not (c & sol):
                    return False
            return True
            
        def coverage_count(sol):
            # Return list of coverage counts for each clique
            counts = [0] * len(cliques_list)
            for v in sol:
                for ci in vertex_to_cliques[v]:
                    counts[ci] += 1
            return counts
            
        # Local Search Loop
        improved = True
        iterations = 0
        max_iters = 5000
        
        while improved and iterations < max_iters and time.time() < deadline - 10:
            improved = False
            iterations += 1
            
            # 1. Removal
            for v in list(solution):
                new_sol = solution - {v}
                if is_valid(new_sol):
                    solution = new_sol
                    improved = True
                    break
            
            if improved:
                continue
                
            # 2. Swap
            # Try swapping v in solution with w not in solution
            # This is O(|S| * (n-|S|) * |C|) which might be slow.
            # Optimization: Only swap if it reduces size? No, swap keeps size same.
            # We want to reduce size. So swap is only useful if it enables a subsequent removal?
            # Actually, standard local search for hitting set:
            # We can try to replace a set of vertices with a smaller set.
            # Given time constraints, simple removal is the most effective for small gains.
            # Let's try a 1-for-2 swap? No, that's complex.
            
            # Let's try a different local move:
            # Pick a random vertex in solution, try to replace it with a random vertex outside.
            # If the new solution is valid, keep it. (This doesn't reduce size, but might help escape local minima for removals)
            # Actually, if we just want to minimize size, removal is the key.
            # If no removal is possible, we are in a local minimum.
            # To escape, we can allow temporary invalidity (Simulated Annealing) or just accept the local minimum.
            # Given the graph sizes (n=30-40), the optimal solution is likely small (maybe 5-15 vertices).
            # Greedy is usually within a factor of ln(n) of optimal.
            
            # Let's try a simple random restart with different greedy orders if we have time.
            # But for now, let's just return the current best.
            break
            
        return solution

    results = []
    
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 10:
            # Fallback: return all vertices (guaranteed valid)
            results.append(list(range(n)))
            continue
            
        edges = random_graph(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(n, edges)
        
        # Filter cliques of size < 2? 
        # Task says "meet every maximal clique with at least two vertices".
        # So we only care about cliques of size >= 2.
        cliques = [c for c in cliques if len(c) >= 2]
        
        if not cliques:
            results.append([])
            continue
            
        best_sol = None
        best_size = float('inf')
        
        # Run greedy multiple times with random tie-breaking to find better local minima
        # Since n is small, we can afford a few restarts.
        num_restarts = 5
        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            # Shuffle vertex order for greedy to get different results
            vertices = list(range(n))
            random.seed(seed + r)
            random.shuffle(vertices)
            
            # Modified greedy: 
            # Instead of strictly max hits, we can use a randomized greedy.
            # But standard greedy is often good. Let's just use the standard one 
            # but maybe the "touched" set from the seed program was just all vertices.
            # My greedy should be much better.
            
            sol = greedy_hitting_set(cliques, n, random_seed=seed + r)
            
            if len(sol) < best_size:
                best_size = len(sol)
                best_sol = sol
                
        if best_sol is None:
            # Fallback
            best_sol = set(range(n))
            
        results.append(sorted(best_sol))
        
    return results
# EVOLVE-BLOCK-END
