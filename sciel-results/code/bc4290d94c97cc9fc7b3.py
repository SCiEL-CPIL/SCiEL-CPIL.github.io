"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 130  # safety margin before 150s kill

    def generate_graph(n, p, seed):
        """Generate a random graph using the same RNG as the seed program."""
        import random
        rng = random.Random(seed)
        edges = set()
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return adj, edges

    def bron_kerbosch_maximal_cliques(adj, n):
        """Enumerate all maximal cliques using Bron-Kerbosch with pivot."""
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Choose pivot u from P ∪ X to minimize branching
            if P:
                pivot_candidates = P
            else:
                pivot_candidates = X
            
            if not pivot_candidates:
                # No pivot, just iterate all of P
                for v in list(P):
                    P.remove(v)
                    bk(R | {v}, P & adj[v], X & adj[v])
                    X.add(v)
            else:
                # Pick pivot with max degree in P
                best_pivot = max(pivot_candidates, key=lambda u: len(adj[u] & P))
                # Only branch on vertices in P that are NOT neighbors of pivot
                for v in list(P - adj[best_pivot]):
                    P.remove(v)
                    bk(R | {v}, P & adj[v], X & adj[v])
                    X.add(v)
        
        bk(frozenset(), frozenset(range(n)), frozenset())
        return cliques

    def solve_min_transversal(cliques, n, time_limit=40):
        """Find minimum vertex set hitting all cliques via branch-and-bound."""
        if not cliques:
            return []
        
        # Represent cliques as frozensets
        # Convert to list of sets for efficiency
        clique_list = [set(c) for c in cliques]
        
        # Sort cliques by size (smaller first for better pruning)
        clique_list.sort(key=len)
        
        # Quick check: if any clique is size 1, that vertex must be in solution
        # But we only have cliques of size >= 2, so no forced singletons
        
        # Greedy initial solution: repeatedly pick vertex covering most uncovered cliques
        uncovered = list(range(len(clique_list)))
        solution = []
        remaining_cliques = [set(c) for c in clique_list]
        
        while uncovered:
            best_v = -1
            best_count = -1
            for v in range(n):
                count = sum(1 for i in uncovered if v in remaining_cliques[i])
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_count <= 0:
                break
            solution.append(best_v)
            new_uncovered = []
            for i in uncovered:
                if best_v not in remaining_cliques[i]:
                    new_uncovered.append(i)
            uncovered = new_uncovered
        
        best_solution = sorted(solution)
        best_size = len(best_solution)
        
        if best_size == 0:
            return []
        
        # Branch and bound
        # State: (cliques_to_cover, current_selection, index)
        # We'll use a recursive approach with memoization-like pruning
        
        sys.setrecursionlimit(100000)
        
        # Precompute: for each vertex, which cliques it covers
        vertex_cliques = [[] for _ in range(n)]
        for i, c in enumerate(clique_list):
            for v in c:
                vertex_cliques[v].append(i)
        
        # Sort vertices by number of cliques they cover (descending) for ordering
        vertex_order = sorted(range(n), key=lambda v: -len(vertex_cliques[v]))
        
        # Remove vertices that don't cover any clique
        useful_vertices = [v for v in vertex_order if vertex_cliques[v]]
        
        if not useful_vertices:
            return []
        
        # Check if we can do better than greedy
        # Lower bound: maximum matching on conflict graph? 
        # Simpler LB: for each clique, we need at least 1 vertex. 
        # Better LB: find a set of pairwise disjoint cliques -> need at least that many vertices
        
        # Find greedy packing of disjoint cliques for lower bound
        def lower_bound_disjoint(clique_indices):
            """Find max number of pairwise disjoint cliques (greedy)."""
            used = set()
            count = 0
            for i in sorted(clique_indices, key=lambda x: len(clique_list[x])):
                c = clique_list[i]
                if not c & used:
                    used |= c
                    count += 1
            return count
        
        global_lb = lower_bound_disjoint(range(len(clique_list)))
        
        # If greedy solution size equals lower bound, it's optimal
        if best_size <= global_lb:
            return best_solution
        
        # Branch and bound
        # We process cliques in order; at each step, pick a vertex from the first uncovered clique
        # Use bitmasks for clique coverage if feasible, else use sets
        
        # For efficiency with up to ~40 vertices and potentially many cliques,
        # use a set of remaining clique indices
        
        best = best_size
        best_sol = best_solution[:]
        
        def bnb(remaining_cliques, current_sel, sel_set):
            nonlocal best, best_sol
            
            if time.time() - start_time > time_limit:
                return
            
            if not remaining_cliques:
                if len(current_sel) < best:
                    best = len(current_sel)
                    best_sol = current_sel[:]
                return
            
            # Lower bound: current size + disjoint packing of remaining
            lb = len(current_sel) + lower_bound_disjoint(remaining_cliques)
            if lb >= best:
                return
            
            # Pick the smallest remaining clique to branch on
            min_idx = min(remaining_cliques, key=lambda i: len(clique_list[i]))
            min_clique = clique_list[min_idx]
            
            # Try each vertex in this clique
            # Order by how many other remaining cliques it covers
            candidates = sorted(min_clique, key=lambda v: -sum(1 for i in remaining_cliques if v in clique_list[i]))
            
            for v in candidates:
                if len(current_sel) + 1 >= best:
                    break
                # Check if v is already selected
                if v in sel_set:
                    continue
                # Add v, remove all cliques it covers
                new_remaining = set()
                for i in remaining_cliques:
                    if v not in clique_list[i]:
                        new_remaining.add(i)
                current_sel.append(v)
                sel_set.add(v)
                bnb(new_remaining, current_sel, sel_set)
                current_sel.pop()
                sel_set.discard(v)
        
        all_cliques = set(range(len(clique_list)))
        bnb(all_cliques, [], set())
        
        return sorted(best_sol)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() - start_time > DEADLINE:
            # Fallback: return greedy solutions for remaining
            for nn, pp, ss in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203))[len(results):]:
                adj, edges = generate_graph(nn, pp, ss)
                touched = {x for e in edges for x in e}
                results.append(sorted(touched))
            break
        
        adj, edges = generate_graph(n, p, seed)
        
        # Enumerate maximal cliques
        cliques = bron_kerbosch_maximal_cliques(adj, n)
        
        if not cliques:
            results.append([])
            continue
        
        # Solve min transversal
        sol = solve_min_transversal(cliques, n, time_limit=35)
        results.append(sol)
    
    return results
# EVOLVE-BLOCK-END
