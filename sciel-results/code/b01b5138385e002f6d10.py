"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def get_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return edges, adj

    def bron_kerbosch(R, P, X, adj, cliques):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        
        # Pivot selection to optimize
        pivot_candidates = list(P) + list(X)
        if not pivot_candidates:
            return
        
        best_pivot = None
        max_count = -1
        for v in pivot_candidates:
            count = len(P & adj[v])
            if count > max_count:
                max_count = count
                best_pivot = v
        
        # Iterate over P \ N(best_pivot)
        if best_pivot is not None:
            candidates = P - adj[best_pivot]
        else:
            candidates = P
            
        for v in list(candidates):
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            bron_kerbosch(new_R, new_P, new_X, adj, cliques)
            P.remove(v)
            X.add(v)

    def maximum_matching(adj, n):
        """Find maximum matching using Edmonds' algorithm (simplified for small graphs)."""
        # Use a simple augmenting path approach
        match = [-1] * n
        
        def try_augment(v, visited):
            for u in adj[v]:
                if not visited[u]:
                    visited[u] = True
                    if match[u] == -1 or try_augment(match[u], visited):
                        match[u] = v
                        match[v] = u
                        return True
            return False
        
        matching_size = 0
        for v in range(n):
            if match[v] == -1:
                visited = [False] * n
                if try_augment(v, visited):
                    matching_size += 1
        
        return matching_size

    def branch_and_bound(n, adj, cliques, time_limit=5.0):
        """Branch and bound to find minimum transversal."""
        if not cliques:
            return []
        
        # Precompute vertex -> list of cliques it covers
        vertex_cliques = {v: [] for v in range(n)}
        for i, C in enumerate(cliques):
            for v in C:
                vertex_cliques[v].append(i)
        
        # Greedy initialization
        selected = []
        remaining_cliques = set(range(len(cliques)))
        while remaining_cliques:
            best_v = None
            max_covered = -1
            for v in range(n):
                if v in selected:
                    continue
                count = 0
                for ci in vertex_cliques[v]:
                    if ci in remaining_cliques:
                        count += 1
                if count > max_covered:
                    max_covered = count
                    best_v = v
            
            if best_v is None or max_covered == 0:
                break
                
            selected.append(best_v)
            for ci in vertex_cliques[best_v]:
                if ci in remaining_cliques:
                    remaining_cliques.remove(ci)
        
        best_solution = list(selected)
        best_size = len(best_solution)
        
        # Local search to improve initial solution
        changed = True
        while changed:
            changed = False
            for v in list(selected):
                safe = True
                for ci in vertex_cliques[v]:
                    C = cliques[ci]
                    has_other = False
                    for u in C:
                        if u != v and u in selected:
                            has_other = True
                            break
                    if not has_other:
                        safe = False
                        break
                if safe:
                    selected.remove(v)
                    changed = True
        
        if len(selected) < best_size:
            best_solution = list(selected)
            best_size = len(selected)
        
        # Branch and bound
        # We'll use a recursive approach with pruning
        # State: current selection, set of uncovered cliques
        # Lower bound: maximum matching on the subgraph induced by uncovered cliques
        
        def lower_bound(uncovered_cliques):
            if not uncovered_cliques:
                return 0
            # Create a graph where vertices are vertices of the original graph
            # that appear in uncovered cliques, and edges connect vertices
            # that are in the same uncovered clique
            # The minimum transversal is at least the maximum matching in this graph
            # Actually, we need a different approach:
            # Each uncovered clique requires at least one vertex from it
            # We can model this as a set cover problem
            # A simple lower bound: find disjoint uncovered cliques
            # Each disjoint clique requires at least one vertex
            # So the lower bound is the number of disjoint cliques we can find
            
            # Greedy: find disjoint cliques
            disjoint_count = 0
            used_vertices = set()
            for ci in uncovered_cliques:
                C = cliques[ci]
                if C.isdisjoint(used_vertices):
                    disjoint_count += 1
                    used_vertices.update(C)
            
            return disjoint_count
        
        # Recursive branch and bound
        def bb(current_selection, uncovered_cliques, depth=0):
            nonlocal best_solution, best_size
            if time.time() > start_time + time_limit:
                return
            if not uncovered_cliques:
                if len(current_selection) < best_size:
                    best_solution = list(current_selection)
                    best_size = len(current_selection)
                return
            
            # Prune if current size + lower bound >= best_size
            lb = lower_bound(uncovered_cliques)
            if len(current_selection) + lb >= best_size:
                return
            
            # Choose a clique to branch on (one with fewest available vertices)
            best_ci = None
            min_options = n + 1
            for ci in uncovered_cliques:
                C = cliques[ci]
                # Count vertices in C that could be added
                options = 0
                for v in C:
                    if v not in current_selection:
                        options += 1
                if options < min_options:
                    min_options = options
                    best_ci = ci
            
            if best_ci is None:
                return
            
            C = cliques[best_ci]
            # Try adding each vertex in C
            for v in C:
                if v in current_selection:
                    continue
                new_selection = current_selection + [v]
                # Update uncovered cliques
                new_uncovered = set()
                for ci in uncovered_cliques:
                    if v not in cliques[ci]:
                        new_uncovered.add(ci)
                bb(new_selection, new_uncovered, depth + 1)
        
        # Start branch and bound
        bb([], set(range(len(cliques))))
        
        return best_solution

    def solve_transversal(n, adj, cliques):
        if not cliques:
            return []
        
        # Use branch and bound
        sol = branch_and_bound(n, adj, cliques, time_limit=5.0)
        
        # Verify and return
        if not sol:
            return []
        
        # Final local search polish
        selected = set(sol)
        vertex_cliques = {v: [] for v in range(n)}
        for i, C in enumerate(cliques):
            for v in C:
                vertex_cliques[v].append(i)
        
        changed = True
        while changed:
            changed = False
            for v in list(selected):
                safe = True
                for ci in vertex_cliques[v]:
                    C = cliques[ci]
                    has_other = False
                    for u in C:
                        if u != v and u in selected:
                            has_other = True
                            break
                    if not has_other:
                        safe = False
                        break
                if safe:
                    selected.remove(v)
                    changed = True
        
        return sorted(selected)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Emergency fallback: return all vertices (valid but large)
            edges, adj = get_graph(n, p, seed)
            touched = {x for e in edges for x in e}
            results.append(sorted(touched))
            continue
            
        edges, adj = get_graph(n, p, seed)
        cliques = []
        bron_kerbosch(set(), set(range(n)), set(), adj, cliques)
        
        # If no cliques found (e.g., no edges), result is []
        if not cliques:
            results.append([])
            continue
            
        sol = solve_transversal(n, adj, cliques)
        results.append(sorted(sol))
        
    return results
# EVOLVE-BLOCK-END
