"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def instance(n, p, seed):
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
        return edges, adj

    def bron_kerbosch(adj, n):
        """Find all maximal cliques of size >= 2 using Bron-Kerbosch with pivoting."""
        maximal_cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(frozenset(R))
                return
            
            # Choose pivot u from P ∪ X to minimize branching
            pivot_candidates = P | X
            if pivot_candidates:
                # Pick pivot with max degree in P
                best_pivot = None
                best_count = -1
                for u in pivot_candidates:
                    count = len(P & adj[u])
                    if count > best_count:
                        best_count = count
                        best_pivot = u
                
                # Branch on vertices in P \ N(u)
                candidates = P - adj[best_pivot]
            else:
                candidates = P
            
            for v in list(candidates):
                bk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return maximal_cliques

    def greedy_transversal(cliques, adj, n):
        """Greedy set cover: repeatedly pick vertex hitting most uncovered cliques."""
        uncovered = list(cliques)
        if not uncovered:
            return set()
        
        # Count how many cliques each vertex appears in
        vertex_clique_count = {}
        for clique in uncovered:
            for v in clique:
                vertex_clique_count[v] = vertex_clique_count.get(v, 0) + 1
        
        solution = set()
        while uncovered:
            # Find vertex that hits the most uncovered cliques
            best_v = None
            best_hits = -1
            for v, count in vertex_clique_count.items():
                if count > best_hits:
                    best_hits = count
                    best_v = v
            
            if best_v is None:
                break
            
            solution.add(best_v)
            
            # Remove all cliques containing best_v
            new_uncovered = []
            for clique in uncovered:
                if best_v not in clique:
                    new_uncovered.append(clique)
                else:
                    # Update counts for remaining vertices in this clique
                    for u in clique:
                        if u != best_v:
                            vertex_clique_count[u] = vertex_clique_count.get(u, 0) - 1
            
            uncovered = new_uncovered
            if best_v in vertex_clique_count:
                del vertex_clique_count[best_v]
        
        return solution

    def local_search_improve(solution, cliques, adj, n, time_limit=5.0):
        """Try to improve solution by removing redundant vertices or swapping."""
        sol = set(solution)
        deadline_ls = time.time() + time_limit
        
        # Try removing each vertex if still valid
        changed = True
        while changed and time.time() < deadline_ls:
            changed = False
            for v in list(sol):
                trial = sol - {v}
                # Check if trial still hits all cliques
                valid = True
                for clique in cliques:
                    if not (clique & trial):
                        valid = False
                        break
                if valid:
                    sol = trial
                    changed = True
        
        # Try single swaps: remove one, add another
        if time.time() < deadline_ls:
            for v in list(sol):
                if time.time() >= deadline_ls:
                    break
                for u in range(n):
                    if u in sol or u == v:
                        continue
                    trial = (sol - {v}) | {u}
                    valid = True
                    for clique in cliques:
                        if not (clique & trial):
                            valid = False
                            break
                    if valid:
                        sol = trial
                        break
        
        return sol

    def branch_and_bound(cliques, adj, n, current_best, time_limit=10.0):
        """Branch and bound for minimum clique transversal."""
        deadline_bb = time.time() + time_limit
        best = set(current_best)
        best_size = len(best)
        
        # Sort cliques by size (smaller first for better pruning)
        sorted_cliques = sorted(cliques, key=len)
        
        # Build candidate set: vertices in any clique
        candidates = set()
        for c in cliques:
            candidates.update(c)
        candidates = list(candidates)
        
        # For each vertex, precompute which cliques it hits
        vertex_hits = {}
        for v in candidates:
            vertex_hits[v] = frozenset(i for i, c in enumerate(sorted_cliques) if v in c)
        
        uncovered = set(range(len(sorted_cliques)))
        
        def bb(uncovered_set, chosen, depth):
            nonlocal best, best_size
            if time.time() > deadline_bb:
                return
            
            if not uncovered_set:
                if len(chosen) < best_size:
                    best = set(chosen)
                    best_size = len(chosen)
                return
            
            # Lower bound: need at least ceil(max_clique_size / max_vertices_per_clique) 
            # Better: each remaining clique needs at least 1 vertex, but one vertex can cover multiple
            # Simple bound: at least 1 more vertex needed
            if len(chosen) + 1 >= best_size:
                return
            
            # Choose the most constrained uncovered clique
            # Pick clique with fewest candidate vertices that could still be chosen
            target_clique = None
            min_options = float('inf')
            for idx in uncovered_set:
                clique = sorted_cliques[idx]
                # Count vertices in this clique not yet excluded
                options = sum(1 for v in clique if v not in chosen)
                if options < min_options:
                    min_options = options
                    target_clique = idx
                if options == 0:
                    return  # Dead end
            
            if target_clique is None:
                return
            
            clique = sorted_cliques[target_clique]
            # Branch on each vertex in this clique
            for v in clique:
                if v in chosen:
                    continue
                new_chosen = chosen | {v}
                # Remove all cliques hit by v
                new_uncovered = uncovered_set - set(vertex_hits[v])
                bb(new_uncovered, new_chosen, depth + 1)
                if time.time() > deadline_bb:
                    return
        
        bb(uncovered, set(), 0)
        return best

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 20:
            # Not enough time left; use greedy for remaining
            edges, adj = instance(n, p, seed)
            cliques = bron_kerbosch(adj, n)
            sol = greedy_transversal(cliques, adj, n)
            results.append(sorted(sol))
            continue
        
        edges, adj = instance(n, p, seed)
        cliques = bron_kerbosch(adj, n)
        
        if not cliques:
            results.append([])
            continue
        
        # Start with greedy
        sol = greedy_transversal(cliques, adj, n)
        
        # Local search improvement
        remaining_time = deadline - time.time()
        ls_time = min(3.0, remaining_time * 0.3)
        sol = local_search_improve(sol, cliques, adj, n, time_limit=ls_time)
        
        # Branch and bound if time permits
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            bb_time = min(remaining_time * 0.5, 15.0)
            improved = branch_and_bound(cliques, adj, n, sol, time_limit=bb_time)
            if len(improved) < len(sol):
                sol = improved
        
        results.append(sorted(sol))
    
    return results
# EVOLVE-BLOCK-END
