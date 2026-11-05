"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    def generate_edges(n, p, seed):
        from random import Random
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges
    
    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
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
            # Choose pivot u from P ∪ X to minimize branching
            pivot_set = P | X
            if pivot_set:
                # Pick pivot with max degree in P
                u = max(pivot_set, key=lambda x: len(adj[x] & P))
                candidates = P - adj[u]
            else:
                candidates = P
            
            for v in list(candidates):
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                bk(new_R, new_P, new_X)
                P.remove(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return cliques
    
    def greedy_hitting_set(cliques, n):
        """Greedy: repeatedly pick vertex hitting most uncovered cliques."""
        # Map each clique to index
        clique_list = list(cliques)
        num_cliques = len(clique_list)
        
        if num_cliques == 0:
            return set()
        
        # For each vertex, which cliques it hits
        vertex_to_cliques = defaultdict(set)
        for ci, clique in enumerate(clique_list):
            for v in clique:
                vertex_to_cliques[v].add(ci)
        
        uncovered = set(range(num_cliques))
        selected = set()
        
        while uncovered:
            best_v = None
            best_count = -1
            for v in vertex_to_cliques:
                count = len(vertex_to_cliques[v] & uncovered)
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v is None or best_count <= 0:
                break
            selected.add(best_v)
            for ci in vertex_to_cliques[best_v]:
                if ci in uncovered:
                    uncovered.remove(ci)
        
        return selected
    
    def improve_local_search(selected, cliques, n, max_iter=200):
        """Try to remove or swap vertices to reduce size while maintaining coverage."""
        selected = set(selected)
        clique_list = list(cliques)
        
        # Build coverage: for each clique, which selected vertices cover it
        def is_valid(sel):
            for clique in clique_list:
                if not (sel & clique):
                    return False
            return True
        
        if not is_valid(selected):
            return selected
        
        for iteration in range(max_iter):
            improved = False
            # Try removing each vertex
            for v in list(selected):
                new_sel = selected - {v}
                if is_valid(new_sel):
                    selected = new_sel
                    improved = True
                    break
            if improved:
                continue
            
            # Try swapping: remove one, add one
            for v_remove in list(selected):
                sel_without = selected - {v_remove}
                # Find uncovered cliques
                uncovered_cliques = [cl for cl in clique_list if not (sel_without & cl)]
                if not uncovered_cliques:
                    selected = sel_without
                    improved = True
                    break
                # Find a vertex that covers all uncovered cliques and is not in sel_without
                # Actually we need to cover all uncovered cliques with one vertex
                candidate_set = set()
                for cl in uncovered_cliques:
                    candidate_set |= cl
                candidate_set -= sel_without
                for v_add in candidate_set:
                    new_sel = sel_without | {v_add}
                    if is_valid(new_sel):
                        selected = new_sel
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        return selected
    
    def solve_instance(n, p, seed):
        edges = generate_edges(n, p, seed)
        adj = build_adj(n, edges)
        cliques = bron_kerbosch_maximal_cliques(adj, n)
        
        if not cliques:
            return set()
        
        # Greedy initial solution
        selected = greedy_hitting_set(cliques, n)
        
        # Local search improvement
        selected = improve_local_search(selected, cliques, n)
        
        return selected
    
    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Fallback: return all vertices as safety
            results.append(set(range(n)))
            continue
        
        selected = solve_instance(n, p, seed)
        results.append(selected)
    
    return [sorted(r) for r in results]
# EVOLVE-BLOCK-END
