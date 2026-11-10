"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    
    def build_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj
    
    def bron_kerbosch_maximal_cliques(adj):
        n = len(adj)
        maximal_cliques = []
        
        def expand(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(frozenset(R))
                return
            # Choose pivot for better pruning
            if P:
                pivot = max(P, key=lambda v: len(P & adj[v]) + len(X & adj[v]))
                candidates = P - adj[pivot]
            else:
                candidates = P
            for v in list(candidates):
                expand(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        expand(set(), set(range(n)), set())
        return maximal_cliques
    
    def solve_set_cover(n, cliques):
        if not cliques:
            return []
        
        num_cliques = len(cliques)
        # For each vertex, which cliques does it belong to?
        vertex_to_cliques = [set() for _ in range(n)]
        for ci, clique in enumerate(cliques):
            for v in clique:
                vertex_to_cliques[v].add(ci)
        
        # Greedy set cover
        uncovered = set(range(num_cliques))
        selected = set()
        
        while uncovered:
            # Find vertex covering most uncovered cliques
            best_v = -1
            best_count = -1
            for v in range(n):
                if v in selected:
                    continue
                count = len(vertex_to_cliques[v] & uncovered)
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v == -1 or best_count == 0:
                break
            selected.add(best_v)
            uncovered -= vertex_to_cliques[best_v]
        
        # Local search: try to remove vertices that are still covered by others
        improved = True
        while improved:
            improved = False
            for v in list(selected):
                # Check if removing v still covers all cliques
                temp_selected = selected - {v}
                all_covered = True
                for ci, clique in enumerate(cliques):
                    if not (clique & temp_selected):
                        all_covered = False
                        break
                if all_covered:
                    selected = temp_selected
                    improved = True
        
        # Local search: try to swap one vertex for another to reduce size
        # Try adding a vertex and then removing more
        for _ in range(3):  # a few rounds of improvement
            improved = False
            for v_out in list(selected):
                for v_in in range(n):
                    if v_in in selected:
                        continue
                    new_selected = (selected - {v_out}) | {v_in}
                    if len(new_selected) >= len(selected):
                        continue
                    # Check if all cliques are covered
                    all_covered = True
                    for ci, clique in enumerate(cliques):
                        if not (clique & new_selected):
                            all_covered = False
                            break
                    if all_covered:
                        selected = new_selected
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        return sorted(selected)
    
    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            out.append(list(range(n)))
            continue
        adj = build_graph(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(adj)
        cliques = [c for c in cliques if len(c) >= 2]
        solution = solve_set_cover(n, cliques)
        out.append(solution)
    
    return out
# EVOLVE-BLOCK-END
