"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from itertools import combinations

def construct():
    deadline = time.time() + 145  # 5s safety margin before 150s kill
    
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
            
            # Pivot selection: choose vertex with max |P ∩ N(v)|
            pivot = max(P | X, key=lambda v: len(P & adj[v])) if (P | X) else None
            candidates = P - adj[pivot] if pivot is not None else P
            
            for v in list(candidates):
                expand(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        expand(frozenset(), set(range(n)), set())
        return maximal_cliques
    
    def greedy_set_cover(cliques, n, time_limit=5.0):
        """Greedy set cover with enhanced local search (1-exchange, 2-exchange)."""
        start = time.time()
        
        clique_list = list(cliques)
        num_cliques = len(clique_list)
        
        if num_cliques == 0:
            return set()
        
        # For each vertex, which cliques does it cover?
        vertex_cliques = {v: set() for v in range(n)}
        for i, clique in enumerate(clique_list):
            for v in clique:
                vertex_cliques[v].add(i)
        
        # Track coverage count per clique for efficient updates
        clique_cover_count = [0] * num_cliques
        
        # Greedy initialization
        selected = set()
        
        while any(c > 0 for c in clique_cover_count) or sum(clique_cover_count) < num_cliques:
            if time.time() - start > time_limit:
                break
            
            # Find vertex covering most uncovered cliques
            best_v = None
            best_count = -1
            for v in range(n):
                if v in selected:
                    continue
                cnt = sum(1 for i in vertex_cliques[v] if clique_cover_count[i] == 0)
                if cnt > best_count:
                    best_count = cnt
                    best_v = v
            
            if best_v is None or best_count <= 0:
                break
            
            selected.add(best_v)
            for i in vertex_cliques[best_v]:
                clique_cover_count[i] += 1
        
        # Local search: try to improve
        for iteration in range(100):
            if time.time() - start > time_limit:
                break
            
            improved = False
            
            # Try removing a vertex (0-1 exchange)
            for v in list(selected):
                # Check if removing v still covers all cliques
                can_remove = True
                for i in vertex_cliques[v]:
                    if clique_cover_count[i] == 1:
                        can_remove = False
                        break
                if can_remove:
                    selected.remove(v)
                    for i in vertex_cliques[v]:
                        clique_cover_count[i] -= 1
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 1-for-1 exchange: remove one, add one
            found_swap = False
            for v_remove in list(selected):
                # Cliques that would become uncovered if we remove v_remove
                at_risk = [i for i in vertex_cliques[v_remove] if clique_cover_count[i] == 1]
                if not at_risk:
                    # This vertex is redundant, should have been caught above
                    continue
                
                for v_add in range(n):
                    if v_add in selected:
                        continue
                    
                    # Check if v_add covers all at-risk cliques
                    covers_all = all(i in vertex_cliques[v_add] for i in at_risk)
                    if covers_all:
                        # Perform swap
                        selected.remove(v_remove)
                        for i in vertex_cliques[v_remove]:
                            clique_cover_count[i] -= 1
                        selected.add(v_add)
                        for i in vertex_cliques[v_add]:
                            clique_cover_count[i] += 1
                        improved = True
                        found_swap = True
                        break
                
                if found_swap:
                    break
            
            if improved:
                continue
            
            # Try 2-for-1 exchange: remove two, add one (reduces size by 1)
            found_2_1 = False
            sel_list = list(selected)
            for idx_a in range(len(sel_list)):
                for idx_b in range(idx_a + 1, len(sel_list)):
                    va = sel_list[idx_a]
                    vb = sel_list[idx_b]
                    
                    # Cliques at risk when removing both va and vb
                    at_risk = set()
                    for i in vertex_cliques[va]:
                        if clique_cover_count[i] <= 2:
                            at_risk.add(i)
                    for i in vertex_cliques[vb]:
                        if clique_cover_count[i] <= 2:
                            at_risk.add(i)
                    
                    if not at_risk:
                        continue
                    
                    for v_add in range(n):
                        if v_add in selected:
                            continue
                        
                        # Check if v_add covers all at-risk cliques
                        covers_all = all(i in vertex_cliques[v_add] for i in at_risk)
                        if covers_all:
                            # Perform 2-for-1 swap
                            selected.remove(va)
                            for i in vertex_cliques[va]:
                                clique_cover_count[i] -= 1
                            selected.remove(vb)
                            for i in vertex_cliques[vb]:
                                clique_cover_count[i] -= 1
                            selected.add(v_add)
                            for i in vertex_cliques[v_add]:
                                clique_cover_count[i] += 1
                            improved = True
                            found_2_1 = True
                            break
                    
                    if found_2_1:
                        break
                if found_2_1:
                    break
            
            if not improved:
                break
        
        return selected
    
    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for n, p, seed in instances:
        if time.time() > deadline:
            # Emergency fallback: return all vertices
            results.append(list(range(n)))
            continue
        
        adj = build_graph(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(adj)
        
        # Filter to only cliques with >= 2 vertices
        cliques = [c for c in cliques if len(c) >= 2]
        
        if not cliques:
            results.append([])
            continue
        
        # Greedy set cover with more time for larger instances
        remaining = deadline - time.time() - 3
        time_budget = min(30.0, max(5.0, remaining / 3))
        selected = greedy_set_cover(cliques, n, time_limit=time_budget)
        
        # Verify coverage
        covered_cliques = set()
        for clique in cliques:
            for v in selected:
                if v in clique:
                    covered_cliques.add(clique)
                    break
        
        if len(covered_cliques) != len(cliques):
            # Fallback: add vertices to cover remaining
            for clique in cliques:
                if clique not in covered_cliques:
                    for v in clique:
                        if v not in selected:
                            selected.add(v)
                            break
        
        results.append(sorted(selected))
    
    return results
# EVOLVE-BLOCK-END
