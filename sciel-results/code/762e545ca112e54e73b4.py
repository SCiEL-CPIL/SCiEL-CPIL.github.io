"""Combined solver: dominance pruning + forced propagation + B&B + polish pass."""
import time
import random
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
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

    def bron_kerbosch_max_cliques(adj, n):
        max_cliques = []
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    max_cliques.append(frozenset(R))
                return
            union = P | X
            if not union:
                return
            best_u = max(union, key=lambda u: len(P & adj[u]))
            candidates = P - adj[best_u]
            for v in list(candidates):
                bk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        bk(set(), set(range(n)), set())
        return max_cliques

    def solve_instance(n, p, seed, time_budget):
        edges = instance(n, p, seed)
        adj = build_adj(n, edges)
        cliques = bron_kerbosch_max_cliques(adj, n)
        
        if not cliques:
            return []
        
        # Build vertex-to-clique mapping
        vertex_to_cliques = defaultdict(set)
        for i, cl in enumerate(cliques):
            for v in cl:
                vertex_to_cliques[v].add(i)
        
        # Dominance pruning: if S_u <= S_v, u is dominated by v
        dominated = set()
        vertices = list(vertex_to_cliques.keys())
        for u in vertices:
            if u in dominated:
                continue
            Su = vertex_to_cliques[u]
            for v in vertices:
                if u == v or v in dominated:
                    continue
                Sv = vertex_to_cliques[v]
                if Su <= Sv:
                    dominated.add(u)
                    break
        
        active_vertices = [v for v in vertices if v not in dominated]
        
        # Forced vertex propagation
        forced = set()
        remaining_cliques = set(range(len(cliques)))
        
        changed = True
        while changed:
            changed = False
            for i in list(remaining_cliques):
                cl = cliques[i]
                active_in_cl = [v for v in cl if v not in dominated and v not in forced]
                if len(active_in_cl) == 1:
                    v = active_in_cl[0]
                    forced.add(v)
                    for ci in vertex_to_cliques[v]:
                        if ci in remaining_cliques:
                            remaining_cliques.remove(ci)
                    changed = True
        
        if not remaining_cliques:
            return sorted(forced)
        
        # B&B on remaining
        best_solution = None
        best_size = float('inf')
        
        def lower_bound(uncovered):
            if not uncovered:
                return 0
            max_cover = 0
            for v in active_vertices:
                cnt = len(vertex_to_cliques[v] & uncovered)
                if cnt > max_cover:
                    max_cover = cnt
            if max_cover == 0:
                return 0
            return (len(uncovered) + max_cover - 1) // max_cover
        
        def bb(uncovered, current_sol):
            nonlocal best_solution, best_size
            
            if time.time() - t0 > DEADLINE - 5:
                return
            
            if not uncovered:
                if len(current_sol) < best_size:
                    best_size = len(current_sol)
                    best_solution = set(current_sol)
                return
            
            lb = lower_bound(uncovered)
            if len(current_sol) + lb >= best_size:
                return
            
            # Select branch clique: one with fewest available vertices
            min_count = float('inf')
            branch_clique_idx = -1
            for i in uncovered:
                cl = cliques[i]
                cnt = 0
                for v in cl:
                    if v not in dominated and v not in current_sol:
                        cnt += 1
                if cnt < min_count:
                    min_count = cnt
                    branch_clique_idx = i
            
            if branch_clique_idx == -1:
                return
                
            branch_clique = cliques[branch_clique_idx]
            
            verts = []
            for v in branch_clique:
                if v not in dominated and v not in current_sol:
                    verts.append(v)
            
            verts.sort(key=lambda v: len(vertex_to_cliques[v] & uncovered), reverse=True)
            
            for v in verts:
                new_sol = set(current_sol)
                new_sol.add(v)
                new_uncovered = uncovered - vertex_to_cliques[v]
                bb(new_uncovered, new_sol)
        
        bb(remaining_cliques, set(forced))
        
        if best_solution is not None:
            result = forced | best_solution
        else:
            # Fallback to greedy
            result = set(forced)
            uncovered = remaining_cliques
            while uncovered:
                best_v = None
                best_count = -1
                for v in active_vertices:
                    if v in result:
                        continue
                    cnt = len(vertex_to_cliques[v] & uncovered)
                    if cnt > best_count:
                        best_count = cnt
                        best_v = v
                if best_v is None:
                    break
                result.add(best_v)
                uncovered -= vertex_to_cliques[best_v]
        
        # Polish pass: try removing vertices
        for _ in range(3):
            improved = False
            for v in list(result):
                ok = True
                for cl in cliques:
                    hits = cl & result
                    if len(hits) == 1 and v in hits:
                        ok = False
                        break
                if ok:
                    result.discard(v)
                    improved = True
            if not improved:
                break
        
        return sorted(result)

    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for idx, (n, p, seed) in enumerate(instances):
        if time.time() - t0 > DEADLINE - 5:
            results.append([])
            continue
        budget = (DEADLINE - (time.time() - t0)) * (0.3 if idx < 2 else 0.4)
        sol = solve_instance(n, p, seed, budget)
        results.append(sol)
    
    return results
