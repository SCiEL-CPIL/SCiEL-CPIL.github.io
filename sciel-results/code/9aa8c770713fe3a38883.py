import time
import random
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 120.0  # Keep safety margin for 150s kill

    def remaining_time():
        return DEADLINE - (time.time() - t0)

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
            # Pivot selection: max |P ∩ N(u)|
            best_u = max(union, key=lambda u: len(P & adj[u]))
            candidates = P - adj[best_u]
            for v in list(candidates):
                bk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        bk(set(), set(range(n)), set())
        return max_cliques

    def solve_hitting_set(n, cliques, time_limit):
        if not cliques:
            return []
        
        # Map clique index to frozenset
        # Map vertex to set of clique indices
        vertex_to_cliques = defaultdict(set)
        for i, cl in enumerate(cliques):
            for v in cl:
                vertex_to_cliques[v].add(i)
        
        # Dominance check: If S_u is subset of S_v, u is dominated by v.
        # We can remove u from consideration.
        # This reduces the search space significantly.
        dominated = set()
        vertices = list(vertex_to_cliques.keys())
        # Only check vertices that appear in cliques
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
        
        # Filter vertices
        active_vertices = [v for v in vertices if v not in dominated]
        
        # If a clique has only one active vertex, that vertex is essential.
        # Actually, if a clique C has vertices V_C, and some are dominated, 
        # we must pick from the non-dominated ones.
        # If only one non-dominated vertex remains for a clique, it's forced.
        
        forced = set()
        remaining_cliques = set(range(len(cliques)))
        
        # Iteratively find forced vertices
        changed = True
        while changed:
            changed = False
            for i in list(remaining_cliques):
                cl = cliques[i]
                # Find active vertices in this clique
                active_in_cl = [v for v in cl if v not in dominated and v not in forced]
                if not active_in_cl:
                    # This shouldn't happen if dominated logic is correct, 
                    # unless all were dominated by forced ones?
                    # If all vertices in cl are dominated, they are dominated by someone.
                    # If the dominator is not in cl, it doesn't help cover cl.
                    # Wait. Dominance: S_u <= S_v. If v is not in cl, v doesn't cover cl.
                    # So u must be covered by u or another vertex in cl.
                    # So active_in_cl should not be empty if cl is not covered.
                    pass
                if len(active_in_cl) == 1:
                    v = active_in_cl[0]
                    forced.add(v)
                    # Remove all cliques covered by v
                    for ci in vertex_to_cliques[v]:
                        if ci in remaining_cliques:
                            remaining_cliques.remove(ci)
                    changed = True
        
        if not remaining_cliques:
            return sorted(forced)
            
        # B&B on remaining
        best_solution = None
        best_size = float('inf')
        
        # Lower bound function
        def lower_bound(uncovered):
            if not uncovered:
                return 0
            # Simple bound: 1
            # Better: Greedy set cover on dual?
            # Let's use: max number of disjoint cliques in uncovered.
            # Greedy packing:
            rem = set(uncovered)
            count = 0
            while rem:
                # Pick clique with smallest number of available vertices?
                # Or just pick any, remove it and all intersecting?
                # For a lower bound on hitting set, we need a packing of cliques 
                # such that no single vertex hits more than one.
                # This is equivalent to finding a matching in the bipartite graph 
                # of vertices and cliques? No.
                # If we pick a clique C, we remove C and all cliques that share a vertex with C.
                # The size of such a greedy packing is a valid lower bound.
                # To maximize the packing, we should pick cliques that are "isolated".
                # Let's just return 1 for simplicity and rely on strong branching.
                # Or, count uncovered cliques / max_cliques_per_vertex.
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
                # Count active vertices in this clique that are not in current_sol
                # (Vertices in current_sol are already picked, so they don't branch)
                # Actually, if a vertex is in current_sol, the clique is already covered.
                # So uncovered cliques contain no vertices from current_sol.
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
            
            # Order vertices by how many other cliques they cover (most first)
            verts = []
            for v in branch_clique:
                if v not in dominated and v not in current_sol:
                    verts.append(v)
            
            verts.sort(key=lambda v: len(vertex_to_cliques[v] & uncovered), reverse=True)
            
            for v in verts:
                new_sol = current_sol | {v}
                new_uncovered = set(uncovered)
                # Remove all cliques covered by v
                for ci in vertex_to_cliques[v]:
                    new_uncovered.discard(ci)
                
                bb(new_uncovered, new_sol)
                
                # If time is tight, stop
                if time.time() - t0 > DEADLINE - 10:
                    return

        bb(remaining_cliques, set(forced))
        
        if best_solution is None:
            # Fallback: Greedy
            return greedy_hit_set(n, cliques)
            
        return sorted(best_solution)

    def greedy_hit_set(n, cliques):
        vertex_to_cliques = defaultdict(set)
        for i, cl in enumerate(cliques):
            for v in cl:
                vertex_to_cliques[v].add(i)
        
        uncovered = set(range(len(cliques)))
        chosen = set()
        
        while uncovered:
            best_v = None
            best_count = -1
            for v in vertex_to_cliques:
                if v in chosen:
                    continue
                cnt = len(vertex_to_cliques[v] & uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_v = v
            if best_v is None:
                break
            chosen.add(best_v)
            uncovered -= vertex_to_cliques[best_v]
        
        return sorted(chosen)

    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for idx, (n, p, seed) in enumerate(instances):
        if time.time() - t0 > DEADLINE - 5:
            results.append([])
            continue
            
        edges = instance(n, p, seed)
        adj = build_adj(n, edges)
        cliques = bron_kerbosch_max_cliques(adj, n)
        
        if not cliques:
            results.append([])
            continue
            
        # Allocate time
        time_spent = time.time() - t0
        remaining = DEADLINE - time_spent
        # Give more time to later instances? Or equal?
        # Let's give each instance a chunk, but keep a global guard.
        budget = remaining / (len(instances) - idx)
        
        sol = solve_hitting_set(n, cliques, budget)
        results.append(sol)
        
    return results
