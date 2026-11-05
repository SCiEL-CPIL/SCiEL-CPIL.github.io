import time
import random
import sys
from collections import defaultdict

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
            P_set = set(P)
            X_set = set(X)
            
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

    def solve_hitting_set_advanced(n, cliques):
        if not cliques:
            return set()
        
        cliques = list(set(cliques))
        m = len(cliques)
        
        # Map vertex to list of clique indices it covers
        v_to_cliques = defaultdict(list)
        for i, c in enumerate(cliques):
            for v in c:
                v_to_cliques[v].append(i)
        
        # Precompute dominance: u is dominated by v if N(u) subset N(v)
        # This allows pruning branches where we pick u while v is available
        dominated_by = {v: None for v in range(n)}
        for u in range(n):
            if not v_to_cliques[u]:
                continue
            for v in range(n):
                if u == v:
                    continue
                if not v_to_cliques[v]:
                    continue
                # Check if N(u) is subset of N(v)
                # Optimization: if len(N(u)) > len(N(v)), cannot be subset
                if len(v_to_cliques[u]) <= len(v_to_cliques[v]):
                    # Check subset
                    is_subset = True
                    set_u = set(v_to_cliques[u])
                    set_v = set(v_to_cliques[v])
                    if not set_u.issubset(set_v):
                        is_subset = False
                    if is_subset:
                        # u is dominated by v
                        # If u is dominated by multiple, pick one with highest frequency? 
                        # For now, just mark that u is dominated by SOME v.
                        # To be precise, we should check if v is "better" than current dominator.
                        # Simpler: just record one dominator.
                        dominated_by[u] = v
                        break

        # Greedy initial solution for upper bound (UB)
        def greedy_solution():
            uncovered = set(range(m))
            sol = set()
            while uncovered:
                best_v = None
                max_hits = 0
                for v in range(n):
                    if v in sol:
                        continue
                    hits = 0
                    for ci in v_to_cliques[v]:
                        if ci in uncovered:
                            hits += 1
                    if hits > max_hits:
                        max_hits = hits
                        best_v = v
                if best_v is None:
                    break
                sol.add(best_v)
                for ci in v_to_cliques[best_v]:
                    if ci in uncovered:
                        uncovered.discard(ci)
            return sol

        initial_sol = greedy_solution()
        best_size = len(initial_sol)
        best_sol = set(initial_sol)
        
        # Packing bound: find maximal set of disjoint uncovered cliques
        def packing_bound(uncovered):
            if not uncovered:
                return 0
            remaining = list(uncovered)
            count = 0
            while remaining:
                # Pick smallest clique
                remaining.sort(key=lambda i: len(cliques[i]))
                c_idx = remaining[0]
                c_set = cliques[c_idx]
                count += 1
                new_remaining = []
                for i in remaining[1:]:
                    if not cliques[i].intersection(c_set):
                        new_remaining.append(i)
                remaining = new_remaining
            return count

        def bb_search(assigned, unassigned, uncovered, cost):
            nonlocal best_size, best_sol
            
            if time.time() > deadline - 10:
                return
                
            if cost >= best_size:
                return
                
            if not uncovered:
                if cost < best_size:
                    best_size = cost
                    best_sol = set(assigned)
                return
            
            # Lower bound using packing
            lb = packing_bound(uncovered)
            if cost + lb >= best_size:
                return
            
            # Heuristic: try to complete with greedy to update UB
            # Only do this if cost is small enough to be worth it, or periodically
            # Simple greedy on remaining
            rem_uncovered = set(uncovered)
            greedy_adds = set()
            while rem_uncovered:
                best_v = None
                max_hits = 0
                for v in unassigned:
                    hits = 0
                    for ci in v_to_cliques[v]:
                        if ci in rem_uncovered:
                            hits += 1
                    if hits > max_hits:
                        max_hits = hits
                        best_v = v
                if best_v is None:
                    break
                greedy_adds.add(best_v)
                for ci in v_to_cliques[best_v]:
                    if ci in rem_uncovered:
                        rem_uncovered.discard(ci)
            
            if cost + len(greedy_adds) < best_size:
                best_size = cost + len(greedy_adds)
                best_sol = set(assigned) | greedy_adds
            
            # Branching
            # Find uncovered clique with fewest valid candidates
            # Valid candidate: v in unassigned, and v is not dominated by another w in unassigned
            # If v is dominated by w, and w is in unassigned, we skip v.
            
            best_clique_idx = None
            best_candidates = []
            min_cand_count = float('inf')
            
            # Precompute which vertices in unassigned are "active" (not dominated by another active vertex)
            # A vertex u is inactive if dominated_by[u] is in unassigned
            active_vertices = set()
            for v in unassigned:
                dom = dominated_by[v]
                if dom is None or dom not in unassigned:
                    active_vertices.add(v)
            
            for ci in uncovered:
                c = cliques[ci]
                candidates = [v for v in c if v in active_vertices]
                if not candidates:
                    # If no active candidates, it means all candidates are dominated by vertices not in unassigned?
                    # If a candidate v is dominated by w, and w is not in unassigned, w might be in assigned.
                    # If w is in assigned, the clique is already covered? 
                    # Wait, if w is in assigned, then all cliques containing w are covered.
                    # So if c is uncovered, no vertex in c can be in assigned.
                    # Therefore, if v is in c, v is not in assigned.
                    # If v is dominated by w, and w is not in unassigned, w must be in assigned (or removed).
                    # But if w is in assigned, c would be covered. Contradiction.
                    # So this case should not happen if logic is consistent.
                    # However, dominated_by might point to a vertex that was removed from unassigned but not added to assigned?
                    # In our branching, we remove v from unassigned and add to assigned.
                    # So if w is not in unassigned, it is either in assigned or never existed.
                    # If w is in assigned, c is covered.
                    # So candidates should not be empty.
                    # But to be safe, if empty, we can't branch. Return.
                    return
                
                if len(candidates) < min_cand_count:
                    min_cand_count = len(candidates)
                    best_clique_idx = ci
                    best_candidates = candidates
                    if min_cand_count == 1:
                        break
            
            if best_clique_idx is None:
                return
                
            # Sort candidates by frequency in uncovered
            best_candidates.sort(key=lambda v: sum(1 for ci in v_to_cliques[v] if ci in uncovered), reverse=True)
            
            for v in best_candidates:
                new_assigned = assigned | {v}
                new_unassigned = unassigned - {v}
                
                new_uncovered = set(uncovered)
                for ci in v_to_cliques[v]:
                    if ci in new_uncovered:
                        new_uncovered.discard(ci)
                        
                bb_search(new_assigned, new_unassigned, new_uncovered, cost + 1)
                
                if time.time() > deadline - 10:
                    return

        all_vertices = set(range(n))
        bb_search(set(), all_vertices, set(range(m)), 0)
        
        return best_sol

    results = []
    
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 15:
            # Fallback: return all vertices (guaranteed valid)
            results.append(list(range(n)))
            continue
            
        edges = random_graph(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(n, edges)
        
        # Filter cliques of size < 2
        cliques = [c for c in cliques if len(c) >= 2]
        
        if not cliques:
            results.append([])
            continue
            
        sol = solve_hitting_set_advanced(n, cliques)
        
        # Fallback if B&B didn't finish or failed
        if sol is None or len(sol) > n:
             v_to_cliques = defaultdict(list)
             for i, c in enumerate(cliques):
                 for v in c:
                     v_to_cliques[v].append(i)
             uncovered = set(range(len(cliques)))
             greedy_sol = set()
             while uncovered:
                 best_v = None
                 max_hits = 0
                 for v in range(n):
                     if v in greedy_sol: continue
                     hits = sum(1 for ci in v_to_cliques[v] if ci in uncovered)
                     if hits > max_hits:
                         max_hits = hits
                         best_v = v
                 if best_v is None: break
                 greedy_sol.add(best_v)
                 for ci in v_to_cliques[best_v]:
                     if ci in uncovered:
                         uncovered.discard(ci)
             sol = greedy_sol
             
        results.append(sorted(sol))
        
    return results

# EVOLVE-BLOCK-END
