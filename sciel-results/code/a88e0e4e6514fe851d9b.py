import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

    def bron_kerbosch(R, P, X, adj, n):
        if not P and not X:
            if len(R) >= 2:
                yield frozenset(R)
            return
        
        if P:
            pivot = max(P, key=lambda p: len(P & adj[p]))
        elif X:
            pivot = max(X, key=lambda p: len(P & adj[p]))
        else:
            return
            
        for v in list(P - adj[pivot]):
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            yield from bron_kerbosch(new_R, new_P, new_X, adj, n)
            P.remove(v)
            X.add(v)

    def find_max_matching(graph, num_nodes):
        """Find maximum matching using Edmonds' algorithm (simplified for bipartite-like structure).
        For general graphs, we use a heuristic approach with augmenting paths."""
        # Use a simple greedy + augmenting path approach
        match = [-1] * num_nodes
        visited = [False] * num_nodes
        
        def find_augmenting_path(u, seen):
            for v in graph[u]:
                if not seen[v]:
                    seen[v] = True
                    if match[v] == -1 or find_augmenting_path(match[v], seen):
                        match[u] = v
                        match[v] = u
                        return True
            return False
        
        # Try to find augmenting paths
        for u in range(num_nodes):
            if match[u] == -1:
                seen = [False] * num_nodes
                find_augmenting_path(u, seen)
                
        return match

    def vertex_cover_from_matching(match, num_nodes, graph):
        """Get vertex cover from matching edges."""
        cover = set()
        for u in range(num_nodes):
            if match[u] != -1:
                cover.add(u)
                cover.add(match[u])
        return cover

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        all_cliques = list(bron_kerbosch(frozenset(), set(range(n)), set(), adj, n))
        valid_cliques = [c for c in all_cliques if len(c) >= 2]
        
        if not valid_cliques:
            return []
            
        C = len(valid_cliques)
        
        vertex_cliques = defaultdict(set)
        for i, c in enumerate(valid_cliques):
            for v in c:
                vertex_cliques[v].add(i)
                
        best_set = None
        best_size = float('inf')
        
        rng_solver = random.Random(seed + 123)
        
        # Build clique intersection graph
        clique_adj = [set() for _ in range(C)]
        for i in range(C):
            for j in range(i+1, C):
                if valid_cliques[i] & valid_cliques[j]:
                    clique_adj[i].add(j)
                    clique_adj[j].add(i)
        
        # Strategy 1: Use maximum matching on clique intersection graph to identify
        # a 2-approximation vertex cover of cliques, then solve vertex selection on
        # the "core" cliques (those in the VC) with extra care.
        match = find_max_matching(clique_adj, C)
        vc_cliques = vertex_cover_from_matching(match, C, clique_adj)
        
        # Core: cliques in the vertex cover + their neighbors (to ensure coverage)
        core_cliques = set(vc_cliques)
        for ci in vc_cliques:
            for cj in clique_adj[ci]:
                core_cliques.add(cj)
        
        # Solve on core cliques first with greedy, then extend to all
        hit_cliques = set()
        current_set = set()
        
        # Order core cliques by size (smaller first for tighter constraints)
        core_list = sorted(core_cliques, key=lambda ci: len(valid_cliques[ci]))
        for ci in core_list:
            if ci in hit_cliques:
                continue
            best_v = -1
            max_hits = -1
            candidates = []
            for v in valid_cliques[ci]:
                hits = 0
                for cj in vertex_cliques[v]:
                    if cj not in hit_cliques:
                        hits += 1
                if hits > max_hits:
                    max_hits = hits
                    best_v = v
                    candidates = [v]
                elif hits == max_hits and hits > 0:
                    candidates.append(v)
            if candidates:
                v = rng_solver.choice(candidates)
                current_set.add(v)
                for cj in vertex_cliques[v]:
                    hit_cliques.add(cj)
        
        # Extend to remaining cliques
        remaining = [ci for ci in range(C) if ci not in hit_cliques]
        remaining.sort(key=lambda ci: len(valid_cliques[ci]))
        for ci in remaining:
            best_v = -1
            max_hits = -1
            candidates = []
            for v in valid_cliques[ci]:
                hits = 0
                for cj in vertex_cliques[v]:
                    if cj not in hit_cliques:
                        hits += 1
                if hits > max_hits:
                    max_hits = hits
                    best_v = v
                    candidates = [v]
                elif hits == max_hits and hits > 0:
                    candidates.append(v)
            if candidates:
                v = rng_solver.choice(candidates)
                current_set.add(v)
                for cj in vertex_cliques[v]:
                    hit_cliques.add(cj)
        
        # Local search: reduction
        improved = True
        while improved:
            improved = False
            vs = list(current_set)
            rng_solver.shuffle(vs)
            for v in vs:
                if time.time() > deadline:
                    break
                can_remove = True
                for ci in vertex_cliques[v]:
                    c = valid_cliques[ci]
                    found_other = False
                    for u in c:
                        if u != v and u in current_set:
                            found_other = True
                            break
                    if not found_other:
                        can_remove = False
                        break
                if can_remove:
                    current_set.remove(v)
                    improved = True
        
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = set(current_set)
        
        # Strategy 2: Standard Greedy with random restarts
        num_restarts = 300
        
        for run in range(num_restarts):
            if time.time() > deadline:
                break
                
            hit_cliques = set()
            current_set = set()
            
            while len(hit_cliques) < C:
                best_v = -1
                max_hits = -1
                candidates = []
                
                for v in range(n):
                    if v in current_set:
                        continue
                    hits = 0
                    for ci in vertex_cliques[v]:
                        if ci not in hit_cliques:
                            hits += 1
                    if hits > max_hits:
                        max_hits = hits
                        best_v = v
                        candidates = [v]
                    elif hits == max_hits and hits > 0:
                        candidates.append(v)
                        
                if max_hits == 0:
                    break
                    
                v = rng_solver.choice(candidates)
                current_set.add(v)
                for ci in vertex_cliques[v]:
                    hit_cliques.add(ci)
                    
            improved = True
            while improved:
                improved = False
                vs = list(current_set)
                rng_solver.shuffle(vs)
                for v in vs:
                    if time.time() > deadline:
                        break
                    can_remove = True
                    for ci in vertex_cliques[v]:
                        c = valid_cliques[ci]
                        found_other = False
                        for u in c:
                            if u != v and u in current_set:
                                found_other = True
                                break
                        if not found_other:
                            can_remove = False
                            break
                    if can_remove:
                        current_set.remove(v)
                        improved = True
                        
            if len(current_set) < best_size:
                best_size = len(current_set)
                best_set = set(current_set)
                
        # Strategy 3: High-precision polish on best_set using swap-based local search
        if best_set:
            polish_set = set(best_set)
            # Try single swaps: remove v, add w if it improves
            for _ in range(5):  # polish iterations
                if time.time() > deadline:
                    break
                improved = False
                for v in list(polish_set):
                    if time.time() > deadline:
                        break
                    # Check if v can be removed
                    can_remove = True
                    for ci in vertex_cliques[v]:
                        c = valid_cliques[ci]
                        found_other = False
                        for u in c:
                            if u != v and u in polish_set:
                                found_other = True
                                break
                        if not found_other:
                            can_remove = False
                            break
                    if can_remove:
                        polish_set.remove(v)
                        improved = True
                        break
                if not improved:
                    break
                
                # Try adding a vertex that allows removal of multiple others
                # (this is complex; skip for now and rely on reduction)
            
            if len(polish_set) < best_size:
                best_size = len(polish_set)
                best_set = polish_set
        
        return sorted(best_set) if best_set else []

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            edges = instance(n, p, seed)
            touched = {x for e in edges for x in e}
            results.append(sorted(touched))
            continue
            
        res = solve_instance(n, p, seed)
        results.append(res)
        
    return results
