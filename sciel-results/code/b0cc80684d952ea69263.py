import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

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

    def find_max_matching_bipartite(adj_left, adj_right, n_left, n_right):
        """Find maximum matching in a bipartite graph using Hopcroft-Karp."""
        pair_u = [-1] * n_left
        pair_v = [-1] * n_right
        dist = [0] * n_left
        
        def bfs():
            from collections import deque
            q = deque()
            for u in range(n_left):
                if pair_u[u] == -1:
                    dist[u] = 0
                    q.append(u)
                else:
                    dist[u] = -1
            found = False
            while q:
                u = q.popleft()
                for v in adj_left[u]:
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == -1:
                        dist[pu] = dist[u] + 1
                        q.append(pu)
            return found
            
        def dfs(u):
            for v in adj_left[u]:
                pu = pair_v[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = -1
            return False
            
        matching = 0
        while bfs():
            for u in range(n_left):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1
        return pair_u, pair_v, matching

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
        
        # Strategy 1: Vertex Cover on Clique Intersection Graph
        # Build intersection graph: nodes are cliques, edge if they share a vertex
        clique_adj = [set() for _ in range(C)]
        for i in range(C):
            for j in range(i+1, C):
                if valid_cliques[i] & valid_cliques[j]:
                    clique_adj[i].add(j)
                    clique_adj[j].add(i)
        
        # Find maximum matching and derive vertex cover
        # We use a simple greedy + augmenting path approach for the general graph
        match = [-1] * C
        visited = [False] * C
        
        def find_augmenting_path(u, seen):
            for v in clique_adj[u]:
                if not seen[v]:
                    seen[v] = True
                    if match[v] == -1 or find_augmenting_path(match[v], seen):
                        match[u] = v
                        match[v] = u
                        return True
            return False
        
        for u in range(C):
            if match[u] == -1:
                seen = [False] * C
                find_augmenting_path(u, seen)
                
        vc_cliques = set()
        for u in range(C):
            if match[u] != -1:
                vc_cliques.add(u)
                vc_cliques.add(match[u])
        
        # For each clique in vertex cover, we need to select vertices
        # Greedily select vertices to cover all cliques in vc_cliques
        hit_cliques = set()
        current_set = set()
        
        # Prioritize cliques in the vertex cover
        cliques_to_cover = list(vc_cliques)
        # Add any remaining cliques not in vertex cover
        for i in range(C):
            if i not in vc_cliques:
                cliques_to_cover.append(i)
        
        for ci in cliques_to_cover:
            if ci in hit_cliques:
                continue
            # Find vertex in this clique that covers most uncovered cliques
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
        num_restarts = 500
        
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
                
        return sorted(best_set) if best_set else []

    results = []
    # Allocate time budget: give more time to harder instances
    instance_budgets = [45, 45, 45]  # Total 135s, with 5s safety margin
    
    for idx, (n, p, seed) in enumerate(((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203))):
        remaining = deadline - time.time()
        if remaining < 5:
            edges = instance(n, p, seed)
            touched = {x for e in edges for x in e}
            results.append(sorted(touched))
            continue
            
        # Temporarily adjust deadline for this instance
        global_deadline = deadline
        deadline = min(deadline, time.time() + min(instance_budgets[idx], remaining - 2))
        
        res = solve_instance(n, p, seed)
        results.append(res)
        
        deadline = global_deadline
        
    return results
