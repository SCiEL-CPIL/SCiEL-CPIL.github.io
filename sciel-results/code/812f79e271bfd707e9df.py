import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return adj, edges

    def bron_kerbosch_maximal_cliques(adj):
        """
        Enumerates all maximal cliques in the graph defined by adjacency lists.
        Uses Bron-Kerbosch with pivot.
        """
        n = len(adj)
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Choose pivot u from P union X
            # Heuristic: choose u with max degree in P
            P_X = P.union(X)
            if not P_X:
                return
            u = max(P_X, key=lambda v: len(adj[v].intersection(P)))
            
            # Iterate over P \ N(u)
            for v in list(P - adj[u]):
                R_new = R.union({v})
                P_new = P.intersection(adj[v])
                X_new = X.intersection(adj[v])
                bk(R_new, P_new, X_new)
                P.remove(v)
                X.add(v)
        
        # Initial call
        # P = all vertices, X = empty
        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance(adj, n, deadline):
        cliques = bron_kerbosch_maximal_cliques(adj)
        if not cliques:
            return []
        
        # Convert to list of sets for easier manipulation
        cliques_list = [set(c) for c in cliques]
        num_cliques = len(cliques_list)
        
        # Precompute vertex -> list of clique indices
        vertex_to_cliques = [[] for _ in range(n)]
        for i, c in enumerate(cliques_list):
            for v in c:
                vertex_to_cliques[v].append(i)
        
        # Precompute bitmask for each vertex: bit ci set if vertex covers clique ci
        vertex_clique_mask = [0] * n
        for v in range(n):
            mask = 0
            for ci in vertex_to_cliques[v]:
                mask |= (1 << ci)
            vertex_clique_mask[v] = mask
        
        # All-cliques mask
        all_cliques_mask = (1 << num_cliques) - 1
        
        def greedy_init():
            S = set()
            cover_mask = 0
            remaining_cliques = num_cliques
            
            while remaining_cliques > 0:
                best_v = -1
                max_score = -1
                for v in range(n):
                    if v in S:
                        continue
                    # Count newly covered cliques
                    score = bin(vertex_clique_mask[v] & ~cover_mask).count('1')
                    if score > max_score:
                        max_score = score
                        best_v = v
                if best_v == -1:
                    break
                S.add(best_v)
                new_mask = vertex_clique_mask[best_v] & ~cover_mask
                remaining_cliques -= bin(new_mask).count('1')
                cover_mask |= vertex_clique_mask[best_v]
            return S

        def prune(S):
            S = set(S)
            # Compute coverage mask for each vertex's unique contribution
            # A vertex is removable if no clique is covered only by it
            cover_count = [0] * num_cliques
            for v in S:
                for ci in vertex_to_cliques[v]:
                    cover_count[ci] += 1
            
            changed = True
            while changed:
                changed = False
                for v in list(S):
                    can_remove = True
                    for ci in vertex_to_cliques[v]:
                        if cover_count[ci] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        S.remove(v)
                        for ci in vertex_to_cliques[v]:
                            cover_count[ci] -= 1
                        changed = True
            return S

        def local_search_bitmask(S, rng, deadline):
            """
            Fast local search using bitmasks for O(1) move evaluation.
            """
            current_S = set(S)
            current_size = len(current_S)
            
            # Cover mask: which cliques are covered
            cover_mask = 0
            for v in current_S:
                cover_mask |= vertex_clique_mask[v]
            
            # For each clique, count how many vertices in S cover it
            cover_count = [0] * num_cliques
            for v in current_S:
                for ci in vertex_to_cliques[v]:
                    cover_count[ci] += 1
            
            no_improve = 0
            max_no_improve = 5000
            
            while time.time() < deadline:
                if no_improve > max_no_improve:
                    break
                
                made_move = False
                
                # 1) Try removals: find vertex whose removal keeps cover_mask intact
                if current_S:
                    # Find vertices that are not critical
                    removable = []
                    for v in current_S:
                        # v is removable if for every clique it covers, some other vertex in S also covers it
                        # i.e., cover_count[ci] > 1 for all ci in vertex_to_cliques[v]
                        is_critical = False
                        for ci in vertex_to_cliques[v]:
                            if cover_count[ci] == 1:
                                is_critical = True
                                break
                        if not is_critical:
                            removable.append(v)
                    
                    if removable:
                        v = rng.choice(removable)
                        current_S.remove(v)
                        for ci in vertex_to_cliques[v]:
                            cover_count[ci] -= 1
                        cover_mask &= ~vertex_clique_mask[v]
                        current_size -= 1
                        no_improve = 0
                        made_move = True
                        continue
                
                # 2) Try swaps: replace v_in with v_out to maintain coverage
                if current_S:
                    # For each v_in in S, find cliques only covered by v_in
                    best_swap = None
                    best_swap_score = 0  # We want to minimize size, so any valid swap is good
                    
                    # Sample a few v_in to try (for speed)
                    sample_size = min(len(current_S), 5)
                    v_in_candidates = rng.sample(list(current_S), sample_size)
                    
                    for v_in in v_in_candidates:
                        # Find cliques uniquely covered by v_in
                        unique_mask = 0
                        for ci in vertex_to_cliques[v_in]:
                            if cover_count[ci] == 1:
                                unique_mask |= (1 << ci)
                        
                        if unique_mask == 0:
                            # v_in is removable (shouldn't happen if removals tried first, but just in case)
                            continue
                        
                        # Find v_out not in S that covers all unique cliques
                        valid_v_outs = []
                        for v_out in range(n):
                            if v_out in current_S:
                                continue
                            if (vertex_clique_mask[v_out] & unique_mask) == unique_mask:
                                valid_v_outs.append(v_out)
                        
                        if valid_v_outs:
                            v_out = rng.choice(valid_v_outs)
                            # Perform swap
                            current_S.remove(v_in)
                            for ci in vertex_to_cliques[v_in]:
                                cover_count[ci] -= 1
                            current_S.add(v_out)
                            for ci in vertex_to_cliques[v_out]:
                                cover_count[ci] += 1
                            cover_mask = (cover_mask & ~vertex_clique_mask[v_in]) | vertex_clique_mask[v_out]
                            no_improve = 0
                            made_move = True
                            break
                    
                    if made_move:
                        continue
                
                # 3) Try 2-for-1: replace two vertices with one
                if len(current_S) >= 2:
                    # Sample pairs
                    sample_pairs = min(10, (len(current_S) * (len(current_S) - 1)) // 2)
                    tried = set()
                    found = False
                    for _ in range(sample_pairs):
                        v1, v2 = rng.sample(list(current_S), 2)
                        if (v1, v2) in tried or (v2, v1) in tried:
                            continue
                        tried.add((v1, v2))
                        tried.add((v2, v1))
                        
                        # Find cliques that would become uncovered if we remove both v1 and v2
                        # A clique ci is uncovered if cover_count[ci] == number of {v1,v2} that cover ci
                        uncovered_mask = 0
                        for ci in range(num_cliques):
                            cnt_in_pair = 0
                            if ci in vertex_to_cliques[v1]:
                                cnt_in_pair += 1
                            if ci in vertex_to_cliques[v2]:
                                cnt_in_pair += 1
                            if cnt_in_pair > 0 and cover_count[ci] == cnt_in_pair:
                                uncovered_mask |= (1 << ci)
                        
                        if uncovered_mask == 0:
                            # Can remove both
                            current_S.remove(v1)
                            for ci in vertex_to_cliques[v1]:
                                cover_count[ci] -= 1
                            current_S.remove(v2)
                            for ci in vertex_to_cliques[v2]:
                                cover_count[ci] -= 1
                            cover_mask &= ~vertex_clique_mask[v1]
                            cover_mask &= ~vertex_clique_mask[v2]
                            current_size -= 2
                            no_improve = 0
                            found = True
                            break
                        
                        # Find v_out that covers all uncovered cliques
                        valid_v_outs = []
                        for v_out in range(n):
                            if v_out in current_S:
                                continue
                            if (vertex_clique_mask[v_out] & uncovered_mask) == uncovered_mask:
                                valid_v_outs.append(v_out)
                        
                        if valid_v_outs:
                            v_out = rng.choice(valid_v_outs)
                            current_S.remove(v1)
                            for ci in vertex_to_cliques[v1]:
                                cover_count[ci] -= 1
                            current_S.remove(v2)
                            for ci in vertex_to_cliques[v2]:
                                cover_count[ci] -= 1
                            current_S.add(v_out)
                            for ci in vertex_to_cliques[v_out]:
                                cover_count[ci] += 1
                            cover_mask = (cover_mask & ~vertex_clique_mask[v1] & ~vertex_clique_mask[v2]) | vertex_clique_mask[v_out]
                            current_size -= 1
                            no_improve = 0
                            found = True
                            break
                    
                    if found:
                        continue
                
                no_improve += 1
            
            return current_S

        # Multiple restarts with different initializations
        best_S = greedy_init()
        best_S = prune(best_S)
        best_size = len(best_S)
        
        # Run multiple local search phases with different random seeds
        current_time = time.time()
        remaining_time = deadline - current_time
        
        # Phase 1: Local search
        phase1_time = remaining_time * 0.4
        if phase1_time > 1:
            rng1 = random.Random(42)
            improved_S = local_search_bitmask(best_S, rng1, current_time + phase1_time)
            if len(improved_S) < best_size:
                best_S = improved_S
                best_size = len(best_S)
            current_time = time.time()
        
        # Phase 2: Another local search from best
        phase2_time = (deadline - current_time) * 0.4
        if phase2_time > 1:
            rng2 = random.Random(123)
            improved_S = local_search_bitmask(best_S, rng2, current_time + phase2_time)
            if len(improved_S) < best_size:
                best_S = improved_S
                best_size = len(best_S)
            current_time = time.time()
        
        # Phase 3: Final long local search
        if time.time() < deadline:
            rng3 = random.Random(456)
            improved_S = local_search_bitmask(best_S, rng3, deadline)
            if len(improved_S) < best_size:
                best_S = improved_S
                best_size = len(best_S)
        
        return sorted(best_S)

    out = []
    # Allocate time: give more to larger instances
    # Instance sizes: 30, 35, 40 nodes
    # Allocate proportional to n^2 (clique enumeration cost)
    n_values = [30, 35, 40]
    weights = [n*n for n in n_values]
    total_weight = sum(weights)
    
    time_alloc = []
    elapsed = 0
    for i, (n, p, seed) in enumerate(zip(n_values, [0.3, 0.25, 0.2], [201, 202, 203])):
        alloc = (deadline - start_time) * weights[i] / total_weight
        time_alloc.append(alloc)
    
    for idx, (n, p, seed) in enumerate(zip(n_values, [0.3, 0.25, 0.2], [201, 202, 203])):
        instance_deadline = start_time + time_alloc[idx]
        # Ensure minimum time for each instance
        instance_deadline = max(instance_deadline, start_time + 15)
        
        if time.time() >= instance_deadline:
            # Fallback: quick greedy
            adj, edges = random_graph(n, p, seed)
            cliques = bron_kerbosch_maximal_cliques(adj)
            cliques_list = [set(c) for c in cliques]
            S = set()
            remaining = list(cliques_list)
            while remaining:
                best_v = None
                max_cov = -1
                for v in range(n):
                    if v in S:
                        continue
                    cov = sum(1 for c in remaining if v in c)
                    if cov > max_cov:
                        max_cov = cov
                        best_v = v
                if best_v is None:
                    break
                S.add(best_v)
                remaining = [c for c in remaining if best_v not in c]
            out.append(sorted(S))
            continue
        
        adj, edges = random_graph(n, p, seed)
        res = solve_instance(adj, n, instance_deadline)
        out.append(res)
        
    return out
