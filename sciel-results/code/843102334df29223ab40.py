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
        
        # Use bitmask for coverage: bit i set means clique i is covered
        # For n <= 40 and typically < 1000 cliques, this is feasible
        
        def greedy_init():
            S = set()
            cover_mask = 0  # Bitmask of covered cliques
            remaining_cliques = num_cliques
            # Track how many times each clique is "needed" (uncovered)
            clique_weight = [1] * num_cliques  # weight = 1 if uncovered
            
            while remaining_cliques > 0:
                best_v = -1
                max_score = -1
                for v in range(n):
                    if v in S:
                        continue
                    score = 0
                    for ci in vertex_to_cliques[v]:
                        if not (cover_mask & (1 << ci)):
                            score += clique_weight[ci]
                    if score > max_score:
                        max_score = score
                        best_v = v
                if best_v == -1:
                    break
                S.add(best_v)
                for ci in vertex_to_cliques[best_v]:
                    if not (cover_mask & (1 << ci)):
                        remaining_cliques -= 1
                        cover_mask |= (1 << ci)
            return S

        def prune(S):
            S = set(S)
            cover_count = [0] * num_cliques
            for v in S:
                for ci in vertex_to_cliques[v]:
                    cover_count[ci] += 1
            
            changed = True
            while changed:
                changed = False
                for v in list(S):
                    # Check if removing v keeps all cliques covered
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

        def local_search(S, rng, deadline):
            """Local search with removals, swaps, and 2-for-1 improvements"""
            current_S = set(S)
            current_size = len(current_S)
            
            # Precompute cover counts
            cover_count = [0] * num_cliques
            for v in current_S:
                for ci in vertex_to_cliques[v]:
                    cover_count[ci] += 1
            
            # Precompute for each vertex, the set of cliques it covers (as a frozenset for fast lookup)
            vertex_clique_sets = {}
            for v in range(n):
                vertex_clique_sets[v] = frozenset(vertex_to_cliques[v])
            
            no_improve = 0
            max_no_improve = 10000
            step = 0
            
            while time.time() < deadline:
                if no_improve > max_no_improve:
                    break
                step += 1
                
                # Try removals first (deterministic pass + random)
                removed = False
                if current_S:
                    # Deterministic: try all vertices for removal
                    for v_out in list(current_S):
                        is_critical = False
                        for ci in vertex_to_cliques[v_out]:
                            if cover_count[ci] == 1:
                                is_critical = True
                                break
                        if not is_critical:
                            current_S.remove(v_out)
                            for ci in vertex_to_cliques[v_out]:
                                cover_count[ci] -= 1
                            current_size -= 1
                            no_improve = 0
                            removed = True
                            break
                    if removed:
                        continue
                    
                    # Random removal attempt
                    v_out = rng.choice(list(current_S))
                    is_critical = False
                    for ci in vertex_to_cliques[v_out]:
                        if cover_count[ci] == 1:
                            is_critical = True
                            break
                    if not is_critical:
                        current_S.remove(v_out)
                        for ci in vertex_to_cliques[v_out]:
                            cover_count[ci] -= 1
                        current_size -= 1
                        no_improve = 0
                        continue
                
                # Try 2-for-1: replace two vertices with one
                if len(current_S) >= 2:
                    v1, v2 = rng.sample(list(current_S), 2)
                    # Check if removing both would leave any clique uncovered
                    # That is, are there cliques covered ONLY by v1 or v2 (not both)
                    # Actually, we need: for every clique, it's covered by at least one of {v1,v2} or by some other vertex in S\{v1,v2}
                    # A clique is "uncovered" after removing v1,v2 iff cover_count[ci] == number of {v1,v2} that cover it
                    # i.e., if a clique is covered by exactly one of v1,v2 and no other vertex in S
                    uncovered_after_removal = []
                    for ci in vertex_to_cliques[v1]:
                        if cover_count[ci] == 1:
                            uncovered_after_removal.append(ci)
                    for ci in vertex_to_cliques[v2]:
                        if cover_count[ci] == 1 and ci not in uncovered_after_removal:
                            # Check if v1 also covers ci
                            if ci in vertex_clique_sets[v1]:
                                # Both v1 and v2 cover ci, so removing both leaves it uncovered
                                # Wait, if cover_count[ci]==1 for v2's perspective, that means only v2 covers it
                                # But we already checked v1. Let me redo this.
                                pass
                    # Simpler: compute which cliques would be uncovered
                    uncovered = set()
                    for ci in range(num_cliques):
                        # Count how many of v1,v2 cover ci
                        cnt = 0
                        if ci in vertex_clique_sets[v1]:
                            cnt += 1
                        if ci in vertex_clique_sets[v2]:
                            cnt += 1
                        if cnt > 0 and cover_count[ci] == cnt:
                            uncovered.add(ci)
                    
                    if uncovered:
                        # Find a single vertex that covers all uncovered cliques
                        v_out_cliques = vertex_clique_sets.get
                        valid_replacements = []
                        for v_out in range(n):
                            if v_out in current_S:
                                continue
                            if uncovered.issubset(vertex_clique_sets[v_out]):
                                valid_replacements.append(v_out)
                        
                        if valid_replacements:
                            v_out = rng.choice(valid_replacements)
                            # Remove v1, v2, add v_out
                            current_S.remove(v1)
                            for ci in vertex_to_cliques[v1]:
                                cover_count[ci] -= 1
                            current_S.remove(v2)
                            for ci in vertex_to_cliques[v2]:
                                cover_count[ci] -= 1
                            current_S.add(v_out)
                            for ci in vertex_to_cliques[v_out]:
                                cover_count[ci] += 1
                            current_size -= 1
                            no_improve = 0
                            continue
                    else:
                        # Can just remove both
                        current_S.remove(v1)
                        for ci in vertex_to_cliques[v1]:
                            cover_count[ci] -= 1
                        current_S.remove(v2)
                        for ci in vertex_to_cliques[v2]:
                            cover_count[ci] -= 1
                        current_size -= 2
                        no_improve = 0
                        continue
                
                # Try swaps: replace v_in with v_out
                if len(current_S) > 0:
                    v_in = rng.choice(list(current_S))
                    unique_cliques = [ci for ci in vertex_to_cliques[v_in] 
                                     if cover_count[ci] == 1]
                    
                    if not unique_cliques:
                        current_S.remove(v_in)
                        for ci in vertex_to_cliques[v_in]:
                            cover_count[ci] -= 1
                        current_size -= 1
                        no_improve = 0
                        continue
                    
                    valid_v_outs = []
                    unique_set = set(unique_cliques)
                    for v_out in range(n):
                        if v_out in current_S:
                            continue
                        if unique_set.issubset(vertex_clique_sets[v_out]):
                            valid_v_outs.append(v_out)
                    
                    if valid_v_outs:
                        v_out = rng.choice(valid_v_outs)
                        current_S.remove(v_in)
                        for ci in vertex_to_cliques[v_in]:
                            cover_count[ci] -= 1
                        current_S.add(v_out)
                        for ci in vertex_to_cliques[v_out]:
                            cover_count[ci] += 1
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            
            return current_S

        # Multiple restarts with different initializations
        best_S = greedy_init()
        best_S = prune(best_S)
        best_size = len(best_S)
        
        # Run multiple local search phases with different random seeds
        current_time = time.time()
        remaining_time = deadline - current_time
        
        # Phase 1: Short local search
        phase1_time = remaining_time * 0.3
        if phase1_time > 1:
            rng1 = random.Random(42)
            improved_S = local_search(best_S, rng1, current_time + phase1_time)
            if len(improved_S) < best_size:
                best_S = improved_S
                best_size = len(best_S)
            current_time = time.time()
        
        # Phase 2: Another local search from best
        phase2_time = (deadline - current_time) * 0.3
        if phase2_time > 1:
            rng2 = random.Random(123)
            improved_S = local_search(best_S, rng2, current_time + phase2_time)
            if len(improved_S) < best_size:
                best_S = improved_S
                best_size = len(best_S)
            current_time = time.time()
        
        # Phase 3: Final long local search
        if time.time() < deadline:
            rng3 = random.Random(456)
            improved_S = local_search(best_S, rng3, deadline)
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
