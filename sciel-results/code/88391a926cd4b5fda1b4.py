import time
import random
from itertools import combinations

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
        
        # Precompute vertex -> list of clique indices
        # This allows O(1) lookup for which cliques a vertex covers
        vertex_to_cliques = [[] for _ in range(n)]
        for i, c in enumerate(cliques_list):
            for v in c:
                vertex_to_cliques[v].append(i)
        
        # Track coverage count for each clique
        # cover_count[i] = number of vertices in S that are in clique i
        # A clique is covered if cover_count[i] > 0
        # Uncovered count = number of i where cover_count[i] == 0
        
        def greedy_init():
            S = set()
            cover_count = [0] * len(cliques_list)
            remaining_cliques = len(cliques_list)
            
            while remaining_cliques > 0:
                best_v = -1
                max_new = -1
                for v in range(n):
                    if v in S:
                        continue
                    new_cover = 0
                    for ci in vertex_to_cliques[v]:
                        if cover_count[ci] == 0:
                            new_cover += 1
                    if new_cover > max_new:
                        max_new = new_cover
                        best_v = v
                if best_v == -1:
                    break
                S.add(best_v)
                for ci in vertex_to_cliques[best_v]:
                    if cover_count[ci] == 0:
                        remaining_cliques -= 1
                    cover_count[ci] += 1
            return S

        def prune(S):
            S = set(S)
            cover_count = [0] * len(cliques_list)
            for v in S:
                for ci in vertex_to_cliques[v]:
                    cover_count[ci] += 1
            remaining_cliques = sum(1 for c in cover_count if c == 0)
            
            changed = True
            while changed:
                changed = False
                for v in list(S):
                    # Check if removing v keeps all cliques covered
                    # v covers cliques in vertex_to_cliques[v]
                    # If any of those have count 1, removing v makes them 0
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

        best_S = greedy_init()
        best_S = prune(best_S)
        
        # Calculate initial score (size)
        # We assume valid. If not, we need to handle invalidity.
        # Greedy ensures validity.
        
        current_S = set(best_S)
        current_size = len(current_S)
        best_size = current_size
        
        # Precompute cover counts for current_S
        cover_count = [0] * len(cliques_list)
        for v in current_S:
            for ci in vertex_to_cliques[v]:
                cover_count[ci] += 1
        
        # Local Search
        # We will use a simple hill climbing with restarts or SA
        # Given the small size, hill climbing with swaps is very effective.
        
        # Helper to check if S is valid (all cover_count > 0)
        # We maintain remaining_cliques count
        remaining_cliques = sum(1 for c in cover_count if c == 0)
        
        # If greedy failed to cover (shouldn't happen), add random vertices
        while remaining_cliques > 0:
             # Pick random vertex that covers most uncovered
             best_v = -1
             max_cov = -1
             for v in range(n):
                 if v in current_S: continue
                 cov = 0
                 for ci in vertex_to_cliques[v]:
                     if cover_count[ci] == 0:
                         cov += 1
                 if cov > max_cov:
                     max_cov = cov
                     best_v = v
             if best_v == -1: break
             current_S.add(best_v)
             for ci in vertex_to_cliques[best_v]:
                 if cover_count[ci] == 0:
                     remaining_cliques -= 1
                 cover_count[ci] += 1
             current_size = len(current_S)
             if current_size < best_size:
                 best_size = current_size
                 best_S = set(current_S)
        
        # Main Loop: Swap-based Local Search
        # We try to swap a vertex in S with a vertex not in S.
        # Or remove a vertex if it doesn't break validity.
        
        rng = random.Random(42)
        no_improve_count = 0
        max_no_improve = 2000
        
        while time.time() < deadline:
            if no_improve_count > max_no_improve:
                break
                
            # Try a random move
            # 1. Remove a vertex (if valid)
            # 2. Swap a vertex (if valid or improves size? Size stays same. We want to improve structure for future removals?)
            # Actually, for Set Cover, minimizing size is the goal.
            # If we swap, size is constant. It might allow a subsequent removal.
            # Let's prioritize Removal.
            
            # Try removal
            removed = False
            if current_S:
                # Try to remove a random vertex
                v_out = rng.choice(list(current_S))
                # Check if v_out is critical
                is_critical = False
                for ci in vertex_to_cliques[v_out]:
                    if cover_count[ci] == 1:
                        is_critical = True
                        break
                
                if not is_critical:
                    # Safe to remove
                    current_S.remove(v_out)
                    for ci in vertex_to_cliques[v_out]:
                        cover_count[ci] -= 1
                    current_size -= 1
                    no_improve_count = 0
                    if current_size < best_size:
                        best_size = current_size
                        best_S = set(current_S)
                    removed = True
            
            if not removed:
                # Try Swap
                # Pick v_in from S, v_out from not S
                # We want to swap if it allows future removals?
                # Or just explore.
                # Let's try to find a swap that reduces "criticality" or allows a removal.
                # Simpler: Random swap to escape local optima.
                if len(current_S) > 0:
                    v_in = rng.choice(list(current_S))
                    # Find candidates for v_out
                    # Prefer vertices that cover many uncovered? No, we are valid.
                    # Prefer vertices that cover cliques currently covered only by v_in?
                    # That would be a "safe" swap that might allow v_in removal?
                    # No, if we swap v_in -> v_out, v_in is removed.
                    # If v_in covers a clique C with count 1, and v_out does NOT cover C, then C becomes uncovered. Invalid.
                    # So we must ensure v_out covers all cliques that v_in uniquely covers.
                    
                    unique_cliques_of_v_in = []
                    for ci in vertex_to_cliques[v_in]:
                        if cover_count[ci] == 1:
                            unique_cliques_of_v_in.append(ci)
                    
                    # If v_in has no unique cliques, we can swap with anyone (or just remove, which we already tried? 
                    # Wait, if no unique cliques, removal is safe. So this case is handled by removal logic above.
                    # So here, v_in MUST have at least one unique clique.
                    
                    if unique_cliques_of_v_in:
                        # v_out must cover all unique_cliques_of_v_in
                        valid_v_outs = []
                        # Optimization: Check vertices not in S
                        for v_out in range(n):
                            if v_out in current_S:
                                continue
                            # Check if v_out covers all unique cliques
                            ok = True
                            # v_out covers clique ci if ci is in vertex_to_cliques[v_out]
                            # We need to check membership in vertex_to_cliques[v_out]
                            # This is O(degree).
                            v_out_cliques_set = set(vertex_to_cliques[v_out]) # Cache this? 
                            # Actually, let's just iterate unique_cliques and check if ci in vertex_to_cliques[v_out]
                            # Since unique_cliques is small, this is fast.
                            for ci in unique_cliques_of_v_in:
                                if ci not in vertex_to_cliques[v_out]:
                                    ok = False
                                    break
                            if ok:
                                valid_v_outs.append(v_out)
                        
                        if valid_v_outs:
                            v_out = rng.choice(valid_v_outs)
                            # Perform Swap
                            current_S.remove(v_in)
                            for ci in vertex_to_cliques[v_in]:
                                cover_count[ci] -= 1
                            
                            current_S.add(v_out)
                            for ci in vertex_to_cliques[v_out]:
                                cover_count[ci] += 1
                            
                            # Size unchanged. 
                            # This move might enable a removal in the next step.
                            # We don't count this as an improvement in size, but it changes state.
                            # To avoid infinite loops of swaps, we rely on the fact that 
                            # if a removal becomes possible, we take it.
                            # If no removal is possible, we might swap forever.
                            # Limit: if we do many swaps without removal, restart or stop.
                            # For now, just count as no_improve? No, that kills the search.
                            # Let's just do it.
                            pass
                        else:
                            no_improve_count += 1
                    else:
                        # Should have been removed
                        no_improve_count += 1
                else:
                    no_improve_count += 1

        return sorted(best_S)

    out = []
    # Allocate time roughly equally, but give more to larger instances?
    # 30, 35, 40 nodes.
    # Let's just run until deadline.
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() - start_time > deadline:
            # Fallback: return empty or best effort? 
            # If we run out of time, we should return what we have or a valid simple set.
            # The greedy init is valid. Let's just run greedy if SA didn't start or if we are late.
            adj, edges = random_graph(n, p, seed)
            # Quick greedy
            cliques = bron_kerbosch_maximal_cliques(adj)
            S = set()
            remaining = [set(c) for c in cliques]
            while remaining:
                best_v = None
                max_cov = -1
                for v in range(n):
                    if v in S: continue
                    cov = sum(1 for c in remaining if v in c)
                    if cov > max_cov:
                        max_cov = cov
                        best_v = v
                if best_v is None: break
                S.add(best_v)
                remaining = [c for c in remaining if best_v not in c]
            out.append(sorted(S))
            continue
            
        adj, edges = random_graph(n, p, seed)
        # Pass remaining time
        res = solve_instance(adj, n, deadline)
        out.append(res)
        
    return out
