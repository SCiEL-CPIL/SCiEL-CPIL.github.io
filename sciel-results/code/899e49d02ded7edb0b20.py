import time
from random import Random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency for C
        adj_C = [set() for _ in range(k)]
        # Build neighborhoods for I
        N_I = [set() for _ in range(n)]
        
        for u, v in edges:
            if u < k and v < k:
                adj_C[u].add(v)
                adj_C[v].add(u)
            elif u < k:
                N_I[v].add(u)
            else:
                N_I[u].add(v)
        
        # Identify active I vertices
        active_I = []
        for v in range(k, n):
            if N_I[v]:
                active_I.append(v)
        
        # Sort active I by neighborhood size descending
        active_I.sort(key=lambda v: len(N_I[v]), reverse=True)
        
        # Precompute neighborhood masks for I
        # mask[v] is a bitmask of C vertices
        N_masks = [0] * n
        for v in active_I:
            mask = 0
            for u in N_I[v]:
                mask |= (1 << u)
            N_masks[v] = mask
        
        # Helper: check if a set of vertices (bitmask) is a clique in C
        def is_clique(mask):
            # Extract vertices
            verts = []
            temp = mask
            while temp:
                lsb = temp & (-temp)
                u = lsb.bit_length() - 1
                verts.append(u)
                temp ^= lsb
            
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    if verts[j] not in adj_C[verts[i]]:
                        return False
            return True

        # We want to partition C (bitmask (1<<k)-1) into m cliques
        # such that there is a matching from active_I to cliques
        # where v is matched to clique Q iff N_masks[v] is subset of Q
        
        full_C_mask = (1 << k) - 1
        
        # Iterate m from len(active_I) to k (max cliques is k, size 1)
        # Actually, max cliques is k. Min is len(active_I).
        min_m = len(active_I)
        max_m = k
        
        # Backtracking state
        # cliques: list of bitmasks
        # uncovered: bitmask of C vertices not yet in any clique
        # matched_v: bitmask of active_I vertices already matched to a clique
        # Note: A clique can be matched to at most one v.
        # We can decide matching on the fly.
        
        def backtrack(m, cliques, uncovered, matched_mask):
            if time.time() > deadline:
                return None
            
            if uncovered == 0:
                # All C vertices covered
                # Check if all active_I are matched
                if matched_mask == (1 << len(active_I)) - 1:
                    return cliques
                else:
                    return None
            
            # Heuristic: Pick the vertex in uncovered with highest degree in uncovered
            # Or just the first one
            # To improve matching, we should prioritize cliques that can serve unmatched v's
            # Let's pick the next vertex u to add to a clique
            
            # Find a candidate vertex u in uncovered
            # Try to extend existing cliques first
            u_temp = uncovered
            # Pick u with max degree in uncovered subgraph to force clique structure
            # Actually, for partitioning into cliques, we just need to ensure the new set is a clique.
            
            # Get list of vertices in uncovered
            verts_uncovered = []
            temp = uncovered
            while temp:
                lsb = temp & (-temp)
                u = lsb.bit_length() - 1
                verts_uncovered.append(u)
                temp ^= lsb
            
            if not verts_uncovered:
                return None

            # Order vertices by degree in uncovered (descending)
            verts_uncovered.sort(key=lambda u: sum(1 for w in verts_uncovered if w in adj_C[u]), reverse=True)
            
            u = verts_uncovered[0]
            
            # Try adding u to an existing clique
            for i, q in enumerate(cliques):
                new_q = q | (1 << u)
                if is_clique(new_q):
                    # Check if we can match some v to this new clique
                    # Or if it was already matched, it stays matched
                    # If it wasn't matched, we might match it now
                    
                    # Option 1: Don't match any new v to this clique (keep matched status)
                    # But if it becomes a superset of N_v for some unmatched v, we *can* match it.
                    # To maximize success, we should match if possible?
                    # No, we might need this clique for a "larger" N_v later?
                    # Actually, if N_v is subset of new_q, and v is unmatched, we can match it.
                    # But maybe another clique will be better for v?
                    # Since we are partitioning, once a clique is formed, it's fixed.
                    # So we should try to match v to the "smallest" possible clique containing N_v?
                    
                    # Let's try all options for matching:
                    # 1. No new match
                    res = backtrack(m, cliques[:i] + [new_q] + cliques[i+1:], uncovered ^ (1 << u), matched_mask)
                    if res is not None:
                        return res
                    
                    # 2. Match an unmatched v to this clique
                    # Find v such that N_masks[v] is subset of new_q and v is not in matched_mask
                    # To reduce branching, pick the "hardest" v (largest N_v) that fits?
                    # Or just try all?
                    # Since k is small, number of v's is ~60.
                    # But we can filter: only v's where N_masks[v] is subset of new_q.
                    
                    # Optimization: Only try to match if the clique was NOT previously matched
                    # If it was matched, it's already covered.
                    # We need to know if clique i was matched.
                    # This state representation is getting complex.
                    
                    # Alternative:
                    # Just form the partition of C first.
                    # Then check if matching exists.
                    # This is simpler.
                    # Let's stick to: Find partition of C into m cliques.
                    # Then check Hall's condition / Bipartite Matching.
                    
            # Option 2: Start a new clique with u
            if len(cliques) < m:
                new_clique = 1 << u
                res = backtrack(m, cliques + [new_clique], uncovered ^ (1 << u), matched_mask)
                if res is not None:
                    return res
            
            return None
        
        # The above backtracking is for finding ANY partition of size m.
        # It doesn't enforce the matching constraint during search, which might be slow if many partitions fail matching.
        # But checking matching is fast (Hopcroft-Karp).
        
        def check_matching(cliques):
            # cliques: list of bitmasks
            # active_I: list of v indices
            # Build bipartite graph: v -> i if N_masks[active_I[v]] is subset of cliques[i]
            # Use simple DFS matching since sizes are small
            
            n_v = len(active_I)
            n_q = len(cliques)
            
            # adj[v_idx] = list of q_idx
            adj = [[] for _ in range(n_v)]
            for i, v in enumerate(active_I):
                mask_v = N_masks[v]
                for j, q_mask in enumerate(cliques):
                    if (mask_v & q_mask) == mask_v:
                        adj[i].append(j)
            
            # Bipartite matching
            match_q = [-1] * n_q # which v is matched to q
            match_v = [-1] * n_v # which q is matched to v
            
            def dfs(v, seen):
                for q in adj[v]:
                    if seen[q]:
                        continue
                    seen[q] = True
                    if match_q[q] == -1 or dfs(match_q[q], seen):
                        match_q[q] = v
                        match_v[v] = q
                        return True
                return False
            
            for v in range(n_v):
                seen = [False] * n_q
                if not dfs(v, seen):
                    return False
            return True

        # Search for minimum m
        for m in range(min_m, max_m + 1):
            if time.time() > deadline:
                break
            
            # Use backtracking to find a partition of C into m cliques
            # To speed up, we can use a greedy initialization or better heuristics
            # For now, simple backtracking with pruning
            
            # Pruning: If m < max size of any N_v, it's impossible?
            # No, N_v can be covered by a clique of size >= |N_v|.
            
            # Let's try to find one valid partition
            found_partition = None
            
            # Simple backtracking
            # State: cliques (list of masks), uncovered (mask)
            def bt_partition(cliques, uncovered):
                if time.time() > deadline:
                    return None
                if uncovered == 0:
                    return cliques
                
                # Pick vertex u
                temp = uncovered
                verts = []
                while temp:
                    lsb = temp & (-temp)
                    u = lsb.bit_length() - 1
                    verts.append(u)
                    temp ^= lsb
                
                if not verts:
                    return None
                
                # Sort by degree in uncovered
                verts.sort(key=lambda u: sum(1 for w in verts if w in adj_C[u]), reverse=True)
                u = verts[0]
                
                # Try adding to existing cliques
                for i, q in enumerate(cliques):
                    new_q = q | (1 << u)
                    if is_clique(new_q):
                        res = bt_partition(cliques[:i] + [new_q] + cliques[i+1:], uncovered ^ (1 << u))
                        if res is not None:
                            return res
                
                # Try new clique
                if len(cliques) < m:
                    res = bt_partition(cliques + [1 << u], uncovered ^ (1 << u))
                    if res is not None:
                        return res
                
                return None

            # Start with empty cliques
            res = bt_partition([], full_C_mask)
            
            if res is not None:
                # Check matching
                if check_matching(res):
                    # Construct final partition
                    final_cliques = []
                    for q_mask in res:
                        # Convert mask to list
                        verts = []
                        temp = q_mask
                        while temp:
                            lsb = temp & (-temp)
                            u = lsb.bit_length() - 1
                            verts.append(u)
                            temp ^= lsb
                        final_cliques.append(verts)
                    
                    # Now assign v's to cliques
                    # Re-run matching to get assignments
                    n_v = len(active_I)
                    n_q = len(res)
                    adj = [[] for _ in range(n_v)]
                    for i, v in enumerate(active_I):
                        mask_v = N_masks[v]
                        for j, q_mask in enumerate(res):
                            if (mask_v & q_mask) == mask_v:
                                adj[i].append(j)
                    
                    match_q = [-1] * n_q
                    match_v = [-1] * n_v
                    
                    def dfs(v, seen):
                        for q in adj[v]:
                            if seen[q]:
                                continue
                            seen[q] = True
                            if match_q[q] == -1 or dfs(match_q[q], seen):
                                match_q[q] = v
                                match_v[v] = q
                                return True
                        return False
                    
                    for v in range(n_v):
                        seen = [False] * n_q
                        dfs(v, seen)
                    
                    # Add v's to cliques
                    for i, v in enumerate(active_I):
                        if match_v[i] != -1:
                            j = match_v[i]
                            final_cliques[j].append(v)
                    
                    return final_cliques
                else:
                    # This partition doesn't work, need to find another one
                    # Backtracking should continue to find next partition
                    # But my bt_partition returns the first one.
                    # I need to modify it to yield all partitions or continue search.
                    # For simplicity, if first one fails, we might need to try next m?
                    # No, a valid partition might exist for this m.
                    # I need a generator or a loop that tries different branches.
                    
                    # Let's implement a generator for partitions
                    def gen_partitions(cliques, uncovered):
                        if time.time() > deadline:
                            return
                        if uncovered == 0:
                            yield cliques
                            return
                        
                        temp = uncovered
                        verts = []
                        while temp:
                            lsb = temp & (-temp)
                            u = lsb.bit_length() - 1
                            verts.append(u)
                            temp ^= lsb
                        
                        if not verts:
                            return
                        
                        verts.sort(key=lambda u: sum(1 for w in verts if w in adj_C[u]), reverse=True)
                        u = verts[0]
                        
                        for i, q in enumerate(cliques):
                            new_q = q | (1 << u)
                            if is_clique(new_q):
                                yield from gen_partitions(cliques[:i] + [new_q] + cliques[i+1:], uncovered ^ (1 << u))
                        
                        if len(cliques) < m:
                            yield from gen_partitions(cliques + [1 << u], uncovered ^ (1 << u))

                    for part in gen_partitions([], full_C_mask):
                        if check_matching(part):
                            # Construct result
                            final_cliques = []
                            for q_mask in part:
                                verts = []
                                temp = q_mask
                                while temp:
                                    lsb = temp & (-temp)
                                    u = lsb.bit_length() - 1
                                    verts.append(u)
                                    temp ^= lsb
                                final_cliques.append(verts)
                            
                            # Matching
                            n_v = len(active_I)
                            n_q = len(part)
                            adj = [[] for _ in range(n_v)]
                            for i, v in enumerate(active_I):
                                mask_v = N_masks[v]
                                for j, q_mask in enumerate(part):
                                    if (mask_v & q_mask) == mask_v:
                                        adj[i].append(j)
                            
                            match_q = [-1] * n_q
                            match_v = [-1] * n_v
                            
                            def dfs(v, seen):
                                for q in adj[v]:
                                    if seen[q]:
                                        continue
                                    seen[q] = True
                                    if match_q[q] == -1 or dfs(match_q[q], seen):
                                        match_q[q] = v
                                        match_v[v] = q
                                        return True
                                return False
                            
                            for v in range(n_v):
                                seen = [False] * n_q
                                dfs(v, seen)
                            
                            for i, v in enumerate(active_I):
                                if match_v[i] != -1:
                                    j = match_v[i]
                                    final_cliques[j].append(v)
                            
                            return final_cliques
                    # If no partition with matching found for this m, continue to next m
                    continue
            else:
                # No partition found (timeout or impossible), continue
                continue

        # If all m failed, fallback to greedy (should not happen)
        # Return empty or best effort
        return []

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > deadline:
            while len(results) < 3:
                results.append([])
            return results
        res = solve_instance(*spec)
        cleaned = []
        for clique in res:
            cleaned.append([int(x) for x in clique])
        results.append(cleaned)
        
    return results
