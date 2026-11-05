import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 118.0  # Keep 2s buffer for 120s limit

    def elapsed():
        return time.time() - start_time

    def time_ok():
        return elapsed() < deadline

    # 1. Construct Polarity Graph of PG(2,5)
    q = 5
    # Points in PG(2,q): (x,y,1) for x,y in F_q, (x,1,0) for x in F_q, (1,0,0)
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    
    n0 = len(points)  # 31

    # Polarity: symmetric bilinear form <p,q> = x1*y2 + y1*x2 + z1*z2 = 0
    def is_orthogonal(p, qpt):
        x1, y1, z1 = p
        x2, y2, z2 = qpt
        return (x1 * y2 + y1 * x2 + z1 * z2) % q == 0

    # Build adjacency
    adj = [set() for _ in range(n0)]
    for i in range(n0):
        for j in range(i + 1, n0):
            if is_orthogonal(points[i], points[j]):
                adj[i].add(j)
                adj[j].add(i)
    
    n = n0

    # 2. Fast independent set check using bitsets
    # Represent subsets as integers (Python big ints)
    # For n up to 43, this is efficient.
    
    def has_independent_set_size_k(adj, k):
        n_v = len(adj)
        if k > n_v:
            return False
        if k <= 1:
            return n_v >= k
        
        # Precompute neighbor masks
        # mask[i] has bit j set if i and j are adjacent
        masks = [0] * n_v
        for i in range(n_v):
            m = 0
            for j in adj[i]:
                m |= (1 << j)
            masks[i] = m
        
        # DFS with pruning
        # We want to find an independent set of size k.
        # State: current independent set size, available vertices (bitmask)
        # Pruning:
        # 1. If current_size + popcount(available) < k: return False
        # 2. Greedy upper bound on independent set size in 'available'?
        #    A simple bound: if we color the subgraph, alpha <= number of colors? No.
        #    Greedy lower bound on clique cover? 
        #    Let's stick to simple pruning + ordering.
        
        # To speed up, we can use a recursive helper that picks a vertex 
        # and branches on include/exclude.
        # To make it fast, we should pick vertices that are "high degree" in the 
        # available subgraph? Or low degree?
        # If we pick a vertex v:
        # Include v: remove v and N(v) from available.
        # Exclude v: remove v from available.
        
        # A better heuristic for finding a LARGE independent set (to prove existence):
        # Try to build it greedily first?
        # If greedy finds size k, return True immediately.
        
        # Greedy attempt
        avail = (1 << n_v) - 1
        current_size = 0
        while avail and current_size < k:
            # Pick a vertex with minimum degree in the available subgraph?
            # Or just first available?
            # Let's pick a random vertex or min-degree to maximize chance of success?
            # For proving existence, any order works if it succeeds.
            # Let's try min-degree in available subgraph.
            min_deg = n_v
            min_v = -1
            a = avail
            while a:
                lsb = a & (-a)
                v = lsb.bit_length() - 1
                a ^= lsb
                # Count neighbors of v in avail
                deg = bin(masks[v] & avail).count('1')
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
                    if min_deg == 0:
                        break
            if min_v == -1:
                break
            current_size += 1
            # Remove v and its neighbors
            avail &= ~(1 << min_v)
            avail &= ~masks[min_v]
            
        if current_size >= k:
            return True
            
        # If greedy failed, we need exact search.
        # But greedy is a heuristic. It might fail even if an IS of size k exists.
        # So we must do exact search if greedy fails?
        # Actually, if we want to check if alpha >= k, and greedy finds one, we return True.
        # If greedy doesn't, we MUST verify if one exists.
        
        # Exact search
        def dfs(available, current_size):
            if current_size == k:
                return True
            if not available:
                return False
            # Prune: max possible size
            if current_size + bin(available).count('1') < k:
                return False
                
            # Pick a vertex
            # Heuristic: pick vertex with max degree in available subgraph?
            # This reduces the search space faster in the "exclude" branch?
            # Or min degree?
            # Let's pick the first available vertex for simplicity, but maybe randomize?
            # Randomization helps avoid getting stuck in bad branches.
            # Let's pick a vertex with high degree to prune the "include" branch less?
            # No, "include" removes neighbors. High degree removes many.
            # "Exclude" removes only v.
            # So high degree vertex is good for "include" branch pruning.
            
            # Let's just pick the first one.
            lsb = available & (-available)
            v = lsb.bit_length() - 1
            
            # Option 1: Include v
            # New available: available without v and N(v)
            new_avail_include = available & ~(1 << v) & ~masks[v]
            if dfs(new_avail_include, current_size + 1):
                return True
                
            # Option 2: Exclude v
            new_avail_exclude = available & ~(1 << v)
            if dfs(new_avail_exclude, current_size):
                return True
            
            return False
        
        return dfs((1 << n_v) - 1, 0)

    # 3. Expansion Strategy
    # We want to add vertices one by one.
    # New vertex v connects to a set S of existing vertices.
    # Constraints:
    # 1. C4-free: No two vertices in S can share a neighbor in the old graph.
    #    i.e., S must be an independent set in the "conflict graph" where edges
    #    represent sharing a neighbor.
    # 2. Alpha < 11: The new graph must not have an independent set of size 11.
    #    New IS can be:
    #    a) Old IS not containing v (size <= 10, guaranteed if old alpha <= 10)
    #    b) Old IS containing v. v is in S's complement? No, v is connected to S.
    #       So v is NOT adjacent to V\S.
    #       An IS containing v must consist of {v} U I, where I is an IS in V\S.
    #       So we need alpha(V\S) <= 9.
    
    # Precompute conflict graph for the base graph
    # conflict[i] contains j if adj[i] & adj[j] is not empty
    conflict = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & adj[j]:
                conflict[i].add(j)
                conflict[j].add(i)
                
    # Helper to check if S is independent in conflict graph
    def is_conflict_independent(S):
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                if S_list[j] in conflict[S_list[i]]:
                    return False
        return True

    # Helper to check alpha of subgraph induced by V \ S
    # We need to create the adjacency for V \ S
    def alpha_of_complement(S, adj, n):
        # Map old indices to new indices
        # Or just filter
        V = set(range(n)) - S
        if len(V) < 10:
            return len(V)
        
        # Create subgraph
        # For speed, we can just pass the filtered adjacency to has_independent_set_size_k
        # But has_independent_set_size_k expects a list of sets of indices 0..n-1.
        # We need to remap.
        
        # Optimization: If S is small, the complement is large.
        # Let's just build the subgraph adj_sub.
        # Map: old_idx -> new_idx
        mapping = {}
        new_idx = 0
        for v in V:
            mapping[v] = new_idx
            new_idx += 1
            
        n_sub = len(V)
        adj_sub = [set() for _ in range(n_sub)]
        for v in V:
            u_new = mapping[v]
            for w in adj[v]:
                if w in mapping:
                    adj_sub[u_new].add(mapping[w])
                    
        # Check if alpha(adj_sub) >= 10
        # We only care if it's >= 10 (which would make new alpha >= 11)
        return has_independent_set_size_k(adj_sub, 10)

    best_n = n
    best_edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                best_edges.append([u, v])

    # Main expansion loop
    while time_ok():
        added = False
        
        # Strategy: Try to find a valid S
        # We want S to be small to keep alpha low?
        # Or large?
        # If S is large, V\S is small, so alpha(V\S) is likely small.
        # But S must be conflict-independent.
        # In the polarity graph, the conflict graph is dense?
        
        # Let's try random S of varying sizes.
        # Start with small S (size 1, 2) as they are easier to find conflict-independent.
        # But wait, if S is small, V\S is large, so alpha(V\S) might be large.
        # If S is large, V\S is small, so alpha(V\S) is small.
        # So we should prefer LARGER S?
        
        # However, finding a large conflict-independent set is hard.
        # Let's try greedy construction of S.
        
        attempts = 0
        max_attempts = 200
        while attempts < max_attempts and time_ok():
            attempts += 1
            
            # Generate a random conflict-independent set S
            S = set()
            available = set(range(n))
            # Shuffle to randomize
            avail_list = list(available)
            random.shuffle(avail_list)
            
            for u in avail_list:
                if u not in available:
                    continue
                # Check if u conflicts with any in S
                ok = True
                for w in S:
                    if w in conflict[u]:
                        ok = False
                        break
                if ok:
                    S.add(u)
                    # Remove u and its conflicts from available
                    available.remove(u)
                    for w in conflict[u]:
                        if w in available:
                            available.remove(w)
            
            if not S:
                continue
            
            # Check C4 constraint (already ensured by construction of S)
            
            # Check Alpha constraint
            # We need alpha(V \ S) <= 9
            if not alpha_of_complement(S, adj, n):
                # Valid!
                # Add vertex n with neighbors S
                new_v = n
                adj.append(set(S))
                for u in S:
                    adj[u].add(new_v)
                
                # Update conflict
                # New conflicts for new_v:
                # new_v conflicts with u if adj[new_v] & adj[u] is not empty.
                # adj[new_v] = S.
                # So new_v conflicts with u if S & adj[u] is not empty.
                conflict.append(set())
                for u in range(n):
                    if S & adj[u]:
                        conflict[u].add(new_v)
                        conflict[new_v].add(u)
                
                # Update conflicts within S?
                # Pairs in S now share new_v as a common neighbor.
                # So all pairs in S become conflicts.
                S_list = list(S)
                for i in range(len(S_list)):
                    for j in range(i+1, len(S_list)):
                        conflict[S_list[i]].add(S_list[j])
                        conflict[S_list[j]].add(S_list[i])
                        
                n += 1
                if n > best_n:
                    best_n = n
                    best_edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                best_edges.append([u, v])
                added = True
                break # Restart search with new state
                
        if not added:
            # If we can't add a vertex with the current strategy, try a different one?
            # Maybe force S to be smaller or larger?
            # Or just break?
            # Let's try a few more iterations with different random seeds or strategies.
            # If still stuck, break.
            
            # Try specific small S
            found = False
            for size in [1, 2, 3]:
                if not time_ok():
                    break
                # Try random subsets of size 'size'
                for _ in range(50):
                    if not time_ok():
                        break
                    S = set(random.sample(range(n), size))
                    if not is_conflict_independent(S):
                        continue
                    if not alpha_of_complement(S, adj, n):
                        # Valid
                        new_v = n
                        adj.append(set(S))
                        for u in S:
                            adj[u].add(new_v)
                        
                        conflict.append(set())
                        for u in range(n):
                            if S & adj[u]:
                                conflict[u].add(new_v)
                                conflict[new_v].add(u)
                        
                        S_list = list(S)
                        for i in range(len(S_list)):
                            for j in range(i+1, len(S_list)):
                                conflict[S_list[i]].add(S_list[j])
                                conflict[S_list[j]].add(S_list[i])
                                
                        n += 1
                        if n > best_n:
                            best_n = n
                            best_edges = []
                            for u in range(n):
                                for v in adj[u]:
                                    if u < v:
                                        best_edges.append([u, v])
                        found = True
                        break
                if found:
                    break
            
            if not found:
                break

    return {'n': best_n, 'edges': best_edges}
