import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 115.0  # Keep 5s buffer for 120s limit
    
    # 1. Construct Polarity Graph of PG(2,5)
    q = 5
    # Points of PG(2,q)
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    
    n = len(points) # 31
    
    # Polarity: x1*y2 + y1*x2 + z1*z2 == 0 mod q
    def is_orthogonal(p, qpt):
        x1, y1, z1 = p
        x2, y2, z2 = qpt
        return (x1 * y2 + y1 * x2 + z1 * z2) % q == 0

    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if is_orthogonal(points[i], points[j]):
                adj[i].add(j)
                adj[j].add(i)
                
    # Initial check: C4-free and alpha < 11
    # PG(2,q) polarity graph is C4-free.
    # Alpha is q+1 = 6.
    
    # Conflict graph: conflict[u] contains v if N(u) & N(v) is not empty
    conflict = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & adj[j]:
                conflict[i].add(j)
                conflict[j].add(i)
                
    best_n = n
    best_edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                best_edges.append([u, v])
                
    # Bitset representation for fast independent set search
    # n <= 43, so we can use Python integers as bitsets.
    
    def check_alpha_le_k(adj, n, k):
        """
        Returns True if alpha(adj) <= k.
        Uses bitset backtracking.
        """
        # Precompute neighbor bitsets
        neigh = [0] * n
        for i in range(n):
            mask = 0
            for j in adj[i]:
                mask |= (1 << j)
            neigh[i] = mask
            
        # We want to know if there is an independent set of size k+1.
        # If we find one, return False.
        # If we exhaust search without finding one, return True.
        
        # To speed up, we can use a greedy upper bound.
        # If greedy finds an independent set of size k+1, return False immediately.
        # Greedy:
        rem = (1 << n) - 1
        size = 0
        while rem:
            # Pick vertex with min degree in remaining? Or just first?
            # For speed, just pick the first set bit.
            # To be more effective, pick vertex with highest degree in remaining.
            # But for a quick check, let's just do a simple DFS with pruning.
            
            pass # We'll do DFS below.
            
        def dfs(rem, current_size):
            if current_size > k:
                return True # Found independent set of size k+1
            if not rem:
                return False
            # Pruning: if current_size + popcount(rem) <= k, we can't exceed k?
            # No, we want to see if we CAN exceed k.
            # If current_size + popcount(rem) <= k, then even if we take all remaining, 
            # we won't exceed k. So return False (no set of size k+1 found in this branch).
            if current_size + bin(rem).count('1') <= k:
                return False
                
            # Pick a vertex
            # To prune better, pick a vertex with high degree in remaining graph.
            # This reduces the remaining set more.
            # Find v in rem with max degree in subgraph induced by rem.
            best_v = -1
            best_deg = -1
            # Iterate through set bits
            r = rem
            while r:
                lsb = r & -r
                v = lsb.bit_length() - 1
                # Count neighbors of v in rem
                # Neighbors of v are neigh[v]. Intersection with rem.
                common = neigh[v] & rem
                deg = bin(common).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                r ^= lsb
                
            if best_v == -1:
                return False
                
            # Option 1: Include best_v
            # New rem: remove best_v and its neighbors
            new_rem = rem & ~(neigh[best_v] | (1 << best_v))
            if dfs(new_rem, current_size + 1):
                return True
                
            # Option 2: Exclude best_v
            new_rem = rem & ~(1 << best_v)
            if dfs(new_rem, current_size):
                return True
                
            return False
            
        return not dfs((1 << n) - 1, 0)

    def try_add_vertex(adj, conflict, n, S):
        """
        Try to add a new vertex with neighborhood S.
        Returns (success, new_adj, new_conflict, new_n)
        """
        # 1. Check C4: S must be independent in conflict
        # We assume S is generated as independent in conflict, but double check for safety
        for u in S:
            for w in S:
                if u != w and w in conflict[u]:
                    return False, None, None, n
            
        # 2. Check Alpha: alpha(new_adj) <= 10
        # new_adj is adj + vertex n connected to S.
        # We need to check if there is an independent set of size 11 in new_adj.
        # This is equivalent to checking if there is an independent set of size 10 in adj[V\S].
        # Because if I is an independent set of size 10 in adj[V\S], then I U {n} is an independent set of size 11 in new_adj.
        # Conversely, any independent set of size 11 in new_adj either contains n or not.
        # If it contains n, it must be subset of {n} U (V\S) and be independent, so the other 10 are in V\S and independent in adj.
        # If it doesn't contain n, it is an independent set of size 11 in adj. But we know alpha(adj) <= 10.
        # So we just need to check if alpha(adj[V\S]) <= 9.
        
        # Construct subgraph adj_sub on V \ S
        # Map old indices to new indices in subgraph
        # V \ S has size n - len(S)
        
        # For speed, we can just build the neighbor lists for the subgraph.
        # But n is small (~30-40).
        
        # Let's just build the full new_adj and run the alpha check on it?
        # No, that's slower.
        # Let's build the subgraph induced by V \ S.
        
        sub_vertices = [i for i in range(n) if i not in S]
        sub_n = len(sub_vertices)
        if sub_n < 10:
            # Alpha is at most sub_n, which is < 10. So condition holds.
            pass
        else:
            # Build adj_sub
            # Map old index to new index
            idx_map = {old: new for new, old in enumerate(sub_vertices)}
            adj_sub = [set() for _ in range(sub_n)]
            for i_old in sub_vertices:
                i_new = idx_map[i_old]
                for j_old in adj[i_old]:
                    if j_old in idx_map:
                        j_new = idx_map[j_old]
                        if i_new < j_new:
                            adj_sub[i_new].add(j_new)
                            adj_sub[j_new].add(i_new)
            
            # Check if alpha(adj_sub) <= 9
            if not check_alpha_le_k(adj_sub, sub_n, 9):
                return False, None, None, n
                
        # If we reach here, it's valid.
        new_n = n + 1
        new_adj = [set(s) for s in adj]
        new_adj.append(set())
        for u in S:
            new_adj[u].add(new_n - 1)
            new_adj[new_n - 1].add(u)
            
        new_conflict = [set(s) for s in conflict]
        new_conflict.append(set())
        
        # Update conflict for new vertex
        for u in range(n):
            if new_adj[u] & new_adj[new_n - 1]:
                new_conflict[u].add(new_n - 1)
                new_conflict[new_n - 1].add(u)
                
        # Update conflicts within S
        # Pairs in S now have common neighbor new_n-1, so they conflict.
        for u in S:
            for w in S:
                if u != w:
                    new_conflict[u].add(w)
                    new_conflict[w].add(u)
                    
        return True, new_adj, new_conflict, new_n

    # Main Loop
    while time.time() < deadline:
        added = False
        # Try to find a valid S
        # Strategy: Random independent sets in conflict graph
        # To bias towards "good" S, we can try to pick S that hits independent sets of size 10.
        # But for now, let's just try many random ones.
        
        attempts = 0
        max_attempts = 200
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            # Generate random independent set in conflict graph
            S = set()
            available = set(range(n))
            avail_list = list(available)
            random.shuffle(avail_list)
            
            for u in avail_list:
                if u in available:
                    # Check if u conflicts with any in S
                    ok = True
                    for w in S:
                        if w in conflict[u]:
                            ok = False
                            break
                    if ok:
                        S.add(u)
                        available.remove(u)
                        for w in conflict[u]:
                            if w in available:
                                available.remove(w)
            
            if not S:
                continue
                
            # Try to add vertex with neighborhood S
            success, new_adj, new_conflict, new_n = try_add_vertex(adj, conflict, n, S)
            if success:
                adj = new_adj
                conflict = new_conflict
                n = new_n
                if n > best_n:
                    best_n = n
                    best_edges = []
                    for u in range(n):
                        for w in adj[u]:
                            if u < w:
                                best_edges.append([u, w])
                added = True
                break
                
        if not added:
            # If we can't add with random S, try smaller S? Or different strategy?
            # If we fail to add, we might be stuck.
            # Let's try S of size 1.
            found_single = False
            for _ in range(20):
                u = random.randint(0, n - 1)
                S = {u}
                success, new_adj, new_conflict, new_n = try_add_vertex(adj, conflict, n, S)
                if success:
                    adj = new_adj
                    conflict = new_conflict
                    n = new_n
                    if n > best_n:
                        best_n = n
                        best_edges = []
                        for x in range(n):
                            for y in adj[x]:
                                if x < y:
                                    best_edges.append([x, y])
                    found_single = True
                    break
            if not found_single:
                break # Can't add more
                
    return {'n': best_n, 'edges': best_edges}
