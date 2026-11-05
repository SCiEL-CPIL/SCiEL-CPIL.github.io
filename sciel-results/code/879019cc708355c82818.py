import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s buffer for 120s limit
    
    # Construct PG(2,5) Polarity Graph
    q = 5
    n0 = q * q + q + 1  # 31
    
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    
    n = len(points)
    
    def is_orthogonal(p, qpt):
        x1, y1, z1 = p
        x2, y2, z2 = qpt
        return (x1 * y2 + y1 * x2 + z1 * z2) % q == 0
    
    # Build adjacency using bitmasks for speed
    adj_mask = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if is_orthogonal(points[i], points[j]):
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    
    # Conflict graph: conflict_mask[i] has bit j set if N(i) & N(j) != empty
    def build_conflict_mask(adj_mask, n):
        conflict_mask = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_mask[i] & adj_mask[j]) != 0:
                    conflict_mask[i] |= (1 << j)
                    conflict_mask[j] |= (1 << i)
        return conflict_mask
    
    conflict_mask = build_conflict_mask(adj_mask, n)
    
    # Fast independent set size k checker using bitmask DFS
    def has_independent_set_size_k(adj_mask, n, k):
        if k > n:
            return False
        if k <= 1:
            return n >= k
        
        # Use iterative DFS with bitmasks
        # remaining: bitmask of available vertices
        # We try to find k non-adjacent vertices
        
        def dfs(remaining, count):
            if count == k:
                return True
            if remaining == 0:
                return False
            if count + popcount(remaining) < k:
                return False
            
            # Pick a vertex - use lowest set bit for determinism
            v_bit = remaining & (-remaining)
            v = v_bit.bit_length() - 1
            
            # Option 1: Include v
            # Remove v and its neighbors from remaining
            new_remaining_include = remaining & ~(v_bit | adj_mask[v])
            if dfs(new_remaining_include, count + 1):
                return True
            
            # Option 2: Exclude v
            new_remaining_exclude = remaining & ~v_bit
            if dfs(new_remaining_exclude, count):
                return True
            
            return False
        
        return dfs((1 << n) - 1, 0)
    
    def popcount(x):
        return bin(x).count('1')
    
    # Try to add a vertex connected to set S
    def try_add_vertex(adj_mask, conflict_mask, n, S_mask):
        # S_mask: bitmask of vertices in S
        if S_mask == 0:
            return None
        
        # Check 1: S is independent in conflict graph
        # For each pair u,v in S, conflict_mask[u] should not have v
        s_list = []
        temp = S_mask
        while temp:
            b = temp & (-temp)
            v = b.bit_length() - 1
            s_list.append(v)
            temp ^= b
        
        for i, u in enumerate(s_list):
            for j, v in enumerate(s_list):
                if i != j and (conflict_mask[u] >> v) & 1:
                    return None
        
        # Check 2: In the new graph, no C4 is formed
        # C4 check: For any two vertices a,b in S, they should not have a common neighbor
        # This is already checked by the conflict graph independence
        
        # Also need to check that no vertex in S has degree > 1 within S
        # Actually, the conflict graph check handles the C4 constraint properly
        
        # Check 3: Alpha constraint
        # New graph has n+1 vertices. New vertex v_new connected to S.
        # We need to check if there's an independent set of size 11 in the new graph.
        # Independent sets in new graph are either:
        #   a) Independent sets in old graph (size <= 10, since old graph has alpha <= 10)
        #   b) Sets containing v_new: {v_new} U T where T is independent in old graph 
        #      and T is disjoint from S (since v_new is adjacent to all in S)
        # So we need: alpha(old graph induced by V\S) <= 9
        
        # Build adj_mask for subgraph induced by V\S
        # Vertices in V\S: all vertices except those in S
        # We need to check if this subgraph has an independent set of size 10
        
        # Create a mapping from old indices to new subgraph indices
        # But for simplicity, let's just check directly
        
        # The subgraph on V\S has vertices {0,...,n-1} \ S
        # We can create a new adj_mask for this subgraph
        # But that's expensive. Instead, let's use a different approach.
        
        # Alternative: Check if there exists an independent set of size 10 in V\S
        # We can do this by modifying the adj_mask to "remove" S vertices
        
        # Create a modified adj_mask where S vertices are "removed"
        # For the purpose of independent set checking, we can just mask out S
        
        # Let's create a function that checks independent set in a subset of vertices
        def has_independent_set_in_subset(adj_mask, n, subset_mask, k):
            """Check if there's an independent set of size k within subset_mask vertices"""
            if k > popcount(subset_mask):
                return False
            if k <= 1:
                return popcount(subset_mask) >= k
            
            def dfs(remaining, count):
                if count == k:
                    return True
                if remaining == 0:
                    return False
                if count + popcount(remaining) < k:
                    return False
                
                v_bit = remaining & (-remaining)
                v = v_bit.bit_length() - 1
                
                # Include v
                new_remaining_include = remaining & ~(v_bit | (adj_mask[v] & subset_mask))
                if dfs(new_remaining_include, count + 1):
                    return True
                
                # Exclude v
                new_remaining_exclude = remaining & ~v_bit
                if dfs(new_remaining_exclude, count):
                    return True
                
                return False
            
            return dfs(subset_mask, 0)
        
        # V\S = all vertices except S
        complement_mask = ((1 << n) - 1) & ~S_mask
        if has_independent_set_in_subset(adj_mask, n, complement_mask, 10):
            return None  # Invalid: would create independent set of size 11
        
        # Valid! Construct new adj_mask
        new_adj_mask = adj_mask[:]  # Copy
        new_n = n + 1
        new_adj_mask.append(S_mask)  # New vertex's neighbors are S
        
        # Update neighbors in S
        for v in s_list:
            new_adj_mask[v] |= (1 << new_n)
        
        # Update conflict_mask
        new_conflict_mask = conflict_mask[:]
        new_conflict_mask.append(0)
        
        # Pairs in S become conflicts
        for i, u in enumerate(s_list):
            for j, v in enumerate(s_list):
                if i != j:
                    new_conflict_mask[u] |= (1 << v)
                    new_conflict_mask[v] |= (1 << u)
        
        # Conflicts involving new vertex
        # New vertex v_new has neighbors S
        # For each u in 0..n-1:
        #   N_new(u) = N_old(u) U {v_new} if u in S, else N_old(u)
        #   Common neighbors of v_new and u:
        #   N(v_new) = S
        #   N(u) as above
        #   Intersection = S & N(u)
        #   If non-empty, conflict
        
        for u in range(n):
            common = S_mask & new_adj_mask[u]
            if common != 0:
                new_conflict_mask[u] |= (1 << new_n)
                new_conflict_mask[new_n] |= (1 << u)
        
        return new_adj_mask, new_conflict_mask, new_n
    
    # Main search loop
    current_adj_mask = adj_mask
    current_conflict_mask = conflict_mask
    current_n = n
    
    best_n = current_n
    best_adj_mask = current_adj_mask[:]
    
    # Precompute some useful values
    all_mask = (1 << current_n) - 1
    
    # Try to add vertices
    max_attempts = 200
    attempt = 0
    
    while current_n < 43 and attempt < max_attempts and time.time() < deadline - 5:
        attempt += 1
        added = False
        
        # Try different strategies for choosing S
        # Strategy 1: Random independent sets in conflict graph
        for _ in range(30):
            if time.time() > deadline - 5:
                break
            
            # Generate a random independent set in conflict graph
            S_mask = 0
            available = all_mask
            avail_list = []
            temp = available
            while temp:
                b = temp & (-temp)
                v = b.bit_length() - 1
                avail_list.append(v)
                temp ^= b
            random.shuffle(avail_list)
            
            for u in avail_list:
                if (available >> u) & 1:
                    # Check if u conflicts with any in S
                    ok = True
                    temp_s = S_mask
                    while temp_s:
                        b = temp_s & (-temp_s)
                        w = b.bit_length() - 1
                        if (current_conflict_mask[u] >> w) & 1:
                            ok = False
                            break
                        temp_s ^= b
                    if ok:
                        S_mask |= (1 << u)
                        available &= ~(1 << u)
                        # Remove conflicts of u from available
                        available &= ~current_conflict_mask[u]
            
            if S_mask == 0:
                continue
            
            result = try_add_vertex(current_adj_mask, current_conflict_mask, current_n, S_mask)
            if result:
                current_adj_mask, current_conflict_mask, current_n = result
                all_mask = (1 << current_n) - 1
                best_n = current_n
                best_adj_mask = current_adj_mask[:]
                added = True
                break
        
        if not added:
            # Try smaller S sets
            for _ in range(20):
                if time.time() > deadline - 5:
                    break
                
                # Try S of size 1 or 2
                s_size = random.randint(1, 2)
                if s_size > current_n:
                    s_size = current_n
                
                # Pick random vertices
                vertices = list(range(current_n))
                random.shuffle(vertices)
                S_list = vertices[:s_size]
                S_mask = 0
                for v in S_list:
                    S_mask |= (1 << v)
                
                result = try_add_vertex(current_adj_mask, current_conflict_mask, current_n, S_mask)
                if result:
                    current_adj_mask, current_conflict_mask, current_n = result
                    all_mask = (1 << current_n) - 1
                    best_n = current_n
                    best_adj_mask = current_adj_mask[:]
                    added = True
                    break
        
        if not added:
            # Try even more aggressively
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Try random S of various sizes
                s_size = random.randint(1, min(5, current_n))
                vertices = list(range(current_n))
                random.shuffle(vertices)
                S_list = vertices[:s_size]
                S_mask = 0
                for v in S_list:
                    S_mask |= (1 << v)
                
                result = try_add_vertex(current_adj_mask, current_conflict_mask, current_n, S_mask)
                if result:
                    current_adj_mask, current_conflict_mask, current_n = result
                    all_mask = (1 << current_n) - 1
                    best_n = current_n
                    best_adj_mask = current_adj_mask[:]
                    added = True
                    break
        
        if not added and attempt > 50:
            # Give up if we've tried many times without success
            if random.random() < 0.1:  # 10% chance to continue
                continue
            else:
                break
    
    # Build edge list from best_adj_mask
    edges = []
    for i in range(best_n):
        mask = best_adj_mask[i]
        temp = mask
        while temp:
            b = temp & (-temp)
            j = b.bit_length() - 1
            if i < j:
                edges.append([i, j])
            temp ^= b
    
    return {'n': best_n, 'edges': edges}
