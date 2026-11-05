import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 100.0  # Keep 20s buffer for 120s limit
    
    # 1. Construct Polarity Graph of PG(2,5)
    # q = 5
    q = 5
    n0 = q * q + q + 1
    # Vertices are points of PG(2,q)
    # We can represent them as (x, y, z) with z=1, plus (x, y, 0) for line at infinity?
    # Standard representation:
    # Points: (1, x, y) for x,y in F_q, and (0, 1, x) for x in F_q, and (0, 0, 1) ... wait.
    # Easier: Use indices 0..n0-1.
    # Polarity: Two points are orthogonal if they satisfy the quadratic form.
    # For q=5, we can just generate the edges by checking the condition.
    
    # Let's use a known construction for PG(2,q) polarity graph.
    # Vertices: (x, y) in F_q x F_q, plus (x) in F_q (infinity), plus (infinity). Total q^2 + q + 1.
    # Let's map:
    # 0..q^2-1: (x, y)
    # q^2..q^2+q-1: (x, inf) -> represented as x
    # q^2+q: (inf, inf)
    
    def get_vertex_coords(idx):
        if idx < q * q:
            return (idx // q, idx % q, 1)
        elif idx < q * q + q:
            return (idx - q * q, 1, 0) # Line at infinity direction? 
            # Actually, standard homogeneous coords for PG(2,q):
            # (x, y, z) not all 0.
            # Let's just use the property: 
            # The graph is strongly regular or similar?
            # No, it's just C4-free.
            
    # Simpler: Just generate the edges for PG(2,5) polarity graph directly.
    # Reference: The polarity graph of PG(2,q) has vertices as points of PG(2,q).
    # Two points P, Q are adjacent if P is on the polar line of Q.
    
    # Let's implement PG(2,q) arithmetic.
    # Points are non-zero vectors in F_q^3 up to scalar.
    # We can fix the last non-zero coordinate to 1.
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    
    n = len(points)
    # n should be 31 for q=5
    
    def is_orthogonal(p, qpt):
        # p . qpt == 0 in F_q
        # Standard dot product? 
        # For PG(2,q), the polarity is given by x1*x1' + x2*x2' + x3*x3' = 0 ?
        # No, that's for hyperbolic quadric.
        # The standard polarity in PG(2,q) is given by the matrix:
        # 0 1 0
        # 1 0 0
        # 0 0 1
        # Wait, that makes (1,0,0) orthogonal to (0,1,0) and (0,0,1).
        # Let's use the standard symmetric bilinear form:
        # <(x1,y1,z1), (x2,y2,z2)> = x1*y2 + y1*x2 + z1*z2 ?
        # If q is odd, this works.
        # Let's check self-orthogonality.
        # <p, p> = 2*x*y + z*z.
        # If 2*x*y + z^2 = 0, p is on its polar line.
        
        x1, y1, z1 = p
        x2, y2, z2 = qpt
        return (x1 * y2 + y1 * x2 + z1 * z2) % q == 0

    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if is_orthogonal(points[i], points[j]):
                adj[i].add(j)
                adj[j].add(i)
                
    # Check C4 freeness
    # In PG(2,q) polarity graph, any two vertices have at most 1 common neighbor.
    # This is a known property.
    
    # Check independence number
    # alpha is q+1 = 6.
    
    # Now, expand
    # State:
    # adj: list of sets
    # conflict: list of sets. conflict[i] contains j if adj[i] & adj[j] is not empty.
    
    conflict = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & adj[j]:
                conflict[i].add(j)
                conflict[j].add(i)
                
    def has_independent_set_size_k(adj, k):
        # Returns True if there is an independent set of size k
        # Simple backtracking with pruning
        # For n <= 40, k=11, this might be slow if not careful.
        # But we only call this if we think it's close.
        
        n_v = len(adj)
        # Greedy upper bound:
        # If we can't find k vertices even greedily, return False?
        # No, greedy finds a set, doesn't prove non-existence.
        
        # Use a simple recursive search
        # Order vertices by degree?
        
        # Optimization: If k > n_v, return False
        if k > n_v:
            return False
            
        # If k == 1, return True (assuming n_v >= 1)
        if k == 1:
            return n_v >= 1
            
        # Try to find independent set of size k
        # We can use a bitset representation for speed?
        # For n=40, int is enough.
        
        # Let's use a simple DFS
        # This might be too slow for 120s if called many times.
        # We will only call it when we are about to add a vertex.
        
        # Better: Check if alpha(adj) >= k.
        # We can compute a lower bound on alpha.
        # But we need to know if alpha < 11.
        
        # Let's implement a fast alpha calculator for small graphs.
        # Branch and bound.
        
        # To speed up, we can use the fact that we are adding one vertex.
        # But let's just write a generic one.
        
        # If this is too slow, we will fall back to a heuristic.
        
        # Heuristic:
        # 1. Try to find an independent set of size k using greedy + local search.
        # 2. If found, return True.
        # 3. If not, assume False (risk of false negative).
        # Given the small n, a proper solver is better.
        
        # Let's use a simple recursive function with memoization? No, subsets are too many.
        # Just simple backtracking.
        
        # To make it fast, we can prune if the maximum possible independent set in the remaining graph + current_size < k.
        # Max independent set in remaining graph <= number of remaining vertices.
        
        def dfs(remaining, current_size):
            if current_size == k:
                return True
            if not remaining:
                return False
            if current_size + len(remaining) < k:
                return False
                
            # Pick a vertex
            # To prune better, pick a vertex with high degree in remaining?
            # Or just pick the first one.
            
            # Let's pick a vertex v from remaining
            # We must decide: include v or not?
            # If we include v, we remove v and its neighbors.
            # If we don't include v, we remove v.
            
            # Optimization: If v is isolated in remaining, we MUST include it to maximize size?
            # No, but if we don't include it, we lose 1 potential vertex.
            
            # Let's just pick the first vertex in remaining.
            v = next(iter(remaining))
            
            # Option 1: Include v
            new_remaining_include = set()
            for u in remaining:
                if u != v and u not in adj[v]:
                    new_remaining_include.add(u)
            if dfs(new_remaining_include, current_size + 1):
                return True
                
            # Option 2: Exclude v
            new_remaining_exclude = set(remaining)
            new_remaining_exclude.remove(v)
            if dfs(new_remaining_exclude, current_size):
                return True
                
            return False
            
        return dfs(set(range(n_v)), 0)

    # Helper to add a vertex
    def try_add_vertex(adj, conflict, S):
        # S is a set of neighbor indices
        # Check if S is independent in conflict
        for u in S:
            for w in S:
                if u != w and w in conflict[u]:
                    return False
        return True

    # Main loop
    best_n = n
    best_edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                best_edges.append([u, v])
                
    while time.time() < deadline:
        # Try to add a vertex
        added = False
        # Generate candidate S
        # Strategy: Try small S first? Or random S?
        # Let's try S of size 1 first (connect to one vertex).
        # Then size 2, etc.
        
        # To maximize n, we just need ANY valid S.
        # Let's try random independent sets in conflict.
        
        attempts = 0
        while attempts < 100 and time.time() < deadline:
            attempts += 1
            # Pick a random vertex to start S
            if not conflict:
                break
            # Find an independent set in conflict
            # Greedy:
            S = set()
            available = set(range(n))
            # Shuffle available for randomness
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
                        # Remove u and its conflicts from available
                        available.remove(u)
                        for w in conflict[u]:
                            if w in available:
                                available.remove(w)
            
            if not S:
                continue
                
            # Check if adding v with neighbors S is valid
            # 1. C4: S is independent in conflict (checked above)
            # 2. Alpha: 
            #    Construct new adj
            new_adj = [set(s) for s in adj]
            new_n = n + 1
            new_adj.append(set()) # v
            v_idx = n
            
            for u in S:
                new_adj[u].add(v_idx)
                new_adj[v_idx].add(u)
                
            # Check alpha(new_adj) <= 10
            # We know alpha(adj) <= 10 (from previous step)
            # We need alpha(new_adj) <= 10.
            # alpha(new_adj) = max(alpha(adj), 1 + alpha(adj \ S))
            # So we need 1 + alpha(adj \ S) <= 10 => alpha(adj \ S) <= 9.
            
            # Let's check if alpha(adj \ S) <= 9.
            # We can run the has_independent_set_size_k on the subgraph induced by V \ S.
            # But has_independent_set_size_k checks for size k.
            # We want to know if there is an independent set of size 10 in adj \ S.
            # If yes, reject.
            
            # Create subgraph adj_sub
            # Indices in adj_sub correspond to V \ S
            # This is expensive.
            
            # Alternative: Just check if the NEW graph has an independent set of size 11.
            # If it does, reject.
            
            if has_independent_set_size_k(new_adj, 11):
                # Invalid
                continue
            else:
                # Valid
                # Update state
                adj = new_adj
                n = new_n
                
                # Update conflict
                new_conflict = [set(s) for s in conflict]
                new_conflict.append(set())
                # Pairs in S become conflicts
                for u in S:
                    for w in S:
                        if u != w:
                            new_conflict[u].add(w)
                            new_conflict[w].add(u)
                # Conflicts with v?
                # v is connected to S.
                # For u in S, w not in S:
                # N_new(u) = N_old(u) U {v}
                # N_new(w) = N_old(w)
                # Intersection is N_old(u) n N_old(w).
                # So no new conflicts between S and V\S.
                # For u, w in V\S:
                # No change.
                # For v and u:
                # N(v) = S.
                # N(u) = N_old(u) U {v} (if u in S) or N_old(u) (if u not in S).
                # If u in S: N(v) n N(u) = S n (N_old(u) U {v}).
                # v is not in N_old(u). So S n N_old(u).
                # If there is w in S such that w in N_old(u), then v and u conflict?
                # Yes, if u and w are neighbors in old graph, and both in S, then v-u-w-v is a triangle? No, v-u-w-v is not a cycle.
                # Conflict means common neighbor.
                # Common neighbors of v and u:
                # N(v) = S.
                # N(u) includes v.
                # Intersection: S n (N_old(u) U {v}) = (S n N_old(u)) U (S n {v}).
                # S n {v} is empty.
                # So intersection is S n N_old(u).
                # If this is non-empty, v and u conflict.
                
                # Let's just recompute conflict for the new vertex v and update existing ones.
                # It's safer to recompute conflict for all pairs involving new vertices or changed neighborhoods.
                # But S is the only change.
                
                # Let's just update conflict for pairs involving v.
                for u in range(n - 1):
                    # Check if u and v have common neighbor
                    # N(v) = S
                    # N(u) in new_adj
                    common = new_adj[u] & new_adj[v]
                    if common:
                        new_conflict[u].add(v)
                        new_conflict[v].add(u)
                        
                # Update conflicts within S
                # We already added them.
                
                conflict = new_conflict
                
                # Update best
                if n > best_n:
                    best_n = n
                    best_edges = []
                    for u in range(n):
                        for w in adj[u]:
                            if u < w:
                                best_edges.append([u, w])
                added = True
                break # Break inner loop, go to next outer loop iteration
                
        if not added:
            # Try a different strategy?
            # Maybe S of size 1 always?
            # Let's try S = {random_vertex}
            found_single = False
            for _ in range(10):
                u = random.randint(0, n - 1)
                S = {u}
                # Check C4: Single vertex is always independent in conflict.
                # Check Alpha
                new_adj = [set(s) for s in adj]
                new_adj.append(set())
                v_idx = n
                new_adj[u].add(v_idx)
                new_adj[v_idx].add(u)
                
                if not has_independent_set_size_k(new_adj, 11):
                    adj = new_adj
                    n = new_n
                    new_conflict = [set(s) for s in conflict]
                    new_conflict.append(set())
                    # Update conflict for v
                    for x in range(n - 1):
                        if new_adj[x] & new_adj[v_idx]:
                            new_conflict[x].add(v_idx)
                            new_conflict[v_idx].add(x)
                    conflict = new_conflict
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
