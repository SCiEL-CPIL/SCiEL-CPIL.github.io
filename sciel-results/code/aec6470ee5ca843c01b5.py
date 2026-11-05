import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for 150s limit
    
    # 1. Construct PG(2,5) Polarity Graph
    q = 5
    n0 = q * q + q + 1 # 31
    
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
        # Bilinear form for PG(2,5) polarity
        # x1*y2 + y1*x2 + z1*z2 = 0
        return (x1 * y2 + y1 * x2 + z1 * z2) % q == 0

    # Adjacency list
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if is_orthogonal(points[i], points[j]):
                adj[i].add(j)
                adj[j].add(i)
                
    # Convert to adjacency matrix for faster ops if needed, but sets are good for now
    # We need to maintain the Conflict Graph H
    # H[u] contains v if N(u) & N(v) is not empty
    
    def build_conflict(adj):
        n_v = len(adj)
        conflict = [set() for _ in range(n_v)]
        for i in range(n_v):
            for j in range(i + 1, n_v):
                if adj[i] & adj[j]:
                    conflict[i].add(j)
                    conflict[j].add(i)
        return conflict

    conflict = build_conflict(adj)
    
    def has_independent_set_size_k(adj, k):
        """
        Returns True if there is an independent set of size k in the graph defined by adj (list of sets).
        Uses simple backtracking with pruning.
        """
        n_v = len(adj)
        if k > n_v:
            return False
        if k == 0:
            return True
        if k == 1:
            return n_v >= 1
            
        # Order vertices by degree descending to prune earlier?
        # Actually, for independent set, we want to pick vertices with low degree?
        # Or just use a simple DFS.
        
        # State: remaining vertices, current size
        # To speed up, we can use a bitset if n_v is small, but n_v <= 39.
        
        def dfs(remaining, current_size):
            if current_size == k:
                return True
            if not remaining:
                return False
            if current_size + len(remaining) < k:
                return False
                
            # Pick a vertex
            # Heuristic: Pick vertex with min degree in remaining?
            # Or just pick the first one.
            # Let's pick the first one for simplicity, but maybe sort?
            # Sorting at every step is expensive.
            # Let's just pick next(iter(remaining))
            
            v = next(iter(remaining))
            
            # Option 1: Include v
            # New remaining: remaining - {v} - N(v)
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

    def try_add_vertex(adj, conflict):
        """
        Tries to add a new vertex to the graph.
        Returns (new_adj, new_conflict) if successful, else None.
        """
        n_v = len(adj)
        # Try several random S
        for _ in range(50):
            # Size of S: 2 to 4
            s_size = random.randint(2, 4)
            if s_size > n_v:
                s_size = n_v
            S = set(random.sample(range(n_v), s_size))
            
            # Check 1: S is independent in conflict graph
            valid = True
            for u in S:
                for v in S:
                    if u != v and v in conflict[u]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
                
            # Check 2: G[S] has max degree <= 1
            # i.e., no vertex in S has >1 neighbor in S
            for u in S:
                deg_in_S = sum(1 for v in S if v != u and v in adj[u])
                if deg_in_S > 1:
                    valid = False
                    break
            if not valid:
                continue
                
            # Tentatively add vertex n_v
            new_adj = [set(s) for s in adj]
            new_adj.append(set(S)) # New vertex's neighbors are S
            
            # Update neighbors in old vertices
            for u in S:
                new_adj[u].add(n_v)
                
            # Check 3: Alpha < 11
            # We need to check if new_adj has independent set of size 11
            # This is the expensive part
            if has_independent_set_size_k(new_adj, 11):
                continue
            else:
                # Valid!
                # Update conflict graph
                new_conflict = build_conflict(new_adj)
                return new_adj, new_conflict
                
        return None

    # Start with PG(2,5)
    current_adj = adj
    current_conflict = conflict
    current_n = n
    
    # Try to add 8 vertices
    target_n = 39
    for _ in range(target_n - current_n):
        if time.time() > deadline - 10:
            break
            
        result = try_add_vertex(current_adj, current_conflict)
        if result:
            current_adj, current_conflict = result
            current_n += 1
        else:
            # If we can't add a vertex, try a few more times with different randomness?
            # Or just break?
            # Let's try 5 more times
            added = False
            for _ in range(5):
                if time.time() > deadline - 10:
                    break
                result = try_add_vertex(current_adj, current_conflict)
                if result:
                    current_adj, current_conflict = result
                    current_n += 1
                    added = True
                    break
            if not added:
                break
                
    # Final check: Ensure alpha < 11
    # The loop ensures this, but let's be safe.
    # If the last addition failed the alpha check, we wouldn't have added it.
    # So current_adj should be valid.
    
    # Convert to edge list
    edges = []
    for i in range(current_n):
        for j in current_adj[i]:
            if i < j:
                edges.append([i, j])
                
    return {'n': current_n, 'edges': edges}

construct()
