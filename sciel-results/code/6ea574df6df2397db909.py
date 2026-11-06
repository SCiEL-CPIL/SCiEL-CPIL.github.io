import random
import time
import sys

def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin
    
    def is_valid(adj, n):
        # Check K4
        for i in range(n):
            for j in range(i+1, n):
                if not (adj[i] & (1 << j)):
                    continue
                # Common neighbors of i and j
                common = adj[i] & adj[j]
                # Remove i and j
                common &= ~(1 << i)
                common &= ~(1 << j)
                # Check if any pair in common is connected
                # Iterate through bits in common
                c = common
                while c:
                    # Get lowest set bit
                    b = c & -c
                    u = b.bit_length() - 1
                    c -= b
                    # Check if u is connected to any other bit in common
                    # (common & adj[u]) & ~b & ~u
                    rest = (common & adj[u]) & ~b
                    if rest:
                        return False # Found K4
        # Check I6
        # Independent set of size 6.
        # This is harder to check incrementally in a simple loop without precomputation.
        # We can check for I6 by looking for 6 vertices with no edges between them.
        # Brute force is O(n^6). For n=35, that's 1.6M.
        # We can optimize by checking if the complement graph has a K6.
        # Let's just do a optimized check.
        # If n is small (< 30), brute force is fine.
        # For n=35, we might need to be careful.
        # Let's use a simple backtracking check for I6.
        
        def has_i6():
            # Try to build an independent set of size 6
            # Use a simple DFS
            stack = [(0, 0, 0, 0, 0, 0)] # (idx, v1, v2, v3, v4, v5, v6) - actually just track set
            # Better: recursive function
            def dfs(start, current_set, size):
                if size == 6:
                    return True
                if start >= n:
                    return False
                # Option 1: Don't include start
                if dfs(start + 1, current_set, size):
                    return True
                # Option 2: Include start if it's independent of current_set
                # Check if start is connected to any in current_set
                ok = True
                for v in current_set:
                    if adj[start] & (1 << v):
                        ok = False
                        break
                if ok:
                    if dfs(start + 1, current_set + [start], size + 1):
                        return True
                return False
            
            return dfs(0, [], 0)
        
        return not has_i6()

    def find_graph(n):
        # Initialize random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Repair loop
        while time.time() - start_time < time_limit:
            # Check validity
            if is_valid(adj, n):
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
            
            # Find a conflict
            # 1. Find a K4
            k4_found = False
            for i in range(n):
                for j in range(i+1, n):
                    if not (adj[i] & (1 << j)):
                        continue
                    common = adj[i] & adj[j] & ~(1 << i) & ~(1 << j)
                    c = common
                    while c:
                        b = c & -c
                        u = b.bit_length() - 1
                        c -= b
                        rest = (common & adj[u]) & ~b
                        if rest:
                            # Found K4: {i, j, u, v} where v is in rest
                            b2 = rest & -rest
                            v = b2.bit_length() - 1
                            # Remove a random edge from this K4
                            # Pick random edge among (i,j), (i,u), (i,v), (j,u), (j,v), (u,v)
                            edges_to_remove = [(i,j), (i,u), (i,v), (j,u), (j,v), (u,v)]
                            e = random.choice(edges_to_remove)
                            adj[e[0]] &= ~(1 << e[1])
                            adj[e[1]] &= ~(1 << e[0])
                            k4_found = True
                            break
                        if k4_found: break
                    if k4_found: break
                if k4_found: break
            if k4_found:
                continue
            
            # 2. Find an I6
            # We need to find 6 vertices with no edges.
            # We can use a similar search to `has_i6` but return the set.
            def find_i6():
                def dfs(start, current_set, size):
                    if size == 6:
                        return current_set
                    if start >= n:
                        return None
                    # Try not including start
                    res = dfs(start + 1, current_set, size)
                    if res:
                        return res
                    # Try including start
                    ok = True
                    for v in current_set:
                        if adj[start] & (1 << v):
                            ok = False
                            break
                    if ok:
                        res = dfs(start + 1, current_set + [start], size + 1)
                        if res:
                            return res
                    return None
                return dfs(0, [], 0)
            
            i6_set = find_i6()
            if i6_set:
                # Add a random edge within this I6
                # Pick two random vertices from the set
                u, v = random.sample(i6_set, 2)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            else:
                # Should not happen if is_valid returned False
                break

        return None

    # Try n from 35 down to 20
    for n in range(35, 19, -1):
        result = find_graph(n)
        if result:
            return result
            
    # Fallback to a small valid graph (e.g., n=10)
    # Simple cycle or complete graph minus edges?
    # C_10 is K4-free? No, C_10 has no K4. Alpha(C_10) = 5. So C_10 is valid for n=10?
    # Wait, R(4,6) > 10.
    # Let's just return a random valid small graph.
    # A random graph with n=10, p=0.5 is likely valid.
    adj = [0] * 10
    for i in range(10):
        for j in range(i+1, 10):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    # Verify and repair if needed
    while not is_valid(adj, 10):
        # Just re-randomize for fallback
        adj = [0] * 10
        for i in range(10):
            for j in range(i+1, 10):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if time.time() - start_time > time_limit:
            break
            
    edges = []
    for i in range(10):
        for j in range(i+1, 10):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return {'n': 10, 'edges': edges}
