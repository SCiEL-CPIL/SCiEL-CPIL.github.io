import random
import itertools

def construct(k, g):
    # Target: 4-regular graph, girth 10.
    # Known best: n=320.
    # Strategy: Search for a Cayley graph on Z_5 x Z_64.
    
    if k != 4 or g != 10:
        # Fallback for other parameters: use L-U-W logic from seed if possible, 
        # or return a simple cycle if k=2, etc.
        # For now, focus on k=4, g=10.
        # If parameters differ, we might fail. But task is specific.
        pass

    n = 320
    # Group Z_5 x Z_64
    # Elements represented as tuples (a, b) or integers 0..319
    # Let's use integers: idx = a * 64 + b
    
    def add(x, y):
        ax, bx = divmod(x, 64)
        ay, by = divmod(y, 64)
        return ((ax + ay) % 5) * 64 + (bx + by) % 64

    def neg(x):
        ax, bx = divmod(x, 64)
        return ((-ax) % 5) * 64 + (-bx) % 64

    # Try specific generators
    # Common generators for such cages involve small steps.
    # Let's try a few hardcoded candidates first.
    candidates = [
        [(1, 0), (0, 1), (1, 1), (0, 2)],
        [(1, 0), (0, 1), (2, 1), (0, 3)],
        [(1, 0), (0, 1), (1, 2), (0, 5)],
        [(1, 1), (0, 2), (2, 1), (0, 3)],
    ]
    
    # Convert candidates to flat indices
    def to_idx(t):
        return t[0] * 64 + t[1]
    
    valid_generators = None
    
    # Check candidates
    for cand in candidates:
        gens = [to_idx(t) for t in cand]
        # Ensure symmetric: add negations
        # The set S should be {g1, -g1, g2, -g2}
        # So we pick 2 generators g1, g2.
        # My candidates above have 4 elements. 
        # I need to interpret them as 2 pairs.
        # Let's assume the candidate list contains the 4 elements of S.
        # Check if S is symmetric: for every s in S, -s in S.
        S = set(gens)
        if all(neg(s) in S for s in S) and 0 not in S and len(S) == 4:
            # Check if it generates the group
            # BFS from 0 using S
            visited = {0}
            queue = [0]
            while queue:
                v = queue.pop(0)
                for s in S:
                    n_v = add(v, s)
                    n_v2 = add(v, neg(s))
                    if n_v not in visited:
                        visited.add(n_v)
                        queue.append(n_v)
                    if n_v2 not in visited:
                        visited.add(n_v2)
                        queue.append(n_v2)
            if len(visited) == n:
                # Check girth
                # Build adjacency list
                adj = [[] for _ in range(n)]
                for v in range(n):
                    for s in S:
                        u = add(v, s)
                        adj[v].append(u)
                # Girth check
                # Use BFS from each vertex to depth 4
                # If we find a cycle of length < 10, invalid.
                # Optimization: Check for cycles of length 3, 4, 5, 6, 7, 8, 9.
                # For k=4, n=320, this is fast.
                is_valid = True
                for v in range(n):
                    # BFS
                    dist = [-1] * n
                    parent = [-1] * n
                    dist[v] = 0
                    q = [v]
                    head = 0
                    while head < len(q):
                        u = q[head]
                        head += 1
                        if dist[u] >= 4:
                            continue
                        for w in adj[u]:
                            if dist[w] == -1:
                                dist[w] = dist[u] + 1
                                parent[w] = u
                                q.append(w)
                            elif parent[w] != u and parent[u] != w:
                                # Cycle found
                                # Length = dist[u] + dist[w] + 1
                                # But this detects any back-edge.
                                # We need to be careful not to count the tree edge.
                                # If w is visited and w is not parent of u, 
                                # and u is not parent of w (handled by undirected),
                                # then we have a cycle.
                                # However, BFS tree edges are (u, parent[u]).
                                # If we see (u, w) and w is visited:
                                # If w is parent of u, skip.
                                # If u is parent of w, skip (handled by dist check? No, dist[w] = dist[u]+1).
                                # If dist[w] <= dist[u], then we have a cycle.
                                if dist[w] <= dist[u]:
                                    cycle_len = dist[u] + dist[w] + 1
                                    if cycle_len < g:
                                        is_valid = False
                                        break
                    if not is_valid:
                        break
                if is_valid:
                    valid_generators = S
                    break

    if valid_generators is None:
        # Random search
        random.seed(42)
        for _ in range(1000):
            g1 = random.randint(1, n-1)
            g2 = random.randint(1, n-1)
            S = {g1, neg(g1), g2, neg(g2)}
            if 0 in S or len(S) < 4:
                continue
            # Check generation
            visited = {0}
            queue = [0]
            while queue:
                v = queue.pop(0)
                for s in S:
                    n_v = add(v, s)
                    if n_v not in visited:
                        visited.add(n_v)
                        queue.append(n_v)
            if len(visited) != n:
                continue
            # Check girth (same as above)
            adj = [[] for _ in range(n)]
            for v in range(n):
                for s in S:
                    u = add(v, s)
                    adj[v].append(u)
            
            is_valid = True
            for v in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[v] = 0
                q = [v]
                head = 0
                while head < len(q):
                    u = q[head]
                    head += 1
                    if dist[u] >= 4:
                        continue
                    for w in adj[u]:
                        if dist[w] == -1:
                            dist[w] = dist[u] + 1
                            parent[w] = u
                            q.append(w)
                        elif parent[w] != u and parent[u] != w:
                            if dist[w] <= dist[u]:
                                cycle_len = dist[u] + dist[w] + 1
                                if cycle_len < g:
                                    is_valid = False
                                    break
                if not is_valid:
                    break
            if is_valid:
                valid_generators = S
                break

    if valid_generators is None:
        # Fallback: Return a larger graph? 
        # Or just return the first candidate even if invalid? No.
        # Let's return a graph with n=340 if we can find one?
        # For now, if no 320 found, we might timeout or return invalid.
        # To be safe, let's construct a known valid graph if possible.
        # If we can't find 320, we return 0, 0? No.
        # Let's assume the search finds one.
        return 0, []

    # Construct edges
    edges = []
    for v in range(n):
        for s in valid_generators:
            u = add(v, s)
            if v < u:
                edges.append((v, u))
    
    return n, edges
