import random
import time

def construct(k, g):
    # If k or g are not 4 and 10, fallback to standard construction
    if k != 4 or g != 10:
        return _fallback_construct(k, g)
    
    deadline = time.time() + 105  # 105s limit to be safe
    
    # Attempt to construct the Moore graph (n=242)
    try:
        n, edges = _construct_moore_candidate(deadline)
        if n is not None:
            # Verify girth
            if _verify_girth(n, edges, g):
                return n, edges
    except Exception:
        pass
    
    # Fallback to original Lazebnik-Ustimenko-Woldar construction
    return _fallback_construct(k, g)

def _construct_moore_candidate(deadline):
    # Build two trees of depth 4, rooted at 0 and 1 (u and v)
    # u=0, v=1
    # Left tree children of 0: 2, 3, 4
    # Right tree children of 1: 121, 122, 123
    
    # 1. Construct Edges for Trees
    edges = []
    # Center edge
    edges.append((0, 1))
    
    # Left Tree
    l1 = [2, 3, 4]
    for node in l1: edges.append((0, node))
    
    l2 = []
    idx = 5
    for node in l1:
        for _ in range(3):
            edges.append((node, idx))
            l2.append(idx)
            idx += 1
            
    l3 = []
    for node in l2:
        for _ in range(3):
            edges.append((node, idx))
            l3.append(idx)
            idx += 1
            
    l4 = []
    for node in l3:
        for _ in range(3):
            edges.append((node, idx))
            l4.append(idx)
            idx += 1
            
    # Right Tree
    r1 = [121, 122, 123]
    for node in r1: edges.append((1, node))
    
    r2 = []
    idx = 124
    for node in r1:
        for _ in range(3):
            edges.append((node, idx))
            r2.append(idx)
            idx += 1
            
    r3 = []
    for node in r2:
        for _ in range(3):
            edges.append((node, idx))
            r3.append(idx)
            idx += 1
            
    r4 = []
    for node in r3:
        for _ in range(3):
            edges.append((node, idx))
            r4.append(idx)
            idx += 1
    
    # 2. Compute Distance Matrices using LCA
    # Build parent pointers and depth for left tree (rooted at 0)
    parents_l = {}
    depth_l = {0: 0}
    parents_l[2]=0; depth_l[2]=1
    parents_l[3]=0; depth_l[3]=1
    parents_l[4]=0; depth_l[4]=1
    for i, node in enumerate(l2):
        par = l1[i//3]
        parents_l[node] = par
        depth_l[node] = depth_l[par] + 1
    for i, node in enumerate(l3):
        par = l2[i//3]
        parents_l[node] = par
        depth_l[node] = depth_l[par] + 1
    for i, node in enumerate(l4):
        par = l3[i//3]
        parents_l[node] = par
        depth_l[node] = depth_l[par] + 1
        
    # Build parent pointers and depth for right tree (rooted at 1)
    parents_r = {}
    depth_r = {1: 0}
    parents_r[121]=1; depth_r[121]=1
    parents_r[122]=1; depth_r[122]=1
    parents_r[123]=1; depth_r[123]=1
    for i, node in enumerate(r2):
        par = r1[i//3]
        parents_r[node] = par
        depth_r[node] = depth_r[par] + 1
    for i, node in enumerate(r3):
        par = r2[i//3]
        parents_r[node] = par
        depth_r[node] = depth_r[par] + 1
    for i, node in enumerate(r4):
        par = r3[i//3]
        parents_r[node] = par
        depth_r[node] = depth_r[par] + 1
    
    # Efficient distance computation in a tree using LCA
    def tree_distance(u, v, parents, depth):
        # Find LCA of u and v
        # Mark all ancestors of u
        ancestors_u = set()
        cur = u
        while cur in parents:
            ancestors_u.add(cur)
            cur = parents[cur]
        # Root (0 for left, 1 for right) is not in parents, so loop stops there
        # Now walk up from v until we find a node in ancestors_u
        cur = v
        lca = v
        while cur not in ancestors_u:
            if cur not in parents:
                # Should not happen if both in same tree
                break
            cur = parents[cur]
        lca = cur
        # Distance = depth[u] + depth[v] - 2*depth[lca]
        return depth[u] + depth[v] - 2 * depth[lca]
    
    # Precompute D_L and D_R
    N = 81
    D_L = [[0]*N for _ in range(N)]
    D_R = [[0]*N for _ in range(N)]
    
    for i in range(N):
        for j in range(i+1, N):
            u = l4[i]
            v = l4[j]
            d = tree_distance(u, v, parents_l, depth_l)
            D_L[i][j] = d
            D_L[j][i] = d
            
            u = r4[i]
            v = r4[j]
            d = tree_distance(u, v, parents_r, depth_r)
            D_R[i][j] = d
            D_R[j][i] = d
            
    # 3. Find Permutation sigma
    # We want min_{i!=j} D_L[i][j] + D_R[sigma[i]][sigma[j]] >= 8
    
    # Local Search with better initialization and acceptance criteria
    best_sigma = None
    best_min_sum = -1
    
    for restart in range(100):
        if time.time() > deadline - 5:
            break
            
        # Start with identity permutation (often good for structured problems)
        if restart == 0:
            sigma = list(range(N))
        else:
            sigma = list(range(N))
            random.shuffle(sigma)
        
        # Compute current min sum
        def compute_min_sum(sigma):
            min_s = float('inf')
            for i in range(N):
                si = sigma[i]
                for j in range(i+1, N):
                    s = D_L[i][j] + D_R[si][sigma[j]]
                    if s < min_s:
                        min_s = s
            return min_s
        
        current_min = compute_min_sum(sigma)
        if current_min >= 8:
            if current_min > best_min_sum:
                best_min_sum = current_min
                best_sigma = sigma[:]
            return 242, edges + [(l4[i], r4[sigma[i]]) for i in range(N)]
        
        # Local search: try swaps that improve the minimum
        for step in range(500):
            if time.time() > deadline - 5:
                break
                
            # Find the worst pair
            min_sum = float('inf')
            min_i, min_j = 0, 1
            for i in range(N):
                si = sigma[i]
                for j in range(i+1, N):
                    s = D_L[i][j] + D_R[si][sigma[j]]
                    if s < min_sum:
                        min_sum = s
                        min_i, min_j = i, j
                        
            if min_sum >= 8:
                if min_sum > best_min_sum:
                    best_min_sum = min_sum
                    best_sigma = sigma[:]
                return 242, edges + [(l4[i], r4[sigma[i]]) for i in range(N)]
            
            # Try to fix the worst pair by swapping sigma[min_i] with sigma[k] for some k
            # Choose k to minimize the new worst pair
            best_k = -1
            best_new_min = min_sum - 1  # Must be worse than current to accept
            
            # Sample some k values to speed up
            candidates = [min_j] + random.sample(range(N), min(20, N-1))
            for k in candidates:
                if k == min_i:
                    continue
                # Try swap
                sigma[min_i], sigma[k] = sigma[k], sigma[min_i]
                
                # Recompute min (expensive, so only do for promising swaps)
                # Quick check: what's the new sum for the worst pair?
                new_worst = D_L[min_i][min_j] + D_R[sigma[min_i]][sigma[min_j]]
                
                # Also check if any pair involving min_i or k became worse
                # For simplicity, just recompute full min for this candidate
                new_min = float('inf')
                for i in range(N):
                    si = sigma[i]
                    for j in range(i+1, N):
                        s = D_L[i][j] + D_R[si][sigma[j]]
                        if s < new_min:
                            new_min = s
                            if new_min <= new_worst:
                                break
                    if new_min <= new_worst:
                        break
                
                if new_min > best_new_min:
                    best_new_min = new_min
                    best_k = k
                else:
                    # Revert
                    sigma[min_i], sigma[k] = sigma[k], sigma[min_i]
            
            if best_k >= 0 and best_new_min > current_min:
                # Apply the best swap
                sigma[min_i], sigma[best_k] = sigma[best_k], sigma[min_i]
                current_min = best_new_min
            else:
                # No improving swap found, try a random perturbation
                a, b = random.sample(range(N), 2)
                sigma[a], sigma[b] = sigma[b], sigma[a]
                current_min = compute_min_sum(sigma)
                
    # Return best found if it satisfies constraint
    if best_sigma is not None and best_min_sum >= 8:
        return 242, edges + [(l4[i], r4[best_sigma[i]]) for i in range(N)]
        
    return None, None

def _verify_girth(n, edges, g):
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check degree
    for i in range(n):
        if len(adj[i]) != 4:
            return False
            
    # Check simple graph
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        if (min(u,v), max(u,v)) in edge_set:
            return False
        edge_set.add((min(u,v), max(u,v)))
        
    # Girth check via BFS from each vertex to find shortest cycle
    from collections import deque
    
    def bfs_shortest_cycle(start):
        # BFS from start, track parent to detect cycles
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        min_cycle = float('inf')
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and parent[v] != u:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    min_cycle = min(min_cycle, cycle_len)
        
        return min_cycle
    
    # Check girth from a sample of vertices (or all if small enough)
    # For n=242, checking all is fine
    for start in range(n):
        cycle = bfs_shortest_cycle(start)
        if cycle < g:
            return False
    
    return True

def _fallback_construct(k, g):
    # Original program logic
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
