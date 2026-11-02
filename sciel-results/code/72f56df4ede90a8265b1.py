import time
import random
import itertools
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin

    # Fallback: LUW graph
    def luw_construct():
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
        i = 0
        while i < len(order):
            if time.time() > deadline:
                break
            vertex = order[i]
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
            i += 1
        n = len(order)
        return n, edges

    if k != 5 or g != 9:
        return luw_construct()

    # Try to find a smaller graph using random generation and repair
    # Target: beat LUW (n=6250)
    # Known best: 1116. Moore bound: 426.
    
    def check_girth(n, adj, min_girth):
        """
        Returns True if girth >= min_girth.
        adj: list of lists of neighbors.
        """
        # BFS from each vertex to depth (min_girth - 1) // 2
        # If we find a cycle shorter than min_girth, return False.
        depth_limit = (min_girth - 1) // 2
        
        for s in range(n):
            # dist[u] = distance from s
            # parent[u] = parent in BFS tree
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = deque([s])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= depth_limit:
                    continue
                
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # v is already visited.
                        # If v is not the parent of u, we have a cycle.
                        # The cycle length is dist[u] + dist[v] + 1?
                        # Not exactly. The cycle goes u -> ... -> v -> u.
                        # The path from s to u and s to v might share nodes.
                        # However, in a BFS tree, if we find an edge (u,v) where u != parent[v] and v != parent[u],
                        # we have a cycle.
                        # The length of the cycle is at most dist[u] + dist[v] + 1.
                        # Actually, if we just check if dist[u] + dist[v] + 1 < min_girth, it's a necessary condition for a short cycle?
                        # No, the actual cycle length could be smaller if the paths diverge earlier.
                        # But if dist[u] + dist[v] + 1 >= min_girth, it *might* still have a short cycle.
                        # However, if dist[u] + dist[v] + 1 < min_girth, we definitely have a cycle of length <= that.
                        # Wait, if the cycle is shorter, dist[u] + dist[v] + 1 is an upper bound on the cycle length?
                        # Yes. The cycle is u->s->v->u? No, s is the root.
                        # The cycle is formed by the path s->...->u, edge (u,v), and path v->...->s.
                        # The length is dist[u] + 1 + dist[v].
                        # If this is < min_girth, we have a short cycle.
                        if v != parent[u] and u != parent[v]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_girth:
                                return False
            # Also need to check if the BFS tree itself has a back edge?
            # The condition above covers edges (u,v) where both are visited.
            # If dist[u] == dist[v] or |dist[u]-dist[v]|==1, it's a cross/back edge.
            # The cycle length is dist[u] + dist[v] + 1 is correct for the cycle formed by the two tree paths and the edge.
            
        return True

    def find_short_cycle(n, adj, min_girth):
        """
        Returns a list of vertices forming a cycle of length < min_girth, or None.
        """
        depth_limit = (min_girth - 1) // 2
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = deque([s])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= depth_limit:
                    continue
                
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if v != parent[u] and u != parent[v]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_girth:
                                # Reconstruct cycle
                                # Path s->u and s->v
                                path_u = []
                                cur = u
                                while cur != s:
                                    path_u.append(cur)
                                    cur = parent[cur]
                                path_u.append(s)
                                path_u.reverse() # s -> ... -> u
                                
                                path_v = []
                                cur = v
                                while cur != s:
                                    path_v.append(cur)
                                    cur = parent[cur]
                                path_v.append(s)
                                path_v.reverse() # s -> ... -> v
                                
                                # Cycle: path_u (s->u) + (u,v) + path_v reversed (v->s)
                                # We want the cycle sequence.
                                # path_u is s, ..., u.
                                # path_v is s, ..., v.
                                # The cycle is u, ..., s, ..., v, u? No.
                                # The edge is (u,v).
                                # The cycle is the path from u to s, then s to v, then edge v-u?
                                # Actually, the cycle is the union of the two paths and the edge.
                                # To get the simple cycle, we find the LCA of u and v.
                                # But for repair, we just need the edges involved.
                                # We can just return the list of edges in the two paths + (u,v).
                                # Or just return the vertices of the cycle.
                                # For simplicity, return the edge (u,v) and the parents.
                                return [u, v] # Just the edge that closed the cycle?
                                # Actually, for repair, we need the cycle edges.
                                # Let's return the full cycle path.
                                # Find LCA
                                set_u = set(path_u)
                                lca = s
                                for node in path_v:
                                    if node in set_u:
                                        lca = node
                                        break
                                
                                # Path from lca to u
                                p1 = []
                                cur = u
                                while cur != lca:
                                    p1.append(cur)
                                    cur = parent[cur]
                                p1.append(lca)
                                p1.reverse()
                                
                                # Path from lca to v
                                p2 = []
                                cur = v
                                while cur != lca:
                                    p2.append(cur)
                                    cur = parent[cur]
                                p2.append(lca)
                                p2.reverse()
                                
                                # Cycle: p1 + p2[1:]
                                cycle = p1 + p2[1:]
                                return cycle
        return None

    def generate_random_regular(n, k):
        # Configuration model
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        random.shuffle(stubs)
        edges = set()
        adj = [[] for _ in range(n)]
        
        # Pair stubs
        for i in range(0, n * k, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                # Loop, re-pair? Just skip for now and fix later?
                # Better to retry generation if loops are frequent.
                return None
            if (u, v) in edges or (v, u) in edges:
                # Multi-edge, retry
                return None
            edges.add((min(u,v), max(u,v)))
            adj[u].append(v)
            adj[v].append(u)
            
        return adj, list(edges)

    def repair(n, adj, edges, g):
        """
        Attempts to increase girth by swapping edges.
        Returns (n, edges) if successful, else None.
        """
        max_iters = 2000
        for _ in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Check girth
            if check_girth(n, adj, g):
                return n, list(edges)
            
            # Find a short cycle
            cycle = find_short_cycle(n, adj, g)
            if cycle is None:
                # Should not happen if check_girth failed
                break
            
            # cycle is a list of vertices [v0, v1, ..., vk-1]
            # Edges are (v0,v1), (v1,v2), ..., (vk-1,v0)
            # Pick two non-adjacent edges in the cycle to swap?
            # Or just pick two random edges in the cycle.
            # Let's pick two edges (u1, v1) and (u2, v2) from the cycle.
            # To break the cycle, we must remove at least one edge.
            # Swapping two edges (a,b) and (c,d) to (a,c) and (b,d) breaks the cycle if a,b,c,d are in the cycle.
            
            # Get edges in cycle
            cycle_edges = []
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i+1) % len(cycle)]
                cycle_edges.append((min(u,v), max(u,v)))
            
            # Pick two distinct edges
            if len(cycle_edges) < 2:
                break
            e1, e2 = random.sample(cycle_edges, 2)
            
            a, b = e1
            c, d = e2
            
            # Try swap 1: (a,c) and (b,d)
            # Check validity
            if a != c and b != d and a != d and b != c: # No loops
                # Check if edges exist
                if (a,c) not in edges and (b,d) not in edges and (c,a) not in edges and (d,b) not in edges:
                    # Perform swap
                    edges.discard(e1)
                    edges.discard(e2)
                    edges.add((min(a,c), max(a,c)))
                    edges.add((min(b,d), max(b,d)))
                    
                    # Update adj
                    adj[a].remove(c) if c in adj[a] else None # c not in adj[a] yet
                    # Actually, we need to update adj carefully.
                    # Remove old edges
                    adj[a].remove(b)
                    adj[b].remove(a)
                    adj[c].remove(d)
                    adj[d].remove(c)
                    
                    # Add new edges
                    adj[a].append(c)
                    adj[c].append(a)
                    adj[b].append(d)
                    adj[d].append(b)
                    
                    # Continue loop (re-check girth)
                    continue
            
            # Try swap 2: (a,d) and (b,c)
            if a != d and b != c:
                if (a,d) not in edges and (b,c) not in edges and (d,a) not in edges and (c,b) not in edges:
                    edges.discard(e1)
                    edges.discard(e2)
                    edges.add((min(a,d), max(a,d)))
                    edges.add((min(b,c), max(b,c)))
                    
                    adj[a].remove(b)
                    adj[b].remove(a)
                    adj[c].remove(d)
                    adj[d].remove(c)
                    
                    adj[a].append(d)
                    adj[d].append(a)
                    adj[b].append(c)
                    adj[c].append(b)
                    
                    continue
            
            # If no valid swap, try another cycle or break
            break
            
        return None

    # Search for n
    # Start from Moore bound + margin
    # Moore bound for 5,9 is 426.
    # Try n from 1000 to 2000 step 2.
    # If found, return.
    # If not, return LUW.
    
    best_n = float('inf')
    best_edges = None
    
    # We can try a few random restarts for each n?
    # Or just one try per n to save time.
    
    # Order of n: try smaller n first.
    # But smaller n is harder to repair.
    # Maybe start with n=1500?
    # Let's try n in range(1000, 2000, 2)
    
    for n in range(1000, 2001, 2):
        if time.time() > deadline - 10: # Leave 10s for LUW
            break
            
        # Try 5 times for each n
        for _ in range(5):
            if time.time() > deadline - 5:
                break
                
            gen = generate_random_regular(n, k)
            if gen is None:
                continue
                
            adj, edges = gen
            result = repair(n, adj, set(edges), g)
            if result:
                rn, r_edges = result
                if rn < best_n:
                    best_n = rn
                    best_edges = r_edges
                # If we found one, we can try to find a smaller one?
                # Or just return this one?
                # Returning a valid graph is good.
                # But maybe a smaller one exists?
                # Given time, return the first found?
                # Or keep searching for a bit?
                # Let's return the first valid one found in the loop.
                # Actually, the loop goes from 1000 up.
                # So the first found is the smallest n we tried.
                return rn, r_edges
                
    # If no graph found, return LUW
    return luw_construct()
