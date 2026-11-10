import random
import time
import sys

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges).
    """
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Moore bound for k=5, g=9 is 426. Lower bound known is 428.
    # We search for n starting from a reasonable lower bound.
    # For k=5, g=9, n must be even? No, 5-regular graph must have even number of vertices?
    # Sum of degrees = 5n must be even, so 5n is even => n is even.
    
    def is_girth_at_least(edges, n, g):
        # Check if girth is at least g.
        # We can do BFS from each vertex to find the shortest cycle.
        # Or specifically check for cycles of length < g.
        # For g=9, we need to check for cycles of length 4, 6, 8.
        # A faster way: build adjacency list, then for each vertex, 
        # check if any neighbor is reachable in < g steps without backtracking?
        # Actually, standard BFS from each node to find shortest cycle is O(n*(n+E)).
        # For n~500, E~1250, this is ~500*1750 ~ 1e6 ops per graph. Very fast.
        
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check for cycles of length < g
        # We can use a simple BFS from each node to find the shortest cycle involving that node.
        # Or just find the global shortest cycle.
        # Since g is small (9), we can just check for existence of cycles of length 4, 6, 8.
        
        # Check for 4-cycles:
        # For each edge (u, v), check if there is a common neighbor w such that (u,w) and (v,w) exist? No, that's a triangle.
        # 4-cycle: u-v-w-x-u. u and w are opposite. v and x are opposite.
        # u and w have two common neighbors.
        # For each pair of vertices (u, w), count common neighbors. If >= 2, there is a 4-cycle.
        # This is O(n^2 * k^2) which is fine.
        
        # Actually, a more robust way for small g is to just run BFS from each node up to depth g//2 - 1?
        # No, shortest cycle length L. If we do BFS from u, and find a back edge to a node at depth d,
        # the cycle length is d + 1? No.
        
        # Let's just do a full BFS from each node to find the shortest cycle.
        # For each node s:
        #   dist[s] = 0, parent[s] = -1
        #   BFS
        #   When we visit a neighbor v of u:
        #     if v == s: cycle length = dist[u] + 1
        #     elif parent[v] != u and dist[v] < dist[u] + 1: # found a cycle
        #         cycle length = dist[u] + dist[v] + 1
        #     We take the min.
        
        # This is O(n * (n+E)). For n=500, E=1250, n+E=1750. 500*1750 = 875,000.
        # Doing this for each candidate graph is fast.
        
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if v != parent[u] and u != parent[v]:
                            # Found a cycle
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    def generate_random_k_regular(n, k):
        # Configuration model
        if n * k % 2 != 0:
            return None
        
        half_edges = []
        for i in range(n):
            for _ in range(k):
                half_edges.append(i)
        
        random.shuffle(half_edges)
        
        edges = set()
        adj = [[] for _ in range(n)]
        for i in range(0, len(half_edges), 2):
            u = half_edges[i]
            v = half_edges[i+1]
            if u == v:
                return None
            if (u, v) in edges or (v, u) in edges:
                return None
            edges.add((u, v))
            adj[u].append(v)
            adj[v].append(u)
        
        # Check if simple and k-regular (should be by construction if no multi-edges/loops)
        # But we might have missed some degree constraints if we just pair?
        # No, configuration model ensures k half-edges per node.
        # If we reject loops and multi-edges, the remaining graph might not be k-regular?
        # No, if we reject, we just return None and try again.
        # So if we return, it is k-regular and simple.
        
        return list(edges)

    # Search for n
    # Start from Moore bound
    # For k=5, g=9, Moore bound is 426.
    # n must be even.
    min_n = 428
    
    # We can try to find a graph with n close to min_n.
    # But maybe n needs to be larger.
    # Let's try a range of n.
    
    best_n = None
    best_edges = None
    
    # Try n from min_n upwards, step 2 (since n must be even)
    # But we have time limit.
    # Let's try a few random n values or just start from min_n and go up.
    # If we find one, we can try to minimize n further?
    # The score is n. So we want the smallest n.
    
    # We can do a binary search on n? No, existence is not monotonic in a simple way for random graphs,
    # but probability of existence increases with n.
    # So if we find one at n, we can try smaller n?
    # But we started from min_n. So we just go up until we find one.
    
    # To be safe, we can try a few random attempts for each n.
    
    for n in range(min_n, min_n + 200, 2):
        if time.time() > deadline:
            break
        
        # Try a few random graphs for this n
        max_attempts = 100
        for _ in range(max_attempts):
            if time.time() > deadline:
                break
            
            edges = generate_random_k_regular(n, k)
            if edges is None:
                continue
            
            if is_girth_at_least(edges, n, g):
                best_n = n
                best_edges = edges
                # We found one. Can we find a smaller one?
                # We are iterating n upwards. So this is the smallest n we found so far.
                # But we might have skipped some? No, we start from min_n.
                # So we can return immediately?
                # Wait, we want the MINIMUM n.
                # So if we find one at n=430, we should return it.
                # But what if there is one at n=428? We tried it.
                # So we can return.
                return n, edges
        
    if best_n is not None:
        return best_n, best_edges
    
    # If not found, return a valid graph with larger n to avoid score 0
    # Fallback to LUW or just a larger random graph
    n = 1000
    while True:
        if time.time() > deadline:
             # Return whatever we have or a dummy
             if best_n: return best_n, best_edges
             return 0, []
        edges = generate_random_k_regular(n, k)
        if edges is not None and is_girth_at_least(edges, n, g):
            return n, edges
        n += 2

