import time
import random
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_n = 39
    best_edges = []
    
    # Try from largest n down to 40
    for target_n in range(47, 39, -1):
        if time.time() > deadline:
            break
            
        min_deg_needed = target_n - 39
        
        found = False
        
        # Use more restarts if we have time, but keep it bounded
        # Current best is 44. 47 is target.
        # We need to find a valid graph for 47 or 46.
        
        for restart in range(2000):
            if time.time() > deadline:
                break
                
            # Different random seeds
            random.seed(restart * 12345 + 789 + target_n)
            
            in_graph = [[False] * target_n for _ in range(target_n)]
            adj = [0] * target_n
            deg = [0] * target_n
            edges = []
            
            def can_add(u, v):
                # Check if adding (u,v) creates a C4.
                # This happens if there is a path of length 3 between u and v.
                # Path u - a - b - v.
                # So a is neighbor of u, b is neighbor of v, and a-b is an edge.
                # Equivalently, (N(u) \ {v}) and N(v) \ {u} have a common neighbor?
                # No.
                # a in N(u), b in N(v), edge(a,b).
                # So we check if there exists a in N(u) such that N(a) intersects N(v).
                # Note: if a==v, edge(v,v) doesn't exist. If b==u, edge(u,u) doesn't exist.
                # Since graph is simple and u!=v, and we are adding u-v, u is not in N(v) yet, v is not in N(u) yet.
                nu = adj[u]
                nv = adj[v]
                c = nu
                while c:
                    a_bit = c & -c
                    a = a_bit.bit_length() - 1
                    c ^= a_bit
                    # Check if any neighbor of a is in nv
                    if (adj[a] & nv) != 0:
                        return False
                return True

            heap = []
            
            # Strategy: 
            # 1. Random order (priority 0)
            # 2. Least degree sum (priority -deg[u]-deg[v] or just use deg[u]+deg[v] but pop min?)
            # Let's try Random first, it's often robust.
            # Or "Least Connected": we want to help low degree vertices.
            # If we add edge to low degree vertices, their degree goes up.
            # If we use max-heap on (deg[u]+deg[v]), we connect high degree vertices.
            # Let's try MIN-heap on (deg[u]+deg[v]) to prioritize low degree vertices.
            
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    # Use min-heap logic by negating? 
                    # Or just push (deg[u]+deg[v], random.random(), u, v)
                    # and pop smallest?
                    # heapq is min-heap.
                    # So pushing deg[u]+deg[v] will pop smallest sum.
                    # This is "Least Connected First".
                    heapq.heappush(heap, (deg[u] + deg[v], random.random(), u, v))
            
            while heap:
                if time.time() > deadline:
                    break
                    
                pri, _, u, v = heapq.heappop(heap)
                
                if in_graph[u][v]:
                    continue
                
                if pri != deg[u] + deg[v]:
                    continue
                
                if can_add(u, v):
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                    deg[u] += 1
                    deg[v] += 1
                    edges.append([u, v])
                    in_graph[u][v] = True
                    in_graph[v][u] = True
                    
                    for w in range(target_n):
                        if w == u or w == v:
                            continue
                        if not in_graph[u][w]:
                            heapq.heappush(heap, (deg[u] + deg[w], random.random(), u, w))
                        if not in_graph[v][w]:
                            heapq.heappush(heap, (deg[v] + deg[w], random.random(), v, w))
                    
            if min(deg) >= min_deg_needed:
                if target_n > best_n:
                    best_n = target_n
                    best_edges = edges
                found = True
                break
            
        if best_n >= target_n:
            break
            
    if best_n >= 40:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 39, 'edges': []}
