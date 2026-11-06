import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Target n values to try, in descending order
    for n in [47, 46, 45, 44]:
        target_deg = n - 39
        
        # Try multiple random restarts for this n
        # Increase restarts for smaller n if needed?
        # 47 is the hardest.
        max_restarts = 1000
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialize graph
            adj = [set() for _ in range(n)]
            
            # Start with a random graph
            # Use p=0.5
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i].add(j)
                        adj[j].add(i)
            
            # Phase 1: Remove edges to make it C4-free
            stable = False
            while not stable and time.time() < deadline:
                stable = True
                # Iterate all pairs
                for u in range(n):
                    for v in range(u+1, n):
                        if v in adj[u]:
                            common = adj[u] & adj[v]
                            if len(common) >= 2:
                                # Found a C4
                                common_list = list(common)
                                # Pick two distinct common neighbors
                                # If len > 2, pick two random?
                                # Just picking first two is fine, or randomize
                                if len(common_list) > 2:
                                    a, b = random.sample(common_list, 2)
                                else:
                                    a, b = common_list[0], common_list[1]
                                
                                # Edges in cycle: (u,a), (a,v), (v,b), (b,u)
                                # Remove one randomly
                                edge_to_remove = random.choice([(u, a), (a, v), (v, b), (b, u)])
                                u_r, v_r = edge_to_remove
                                if u_r in adj[v_r]:
                                    adj[u_r].remove(v_r)
                                    adj[v_r].remove(u_r)
                                stable = False
                                break
                    if not stable:
                        break
            
            # Check min degree
            min_deg = min(len(adj[i]) for i in range(n))
            if min_deg >= target_deg:
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
    
    # Fallback
    return {'n': 39, 'edges': []}
