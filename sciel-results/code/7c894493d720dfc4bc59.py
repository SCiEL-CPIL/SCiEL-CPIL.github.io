import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 40
    best_edges = 0
    best_graph = None
    
    # List of all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # We will try multiple random restarts
    # Each restart:
    # 1. Initialize with a random subset of edges (or empty)
    # 2. Greedily add edges in random order if no C6 created
    
    # To improve, we can start with a random graph of density p
    # or just empty and add.
    # Starting empty might be slow to reach 100+ edges if we only add.
    # But greedy addition is monotonic.
    
    # Let's try starting with a random graph of low density and then completing it.
    # Or just empty.
    
    # Heuristic: The order matters.
    # Random shuffling of edges.
    
    for attempt in range(1000): # Max attempts, limited by time
        if time.time() > deadline:
            break
            
        # Initialize graph
        # Option 1: Empty
        # Option 2: Random sparse
        # Let's stick to empty for simplicity, but random order of addition is key.
        
        adj = [set() for _ in range(n)]
        edges = []
        
        # Randomize edge order
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > deadline:
                break
                
            # Check if adding (u, v) creates a C6
            # Condition: No simple path of length 4 between u and v
            
            # Quick check: if u and v are already connected? No, simple graph.
            # Check for path of length 4: u-a-b-c-v
            
            # N(u) and N(v)
            nu = adj[u]
            nv = adj[v]
            
            # If nu or nv is empty, no path of length 4
            if not nu or not nv:
                edges.append([u, v])
                adj[u].add(v)
                adj[v].add(u)
                continue
                
            c6_found = False
            # Iterate a in N(u), c in N(v)
            # To speed up, we can iterate over the smaller set first?
            # Or just nested loop.
            # n=40, max degree 39. 39*39 = 1521 pairs.
            # Intersection check is fast.
            
            for a in nu:
                if a == v: # Should not happen if u,v not connected
                    continue
                for c in nv:
                    if a == c:
                        continue
                    # Check intersection of N(a) and N(c)
                    # Exclude u, v, a, c
                    # If there is any other vertex b, then path u-a-b-c-v exists
                    # Note: b must be distinct from u, v, a, c.
                    
                    # Optimization:
                    # If len(N(a)) + len(N(c)) < 4? No.
                    # Just do set intersection.
                    inter = adj[a] & adj[c]
                    # Remove forbidden vertices
                    # We need b not in {u, v, a, c}
                    # Since u is in N(a) and N(c)?
                    # u in N(a) is True.
                    # u in N(c)? If u in N(c), then u is in inter.
                    # v in N(a)? If v in N(a), then v is in inter.
                    # a in N(a)? No.
                    # c in N(c)? No.
                    
                    # So we just need to check if inter contains any vertex
                    # other than u, v, a, c.
                    
                    # Actually, if u in N(c), then u is a candidate for b?
                    # Path u-a-u-c-v. Not simple.
                    # So u is forbidden.
                    # Similarly v is forbidden.
                    # a and c are forbidden.
                    
                    # So:
                    if u in inter:
                        inter.discard(u)
                    if v in inter:
                        inter.discard(v)
                    if a in inter:
                        inter.discard(a)
                    if c in inter:
                        inter.discard(c)
                        
                    if inter:
                        c6_found = True
                        break
                if c6_found:
                    break
            
            if not c6_found:
                edges.append([u, v])
                adj[u].add(v)
                adj[v].add(u)
                
        if len(edges) > best_edges:
            best_edges = len(edges)
            best_graph = edges[:]
            
        # If we reached 105, we can stop early?
        # The task says best known is 105.
        # If we hit 105, we might want to try to exceed it.
        # But 105 is a strong bound.
        # Let's keep going until deadline.
        
    return best_graph if best_graph else []
