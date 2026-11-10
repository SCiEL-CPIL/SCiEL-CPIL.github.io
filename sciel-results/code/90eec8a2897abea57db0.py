import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    n = 16
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    best_edges = 0
    best_graph = set()
    
    # Helper to check if adding edge (u, v) creates Q3
    def creates_q3(graph, u, v):
        # graph is a set of edges
        # We need to check if there is a Q3 containing (u, v)
        # 1. Find pairs (a, b) such that u-a, a-b, b-v are in graph
        pairs_1 = []
        for a in range(n):
            if a == u or a == v: continue
            if (min(u,a), max(u,a)) not in graph: continue
            for b in range(n):
                if b == u or b == v or b == a: continue
                if (min(v,b), max(v,b)) not in graph: continue
                if (min(a,b), max(a,b)) not in graph: continue
                pairs_1.append((a, b))
        
        if not pairs_1:
            return False
            
        # 2. Find pairs (c, d) such that u-c, c-d, d-v are in graph
        # Note: (c, d) could be same as (a, b) or different.
        # In Q3, the two squares sharing an edge are distinct.
        pairs_2 = []
        for c in range(n):
            if c == u or c == v: continue
            if (min(u,c), max(u,c)) not in graph: continue
            for d in range(n):
                if d == u or d == v or d == c: continue
                if (min(v,d), max(v,d)) not in graph: continue
                if (min(c,d), max(c,d)) not in graph: continue
                pairs_2.append((c, d))
                
        # 3. Check combinations
        for a, b in pairs_1:
            for c, d in pairs_2:
                # Distinctness
                if a == c or a == d or b == c or b == d:
                    continue
                # We have vertices u, v, a, b, c, d
                # We need e, f
                # e connects to a, c
                # f connects to b, d
                # e-f connects
                # e, f not in {u, v, a, b, c, d}
                
                used = {u, v, a, b, c, d}
                candidates_e = []
                for e in range(n):
                    if e in used: continue
                    if (min(e,a), max(e,a)) in graph and (min(e,c), max(e,c)) in graph:
                        candidates_e.append(e)
                
                if not candidates_e:
                    continue
                
                for e in candidates_e:
                    candidates_f = []
                    for f in range(n):
                        if f in used or f == e: continue
                        if (min(f,b), max(f,b)) in graph and (min(f,d), max(f,d)) in graph:
                            # Check e-f
                            if (min(e,f), max(e,f)) in graph:
                                return True
        return False

    # Multi-restart greedy
    for _ in range(500): # Limit restarts
        if time.time() > deadline:
            break
            
        random.shuffle(all_edges)
        current_graph = set()
        
        for edge in all_edges:
            u, v = edge
            # Try to add
            if not creates_q3(current_graph, u, v):
                current_graph.add((min(u,v), max(u,v)))
        
        if len(current_graph) > best_edges:
            best_edges = len(current_graph)
            best_graph = set(current_graph)
            
        # Optional: Local search refinement
        # Try to remove one edge and add two? Or just swap?
        # Given time constraints, simple greedy might be enough.
        # But let's add a simple local improvement loop.
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding any missing edge
            for edge in all_edges:
                if edge in current_graph:
                    continue
                u, v = edge
                if not creates_q3(current_graph, u, v):
                    current_graph.add(edge)
                    improved = True
                    break # Restart scan after change? Or continue?
                    # Continuing might be faster if we don't rescan all
            # If no edge added, try removing one and see if we can add more?
            # This is complex. Let's stick to greedy for now.
            
    # Convert to list of lists
    return [[u, v] for u, v in best_graph]
