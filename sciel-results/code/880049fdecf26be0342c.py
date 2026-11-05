import random

def construct():
    n = 20
    best_edges = []
    best_count = 0
    
    # Run multiple random greedy trials
    for trial in range(200):
        random.seed(trial)
        masks = [0] * n
        edges = []
        
        # Generate all possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if can_add_edge(u, v, masks, n):
                # Add edge
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
                edges.append([u, v])
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            
    return best_edges

def can_add_edge(u, v, masks, n):
    # Check if adding (u, v) violates the condition
    # Condition: No 3 vertices have 3 common neighbors.
    # Equivalent: For all distinct a,b,c, |N(a) & N(b) & N(c)| <= 2.
    
    # Adding (u,v) affects triples containing u or v.
    # Triples with u (not v): {u, a, b}
    # New common neighbors: (N(u) U {v}) & N(a) & N(b)
    # = (N(u) & N(a) & N(b)) U ({v} & N(a) & N(b))
    # Size increases by 1 if v is in N(a) & N(b).
    # So we need: if v in N(a) & N(b), then |N(u) & N(a) & N(b)| <= 1.
    
    # Triples with v (not u): {v, a, b}
    # Similar: if u in N(a) & N(b), then |N(v) & N(a) & N(b)| <= 1.
    
    # Triples with both u and v: {u, v, a}
    # Common neighbors: N'(u) & N'(v) & N(a)
    # N'(u) = N(u) U {v}, N'(v) = N(v) U {u}
    # Intersection is N(u) & N(v) & N(a) (since v not in N(v), u not in N(u))
    # This is unchanged by the addition. So no new violation here.
    
    # So we only need to check pairs (a,b) for the two cases above.
    
    # Optimization: Pre-compute common neighbors for all pairs?
    # masks[a] & masks[b] is fast.
    
    # Case 1: v is a common neighbor of a and b.
    # Iterate a, b
    for a in range(n):
        if a == u or a == v:
            continue
        for b in range(a + 1, n):
            if b == u or b == v:
                continue
            
            common_ab = masks[a] & masks[b]
            
            # Check if v is in common_ab
            if common_ab & (1 << v):
                # v is a common neighbor of a and b
                # Check intersection of N(u) and common_ab
                inter = masks[u] & common_ab
                if bin(inter).count('1') > 1:
                    return False
    
    # Case 2: u is a common neighbor of a and b.
    for a in range(n):
        if a == u or a == v:
            continue
        for b in range(a + 1, n):
            if b == u or b == v:
                continue
            
            common_ab = masks[a] & masks[b]
            
            # Check if u is in common_ab
            if common_ab & (1 << u):
                # u is a common neighbor of a and b
                # Check intersection of N(v) and common_ab
                inter = masks[v] & common_ab
                if bin(inter).count('1') > 1:
                    return False
    
    return True
