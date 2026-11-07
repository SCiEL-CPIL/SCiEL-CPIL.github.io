import time

def construct():
    start_time = time.time()
    deadline = 58.0  # Safety margin before 60s limit

    # Candidate: Two disjoint K_3,3
    # Vertices 0-5 form one K_3,3
    # Vertices 6-11 form another K_3,3
    edges = []
    
    # First K_3,3: A={0,1,2}, B={3,4,5}
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
            
    # Second K_3,3: A={6,7,8}, B={9,10,11}
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])
            
    # Verification logic to ensure validity
    # 1. Check degrees
    n = 12
    deg = {i: 0 for i in range(n)}
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    
    if min(deg.values()) < 3:
        # Fallback, though this shouldn't happen for K_3,3
        return []

    # 2. Check for triangles (girth >= 4)
    # Build adjacency set for O(1) lookup
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    for i in range(n):
        for j in adj[i]:
            if i < j:
                # Check if any common neighbor
                common = adj[i] & adj[j]
                if common:
                    # Triangle found, invalid
                    return []

    # 3. Check if it contains a cycle
    # K_3,3 definitely contains cycles (4-cycles).
    # We can do a quick BFS/DFS to confirm connectivity/cycles if needed,
    # but for K_3,3, it's guaranteed.
    
    # Check time
    if time.time() - start_time > deadline:
        return edges
        
    return edges
