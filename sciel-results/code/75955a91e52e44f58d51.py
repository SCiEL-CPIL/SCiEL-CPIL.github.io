import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin

    # 1. Construct Grötzsch graph (Mycielski of C5)
    # Vertices 0..4 form C5
    # Vertices 5..9 are copies of 0..4
    # Vertex 10 is the root
    
    edges = []
    # Cycle 0-1-2-3-4-0
    cycle_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    for u, v in cycle_edges:
        edges.append([u, v])
        
    # Copies: u -> 5+u. 5+u connected to neighbors of u in cycle.
    # Neighbors of 0: 1, 4. So 5 connected to 1, 4.
    # Neighbors of 1: 0, 2. So 6 connected to 0, 2.
    # Neighbors of 2: 1, 3. So 7 connected to 1, 3.
    # Neighbors of 3: 2, 4. So 8 connected to 2, 4.
    # Neighbors of 4: 3, 0. So 9 connected to 3, 0.
    
    copy_connections = {
        5: [1, 4],
        6: [0, 2],
        7: [1, 3],
        8: [2, 4],
        9: [3, 0]
    }
    
    for copy, neighbors in copy_connections.items():
        for n in neighbors:
            edges.append([copy, n])
            
    # Root 10 connected to all copies 5..9
    for i in range(5, 10):
        edges.append([10, i])

    # 2. Find Independent Set S in Grötzsch that minimizes |N(S)|
    # N(S) = {v not in S | exists u in S such that (u,v) in E}
    
    n_base = 11
    # Adjacency list for base graph
    adj = [set() for _ in range(n_base)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    best_S = None
    best_N_size = n_base + 1
    
    # Brute force all subsets? 2^11 = 2048. Very fast.
    for mask in range(1 << n_base):
        S = []
        for i in range(n_base):
            if (mask >> i) & 1:
                S.append(i)
                
        # Check if S is independent
        is_indep = True
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                if S[j] in adj[S[i]]:
                    is_indep = False
                    break
            if not is_indep:
                break
        
        if not is_indep:
            continue
            
        # Calculate N(S)
        N_S = set()
        for u in S:
            for v in adj[u]:
                if v not in S:
                    N_S.add(v)
                    
        if len(N_S) < best_N_size:
            best_N_size = len(N_S)
            best_S = S[:]
            
    if best_S is None:
        # Fallback to empty set or single vertex
        best_S = [0]
        best_N_size = len(adj[0])

    # 3. Construct final graph
    # Vertices 0..10 are Grötzsch
    # Vertices 11..15 are new
    # A = S union (V \ (S union N(S)))
    # A = {v in V | v in S OR (v not in S AND v not in N(S))}
    # Note: v not in S AND v not in N(S) means v is not adjacent to any u in S.
    
    S_set = set(best_S)
    N_S_set = set()
    for u in best_S:
        for v in adj[u]:
            if v not in S_set:
                N_S_set.add(v)
                
    A = []
    for v in range(n_base):
        if v in S_set:
            A.append(v)
        elif v not in N_S_set:
            A.append(v)
            
    # Add 5 new vertices 11..15
    # Connect each to all vertices in A
    for new_v in range(11, 16):
        for a in A:
            edges.append([new_v, a])
            
    # Check time
    if time.time() > deadline:
        # Return what we have, it should be valid
        pass

    return edges
