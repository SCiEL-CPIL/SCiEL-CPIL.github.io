def construct():
        N = 40
        edges = set()
        # Add cycle
        for i in range(N):
            edges.add((i, (i+1)%N))
        
        # Add chords (0, k) for k in 2..20
        for k in range(2, 21):
            edges.add((0, k))
        
        # Convert to list of tuples for output
        # Ensure consistent ordering? "undirected edge pairs". Usually (min, max) is good.
        edge_list = []
        for u, v in edges:
            if u > v:
                u, v = v, u
            edge_list.append((u, v))
        
        cycles = []
        for L in range(3, N+1):
            if L == N:
                cyc = list(range(N))
            elif L <= 21:
                # 0, 1, ..., L-1
                cyc = list(range(L))
            else:
                # 0, 39, ..., 41-L
                k = 41 - L
                # Path 0 -> 39 -> ... -> k
                # Vertices: 0, 39, 38, ..., k
                cyc = [0] + list(range(39, k-1, -1))
            cycles.append(cyc)
            
        return {'edges': edge_list, 'cycles': cycles}
