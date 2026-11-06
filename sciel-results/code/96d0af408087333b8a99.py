N = 40

def construct():
        edges = set()
        # Base cycle
        for i in range(N):
            edges.add((i, (i+1)%N))
        
        # Chords from 0 to 2..20
        for j in range(2, 21):
            edges.add((0, j))
            
        # Convert to list of tuples
        edge_list = [tuple(sorted(e)) for e in edges]
        
        cycles = []
        for k in range(3, N+1):
            if k <= 21:
                # Cycle 0-1-...-(k-1)-0
                cyc = list(range(k))
            elif k == 40:
                # Base cycle
                cyc = list(range(N))
            else:
                # 22 <= k <= 39
                j = 41 - k
                # Cycle 0-39-...-j-0
                cyc = [0] + list(range(39, j-1, -1))
            cycles.append(cyc)
            
        return {'edges': edge_list, 'cycles': cycles}
