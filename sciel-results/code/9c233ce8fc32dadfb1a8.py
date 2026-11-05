import time

N = 40

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    edges = []
    edge_set = set()
    
    def add_edge(u, v):
        if u > v:
            u, v = v, u
        if (u, v) not in edge_set:
            edge_set.add((u, v))
            edges.append((u, v))
            return True
        return False
    
    # Try ladder graph: 2 x 20 grid = 58 edges
    # Path1: 0-1-...-19
    # Path2: 20-21-...-39
    # Rungs: (i, 20+i) for i=0..19
    
    for i in range(19):
        add_edge(i, i + 1)
    for i in range(20, 39):
        add_edge(i, i + 1)
    for i in range(20):
        add_edge(i, 20 + i)
    
    # Construct cycles for ladder graph
    # Ladder graph L_n = P_2 x P_n is pancyclic for n >= 4
    cycles = []
    valid = True
    for L in range(3, N + 1):
        if L == 40:
            # Outer perimeter: 0-1-...-19-39-38-...-20-0
            cycle = list(range(20)) + list(range(39, 19, -1))
        elif L == 3:
            # Triangle doesn't exist in bipartite ladder. Ladder is bipartite!
            # So no odd cycles. This won't work.
            valid = False
            break
        elif L % 2 == 0:
            k = L // 2
            if k <= 19:
                # Rectangle of width k: 0-1-...-(k-1)-(20+k-1)-(20+k-2)-...-20-0
                cycle = list(range(k)) + list(range(20 + k - 1, 19, -1))
            else:
                valid = False
                break
        else:
            # Odd cycle in bipartite graph: impossible
            valid = False
            break
        
        if len(set(cycle)) != len(cycle):
            valid = False
            break
        cycles.append(cycle)
    
    if not valid or len(cycles) != 38:
        # Fall back to fan construction
        edges.clear()
        edge_set.clear()
        
        # 1. Add Hamiltonian Cycle: 0-1-...-39-0
        for i in range(N):
            add_edge(i, (i + 1) % N)
            
        # 2. Add Chords from 0 to k for k in 2..20
        for k in range(2, 21):
            add_edge(0, k)
            
        # 3. Generate Cycles
        cycles = []
        for L in range(3, N + 1):
            if L == N:
                cycle = list(range(N))
            elif L <= 21:
                k = L - 1
                cycle = list(range(k + 1))
            else:
                k = 41 - L
                cycle = [0] + list(range(N - 1, k - 1, -1))
            cycles.append(cycle)
    
    return {'edges': edges, 'cycles': cycles}
