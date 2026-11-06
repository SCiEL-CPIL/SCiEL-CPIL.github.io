import time
import sys

N = 40

def construct():
    t0 = time.time()
    
    # Base cycle edges
    edges = []
    edge_set = set()
    
    for i in range(N):
        u = i
        v = (i + 1) % N
        if u < v:
            e = (u, v)
        else:
            e = (v, u)
        edges.append(e)
        edge_set.add(e)
        
    # Add chords to vertex 0
    # Chords (0, d) for d in 2..20
    for d in range(2, 21):
        e = (0, d)
        edges.append(e)
        edge_set.add(e)
        
    # Generate cycles
    cycles = []
    for k in range(3, N + 1):
        if k == N:
            cycle = list(range(N))
        else:
            # Try forward path 0..k-1
            d1 = k - 1
            if (0, d1) in edge_set:
                cycle = list(range(d1 + 1))
            else:
                # Try backward path 0, N-1, ..., d2
                d2 = N - (k - 1) # This is 41-k
                if (0, d2) in edge_set:
                    # Vertices: 0, N-1, N-2, ..., d2
                    cycle = [0] + list(range(N - 1, d2 - 1, -1))
                else:
                    # Fallback: should not happen
                    cycle = [0, 1, 2] # Invalid placeholder, but logic above covers all
        cycles.append(cycle)
        
    return {'edges': edges, 'cycles': cycles}
