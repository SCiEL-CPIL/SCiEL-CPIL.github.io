import time
import sys

N = 40

def construct():
    t0 = time.time()
    
    # Base cycle edges
    edges = [(i, (i + 1) % N) for i in range(N)]
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
        
    # Add chords to vertex 0
    # Even chords: 2, 4, ..., 20
    for d in range(2, 21, 2):
        edges.append((0, d))
        edge_set.add((0, d))
        
    # Odd chords: 3, 5, ..., 19
    for d in range(3, 20, 2):
        edges.append((0, d))
        edge_set.add((0, d))
        
    # Verify edge count
    # Base: 40
    # Even chords: 10
    # Odd chords: 9
    # Total: 59
    
    # Generate cycles
    cycles = []
    for k in range(3, N + 1):
        if k == N:
            cycle = list(range(N))
        else:
            # Check if d = k - 1 works (cycle 0..k-1)
            d1 = k - 1
            if (min(0, d1), max(0, d1)) in edge_set:
                cycle = list(range(d1 + 1))
            else:
                # Check if d = 41 - k works (cycle 0, 39, ..., 41-k)
                d2 = 41 - k
                if (min(0, d2), max(0, d2)) in edge_set:
                    # Vertices: 0, 39, 38, ..., d2
                    cycle = [0] + list(range(N - 1, d2 - 1, -1))
                else:
                    # Fallback: should not happen, but find any cycle if logic fails
                    # This is a safety net, not expected to be triggered
                    cycle = None
            
        cycles.append(cycle)
        
    return {'edges': edges, 'cycles': cycles}
