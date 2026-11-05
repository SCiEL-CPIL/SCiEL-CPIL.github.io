import time

N = 40

def construct():
    start_time = time.time()
    
    # 1. Construct the specific sparse pancyclic graph
    # Base Hamiltonian Cycle: 0-1-...-39-0
    edges = set()
    for i in range(N):
        u = i
        v = (i + 1) % N
        if u > v:
            u, v = v, u
        edges.add((u, v))
        
    # Chords from 0 to m for m in 2..20
    # These cover all cycle lengths 3..40
    # Length L < 21: uses chord (0, L-1)
    # Length L >= 21: uses chord (0, 40-L+1)
    # We need chords for m in {2, ..., 20}
    for m in range(2, 21):
        u = 0
        v = m
        if u > v:
            u, v = v, u
        edges.add((u, v))
        
    # Total edges: 39 (cycle) + 19 (chords) = 58
    # Value: 58 - 40 = 18
    
    # 2. Construct explicit cycles
    cycles = []
    
    for k in range(3, N + 1):
        if k <= 21:
            # Cycle: 0 -> 1 -> ... -> k-1 -> 0
            # Chord (0, k-1) is needed. k-1 is in 2..20.
            cycle = list(range(0, k))
            # Check: 0, 1, ..., k-1
            # Edges: (0,1), ..., (k-2, k-1), (k-1, 0)
            # (k-1, 0) is the chord.
            cycles.append(cycle)
        else:
            # Cycle: 0 -> m -> m+1 -> ... -> 39 -> 0
            # Length is 1 + (39 - m + 1) = 41 - m
            # We want length k, so 41 - m = k => m = 41 - k
            m = 41 - k
            # m will be in range 20..2 (since k is 22..40)
            # For k=22, m=19. For k=40, m=1 (but k=40 is handled by base cycle? No, k=40 uses m=1? 
            # Wait, for k=40, the cycle is the Hamiltonian cycle.
            # My formula for k > 21:
            # If k=40, m=1. Chord (0,1) is the base edge.
            # The cycle 0->1->...->39->0 is the Hamiltonian cycle.
            # So we can use the same logic if we treat (0,1) as available.
            # But the list of vertices would be 0, 1, ..., 39.
            # Let's check the vertex list construction for k=40.
            # m=1.
            # cycle = [0] + list(range(m, N))
            # [0] + [1, ..., 39] = [0, 1, ..., 39].
            # This is correct.
            
            # However, we must ensure vertices are distinct.
            # 0 is not in range(m, N) if m >= 1.
            # So distinct.
            
            cycle = [0] + list(range(m, N))
            cycles.append(cycle)
            
    edge_list = list(edges)
    return {'edges': edge_list, 'cycles': cycles}

if __name__ == "__main__":
    res = construct()
    edges = set(tuple(sorted(e)) for e in res['edges'])
    valid = True
    for idx, c in enumerate(res['cycles']):
        if len(c) != idx + 3: # idx 0 is len 3
             # Actually cycles list index 0 corresponds to k=3
             pass
        for i in range(len(c)):
            u, v = c[i], c[(i+1)%len(c)]
            if (u,v) not in edges and (v,u) not in edges:
                print(f"Missing edge {u}-{v} for cycle {c} at index {idx}")
                valid = False
    if valid:
        print(f"Edges: {len(res['edges'])}, Score: {len(res['edges'])-40}")
    else:
        print("Invalid cycles found")
