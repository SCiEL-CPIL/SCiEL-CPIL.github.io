import time

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 60.0
    
    # Base Hamiltonian cycle edges
    edges = set()
    for i in range(N):
        edges.add((min(i, (i+1)%N), max(i, (i+1)%N)))
        
    # Chords from 0 to k
    # K = {2, 4, ..., 38}
    K = list(range(2, 39, 2))
    for k in K:
        edges.add((0, k))
        
    # Convert to list for return
    edge_list = sorted(list(edges))
    edge_set = set(edge_list)
    
    # Generate cycles
    cycles = []
    
    # Pre-compute membership for speed
    K_set = set(K)
    
    for L in range(3, N+1):
        if L == N:
            # Hamiltonian cycle
            cycle = list(range(N))
        elif L - 1 in K_set:
            # Cycle 0 -> 1 -> ... -> L-1 -> 0
            cycle = list(range(L))
        elif N - L + 1 in K_set:
            # Cycle 0 -> N-1 -> ... -> N-L+1 -> 0
            k = N - L + 1
            cycle = [0] + list(range(N-1, k-1, -1))
        else:
            # Find a, b in K such that |a - b| = L - 2
            diff = L - 2
            found = False
            for a in K:
                b1 = a + diff
                b2 = a - diff
                if b1 in K_set:
                    # a < b1 (since diff > 0)
                    cycle = [0] + list(range(a, b1 + 1))
                    found = True
                    break
                if b2 in K_set:
                    # b2 < a (since diff > 0)
                    cycle = [0] + list(range(b2, a + 1))
                    found = True
                    break
            if not found:
                raise ValueError(f"Could not find cycle for length {L}")
        
        cycles.append(cycle)
        
    # Verification
    for c in cycles:
        if len(set(c)) != len(c):
            raise ValueError(f"Duplicate vertices in cycle {c}")
        for i in range(len(c)):
            u, v = c[i], c[(i+1)%len(c)]
            if (min(u,v), max(u,v)) not in edge_set:
                raise ValueError(f"Missing edge ({u},{v}) in cycle {c}")

    return {'edges': edge_list, 'cycles': cycles}
