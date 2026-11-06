N = 40
S = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 31, 32, 33, 34, 35, 36, 37]

def construct():
    edges = []
    edge_set = set()
    
    def add_edge(u, v):
        if u == v: return
        e = (min(u, v), max(u, v))
        if e in edge_set: return
        edge_set.add(e)
        edges.append((u, v))

    for i in range(N):
        add_edge(i, (i + 1) % N)
        
    for s in S:
        add_edge(0, s)

    cycles = []
    for L in range(3, N + 1):
        if L == N:
            cycle = list(range(N))
        else:
            # Try A1
            s1 = L - 1
            if s1 in S:
                cycle = list(range(0, s1 + 1))
            # Try A2
            elif (41 - L) in S:
                s2 = 41 - L
                cycle = [0] + list(range(39, s2 - 1, -1))
            # Try B
            else:
                found = False
                for i in range(len(S)):
                    for j in range(i+1, len(S)):
                        if abs(S[i] - S[j]) + 2 == L:
                            u, v = S[i], S[j]
                            if u < v:
                                cycle = [0] + list(range(u, v + 1))
                            else:
                                cycle = [0] + list(range(v, u + 1))
                            found = True
                            break
                    if found: break
                if not found:
                    raise ValueError(f"No cycle for {L}")
            
            # Verify
            if len(cycle) != L:
                raise ValueError(f"Len mismatch {L}")
            if len(set(cycle)) != L:
                raise ValueError(f"Dup vertices {L}")
            for i in range(L):
                u = cycle[i]
                v = cycle[(i+1)%L]
                e = (min(u, v), max(u, v))
                if e not in edge_set:
                    raise ValueError(f"Edge missing {e}")
            
            cycles.append(cycle)
            
    return {'edges': edges, 'cycles': cycles}
