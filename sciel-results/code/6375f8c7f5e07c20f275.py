import time

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    # Build edge set for fast lookup
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add((min(u,v), max(u,v)))
    
    def valid_cycle(cyc):
        # distinct vertices, size >= 3, all consecutive pairs (incl last->first) are edges
        if len(cyc) < 3:
            return False
        if len(set(cyc)) != len(cyc):
            return False
        for i in range(len(cyc)):
            a = cyc[i]
            b = cyc[(i+1) % len(cyc)]
            if (min(a,b), max(a,b)) not in edge_set:
                return False
        return True
    
    # The graph is a 6-regular circulant graph on 21 vertices.
    # Max number of edge-disjoint Hamiltonian cycles is 6/2 = 3.
    # We construct them using steps 1, 2, 5.
    # Step k generates edges (i, i+k) and (i, i-k).
    # Step 1: diffs 1, 20
    # Step 2: diffs 2, 19
    # Step 5: diffs 5, 16
    # These are disjoint sets of edge differences.
    
    cycles = []
    for k in (1, 2, 5):
        cyc = []
        cur = 0
        for _ in range(n):
            cyc.append(cur)
            cur = (cur + k) % n
        if valid_cycle(cyc):
            cycles.append(cyc)
    
    # Verify edge-disjointness
    used = set()
    ok = True
    for cyc in cycles:
        for i in range(len(cyc)):
            a = cyc[i]
            b = cyc[(i+1) % len(cyc)]
            e = (min(a,b), max(a,b))
            if e in used:
                ok = False
                break
            used.add(e)
        if not ok:
            break
    
    if ok and len(cycles) >= 1:
        return [list(c) for c in cycles[:10]]
    
    # Fallback: return the seed cycle
    return [list(range(n))]
# EVOLVE-BLOCK-END
