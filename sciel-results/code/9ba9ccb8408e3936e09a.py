import time
import itertools
import random

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    n = P['n']
    
    # The graph is a circulant graph C_21(1, 2, 5).
    # We can construct 3 edge-disjoint Hamiltonian cycles using steps 1, 2, and 5.
    
    # Cycle 1: Step 1
    c1 = [int(i) for i in range(n)]
    
    # Cycle 2: Step 2
    c2 = []
    curr = 0
    for _ in range(n):
        c2.append(curr)
        curr = (curr + 2) % n
        
    # Cycle 3: Step 5
    c3 = []
    curr = 0
    for _ in range(n):
        c3.append(curr)
        curr = (curr + 5) % n
        
    # Verify validity
    def validate(cycles):
        if not cycles:
            return False
        vset = set(cycles[0])
        if len(vset) != n:
            return False
        seen_edges = set()
        for c in cycles:
            if set(c) != vset:
                return False
            if len(c) != n:
                return False
            # Check distinct vertices
            if len(set(c)) != n:
                return False
            for i in range(len(c)):
                u = c[i]
                v = c[(i+1)%len(c)]
                key = (min(u,v), max(u,v))
                if key in seen_edges:
                    return False
                seen_edges.add(key)
                # Check if edge exists in P
                edge_exists = False
                for e in P['edges']:
                    if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                        edge_exists = True
                        break
                if not edge_exists:
                    return False
        return True

    cycles = [c1, c2, c3]
    
    if validate(cycles):
        # Ensure types are strictly Python int
        return [list(map(int, c)) for c in cycles]
    else:
        # Fallback: This should not happen for this specific graph, 
        # but we return an empty list or try a simple search if time permits.
        # Given the constraints and the known structure, the constructive solution is robust.
        return []

# EVOLVE-BLOCK-END
