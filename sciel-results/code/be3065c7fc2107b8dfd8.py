import time
import itertools
import random

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    
    # The graph is a circulant graph C_21(1, 2, 5).
    # It has 63 edges, all of which are partitioned into 3 edge-disjoint
    # Hamiltonian cycles using steps 1, 2, and 5. This is optimal since
    # 3 * 21 = 63 = total edges.
    
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
    
    # Quick validation
    edge_set = set()
    for e in P['edges']:
        edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
    
    seen = set()
    valid = True
    vset = set(c1)
    if len(vset) != n:
        valid = False
    for c in [c1, c2, c3]:
        if set(c) != vset or len(c) != n or len(set(c)) != n:
            valid = False
            break
        for i in range(n):
            u, v = c[i], c[(i+1) % n]
            key = (min(u, v), max(u, v))
            if key in seen or key not in edge_set:
                valid = False
                break
            seen.add(key)
        if not valid:
            break
    
    if valid:
        return [c1, c2, c3]
    else:
        return []

# EVOLVE-BLOCK-END
