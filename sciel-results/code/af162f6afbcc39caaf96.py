import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    # Compute conflicts
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Build adjacency lists for sparse graph
    adj_list = [np.where(conflicts[i])[0] for i in range(N4)]
    
    best_size = 0
    best_mask = np.zeros(N4, dtype=bool)
    
    def greedy_color(P):
        """Greedy coloring of P. P is a list of vertex indices."""
        C = []
        uncolored = list(P)
        while uncolored:
            color_class = []
            remaining = []
            for v in uncolored:
                # v can join if not adjacent to any in color_class
                if not any(conflicts[v, u] for u in color_class):
                    color_class.append(v)
                else:
                    remaining.append(v)
            C.append(color_class)
            uncolored = remaining
        return C
    
    def tomita(P, X, C):
        nonlocal best_size, best_mask
        
        if len(P) == 0:
            if len(X) > best_size:
                best_size = len(X)
                best_mask = np.zeros(N4, dtype=bool)
                best_mask[X] = True
            return
        
        if len(X) + len(C) <= best_size:
            return
        
        for ci in range(len(C) - 1, -1, -1):
            if not C[ci]:
                continue
            v = C[ci][0]
            P.remove(v)
            C[ci].remove(v)
            
            new_P = [u for u in P if not conflicts[v, u]]
            new_X = X + [v]
            new_C = greedy_color(new_P)
            
            tomita(new_P, new_X, new_C)
            
            P.append(v)
            C[ci].append(v)
            
            if len(X) + len(P) <= best_size:
                return
    
    # Initial ordering by degree
    degrees = np.array([len(adj_list[i]) for i in range(N4)])
    order = np.argsort(degrees)
    
    P_init = list(order)
    C_init = greedy_color(P_init)
    tomita(P_init, [], C_init)
    
    result_v4 = vectors_v4[best_mask]
    final_vectors = np.vstack([result_v4, vectors_v1])
    return final_vectors
