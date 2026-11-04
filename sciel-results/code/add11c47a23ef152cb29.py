import numpy as np
import itertools
import time
import scipy.optimize as opt

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    D = 11
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    N_S = len(supports)
    
    # Map support to index
    support_to_idx = {s: i for i, s in enumerate(supports)}
    
    # Build conflict edges
    # Conflict if intersection size >= 3
    # Precompute masks for speed
    masks = []
    for s in supports:
        m = 0
        for i in s:
            m |= (1 << i)
        masks.append(m)
        
    edges = []
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = masks[i] & masks[j]
            # Count bits
            if bin(inter).count('1') >= 3:
                edges.append((i, j))
                
    # MILP Formulation
    # Maximize sum(x_i)
    # Subject to x_i + x_j <= 1 for all (i, j) in edges
    # x_i in {0, 1}
    
    c = -np.ones(N_S)
    
    # A_ub is sparse
    from scipy.sparse import lil_matrix
    A = lil_matrix((len(edges), N_S), dtype=np.float64)
    b = np.ones(len(edges))
    
    for k, (i, j) in enumerate(edges):
        A[k, i] = 1
        A[k, j] = 1
        
    # Convert to csc for milp
    A_csc = A.tocsr()
    
    constraints = opt.LinearConstraint(A_csc, None, 1)
    
    integrality = np.ones(N_S)
    bounds = [(0, 1)] * N_S
    
    try:
        res = opt.milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0: # Success
            x = res.x.astype(int)
            selected_indices = np.where(x == 1)[0]
        else:
            # Fallback to greedy if MILP fails
            # Simple greedy: sort by degree (all same), pick first, remove neighbors
            x = np.zeros(N_S, dtype=bool)
            remaining = set(range(N_S))
            # Build adjacency list
            adj = {i: set() for i in range(N_S)}
            for i, j in edges:
                adj[i].add(j)
                adj[j].add(i)
                
            # Greedy
            order = list(range(N_S))
            np.random.RandomState(42).shuffle(order)
            for v in order:
                if v in remaining:
                    x[v] = True
                    remaining.discard(v)
                    for u in adj[v]:
                        remaining.discard(u)
            selected_indices = np.where(x)[0]
            
    except Exception as e:
        # Fallback to greedy
        x = np.zeros(N_S, dtype=bool)
        remaining = set(range(N_S))
        adj = {i: set() for i in range(N_S)}
        for i, j in edges:
            adj[i].add(j)
            adj[j].add(i)
        order = list(range(N_S))
        np.random.RandomState(42).shuffle(order)
        for v in order:
            if v in remaining:
                x[v] = True
                remaining.discard(v)
                for u in adj[v]:
                    remaining.discard(u)
        selected_indices = np.where(x)[0]

    # Generate vectors
    vectors_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors_A.append(v2)
    vectors_A = np.array(vectors_A, dtype=np.int64)
    
    vectors_D = []
    for idx in selected_indices:
        support = supports[idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx_s, sign in zip(support, signs):
                v[idx_s] = sign
            vectors_D.append(v)
            
    vectors_D = np.array(vectors_D, dtype=np.int64)
    
    final_vectors = np.vstack([vectors_A, vectors_D])
    return final_vectors
