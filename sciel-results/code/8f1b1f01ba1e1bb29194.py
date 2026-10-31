import numpy as np
import itertools

D = 11

def construct():
    # Generate all weight 4 vectors with entries +/- 1
    indices = list(range(D))
    vectors = []
    for comb in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            vectors.append(v)
    
    V = np.array(vectors)
    N = len(V)
    
    # Compute dot products
    dots = V @ V.T
    
    # Conflict if dot > 2
    # We want independent set where dot <= 2
    compat = (dots <= 2)
    np.fill_diagonal(compat, False)
    
    # Greedy Independent Set
    degrees = np.sum(~compat, axis=1)
    order = np.argsort(degrees)
    
    selected = []
    mask = np.ones(N, dtype=bool)
    
    for i in order:
        if mask[i]:
            selected.append(i)
            mask &= compat[i]
            
    result = V[selected]
    return result
