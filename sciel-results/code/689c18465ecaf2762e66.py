import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    # Generate all weight-3 vectors
    # Indices are 0..10
    indices = range(D)
    triples = list(itertools.combinations(indices, 3))
    
    # We will select a subset of these triples such that
    # no two triples share exactly one index.
    
    selected_triples = []
    
    # Sort triples to ensure deterministic behavior and potentially better packing
    # Sorting by sum of indices or just lexicographical
    triples.sort()
    
    for tri in triples:
        if time.time() > deadline:
            break
            
        # Check compatibility with all currently selected triples
        compatible = True
        for sel_tri in selected_triples:
            # Calculate intersection size
            # Since they are tuples of sorted indices, we can use set or manual check
            # Manual check is faster for small tuples
            inter_count = 0
            for i in tri:
                if i in sel_tri:
                    inter_count += 1
            
            if inter_count == 1:
                compatible = False
                break
        
        if compatible:
            selected_triples.append(tri)
    
    # Construct the vectors
    # For each selected triple, add all 8 sign combinations
    vectors = []
    for tri in selected_triples:
        i, j, k = tri
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    vectors.append(v)
    
    if not vectors:
        # Fallback to seed if something went wrong or empty
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
    
    return np.array(vectors, dtype=np.int64)
