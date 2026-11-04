import numpy as np
import itertools
import random

D = 11

def construct():
    # Generate all 4-subsets of range(11)
    all_sets = list(itertools.combinations(range(D), 4))
    
    best_vectors = []
    best_count = 0
    
    # Try greedy with a few random shuffles to find a good packing
    for _ in range(10):
        random.shuffle(all_sets)
        used_triples = set()
        selected_sets = []
        
        for s in all_sets:
            # Generate triples of s
            triples = list(itertools.combinations(s, 3))
            # Check if any triple is already used
            if any(t in used_triples for t in triples):
                continue
            
            # If no conflict, add to used and selected
            for t in triples:
                used_triples.add(t)
            selected_sets.append(s)
        
        # Generate vectors for this selection
        vectors = []
        for s in selected_sets:
            i, j, k, l = s
            for si in (1, -1):
                for sj in (1, -1):
                    for sk in (1, -1):
                        for sl in (1, -1):
                            v = [0] * D
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            v[l] = sl
                            vectors.append(v)
        
        if len(vectors) > best_count:
            best_count = len(vectors)
            best_vectors = vectors
            
    return np.array(best_vectors, dtype=np.int64)
