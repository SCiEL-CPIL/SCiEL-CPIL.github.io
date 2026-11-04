import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    # Class 1: Vectors with two +/- 2 and rest 0. Norm^2 = 8.
    # There are C(11,2) * 4 = 55 * 4 = 220 such vectors.
    class1 = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = 2 * si
                v[j] = 2 * sj
                class1.append(v)
    
    # Class 2: Vectors with four +/- 1 and rest 0. Norm^2 = 4.
    # Wait, norm^2 = 4 is NOT sqrt(8).
    # My previous analysis: (1,1,1,1) has norm^2 = 4.
    # (2,2,0,0) has norm^2 = 8.
    # So Class 1 has norm sqrt(8). Class 2 has norm 2.
    # If we mix them, max norm is sqrt(8).
    # Class 2 vectors must have dist >= sqrt(8) from Class 1.
    # And dist >= sqrt(8) from each other? No, dist >= max_norm.
    # If we include Class 2 (norm 2), max norm is still sqrt(8) (from Class 1).
    # So ALL pairs must have dist^2 >= 8.
    
    # Let's check dist between Class 2 vectors.
    # b1, b2 in Class 2. Norm^2 = 4.
    # Dist^2 = 4 + 4 - 2 b1.b2 = 8 - 2 b1.b2.
    # Need 8 - 2 b1.b2 >= 8 => b1.b2 <= 0.
    # So any two Class 2 vectors must have non-positive inner product.
    # And dist between Class 1 (a) and Class 2 (b).
    # Norm a^2 = 8, Norm b^2 = 4.
    # Dist^2 = 8 + 4 - 2 a.b = 12 - 2 a.b.
    # Need 12 - 2 a.b >= 8 => 2 a.b <= 4 => a.b <= 2.
    
    # So we want to add as many Class 2 vectors as possible to the 220 Class 1 vectors,
    # such that:
    # 1. For any b in B, a.b <= 2 for all a in A.
    # 2. For any b1, b2 in B, b1.b2 <= 0.
    
    # Let's generate Class 2 vectors.
    class2 = []
    for indices in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, sign in zip(indices, signs):
                v[idx] = sign
            class2.append(v)
            
    class2 = np.array(class2)
    class1 = np.array(class1)
    
    # Check compatibility of class2 with class1
    # a.b <= 2
    # We can vectorize this check.
    # But 5280 * 220 is ~1M ops. Fast.
    
    # First, filter class2 vectors that are compatible with ALL class1 vectors.
    # a.b is computed for all pairs.
    # dot = class2 @ class1.T
    # valid_mask = np.all(dot <= 2, axis=1)
    
    # This might be memory heavy if done at once, but 5280x220 is small.
    dot = class2 @ class1.T
    valid_c2 = np.where(np.all(dot <= 2, axis=1))[0]
    class2_valid = class2[valid_c2]
    
    # Now we need to select a subset of class2_valid such that b1.b2 <= 0 for all pairs.
    # This is a max independent set in a graph where edges connect incompatible vectors.
    # Or max clique in compatibility graph.
    # Greedy approach:
    selected_c2 = []
    # Sort by some heuristic? Random?
    # To maximize count, maybe sort by number of conflicts?
    
    # Compute pairwise dot products of class2_valid
    # 5280 might be too many for O(N^2) in Python if not vectorized.
    # But we can do it iteratively.
    
    remaining_c2 = list(range(len(class2_valid)))
    # Shuffle to avoid bias?
    np.random.shuffle(remaining_c2)
    
    for idx in remaining_c2:
        if time.time() > deadline:
            break
        b = class2_valid[idx]
        compatible = True
        for sel_idx in selected_c2:
            if np.dot(b, class2_valid[sel_idx]) > 0:
                compatible = False
                break
        if compatible:
            selected_c2.append(idx)
            
    # Construct final array
    final_vectors = np.vstack([class1, class2_valid[selected_c2]])
    
    return final_vectors
