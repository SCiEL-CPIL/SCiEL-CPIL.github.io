import numpy as np
import time
import random

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 590  # Safety margin for 630s limit
    
    # Generate all norm^2 = 2 vectors: two +/-1 entries, rest 0
    # Count: C(11,2) * 4 = 55 * 4 = 220
    norm2_vectors = []
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    norm2_vectors.append(v)
    
    base_set = norm2_vectors[:]
    base_size = len(base_set)
    
    best_set = base_set[:]
    best_size = base_size
    
    # Try norm^2=3 vectors with randomized greedy MIS
    if time.time() < deadline - 300:
        norm3_vectors = []
        for i in range(d):
            for j in range(i + 1, d):
                for k in range(j + 1, d):
                    for s1 in [1, -1]:
                        for s2 in [1, -1]:
                            for s3 in [1, -1]:
                                v = np.zeros(d, dtype=np.int64)
                                v[i] = s1
                                v[j] = s2
                                v[k] = s3
                                norm3_vectors.append(v)
        
        n3 = len(norm3_vectors)
        arr3 = np.array(norm3_vectors, dtype=np.int64)
        dots = arr3 @ arr3.T
        conflict = dots >= 2
        np.fill_diagonal(conflict, False)
        
        # Randomized greedy: try many random orderings
        for trial in range(500):
            if time.time() > deadline - 200:
                break
            
            available = np.ones(n3, dtype=bool)
            selected = []
            
            # Random ordering
            order = np.random.permutation(n3)
            
            for idx in order:
                if not available[idx]:
                    continue
                selected.append(idx)
                # Remove this node and its neighbors
                to_remove = conflict[idx] | (np.arange(n3) == idx)
                available[to_remove] = False
            
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = [norm3_vectors[i] for i in selected]
    
    # Try norm^2=4 vectors with randomized greedy
    if time.time() < deadline - 150:
        norm4_vectors = []
        for i in range(d):
            v = np.zeros(d, dtype=np.int64)
            v[i] = 2
            norm4_vectors.append(v)
            norm4_vectors.append(-v)
        for i in range(d):
            for j in range(i + 1, d):
                for k in range(j + 1, d):
                    for l in range(k + 1, d):
                        for s1 in [1, -1]:
                            for s2 in [1, -1]:
                                for s3 in [1, -1]:
                                    for s4 in [1, -1]:
                                        v = np.zeros(d, dtype=np.int64)
                                        v[i] = s1
                                        v[j] = s2
                                        v[k] = s3
                                        v[l] = s4
                                        norm4_vectors.append(v)
        
        n4 = len(norm4_vectors)
        arr4 = np.array(norm4_vectors, dtype=np.int64)
        dots4 = arr4 @ arr4.T
        conflict4 = dots4 >= 3
        np.fill_diagonal(conflict4, False)
        
        for trial in range(100):
            if time.time() > deadline - 100:
                break
            
            available = np.ones(n4, dtype=bool)
            selected = []
            
            order = np.random.permutation(n4)
            
            for idx in order:
                if not available[idx]:
                    continue
                selected.append(idx)
                to_remove = conflict4[idx] | (np.arange(n4) == idx)
                available[to_remove] = False
            
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = [norm4_vectors[i] for i in selected]
    
    # Verify and return
    if best_size >= 2:
        result = np.array(best_set, dtype=np.int64)
        result = np.around(result).astype(np.int64)
        
        max_norm = np.max(np.sum(result**2, axis=1))
        valid = True
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                d2 = int(np.dot(result[i] - result[j], result[i] - result[j]))
                if d2 < max_norm:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            return result
    
    # Fallback: return the 220 norm^2=2 vectors
    result = np.array(base_set, dtype=np.int64)
    return np.around(result).astype(np.int64)
