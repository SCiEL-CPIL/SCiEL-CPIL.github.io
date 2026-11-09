import numpy as np
import time
import random

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 580  # Safety margin for 630s limit
    
    # Generate all norm^2 = 2 vectors: two +/-1 entries, rest 0
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
    best_size = len(base_set)
    best_set = base_set[:]
    
    # Try norm^2=3 vectors with fast greedy MIS
    if time.time() < deadline - 80:
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
        
        n3 = len(norm3_vectors)  # 1320
        arr3 = np.array(norm3_vectors, dtype=np.int8)
        dots = arr3 @ arr3.T
        conflict = (dots >= 2).astype(np.uint8)
        np.fill_diagonal(conflict, 0)
        
        # Fast greedy: iterate in various orders, mark conflicts
        for trial in range(40):
            if time.time() > deadline - 50:
                break
            
            if trial == 0:
                order = np.arange(n3)
            elif trial == 1:
                order = np.argsort(conflict.sum(axis=1))
            elif trial == 2:
                order = np.argsort(-conflict.sum(axis=1))
            else:
                order = np.random.permutation(n3)
            
            available = np.ones(n3, dtype=bool)
            selected = []
            for idx in order:
                if not available[idx]:
                    continue
                selected.append(idx)
                available[conflict[idx] == 1] = False
                available[idx] = False
            
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = [norm3_vectors[i] for i in selected]
    
    # Try mixing: start with subset of norm^2=2, add compatible norm^2=3
    if time.time() < deadline - 60:
        # For a norm^2=3 vector v, it's compatible with norm^2=2 vector u iff:
        # dist^2 = 3 + 2 - 2*u.v = 5 - 2*u.v >= 3 (max_norm=3) => u.v <= 1
        # Since u.v is integer, u.v <= 1
        # u.v >= -2 (min when opposite signs on shared coords)
        
        # Try: keep all norm^2=2 that are compatible with a given norm^2=3 vector
        # This is complex; instead try greedy on combined pool with norm-dependent conflict
        
        # Combined pool: all norm^2=2 and norm^2=3 vectors
        # Conflict between u (norm^2=a) and v (norm^2=b): dist^2 = a+b-2*u.v >= max(a,b)
        # If a=b=2: 4-2*u.v >= 2 => u.v <= 1 (always true for distinct, since max u.v=1)
        # If a=b=3: 6-2*u.v >= 3 => u.v <= 1
        # If a=2, b=3: 5-2*u.v >= 3 => u.v <= 1
        # If a=3, b=2: same as above
        
        # So the conflict rule is: u.v >= 2 for same-norm, u.v >= 2 for cross-norm too?
        # Wait: for a=2, b=3: need 5-2*u.v >= 3, so u.v <= 1. Conflict if u.v >= 2.
        # For a=2, b=2: need 4-2*u.v >= 2, so u.v <= 1. Conflict if u.v >= 2.
        # For a=3, b=3: need 6-2*u.v >= 3, so u.v <= 1.5, i.e., u.v <= 1. Conflict if u.v >= 2.
        
        # So in all cases, conflict iff u.v >= 2!
        
        combined = norm2_vectors + norm3_vectors
        n_combined = len(combined)  # 220 + 1320 = 1540
        arr_combined = np.array(combined, dtype=np.int8)
        dots_c = arr_combined @ arr_combined.T
        conflict_c = (dots_c >= 2).astype(np.uint8)
        np.fill_diagonal(conflict_c, 0)
        
        for trial in range(20):
            if time.time() > deadline - 50:
                break
            
            if trial == 0:
                order = np.arange(n_combined)
            elif trial == 1:
                order = np.argsort(conflict_c.sum(axis=1))
            else:
                order = np.random.permutation(n_combined)
            
            available = np.ones(n_combined, dtype=bool)
            selected = []
            for idx in order:
                if not available[idx]:
                    continue
                selected.append(idx)
                available[conflict_c[idx] == 1] = False
                available[idx] = False
            
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = [combined[i] for i in selected]
    
    # Local refinement: try to add more vectors from the pool
    if best_size >= 2 and time.time() < deadline - 30:
        current_set = set(tuple(v) for v in best_set)
        all_candidates = norm2_vectors + norm3_vectors
        random.shuffle(all_candidates)
        
        max_norm = max(int(np.dot(v, v)) for v in best_set)
        improved = True
        while improved and time.time() < deadline - 30:
            improved = False
            for cand in all_candidates:
                if time.time() > deadline - 30:
                    break
                if tuple(cand) in current_set:
                    continue
                compatible = True
                for r in best_set:
                    d2 = int(np.dot(cand - r, cand - r))
                    if d2 < max_norm:
                        compatible = False
                        break
                if compatible:
                    best_set.append(cand)
                    best_size += 1
                    current_set.add(tuple(cand))
                    max_norm = max(max_norm, int(np.dot(cand, cand)))
                    improved = True
                    break
    
    # Verify and return
    if best_size >= 2:
        result = np.array(best_set, dtype=np.int64)
        result = np.around(result).astype(np.int64)
        
        max_norm = int(np.max(np.sum(result**2, axis=1)))
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
    
    # Fallback
    result = np.array(base_set, dtype=np.int64)
    return np.around(result).astype(np.int64)
