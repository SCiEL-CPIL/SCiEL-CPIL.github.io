import numpy as np
import time
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    DEADLINE = 600.0

    # Known valid solution: all 220 weight-2 vectors (norm²=2)
    # For any two distinct weight-2 vectors, dot product ≤ 1, so dist² ≥ 2 = max_norm²
    def make_weight2_set():
        pts = []
        for i in range(d):
            for j in range(i + 1, d):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = s1
                        v[j] = s2
                        pts.append(v)
        return np.array(pts, dtype=np.int64)

    best_points = make_weight2_set()
    best_k = len(best_points)

    def validate(points):
        """Check if points form a valid configuration."""
        k = len(points)
        if k < 2:
            return False
        norms = np.sum(points**2, axis=1)
        max_norm = int(np.max(norms))
        for i in range(k):
            if np.all(points[i] == 0):
                return False
        for i in range(k):
            for j in range(i + 1, k):
                d2 = int(np.sum((points[i] - points[j])**2))
                if d2 < max_norm:
                    return False
        return True

    def generate_candidates(max_norm_sq, dim):
        """Generate all integer vectors with norm² ≤ max_norm_sq."""
        cands = []
        seen = set()
        
        def add(v):
            key = tuple(v)
            if key not in seen:
                seen.add(key)
                cands.append(v)
        
        # Generate by number of non-zero entries and their values
        # For efficiency, limit to small norms
        for nz in range(1, min(max_norm_sq + 1, 6)):
            for indices in combinations(range(dim), nz):
                # Generate all sign/value assignments
                if nz == 1:
                    for val in range(1, int(np.sqrt(max_norm_sq)) + 1):
                        if val * val <= max_norm_sq:
                            for sign in [1, -1]:
                                v = np.zeros(dim, dtype=np.int64)
                                v[indices[0]] = sign * val
                                add(v)
                elif nz == 2:
                    for v1 in range(1, int(np.sqrt(max_norm_sq)) + 1):
                        for v2 in range(1, int(np.sqrt(max_norm_sq - v1*v1)) + 1):
                            if v1*v1 + v2*v2 <= max_norm_sq:
                                for s1 in [1, -1]:
                                    for s2 in [1, -1]:
                                        v = np.zeros(dim, dtype=np.int64)
                                        v[indices[0]] = s1 * v1
                                        v[indices[1]] = s2 * v2
                                        add(v)
                elif nz == 3:
                    for v1 in range(1, int(np.sqrt(max_norm_sq)) + 1):
                        for v2 in range(1, int(np.sqrt(max_norm_sq - v1*v1)) + 1):
                            rem = max_norm_sq - v1*v1 - v2*v2
                            if rem < 1:
                                continue
                            for v3 in range(1, int(np.sqrt(rem)) + 1):
                                if v1*v1 + v2*v2 + v3*v3 <= max_norm_sq:
                                    for s1 in [1, -1]:
                                        for s2 in [1, -1]:
                                            for s3 in [1, -1]:
                                                v = np.zeros(dim, dtype=np.int64)
                                                v[indices[0]] = s1 * v1
                                                v[indices[1]] = s2 * v2
                                                v[indices[2]] = s3 * v3
                                                add(v)
                elif nz == 4:
                    for v1 in range(1, 3):
                        for v2 in range(1, 3):
                            for v3 in range(1, 3):
                                rem = max_norm_sq - v1*v1 - v2*v2 - v3*v3
                                if rem < 1:
                                    continue
                                for v4 in range(1, int(np.sqrt(rem)) + 1):
                                    if v1*v1 + v2*v2 + v3*v3 + v4*v4 <= max_norm_sq:
                                        for s1 in [1, -1]:
                                            for s2 in [1, -1]:
                                                for s3 in [1, -1]:
                                                    for s4 in [1, -1]:
                                                        v = np.zeros(dim, dtype=np.int64)
                                                        v[indices[0]] = s1 * v1
                                                        v[indices[1]] = s2 * v2
                                                        v[indices[2]] = s3 * v3
                                                        v[indices[3]] = s4 * v4
                                                        add(v)
        
        return cands

    def greedy_select(candidates, norms_sq, max_M, order):
        """Greedy selection."""
        selected = []
        current_M = 0
        n_cands = len(candidates)
        
        for idx in order:
            if time.time() - start_time > DEADLINE - 30:
                break
            
            cand = candidates[idx]
            m = int(norms_sq[idx])
            new_M = max(current_M, m)
            
            if new_M > max_M:
                continue
            
            # Check compatibility
            compatible = True
            for s_idx in selected:
                s = candidates[s_idx]
                d2 = int(np.sum((cand - s)**2))
                if d2 < new_M:
                    compatible = False
                    break
            
            if compatible:
                selected.append(idx)
                current_M = new_M
        
        return selected

    # Try greedy with various M values and candidate pools
    np.random.seed(42)
    
    for max_M in [2, 3, 4, 5, 6, 8]:
        if time.time() - start_time > DEADLINE - 60:
            break
        
        cands = generate_candidates(max_M, d)
        if len(cands) == 0:
            continue
        
        cand_array = np.array(cands, dtype=np.int64)
        norms_sq = np.sum(cand_array**2, axis=1).astype(np.int64)
        n_cands = len(cand_array)
        
        # Try ascending norm order
        order = np.argsort(norms_sq)
        selected = greedy_select(cand_array, norms_sq, max_M, order)
        if len(selected) > best_k:
            best_k = len(selected)
            best_points = cand_array[selected]
            if not validate(best_points):
                best_points = make_weight2_set()
                best_k = len(best_points)
        
        # Try a few random orders
        for trial in range(3):
            if time.time() - start_time > DEADLINE - 60:
                break
            order = np.random.permutation(n_cands)
            selected = greedy_select(cand_array, norms_sq, max_M, order)
            if len(selected) > best_k:
                best_k = len(selected)
                best_points = cand_array[selected]
                if not validate(best_points):
                    best_points = make_weight2_set()
                    best_k = len(best_points)
    
    # Final validation
    if not validate(best_points):
        best_points = make_weight2_set()
        best_k = len(best_points)

    return best_points
