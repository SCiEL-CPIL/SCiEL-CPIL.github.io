import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 620  # safety margin under 630s wall-clock

    # Generate all norm-1 vectors (single ±1) and norm-2 vectors (two ±1)
    vecs = []
    # Norm-1 vectors: 22 total
    for i in range(d):
        for s in (1, -1):
            v = [0] * d
            v[i] = s
            vecs.append(tuple(v))
    # Norm-2 vectors: 220 total
    for i, j in combinations(range(d), 2):
        for s1 in (1, -1):
            for s2 in (1, -1):
                v = [0] * d
                v[i] = s1
                v[j] = s2
                vecs.append(tuple(v))
    
    n = len(vecs)
    arr = np.array(vecs, dtype=np.int64)
    
    # Max norm is 2 (norm-2 vectors)
    # Conflict if dist^2 < 2, i.e., dist^2 == 1
    # dist^2 = |a|^2 + |b|^2 - 2*a.b
    # For two norm-1: dist^2 = 2 - 2*dot, dot in {-1,0,1}, dist^2 in {0,2,4}
    # For norm-1 and norm-2: dist^2 = 3 - 2*dot, dot in {-1,0,1}, dist^2 in {1,3,5}
    # For two norm-2: dist^2 = 4 - 2*dot, dot in {-2,-1,0,1,2}, dist^2 in {0,2,4,6,8}
    # Conflict (dist^2 < 2) only when dist^2 == 1: norm-1 & norm-2 with dot=1
    
    # Compute conflict mask
    dots = arr @ arr.T
    norms = np.sum(arr * arr, axis=1)
    dist_sq = norms[:, None] + norms[None, :] - 2 * dots
    conflict_mask = (dist_sq < 2) & (dist_sq > 0)  # dist^2 == 1
    np.fill_diagonal(conflict_mask, False)
    
    # Find maximum independent set using greedy + local search
    # Greedy: pick vertices with fewest conflicts first
    deg = conflict_mask.sum(axis=1)
    order = np.argsort(deg)  # ascending degree
    
    selected = []
    selected_set = set()
    for idx in order:
        idx = int(idx)
        if time.time() > deadline - 30:
            break
        # Check if idx conflicts with any selected
        can_add = True
        for s in selected_set:
            if conflict_mask[idx, s]:
                can_add = False
                break
        if can_add:
            selected.append(idx)
            selected_set.add(idx)
    
    # Local search: try to swap to improve
    improved = True
    max_iter = 50
    for _ in range(max_iter):
        if time.time() > deadline - 30:
            break
        improved = False
        for a in list(selected):
            for b in range(n):
                if b in selected_set:
                    continue
                # Try swapping a with b
                can = True
                for c in selected_set:
                    if c == a:
                        continue
                    if conflict_mask[b, c]:
                        can = False
                        break
                if can:
                    # Check if this swap helps (reduces total conflicts)
                    old_conf = sum(1 for c in selected_set if c != a and conflict_mask[a, c])
                    new_conf = sum(1 for c in selected_set if c != a and conflict_mask[b, c])
                    if new_conf < old_conf:
                        selected_set.remove(a)
                        selected_set.add(b)
                        selected = list(selected_set)
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    
    points = [tuple(int(x) for x in arr[i]) for i in selected]
    
    # Validate
    if len(points) < 2:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
    
    max_norm = 0
    for p in points:
        nn = sum(x * x for x in p)
        if nn > max_norm:
            max_norm = nn
    
    valid = True
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d2 = sum((a - b) ** 2 for a, b in zip(points[i], points[j]))
            if d2 < max_norm:
                valid = False
                break
        if not valid:
            break
    
    if valid:
        result = np.array(points, dtype=np.int64)
        return np.around(result).astype(np.int64)
    
    # Fallback
    return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)

if __name__ == "__main__":
    res = kissing_number11()
    print("Size:", len(res))
