import numpy as np
from itertools import combinations
import time

def _validate(pts):
    pts = np.asarray(pts, dtype=np.int64)
    if pts.ndim != 2 or pts.shape[1] != 11:
        return False
    k = pts.shape[0]
    if k < 2:
        return False
    if np.any((pts == 0).all(axis=1)):
        return False
    norms = np.einsum('ij,ij->i', pts, pts)
    maxn = int(norms.max())
    for i in range(k):
        for j in range(i+1, k):
            d = pts[i] - pts[j]
            dsq = int(np.dot(d, d))
            if dsq < maxn:
                return False
    return True

def kissing_number11() -> np.ndarray:
    deadline = time.time() + 540
    
    points = []
    
    # Type 3: +/- 2 in two positions
    for i in range(11):
        for j in range(i+1, 11):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(11, dtype=np.int64)
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    points.append(v)
                    
    # Type 2: +/- 1 in four positions
    # Need 4-subsets with pairwise intersection <= 2
    # Greedy packing
    all_4_subsets = list(combinations(range(11), 4))
    # To maximize count, we can just iterate and check
    # There are 330 subsets.
    # We want ~41.
    selected_subsets = []
    for sub in all_4_subsets:
        ok = True
        for s in selected_subsets:
            inter = len(set(sub) & set(s))
            if inter > 2:
                ok = False
                break
        if ok:
            selected_subsets.append(sub)
            
    for sub in selected_subsets:
        for signs in range(16):
            v = np.zeros(11, dtype=np.int64)
            for k, idx in enumerate(sub):
                v[idx] = 1 if (signs >> k) & 1 == 0 else -1
            points.append(v)
            
    pts = np.array(points)
    
    if _validate(pts):
        return pts
    else:
        # Fallback to m=2 set
        points2 = []
        for i in range(11):
            for j in range(i+1, 11):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(11, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        points2.append(v)
        return np.array(points2)
