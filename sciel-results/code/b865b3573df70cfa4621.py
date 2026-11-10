# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using greedy optimization with incremental validity checking.
    """
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    num_directions = 50
    
    def point_to_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb
    
    def idx_to_point(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p*p)) % p
        xa = (idx // (p*p*p)) % p
        return (xa, xb, ya, yb)
    
    # Precompute lines
    lines_other = np.zeros((total_points, num_directions, 48), dtype=np.int32)
    
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    idx = point_to_idx(xa, xb, ya, yb)
                    for ma in range(p):
                        for mb in range(p):
                            dir_idx = ma * p + mb
                            other = []
                            for ta in range(p):
                                for tb in range(p):
                                    if ta == 0 and tb == 0:
                                        continue
                                    nxa = (xa + ta) % p
                                    nxb = (xb + tb) % p
                                    nya = (ya + ta + tb * ma) % p
                                    nyb = (yb + ta * mb + tb) % p
                                    other.append(point_to_idx(nxa, nxb, nya, nyb))
                            for j in range(48):
                                lines_other[idx, dir_idx, j] = other[j]
                    dir_idx = 49
                    other = []
                    for ta in range(p):
                        for tb in range(p):
                            if ta == 0 and tb == 0:
                                continue
                            nxa = xa
                            nxb = xb
                            nya = (ya + ta) % p
                            nyb = (yb + tb) % p
                            other.append(point_to_idx(nxa, nxb, nya, nyb))
                    for j in range(48):
                        lines_other[idx, dir_idx, j] = other[j]
    
    # Precompute y_to_affected: for each y, list of (x, d) where y is in lines_other[x, d, :]
    y_to_affected = [[] for _ in range(total_points)]
    for x in range(total_points):
        for d in range(num_directions):
            for j in range(48):
                y = lines_other[x, d, j]
                y_to_affected[y].append((x, d))
    
    # State
    excluded = np.zeros(total_points, dtype=bool)
    
    # line_excluded_count[x, d] = number of excluded points in lines_other[x, d, :]
    line_excl_count = np.zeros((total_points, num_directions), dtype=np.int32)
    
    # cover_count[x] = number of directions d where line_excl_count[x, d] == 0
    cover_count = np.zeros(total_points, dtype=np.int32)
    
    def recompute_state():
        """Recompute line_excl_count and cover_count from scratch."""
        for x in range(total_points):
            for d in range(num_directions):
                cnt = 0
                for j in range(48):
                    if excluded[lines_other[x, d, j]]:
                        cnt += 1
                line_excl_count[x, d] = cnt
            cc = 0
            for d in range(num_directions):
                if line_excl_count[x, d] == 0:
                    cc += 1
            cover_count[x] = cc
    
    def add_excluded(y):
        """Add point y to excluded set, update state. Return True if valid."""
        excluded[y] = True
        min_cover_before = np.min(cover_count)
        
        for (x, d) in y_to_affected[y]:
            line_excl_count[x, d] += 1
            if line_excl_count[x, d] == 1:
                cover_count[x] -= 1
        
        if np.min(cover_count) <= 0:
            # Invalid, revert
            excluded[y] = False
            for (x, d) in y_to_affected[y]:
                line_excl_count[x, d] -= 1
                if line_excl_count[x, d] == 0:
                    cover_count[x] += 1
            return False
        return True
    
    def remove_excluded(y):
        """Remove point y from excluded set, update state."""
        excluded[y] = False
        for (x, d) in y_to_affected[y]:
            line_excl_count[x, d] -= 1
            if line_excl_count[x, d] == 0:
                cover_count[x] += 1
    
    # Initialize with baseline
    k_const = 3
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add((u*u % p, v))
    
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2*xa + ya*ya - yb*yb*k_const) % p == 0:
                        if (ya, yb) not in y_special:
                            excluded[point_to_idx(xa, xb, ya, yb)] = True
    
    recompute_state()
    
    # Verify
    if np.min(cover_count) <= 0:
        points = []
        for idx in range(total_points):
            xa, xb, ya, yb = idx_to_point(idx)
            points.append([[xa, xb], [ya, yb]])
        return np.array(points, dtype=np.int64)
    
    # Greedy: try to add more excluded points
    candidates = [i for i in range(total_points) if not excluded[i]]
    candidates.sort(key=lambda y: len(y_to_affected[y]))
    
    for y in candidates:
        if not excluded[y]:
            add_excluded(y)
    
    # Second pass
    candidates = [i for i in range(total_points) if not excluded[i]]
    random.shuffle(candidates)
    for y in candidates:
        if not excluded[y]:
            add_excluded(y)
    
    # Build result
    n_points = []
    for idx in range(total_points):
        if not excluded[idx]:
            xa, xb, ya, yb = idx_to_point(idx)
            n_points.append([[xa, xb], [ya, yb]])
    
    return np.array(n_points, dtype=np.int64)


# EVOLVE-BLOCK-END
