# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_q^2 with q = p^2, p = 7.
    Uses greedy removal with efficient critical-point checking.
    """
    p = 7
    q = p * p  # 49
    total = q * q  # 2401
    num_dirs = 50

    # Quadratic non-residue
    k = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k = c
            break

    # Point indexing
    def pt_to_idx(xa, xb, ya, yb):
        return (xa * q + xb) * q + (ya * q + yb)

    def idx_to_pt(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p*p)) % p
        xa = (idx // (p*p*p)) % p
        return (xa, xb, ya, yb)

    # Precompute lines
    line_other = np.zeros((total, num_dirs, q-1), dtype=np.int32)
    
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    pt_idx = pt_to_idx(xa, xb, ya, yb)
                    for ma in range(p):
                        for mb in range(p):
                            dir_idx = ma * p + mb
                            count = 0
                            for ta in range(p):
                                for tb in range(p):
                                    if ta == 0 and tb == 0:
                                        continue
                                    nxa = (xa + ta) % p
                                    nxb = (xb + tb) % p
                                    nya = (ya + ta*ma + tb*mb*k) % p
                                    nyb = (yb + ta*mb + tb*ma) % p
                                    nidx = pt_to_idx(nxa, nxb, nya, nyb)
                                    line_other[pt_idx, dir_idx, count] = nidx
                                    count += 1
                    dir_idx = 49
                    count = 0
                    for ta in range(p):
                        for tb in range(p):
                            if ta == 0 and tb == 0:
                                continue
                            nxa = xa
                            nxb = xb
                            nya = (ya + tb*k) % p
                            nyb = (yb + ta) % p
                            nidx = pt_to_idx(nxa, nxb, nya, nyb)
                            line_other[pt_idx, dir_idx, count] = nidx
                            count += 1

    # Build initial N from baseline
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
    
    in_N = np.ones(total, dtype=bool)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    idx = pt_to_idx(xa, xb, ya, yb)
                    in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        in_N[idx] = False

    # Compute good directions
    good_dirs = np.zeros((total, num_dirs), dtype=bool)
    for pt_idx in range(total):
        for d in range(num_dirs):
            others = line_other[pt_idx, d]
            if np.all(in_N[others]):
                good_dirs[pt_idx, d] = True
    
    good_count = np.sum(good_dirs, axis=1)
    
    start_time = time.time()
    deadline = start_time + 570
    
    # Greedy removal
    for pass_num in range(20):
        if time.time() > deadline - 30:
            break
        any_removed = False
        for y in range(total):
            if time.time() > deadline - 30:
                break
            if not in_N[y]:
                continue
            if good_count[y] == 0:
                continue
            
            # Check if removing y is safe
            can_remove = True
            for d in range(num_dirs):
                others = line_other[y, d]
                for x in others:
                    if x == y:
                        continue
                    if good_dirs[x, d] and good_count[x] <= 1:
                        can_remove = False
                        break
                if not can_remove:
                    break
            
            if can_remove:
                in_N[y] = False
                any_removed = True
                for d in range(num_dirs):
                    others = line_other[y, d]
                    for x in others:
                        if x == y:
                            continue
                        if good_dirs[x, d]:
                            good_dirs[x, d] = False
                            good_count[x] -= 1
        
        if not any_removed:
            break
    
    # Build output
    points = []
    for pt_idx in range(total):
        if in_N[pt_idx]:
            xa, xb, ya, yb = idx_to_pt(pt_idx)
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
