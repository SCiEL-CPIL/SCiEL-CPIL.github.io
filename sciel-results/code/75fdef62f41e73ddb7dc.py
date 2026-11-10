import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    deadline = time.time() + 600.0

    k_const = 3

    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)

    def sub(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)

    def mul(a, b):
        return ((a[0]*b[0] + a[1]*b[1]*k_const) % p,
                (a[0]*b[1] + a[1]*b[0]) % p)

    def pt_to_idx(pt):
        xa, xb = pt[0]
        ya, yb = pt[1]
        return xa * 343 + xb * 49 + ya * 7 + yb

    def idx_to_pt(idx):
        xa = idx // 343
        idx %= 343
        xb = idx // 49
        idx %= 49
        ya = idx // 7
        yb = idx % 7
        return [[xa, xb], [ya, yb]]

    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))
    directions.append(((0, 0), (1, 0)))
    
    num_dirs = len(directions)

    # Precompute line members for each point and direction?
    # line_points[x_idx][d_idx] = list of other points
    # This is 2401 * 50 lists of 48 ints.
    # Let's generate on the fly but cache? Or precompute a flat array.
    # 2401 * 50 * 48 = 5.7M ints. int32 is 23MB.
    
    line_others = np.zeros((2401, num_dirs, 48), dtype=np.int32)
    for x_idx in range(2401):
        x_pt = idx_to_pt(x_idx)
        px, py = x_pt[0], x_pt[1]
        for d_idx, (vx, vy) in enumerate(directions):
            for t_idx in range(48):
                # Map t_idx 0..47 to field elements t != 0
                # We need a mapping from 0..47 to F_49 \ {0}
                # F_49 has 49 elements. 0 is (0,0).
                # Let's just iterate all ta, tb and skip 0,0.
                pass # This is inefficient to fill precomputed array this way.
                
    # Let's just precompute the "affected points" for each p and d.
    # affected[p_idx][d_idx] = list of x_idx != p_idx on the line through p in dir d.
    
    affected = []
    for p_idx in range(2401):
        p_pt = idx_to_pt(p_idx)
        px, py = p_pt[0], p_pt[1]
        d_list = []
        for d_idx in range(num_dirs):
            vx, vy = directions[d_idx]
            pts = []
            for ta in range(p):
                for tb in range(p):
                    if ta == 0 and tb == 0: continue
                    tvx = mul((ta, tb), vx)
                    tvy = mul((ta, tb), vy)
                    nx = add(px, tvx)
                    ny = add(py, tvy)
                    n_idx = pt_to_idx([[nx[0], nx[1]], [ny[0], ny[1]]])
                    pts.append(n_idx)
            d_list.append(pts)
        affected.append(d_list)

    # Initialize S
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))

    S = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2 * xa + ya * ya - yb * yb * k_const) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        S.add(pt_to_idx([[xa, xb], [ya, yb]]))

    line_count = np.zeros((2401, num_dirs), dtype=np.int32)
    for s_idx in S:
        for d_idx in range(num_dirs):
            for n_idx in affected[s_idx][d_idx]:
                line_count[n_idx, d_idx] += 1

    zero_counts = np.sum(line_count == 0, axis=1)

    def is_valid():
        return np.all(zero_counts > 0)

    assert is_valid()

    best_S = set(S)
    best_size = len(S)
    
    # Local Search
    candidates = [i for i in range(2401) if i not in S]
    np.random.seed(42)
    
    while time.time() < deadline - 10:
        np.random.shuffle(candidates)
        for p_idx in candidates:
            if p_idx in S: continue
            # Check if adding p_idx is valid
            valid = True
            changes = []
            for d_idx in range(num_dirs):
                for n_idx in affected[p_idx][d_idx]:
                    if line_count[n_idx, d_idx] == 0:
                        if zero_counts[n_idx] == 1:
                            valid = False
                            break
                        else:
                            changes.append((n_idx, d_idx))
                if not valid: break
            if valid:
                S.add(p_idx)
                for n_idx, d_idx in changes:
                    line_count[n_idx, d_idx] += 1
                    zero_counts[n_idx] -= 1
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
                # Remove from candidates? Or just let it be skipped next time.
                # To speed up, we could remove it, but list removal is O(n).
                # We'll just check `if p_idx in S` at the start of the loop.
                
    S = best_S
    points = []
    for i in range(2401):
        if i not in S:
            points.append(idx_to_pt(i))
            
    return np.array(points, dtype=np.int64)
