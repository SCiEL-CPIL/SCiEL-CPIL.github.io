import numpy as np
import time
from math import gcd

def normalize_line(A, B, C):
    if A == 0 and B == 0:
        return None
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        g = 1
    A //= g
    B //= g
    C //= g
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def count_incidences(points, lines):
    P = points.astype(np.int64)
    L = lines.astype(np.int64)
    vals = P[:,0,None]*L[None,:,0] + P[:,1,None]*L[None,:,1] + L[None,:,2]
    return int(np.sum(vals == 0))

def build_lines(v_params, h_params, dp_params, dm_params):
    """Build lines from 4 families: V, H, D+ (x-y=c), D- (x+y=c)"""
    lines = []
    for v in v_params:
        lines.append((1, 0, -v))
    for h in h_params:
        lines.append((0, 1, -h))
    for c in dp_params:
        lines.append((1, -1, -c))
    for c in dm_params:
        lines.append((1, 1, -c))
    while len(lines) < 27:
        lines.append((1, 0, -(len(lines))))
    return np.array(lines[:27], dtype=np.int64)

def compute_all_incidences(v_params, h_params, dp_params, dm_params, xmin, xmax, ymin, ymax):
    """Compute incidence count at every grid point in [xmin,xmax]x[ymin,ymax]"""
    v_set = set(v_params)
    h_set = set(h_params)
    dp_set = set(dp_params)
    dm_set = set(dm_params)
    
    xs = np.arange(xmin, xmax+1)
    ys = np.arange(ymin, ymax+1)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    inc = np.zeros(XX.shape, dtype=np.int64)
    inc[np.isin(XX, list(v_set))] += 1
    inc[np.isin(YY, list(h_set))] += 1
    inc[np.isin(XX - YY, list(dp_set))] += 1
    inc[np.isin(XX + YY, list(dm_set))] += 1
    
    # Return flat array of (x, y, inc) for all points with inc > 0
    mask = inc > 0
    pts = np.column_stack([XX[mask], YY[mask]])
    incs = inc[mask]
    return pts, incs

def select_top_points(pts, incs, k=54):
    """Select k points with highest incidence, breaking ties by (x,y)"""
    if len(pts) < k:
        return pts.copy(), int(incs.sum()) if len(incs) > 0 else 0
    # Sort by (-inc, x, y)
    order = np.lexsort((pts[:,1], pts[:,0], -incs))
    top_idx = order[:k]
    return pts[top_idx].copy(), int(incs[top_idx].sum())

def evaluate_config(v_params, h_params, dp_params, dm_params):
    """Evaluate a family configuration: return (lines, points, I)"""
    lines = build_lines(v_params, h_params, dp_params, dm_params)
    
    # Determine bounding box from all line intersections
    all_x = list(v_params) + [dp + h for dp in dp_params for h in h_params] + [dm - h for dm in dm_params for h in h_params]
    all_y = list(h_params) + [v - dp for v in v_params for dp in dp_params] + [dm - v for v in v_params for dm in dm_params]
    
    if not all_x or not all_y:
        all_x = list(range(-10, 11))
        all_y = list(range(-10, 11))
    
    xmin, xmax = min(all_x)-2, max(all_x)+2
    ymin, ymax = min(all_y)-2, max(all_y)+2
    
    # Cap the box size for performance
    if xmax - xmin > 60:
        mid = (xmin + xmax) // 2
        xmin, xmax = mid-30, mid+30
    if ymax - ymin > 60:
        mid = (ymin + ymax) // 2
        ymin, ymax = mid-30, mid+30
    
    pts, incs = compute_all_incidences(v_params, h_params, dp_params, dm_params, xmin, xmax, ymin, ymax)
    
    if len(pts) < 54:
        # Pad with nearby points
        existing = set(map(tuple, pts))
        extra = []
        for dx in range(xmin-5, xmax+6):
            for dy in range(ymin-5, ymax+6):
                if (dx, dy) not in existing:
                    extra.append((dx, dy))
                    if len(extra) >= 54 - len(pts):
                        break
            if len(extra) >= 54 - len(pts):
                break
        pts = np.vstack([pts, np.array(extra)])
        incs = np.concatenate([incs, np.zeros(len(extra), dtype=np.int64)])
    
    pts_sel, I = select_top_points(pts, incs, 54)
    return lines, pts_sel, I

def local_search(v_params, h_params, dp_params, dm_params, deadline, max_iters=2000):
    """Local search: perturb one parameter at a time"""
    best_lines, best_pts, best_I = evaluate_config(v_params, h_params, dp_params, dm_params)
    if best_lines is None:
        return None, None, 0
    
    v_params = list(v_params)
    h_params = list(h_params)
    dp_params = list(dp_params)
    dm_params = list(dm_params)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        # Try perturbing each parameter in each family
        for param_idx in range(len(v_params)):
            for delta in [-2, -1, 1, 2]:
                if time.time() > deadline:
                    break
                new_vp = list(v_params)
                new_vp[param_idx] += delta
                new_vp = sorted(new_vp)
                if len(set(new_vp)) != len(new_vp):
                    continue
                bl, bp, bi = evaluate_config(new_vp, h_params, dp_params, dm_params)
                if bi > best_I:
                    best_I = bi
                    best_lines = bl
                    best_pts = bp
                    v_params = new_vp
                    improved = True
                    break
        if time.time() > deadline:
            break
        for param_idx in range(len(h_params)):
            for delta in [-2, -1, 1, 2]:
                if time.time() > deadline:
                    break
                new_hp = list(h_params)
                new_hp[param_idx] += delta
                new_hp = sorted(new_hp)
                if len(set(new_hp)) != len(new_hp):
                    continue
                bl, bp, bi = evaluate_config(v_params, new_hp, dp_params, dm_params)
                if bi > best_I:
                    best_I = bi
                    best_lines = bl
                    best_pts = bp
                    h_params = new_hp
                    improved = True
                    break
        if time.time() > deadline:
            break
        for param_idx in range(len(dp_params)):
            for delta in [-2, -1, 1, 2]:
                if time.time() > deadline:
                    break
                new_dpp = list(dp_params)
                new_dpp[param_idx] += delta
                new_dpp = sorted(new_dpp)
                if len(set(new_dpp)) != len(new_dpp):
                    continue
                bl, bp, bi = evaluate_config(v_params, h_params, new_dpp, dm_params)
                if bi > best_I:
                    best_I = bi
                    best_lines = bl
                    best_pts = bp
                    dp_params = new_dpp
                    improved = True
                    break
        if time.time() > deadline:
            break
        for param_idx in range(len(dm_params)):
            for delta in [-2, -1, 1, 2]:
                if time.time() > deadline:
                    break
                new_dmp = list(dm_params)
                new_dmp[param_idx] += delta
                new_dmp = sorted(new_dmp)
                if len(set(new_dmp)) != len(new_dmp):
                    continue
                bl, bp, bi = evaluate_config(v_params, h_params, dp_params, new_dmp)
                if bi > best_I:
                    best_I = bi
                    best_lines = bl
                    best_pts = bp
                    dm_params = new_dmp
                    improved = True
                    break
        if not improved:
            break
    
    return best_lines, best_pts, best_I

def szemeredi_trotter_incidences():
    deadline = time.time() + 590.0
    
    best_I_global = -1
    best_lines_global = None
    best_pts_global = None
    
    # Strategy 1: Try various symmetric and near-symmetric configurations
    # 4 families: V(9), H(9), D+(9), D-(0) or V(8), H(8), D+(6), D-(5) etc.
    
    configs_to_try = []
    
    # Pure V/H/D+ (9+9+9)
    for offset in range(-5, 6):
        v = [offset + i for i in range(9)]
        h = [offset + i for i in range(9)]
        dp = [offset + i for i in range(9)]
        dm = []
        configs_to_try.append((v, h, dp, dm))
    
    # V/H/D+/D- (7+7+7+6)
    for offset in range(-5, 6):
        v = [offset + i for i in range(7)]
        h = [offset + i for i in range(7)]
        dp = [offset + i for i in range(7)]
        dm = [offset + i for i in range(6)]
        configs_to_try.append((v, h, dp, dm))
    
    # V/H/D+/D- (8+8+6+5)
    for offset in range(-4, 5):
        v = [offset + i for i in range(8)]
        h = [offset + i for i in range(8)]
        dp = [offset + i for i in range(6)]
        dm = [offset + i for i in range(5)]
        configs_to_try.append((v, h, dp, dm))
    
    # V/H/D+/D- (6+6+8+7)
    for offset in range(-4, 5):
        v = [offset + i for i in range(6)]
        h = [offset + i for i in range(6)]
        dp = [offset + i for i in range(8)]
        dm = [offset + i for i in range(7)]
        configs_to_try.append((v, h, dp, dm))
    
    # V/H/D+/D- (5+5+9+8)
    for offset in range(-4, 5):
        v = [offset + i for i in range(5)]
        h = [offset + i for i in range(5)]
        dp = [offset + i for i in range(9)]
        dm = [offset + i for i in range(8)]
        configs_to_try.append((v, h, dp, dm))
    
    # Asymmetric offsets
    for vo in range(-3, 4):
        for ho in range(-3, 4):
            for dpo in range(-3, 4):
                v = [vo + i for i in range(9)]
                h = [ho + i for i in range(9)]
                dp = [dpo + i for i in range(9)]
                dm = []
                configs_to_try.append((v, h, dp, dm))
    
    # Evaluate all configs, keep top ones for local search
    eval_results = []
    for v, h, dp, dm in configs_to_try:
        if time.time() > deadline - 60:
            break
        bl, bp, bi = evaluate_config(v, h, dp, dm)
        if bi > 0:
            eval_results.append((bi, v, h, dp, dm, bl, bp))
    
    eval_results.sort(key=lambda x: -x[0])
    
    # Take best from initial evaluation
    if eval_results:
        best_I_global = eval_results[0][0]
        best_lines_global = eval_results[0][5]
        best_pts_global = eval_results[0][6]
    
    # Strategy 2: Local search from top configurations
    top_n = min(10, len(eval_results))
    for i in range(top_n):
        if time.time() > deadline - 30:
            break
        bi, v, h, dp, dm, bl, bp = eval_results[i]
        bl2, bp2, bi2 = local_search(v, h, dp, dm, deadline - 15, max_iters=500)
        if bi2 > best_I_global:
            best_I_global = bi2
            best_lines_global = bl2
            best_pts_global = bp2
    
    # Strategy 3: Try more diverse random starting points
    import random
    random.seed(123)
    for _ in range(30):
        if time.time() > deadline - 15:
            break
        # Random family sizes that sum to 27
        sizes = [random.randint(3, 12) for _ in range(4)]
        total = sum(sizes)
        # Adjust to sum to 27
        while total > 27:
            idx = random.randint(0, 3)
            if sizes[idx] > 1:
                sizes[idx] -= 1
                total -= 1
        while total < 27:
            idx = random.randint(0, 3)
            sizes[idx] += 1
            total += 1
        
        v = sorted(random.sample(range(-15, 16), sizes[0]))
        h = sorted(random.sample(range(-15, 16), sizes[1]))
        dp = sorted(random.sample(range(-15, 16), sizes[2]))
        dm = sorted(random.sample(range(-15, 16), sizes[3]))
        
        bl, bp, bi = evaluate_config(v, h, dp, dm)
        if bi > best_I_global:
            best_I_global = bi
            best_lines_global = bl
            best_pts_global = bp
        
        # Quick local search
        if time.time() > deadline - 10:
            continue
        bl2, bp2, bi2 = local_search(v, h, dp, dm, deadline - 5, max_iters=100)
        if bi2 > best_I_global:
            best_I_global = bi2
            best_lines_global = bl2
            best_pts_global = bp2
    
    # Fallback
    if best_pts_global is None:
        v = list(range(9))
        h = list(range(9))
        dp = list(range(9))
        dm = []
        best_lines_global, best_pts_global, best_I_global = evaluate_config(v, h, dp, dm)
    
    # Final validation
    # Ensure 54 unique points
    uniq, idx = np.unique(best_pts_global, axis=0, return_index=True)
    if len(uniq) < 54:
        existing = set(map(tuple, uniq))
        for dx in range(-20, 21):
            for dy in range(-20, 21):
                if len(uniq) >= 54:
                    break
                if (dx, dy) not in existing:
                    uniq = np.vstack([uniq, [[dx, dy]]])
                    existing.add((dx, dy))
            if len(uniq) >= 54:
                break
        best_pts_global = uniq[:54]
    else:
        best_pts_global = uniq[idx]
    
    # Ensure 27 unique valid lines
    ul = []
    for l in best_lines_global:
        nl = normalize_line(int(l[0]), int(l[1]), int(l[2]))
        if nl and nl not in ul:
            ul.append(nl)
    while len(ul) < 27:
        ul.append(normalize_line(1, 0, len(ul)))
    best_lines_global = np.array(ul[:27], dtype=np.int64)
    
    return best_pts_global, best_lines_global
