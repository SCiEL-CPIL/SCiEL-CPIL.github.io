# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = 600.0 - 30.0
    
    p = 7
    k = 3
    
    def fe_to_idx(a, b):
        return a + 7 * b
    def idx_to_fe(idx):
        return (idx % 7, idx // 7)
    def fe_add(i1, i2):
        a1,b1 = idx_to_fe(i1); a2,b2 = idx_to_fe(i2)
        return fe_to_idx((a1+a2)%p, (b1+b2)%p)
    def fe_mul(i1, i2):
        a1,b1 = idx_to_fe(i1); a2,b2 = idx_to_fe(i2)
        return fe_to_idx((a1*a2+k*b1*b2)%p, (a1*b2+b1*a2)%p)
    def pt_idx(xa,xb,ya,yb):
        return xa+7*xb+49*ya+343*yb
    def idx_pt(idx):
        return (idx%7, (idx//7)%7, (idx//49)%7, (idx//343)%7)
    
    n_pts, n_fe = 2401, 49
    
    add_t = np.zeros((n_fe,n_fe), dtype=np.int32)
    mul_t = np.zeros((n_fe,n_fe), dtype=np.int32)
    for i in range(n_fe):
        for j in range(n_fe):
            add_t[i,j] = fe_add(i,j)
            mul_t[i,j] = fe_mul(i,j)
    
    # Build lines
    lines = []
    for m in range(n_fe):
        for c in range(n_fe):
            pts = np.array([pt_idx(*idx_to_fe(x), *idx_to_fe(add_t[mul_t[m,x],c])) for x in range(n_fe)], dtype=np.int32)
            lines.append(pts)
    for x0 in range(n_fe):
        xa,xb = idx_to_fe(x0)
        pts = np.array([pt_idx(xa,xb,*idx_to_fe(y)) for y in range(n_fe)], dtype=np.int32)
        lines.append(pts)
    
    n_lines = len(lines)
    
    # point_lines
    point_lines = [[] for _ in range(n_pts)]
    for li in range(n_lines):
        for pt in lines[li]:
            point_lines[pt].append(li)
    
    in_N = np.ones(n_pts, dtype=bool)
    good = [set(point_lines[x]) for x in range(n_pts)]
    lines_thru = [set(point_lines[p]) for p in range(n_pts)]
    
    # Greedy removal - limit to first N points for speed
    max_check = min(n_pts, 500)  # limit for speed
    for p in range(max_check):
        if time.time() - start_time > deadline:
            break
        if not in_N[p]:
            continue
        pl = lines_thru[p]
        affected = set()
        for li in pl:
            for x in lines[li]:
                if x != p:
                    affected.add(x)
        rem = True
        for x in affected:
            if not (good[x] - pl):
                rem = False
                break
        if rem:
            in_N[p] = False
            for li in pl:
                for x in lines[li]:
                    if x != p and li in good[x]:
                        good[x].discard(li)
    
    ridx = np.where(in_N)[0]
    res = np.zeros((len(ridx),2,2), dtype=np.int64)
    for i,idx in enumerate(ridx):
        xa,xb,ya,yb = idx_pt(idx)
        res[i,0,0]=xa; res[i,0,1]=xb; res[i,1,0]=ya; res[i,1,1]=yb
    return res

# EVOLVE-BLOCK-END
