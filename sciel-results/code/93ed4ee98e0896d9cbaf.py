import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    k = 3  # alpha^2 = 3 in F_7
    
    # Field operations on F_49 = F_7[alpha] where alpha^2 = 3
    def field_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def field_sub(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)
    
    def field_mul(a, b):
        return ((a[0]*b[0] + a[1]*b[1]*k) % p, (a[0]*b[1] + a[1]*b[0]) % p)
    
    # Enumerate all points in F_49^2
    all_points = []
    pt_to_idx = {}
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    pt = ((xa, xb), (ya, yb))
                    pt_to_idx[pt] = len(all_points)
                    all_points.append(pt)
    
    n_all = len(all_points)  # 2401
    
    # Enumerate the Hermitian unital U: {(x,y) : Trace(x) + Norm(y) = 0}
    # Trace((a,b)) = 2a, Norm((a,b)) = a^2 - b^2*k
    U_points = []
    U_indices = []
    for i, pt in enumerate(all_points):
        xa, xb = pt[0]
        ya, yb = pt[1]
        if (2*xa + ya*ya - yb*yb*k) % p == 0:
            U_points.append(pt)
            U_indices.append(i)
    
    n_U = len(U_points)
    
    # Special y-coordinate set Y: y_a is a non-zero square mod p
    y_special = set()
    for u in range(1, p):
        for v in range(p):
            y_special.add((u*u % p, v))
    
    # Base excluded set: points in U where y is NOT in y_special
    base_excluded_U_idx = set()
    for i, pt in enumerate(U_points):
        ya, yb = pt[1]
        if (ya, yb) not in y_special:
            base_excluded_U_idx.add(i)
    
    # Directions: (1, m) for m in F_49, and (0, 1)
    slopes = [(ma, mb) for ma in range(p) for mb in range(p)]
    n_dirs = 50
    
    # Precompute for each point and each direction, which U points are on the line
    # pt_U_on_line[i][d] = set of U indices on the line through point i in direction d
    pt_U_on_line = [None] * n_all
    for i in range(n_all):
        pt_U_on_line[i] = [set() for _ in range(n_dirs)]
        pt = all_points[i]
        for d in range(n_dirs):
            if d < 49:
                m = slopes[d]
                for ta in range(p):
                    for tb in range(p):
                        if ta == 0 and tb == 0:
                            continue  # skip t=0
                        t = (ta, tb)
                        new_x = field_add(pt[0], t)
                        mt = field_mul(m, t)
                        new_y = field_add(pt[1], mt)
                        other = (new_x, new_y)
                        j = pt_to_idx[other]
                        if j in pt_to_idx and any(all_points[j] == U_points[u] for u in range(n_U)):
                            # Check if other is in U
                            for u in range(n_U):
                                if U_points[u] == other:
                                    pt_U_on_line[i][d].add(u)
                                    break
            else:
                # Vertical direction (0,1)
                for ta in range(p):
                    for tb in range(p):
                        if ta == 0 and tb == 0:
                            continue
                        t = (ta, tb)
                        new_y = field_add(pt[1], t)
                        other = (pt[0], new_y)
                        j = pt_to_idx[other]
                        for u in range(n_U):
                            if U_points[u] == other:
                                pt_U_on_line[i][d].add(u)
                                break
    
    # Build initial excluded set S (as set of U indices)
    S = set(base_excluded_U_idx)
    
    # Precompute for efficiency: for each U index, which points have this U point on their line in each direction
    # u_line_membership[u][d] = set of point indices i such that u is in pt_U_on_line[i][d]
    u_line_membership = [None] * n_U
    for u in range(n_U):
        u_line_membership[u] = [set() for _ in range(n_dirs)]
    
    for i in range(n_all):
        for d in range(n_dirs):
            for u in pt_U_on_line[i][d]:
                u_line_membership[u][d].add(i)
    
    # For each U point u and direction d, precompute which other U points are on the same line
    u_on_line = [None] * n_U
    for u in range(n_U):
        u_on_line[u] = [set() for _ in range(n_dirs)]
        pt = U_points[u]
        i = U_indices[u]
        for d in range(n_dirs):
            for u2 in pt_U_on_line[i][d]:
                if u2 != u:
                    u_on_line[u][d].add(u2)
    
    def check_validity(S_set):
        """Check if S (set of U indices) is a valid excluded set."""
        # Condition 1: for each s in S, exists d with u_on_line[s][d] ∩ S_set = ∅
        for s in S_set:
            has_iso = False
            for d in range(n_dirs):
                if not u_on_line[s][d].intersection(S_set):
                    has_iso = True
                    break
            if not has_iso:
                return False
        
        # Condition 2: for each x not in S, exists d with pt_U_on_line[x][d] ∩ S_set = ∅
        for x in range(n_all):
            # Check if x is in S
            in_S = False
            if x in pt_to_idx:
                pt = all_points[x]
                for u in range(n_U):
                    if U_points[u] == pt and u in S_set:
                        in_S = True
                        break
            if in_S:
                continue
            has_clean = False
            for d in range(n_dirs):
                if not pt_U_on_line[x][d].intersection(S_set):
                    has_clean = True
                    break
            if not has_clean:
                return False
        
        return True
    
    # Greedy expansion: try to add each U point not in S
    start_time = time.time()
    deadline = 580.0  # Leave margin before 630s kill
    
    # Sort U points not in S by some heuristic (e.g., number of conflicts)
    candidates = list(range(n_U))
    candidates.sort(key=lambda u: len(u_on_line[u][0]) if u not in S else -1)
    
    for u in candidates:
        if time.time() - start_time > deadline:
            break
        if u in S:
            continue
        S_new = S | {u}
        if check_validity(S_new):
            S = S_new
    
    # Build N: all points except those in S
    S_points = set(U_points[i] for i in S)
    points = []
    for pt in all_points:
        if pt not in S_points:
            xa, xb = pt[0]
            ya, yb = pt[1]
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)
