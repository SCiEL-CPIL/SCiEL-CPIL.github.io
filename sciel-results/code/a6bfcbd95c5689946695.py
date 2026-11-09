import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    
    # Baseline construction
    points_in_N = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2*xa + ya*ya - 3*yb*yb) % p == 0
                    in_y_special = ya in {0, 1, 2, 4}
                    if in_unital and not in_y_special:
                        continue
                    points_in_N.add((xa, xb, ya, yb))
                    
    point_list = list(points_in_N)
    point_to_idx = {pt: i for i, pt in enumerate(point_list)}
    idx_to_point = {i: pt for i, pt in enumerate(point_list)}
    
    valid_dirs = [0] * 2401
    
    for x_idx in range(2401):
        x = idx_to_point[x_idx]
        xa, xb, ya, yb = x
        mask = 0
        for v_idx in range(50):
            if v_idx < 49:
                m_a, m_b = divmod(v_idx, 7)
                ok = True
                for t in range(1, p):
                    t_ma = (t * m_a) % p
                    t_mb = (t * m_b) % p
                    nx_a = (xa + t) % p
                    nx_b = (xb + t_mb) % p
                    ny_a = (ya + t_ma) % p
                    ny_b = (yb + t_mb) % p
                    pt = (nx_a, nx_b, ny_a, ny_b)
                    if pt not in points_in_N:
                        ok = False
                        break
                if ok:
                    mask |= (1 << v_idx)
            else:
                ok = True
                for t in range(1, p):
                    nx_a = xa
                    nx_b = xb
                    ny_a = (ya + t) % p
                    ny_b = yb
                    pt = (nx_a, nx_b, ny_a, ny_b)
                    if pt not in points_in_N:
                        ok = False
                        break
                if ok:
                    mask |= (1 << v_idx)
        valid_dirs[x_idx] = mask
        
    N_indices = set(point_to_idx.values())
    
    critical_x = []
    for x in range(2401):
        if valid_dirs[x] != 0:
            if (valid_dirs[x] & (valid_dirs[x] - 1)) == 0:
                critical_x.append(x)
                
    def is_removable(p_idx):
        if not critical_x:
            return True
        p = idx_to_point[p_idx]
        pa, pb, pca, pcb = p
        for x_idx in critical_x:
            x = idx_to_point[x_idx]
            xa, xb, ya, yb = x
            da = (pa - xa) % p
            db = (pb - xb) % p
            if da == 0 and db == 0:
                continue
            if da == 0:
                d_idx = 49
            else:
                da_inv = pow(da, -1, p)
                m_val = (db * da_inv) % p
                d_idx = m_val * 7 + 0
            if (valid_dirs[x_idx] >> d_idx) & 1:
                return False
        return True

    start_time = time.time()
    deadline = start_time + 590
    
    while time.time() < deadline:
        N_list = list(N_indices)
        if not N_list:
            break
            
        found = False
        for p_idx in N_list:
            if is_removable(p_idx):
                N_indices.remove(p_idx)
                p = idx_to_point[p_idx]
                pa, pb, pca, pcb = p
                for x_idx in range(2401):
                    x = idx_to_point[x_idx]
                    xa, xb, ya, yb = x
                    da = (pa - xa) % p
                    db = (pb - xb) % p
                    if da == 0 and db == 0:
                        continue
                    if da == 0:
                        d_idx = 49
                    else:
                        da_inv = pow(da, -1, p)
                        m_val = (db * da_inv) % p
                        d_idx = m_val * 7 + 0
                    if (valid_dirs[x_idx] >> d_idx) & 1:
                        valid_dirs[x_idx] &= ~(1 << d_idx)
                        
                critical_x = []
                for x in range(2401):
                    if valid_dirs[x] != 0:
                        if (valid_dirs[x] & (valid_dirs[x] - 1)) == 0:
                            critical_x.append(x)
                found = True
                break
                
        if not found:
            break
            
    result_points = []
    for p_idx in N_indices:
        pt = idx_to_point[p_idx]
        result_points.append([[pt[0], pt[1]], [pt[2], pt[3]]])
        
    return np.array(result_points, dtype=np.int64)
