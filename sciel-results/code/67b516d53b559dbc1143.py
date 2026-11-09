"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as so

def pack(boxes, container):
    """
    3D Bin Packing using Mixed Integer Programming (MIP).
    """
    W, H, D = container
    n = len(boxes)
    
    def heuristic_pack(boxes, container):
        W, H, D = container
        n = len(boxes)
        sorted_indices = sorted(range(n), key=lambda i: (-boxes[i][0]*boxes[i][1]*boxes[i][2], 
                                                        -max(boxes[i][0], boxes[i][1], boxes[i][2])))
        bins = []
        placements = [None] * n
        
        def overlaps(b1, b2):
            x1, y1, z1, w1, h1, d1 = b1
            x2, y2, z2, w2, h2, d2 = b2
            return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                        y1 + h1 <= y2 or y2 + h2 <= y1 or
                        z1 + d1 <= z2 or z2 + d2 <= z1)
            
        def find_valid_positions(box, bin_boxes):
            w, h, d = box
            candidates = set()
            candidates.add((0, 0, 0))
            for (x, y, z, bw, bh, bd) in bin_boxes:
                candidates.add((x + bw, y, z))
                candidates.add((x, y + bh, z))
                candidates.add((x, y, z + bd))
                candidates.add((x + bw, y + bh, z))
                candidates.add((x + bw, y, z + bd))
                candidates.add((x, y + bh, z + bd))
                candidates.add((x + bw, y + bh, z + bd))
            
            valid = []
            for (x, y, z) in candidates:
                if x + w > W or y + h > H or z + d > D:
                    continue
                new_box = (x, y, z, w, h, d)
                if any(overlaps(new_box, b) for b in bin_boxes):
                    continue
                valid.append((x, y, z))
            return valid
        
        def position_score(pos, box, bin_boxes):
            x, y, z = pos
            w, h, d = box
            base_score = z * 10000 + x * 100 + y
            return base_score
        
        for idx in sorted_indices:
            w, h, d = boxes[idx]
            best_bin = -1
            best_pos = None
            best_score = float('inf')
            
            for bin_idx in range(len(bins)):
                positions = find_valid_positions((w, h, d), bins[bin_idx])
                for pos in positions:
                    s = position_score(pos, (w, h, d), bins[bin_idx])
                    if s < best_score:
                        best_score = s
                        best_bin = bin_idx
                        best_pos = pos
            
            if best_bin == -1:
                best_bin = len(bins)
                bins.append([])
                best_pos = (0, 0, 0)
            
            x, y, z = best_pos
            placements[idx] = (best_bin, x, y, z, w, h, d)
            bins[best_bin].append((x, y, z, w, h, d))
        
        for _ in range(10):
            if not bins: break
            B = len(bins)
            if B <= 1: break
            last_bin = B - 1
            boxes_in_last = [i for i in range(n) if placements[i][0] == last_bin]
            moved = False
            for box_idx in sorted(boxes_in_last, key=lambda i: -(boxes[i][0]*boxes[i][1]*boxes[i][2])):
                for target in range(last_bin):
                    w, h, d = boxes[box_idx]
                    positions = find_valid_positions((w, h, d), bins[target])
                    if positions:
                        best_pos = min(positions, key=lambda p: position_score(p, (w, h, d), bins[target]))
                        x, y, z = best_pos
                        bins[last_bin] = [b for b in bins[last_bin] if not (b[0]==placements[box_idx][1] and b[1]==placements[box_idx][2] and b[2]==placements[box_idx][3] and b[3]==w and b[4]==h and b[5]==d)]
                        bins[target].append((x, y, z, w, h, d))
                        placements[box_idx] = (target, x, y, z, w, h, d)
                        moved = True
                        break
                if moved: break
            if not moved: break
        
        return placements

    heuristic_placements = heuristic_pack(boxes, container)
    heuristic_bins = max(p[0] for p in heuristic_placements) + 1
    
    B_max = heuristic_bins
    
    idx_b = lambda j: j
    idx_x = lambda i, j: B_max + i * B_max + j
    idx_X = lambda i, j: B_max + n * B_max + (i * B_max + j) * 3 + 0
    idx_Y = lambda i, j: B_max + n * B_max + (i * B_max + j) * 3 + 1
    idx_Z = lambda i, j: B_max + n * B_max + (i * B_max + j) * 3 + 2
    
    pairs = []
    for i in range(n):
        for k in range(i+1, n):
            pairs.append((i, k))
    n_pairs = len(pairs)
    
    idx_u = lambda i, k, j, dim, dir: B_max + n * B_max + 3 * n * B_max + ( (i*n + k) * B_max + j ) * 6 + dim * 2 + dir
    
    n_vars = B_max + n * B_max + 3 * n * B_max + 6 * n_pairs * B_max
    
    c = np.zeros(n_vars)
    for j in range(B_max):
        c[idx_b(j)] = 1.0
        
    A_rows = []
    b_vals = []
    lb = np.zeros(n_vars)
    ub = np.zeros(n_vars)
    integrality = np.zeros(n_vars, dtype=int)
    
    for j in range(B_max):
        ub[idx_b(j)] = 1.0
        integrality[idx_b(j)] = 1
        
    for i in range(n):
        for j in range(B_max):
            ub[idx_x(i, j)] = 1.0
            integrality[idx_x(i, j)] = 1
            ub[idx_X(i, j)] = W
            ub[idx_Y(i, j)] = H
            ub[idx_Z(i, j)] = D
            
    for p_idx, (i, k) in enumerate(pairs):
        for j in range(B_max):
            for dim in range(3):
                for dir in range(2):
                    ub[idx_u(i, k, j, dim, dir)] = 1.0
                    integrality[idx_u(i, k, j, dim, dir)] = 1
                    
    for i in range(n):
        row = np.zeros(n_vars)
        for j in range(B_max):
            row[idx_x(i, j)] = 1.0
        A_rows.append(row)
        b_vals.append(1.0)
        
    for i in range(n):
        for j in range(B_max):
            row = np.zeros(n_vars)
            row[idx_b(j)] = 1.0
            row[idx_x(i, j)] = -1.0
            A_rows.append(row)
            b_vals.append(0.0)
            
    for i in range(n):
        for j in range(B_max):
            row = np.zeros(n_vars)
            row[idx_X(i, j)] = 1.0
            row[idx_b(j)] = -W
            A_rows.append(row)
            b_vals.append(-boxes[i][0])
            
            row = np.zeros(n_vars)
            row[idx_Y(i, j)] = 1.0
            row[idx_b(j)] = -H
            A_rows.append(row)
            b_vals.append(-boxes[i][1])
            
            row = np.zeros(n_vars)
            row[idx_Z(i, j)] = 1.0
            row[idx_b(j)] = -D
            A_rows.append(row)
            b_vals.append(-boxes[i][2])
            
    M = max(W, H, D)
    
    for p_idx, (i, k) in enumerate(pairs):
        w_i, h_i, d_i = boxes[i]
        w_k, h_k, d_k = boxes[k]
        
        for j in range(B_max):
            row = np.zeros(n_vars)
            row[idx_X(i, j)] = 1.0
            row[idx_X(k, j)] = -1.0
            row[idx_u(i, k, j, 0, 0)] = -M
            A_rows.append(row)
            b_vals.append(w_i)
            
            row = np.zeros(n_vars)
            row[idx_X(k, j)] = 1.0
            row[idx_X(i, j)] = -1.0
            row[idx_u(i, k, j, 0, 1)] = -M
            A_rows.append(row)
            b_vals.append(w_k)
            
            row = np.zeros(n_vars)
            row[idx_Y(i, j)] = 1.0
            row[idx_Y(k, j)] = -1.0
            row[idx_u(i, k, j, 1, 0)] = -M
            A_rows.append(row)
            b_vals.append(h_i)
            
            row = np.zeros(n_vars)
            row[idx_Y(k, j)] = 1.0
            row[idx_Y(i, j)] = -1.0
            row[idx_u(i, k, j, 1, 1)] = -M
            A_rows.append(row)
            b_vals.append(h_k)
            
            row = np.zeros(n_vars)
            row[idx_Z(i, j)] = 1.0
            row[idx_Z(k, j)] = -1.0
            row[idx_u(i, k, j, 2, 0)] = -M
            A_rows.append(row)
            b_vals.append(d_i)
            
            row = np.zeros(n_vars)
            row[idx_Z(k, j)] = 1.0
            row[idx_Z(i, j)] = -1.0
            row[idx_u(i, k, j, 2, 1)] = -M
            A_rows.append(row)
            b_vals.append(d_k)
            
            row = np.zeros(n_vars)
            row[idx_u(i, k, j, 0, 0)] = -1.0
            row[idx_u(i, k, j, 0, 1)] = -1.0
            row[idx_u(i, k, j, 1, 0)] = -1.0
            row[idx_u(i, k, j, 1, 1)] = -1.0
            row[idx_u(i, k, j, 2, 0)] = -1.0
            row[idx_u(i, k, j, 2, 1)] = -1.0
            row[idx_x(i, j)] = 1.0
            row[idx_x(k, j)] = 1.0
            A_rows.append(row)
            b_vals.append(1.0)
            
    A = np.array(A_rows)
    b = np.array(b_vals)
    
    try:
        res = so.milp(c, constraints=so.LinearConstraint(A, lb=-np.inf, ub=b), 
                      integrality=integrality, 
                      bounds=so.Bounds(lb, ub), 
                      options={"time_limit": 100, "mip_rel_gap": 0.0})
        
        if res.status == 0 and res.x is not None:
            x_sol = res.x
            placements = [None] * n
            for i in range(n):
                for j in range(B_max):
                    if x_sol[idx_x(i, j)] > 0.5:
                        x_pos = x_sol[idx_X(i, j)]
                        y_pos = x_sol[idx_Y(i, j)]
                        z_pos = x_sol[idx_Z(i, j)]
                        placements[i] = (j, x_pos, y_pos, z_pos, boxes[i][0], boxes[i][1], boxes[i][2])
                        break
                if placements[i] is None:
                    return heuristic_placements
            
            return placements
        else:
            return heuristic_placements
    except Exception:
        return heuristic_placements

# EVOLVE-BLOCK-END
