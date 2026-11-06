"""Rewritten 3D Container Loading with skyline/floor-map approach."""
import time
import itertools
import random
import math

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, b[0], b[1], b[2]) for b in boxes]
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    container_vol = W * H * D
    
    # Precompute unique orientations for each box
    orient = []
    for w, h, d in boxes:
        opts = set()
        for perm in itertools.permutations((w, h, d)):
            opts.add(perm)
        orient.append(list(opts))
    
    # Skyline representation: 2D grid of minimum z-heights
    # Grid resolution: use cells of size 1 if feasible, else coarser
    # For containers up to ~200x200x200, use 1-unit cells
    # For larger, use coarser grid
    max_dim = max(W, H, D)
    if max_dim <= 300:
        cell_size = 1
    elif max_dim <= 600:
        cell_size = 2
    else:
        cell_size = max(1, max_dim // 200)
    
    gw = (W + cell_size - 1) // cell_size
    gh = (H + cell_size - 1) // cell_size
    
    def make_skyline():
        return [[0] * gh for _ in range(gw)]
    
    def skyline_fits(sky, x, y, w, h):
        """Check if a box of width w, height h can be placed at (x,y) on the skyline."""
        cx1 = x // cell_size
        cy1 = y // cell_size
        cx2 = (x + w - 1) // cell_size
        cy2 = (y + h - 1) // cell_size
        if cx1 >= gw or cy1 >= gh:
            return False
        cx2 = min(cx2, gw - 1)
        cy2 = min(cy2, gh - 1)
        for cx in range(cx1, cx2 + 1):
            for cy in range(cy1, cy2 + 1):
                if sky[cx][cy] > 0:
                    return False
        return True
    
    def skyline_update(sky, x, y, w, h, z):
        """Update skyline after placing box at (x,y,z) with dims (w,h,d)."""
        cx1 = x // cell_size
        cy1 = y // cell_size
        cx2 = (x + w - 1) // cell_size
        cy2 = (y + h - 1) // cell_size
        cx2 = min(cx2, gw - 1)
        cy2 = min(cy2, gh - 1)
        for cx in range(cx1, cx2 + 1):
            for cy in range(cy1, cy2 + 1):
                sky[cx][cy] = max(sky[cx][cy], z)
    
    def skyline_remove(sky, x, y, w, h, z):
        """Remove box from skyline (approximate - set to min of neighbors or 0)."""
        cx1 = x // cell_size
        cy1 = y // cell_size
        cx2 = (x + w - 1) // cell_size
        cy2 = (y + h - 1) // cell_size
        cx2 = min(cx2, gw - 1)
        cy2 = min(cy2, gh - 1)
        for cx in range(cx1, cx2 + 1):
            for cy in range(cy1, cy2 + 1):
                # Recompute: find max z of any placed box that still covers this cell
                # For simplicity, just set to 0 (crude but safe for search)
                sky[cx][cy] = 0
    
    # Precise overlap check using placed boxes list
    def precise_fits(x, y, z, w, h, d, placed):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in placed:
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True
    
    # Generate candidate (x, y) positions from placed boxes
    def get_xy_candidates(placed):
        cands = set()
        cands.add((0, 0))
        for (px, py, pz, pw, ph, pd) in placed:
            candidates = [
                (px + pw, py),
                (px, py + ph),
                (px + pw, py + ph),
                (px, py),
                (px + pw, py),
                (px, py + ph),
            ]
            for (nx, ny) in candidates:
                if 0 <= nx < W and 0 <= ny < H:
                    cands.add((nx, ny))
        return list(cands)
    
    # Find the lowest valid z for a given (x, y, w, h) using skyline
    def find_lowest_z(sky, x, y, w, h, d, placed):
        """Find lowest z where box fits at (x,y) with footprint (w,h) and depth d."""
        # Use skyline to get a lower bound
        cx1 = x // cell_size
        cy1 = y // cell_size
        cx2 = (x + w - 1) // cell_size
        cy2 = (y + h - 1) // cell_size
        cx2 = min(cx2, gw - 1)
        cy2 = min(cy2, gh - 1)
        
        max_h = 0
        for cx in range(cx1, cx2 + 1):
            for cy in range(cy1, cy2 + 1):
                if sky[cx][cy] > max_h:
                    max_h = sky[cx][cy]
        
        # Try z = max_h first, then increment
        z = max_h
        while z + d <= D:
            if precise_fits(x, y, z, w, h, d, placed):
                return z
            z += 1
        return None
    
    def position_score(x, y, z, w, h, d):
        return z * W * H + y * W + x
    
    def place_one(box_idx, placed, sky):
        best = None
        best_score = float('inf')
        for (ow, oh, od) in orient[box_idx]:
            if ow > W or oh > H or od > D:
                continue
            for (ax, ay) in get_xy_candidates(placed):
                if ax + ow > W or ay + oh > H:
                    continue
                z = find_lowest_z(sky, ax, ay, ow, oh, od, placed)
                if z is not None:
                    s = position_score(ax, ay, z, ow, oh, od)
                    if s < best_score:
                        best_score = s
                        best = (ax, ay, z, ow, oh, od)
        return best
    
    def evaluate(result):
        vol = 0
        for p in result:
            if p[0] == 0:
                vol += p[4] * p[5] * p[6]
        return vol / container_vol
    
    def greedy(order):
        placed = []
        sky = make_skyline()
        result = [None] * n
        for idx in order:
            pos = place_one(idx, placed, sky)
            if pos is not None:
                x, y, z, ow, oh, od = pos
                placed.append((x, y, z, ow, oh, od))
                skyline_update(sky, x, y, ow, oh, z)
                result[idx] = (0, x, y, z, ow, oh, od)
            else:
                w, h, d = boxes[idx]
                result[idx] = (-1, 0, 0, 0, w, h, d)
        return result
    
    indices = list(range(n))
    
    def order_by_volume():
        return sorted(indices, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    def order_by_maxdim():
        return sorted(indices, key=lambda i: max(boxes[i]), reverse=True)
    
    def order_by_mindim():
        return sorted(indices, key=lambda i: min(boxes[i]), reverse=True)
    
    def order_by_surface():
        return sorted(indices, key=lambda i: 2*(boxes[i][0]*boxes[i][1]+boxes[i][1]*boxes[i][2]+boxes[i][2]*boxes[i][0]), reverse=True)
    
    def order_by_aspect():
        def asp(i):
            w,h,d = boxes[i]
            return max(w,h,d)/min(w,h,d) if min(w,h,d)>0 else 0
        return sorted(indices, key=lambda i: asp(i))
    
    def order_by_min_volume():
        return sorted(indices, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2])
    
    def order_by_maxdim_min():
        return sorted(indices, key=lambda i: max(boxes[i]) + min(boxes[i]), reverse=True)
    
    def order_random():
        r = list(indices)
        random.shuffle(r)
        return r
    
    def local_improve(result, budget):
        improved = True
        while improved and time.time() < budget:
            improved = False
            left_out = [i for i in range(n) if result[i][0] == -1]
            left_out.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
            for idx in left_out:
                if time.time() >= budget:
                    break
                # Rebuild placed and sky from result
                placed = [(p[1],p[2],p[3],p[4],p[5],p[6]) for p in result if p[0]==0]
                sky = make_skyline()
                for (px, py, pz, pw, ph, pd) in placed:
                    skyline_update(sky, px, py, pw, ph, pz)
                pos = place_one(idx, placed, sky)
                if pos is not None:
                    x, y, z, ow, oh, od = pos
                    result[idx] = (0, x, y, z, ow, oh, od)
                    improved = True
        return result
    
    best_result = None
    best_fill = 0
    
    strategies = [order_by_volume, order_by_maxdim, order_by_mindim, order_by_surface, 
                  order_by_aspect, order_by_min_volume, order_by_maxdim_min]
    
    for strat in strategies:
        if time.time() > deadline - 5:
            break
        order = strat()
        res = greedy(order)
        fill = evaluate(res)
        if fill > best_fill:
            best_fill = fill
            best_result = res
            if time.time() < deadline - 10:
                res = local_improve(res, deadline - 10)
                fill2 = evaluate(res)
                if fill2 > best_fill:
                    best_fill = fill2
                    best_result = res
    
    n_random = 40
    for r in range(n_random):
        if time.time() > deadline - 3:
            break
        order = order_random()
        res = greedy(order)
        fill = evaluate(res)
        if fill > best_fill:
            best_fill = fill
            best_result = res
            if time.time() < deadline - 8:
                res = local_improve(res, deadline - 8)
                fill2 = evaluate(res)
                if fill2 > best_fill:
                    best_fill = fill2
                    best_result = res
    
    # Final pass: try to fit any remaining boxes
    if best_result is not None and time.time() < deadline - 1:
        placed = [(p[1],p[2],p[3],p[4],p[5],p[6]) for p in best_result if p[0]==0]
        sky = make_skyline()
        for (px, py, pz, pw, ph, pd) in placed:
            skyline_update(sky, px, py, pw, ph, pz)
        for idx in range(n):
            if best_result[idx][0] == -1:
                pos = place_one(idx, placed, sky)
                if pos is not None:
                    x,y,z,ow,oh,od = pos
                    placed.append((x,y,z,ow,oh,od))
                    skyline_update(sky, x, y, ow, oh, z)
                    best_result[idx] = (0,x,y,z,ow,oh,od)
    
    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_result
# EVOLVE-BLOCK-END
