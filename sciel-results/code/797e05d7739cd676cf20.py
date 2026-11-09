"""
3D Bin Packing Solver - Stronger Algorithm
Strategy:
1. Multiple diverse sorting orders (volume, max-dim, surface, aspect-ratio, etc.)
2. Fast greedy packing with efficient candidate generation and vectorized overlap
3. More aggressive local search: box relocation between bins with proper state tracking
4. Multi-start with randomized perturbations of ordering
5. Deadline guard for 150s wall-clock with safety margin
"""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _fast_overlap_check(placed_arr, x, y, z, w, h, d):
    """
    Vectorized overlap check against all placed boxes.
    placed_arr: Nx6 array of (x,y,z,w,h,d)
    Returns True if placement (x,y,z,w,h,d) does NOT overlap any placed box.
    """
    if placed_arr.shape[0] == 0:
        return True
    px, py, pz, pw, ph, pd = placed_arr[:, 0], placed_arr[:, 1], placed_arr[:, 2], placed_arr[:, 3], placed_arr[:, 4], placed_arr[:, 5]
    # Separation: no overlap in at least one axis
    sep = ((x + w <= px) | (px + pw <= x) |
           (y + h <= py) | (py + ph <= y) |
           (z + d <= pz) | (pz + pd <= z))
    return bool(np.all(sep))


def _gen_candidates(placed_arr, x_max, y_max, z_max, bw, bh, bd):
    """
    Generate candidate positions for a box of size (bw,bh,bd) given placed boxes.
    Returns list of (x, y, z) sorted by (y, z, x) for BLF preference.
    """
    cands = set()
    cands.add((0, 0, 0))
    
    if placed_arr.shape[0] > 0:
        px, py, pz, pw, ph, pd = placed_arr[:, 0], placed_arr[:, 1], placed_arr[:, 2], placed_arr[:, 3], placed_arr[:, 4], placed_arr[:, 5]
        
        # X candidates
        for pxv, pwv in zip(px, pw):
            c1 = int(pxv) - bw
            c2 = int(pxv + pwv)
            if 0 <= c1 < x_max:
                cands.add((c1, 0, 0))
                cands.add((c1, int(py[np.where(px == pxv)[0][0]]), int(pz[np.where(px == pxv)[0][0]])))
            if 0 <= c2 < x_max:
                cands.add((c2, 0, 0))
                cands.add((c2, int(py[np.where(px == pxv)[0][0]]), int(pz[np.where(px == pxv)[0][0]])))
        
        # Y candidates
        for pyv, phv, pxv, pzv in zip(py, ph, px, pz):
            c1 = int(pyv) - bh
            c2 = int(pyv + phv)
            if 0 <= c1 < y_max:
                cands.add((0, c1, 0))
                cands.add((int(pxv), c1, int(pzv)))
            if 0 <= c2 < y_max:
                cands.add((0, c2, 0))
                cands.add((int(pxv), c2, int(pzv)))
        
        # Z candidates
        for pzv, pdv, pxv, pyv in zip(pz, pd, px, py):
            c1 = int(pzv) - bd
            c2 = int(pzv + pdv)
            if 0 <= c1 < z_max:
                cands.add((0, 0, c1))
                cands.add((int(pxv), int(pyv), c1))
            if 0 <= c2 < z_max:
                cands.add((0, 0, c2))
                cands.add((int(pxv), int(pyv), c2))
    
    valid = []
    for (cx, cy, cz) in cands:
        if cx < 0 or cy < 0 or cz < 0:
            continue
        if cx + bw > x_max or cy + bh > y_max or cz + bd > z_max:
            continue
        if _fast_overlap_check(placed_arr, cx, cy, cz, bw, bh, bd):
            valid.append((cx, cy, cz))
    
    valid.sort(key=lambda p: (p[1], p[2], p[0]))
    return valid


def _greedy_pack(boxes, container, order):
    """
    Greedy pack boxes in given order. Returns list of (bin, x, y, z, w, h, d).
    """
    W, H, D = container
    n = len(boxes)
    bins = []  # each bin is a list of (x,y,z,w,h,d)
    result = [None] * n
    
    for idx in order:
        w, h, d = boxes[idx]
        placed = None
        best_bin = -1
        best_pos = None
        
        for b_i in range(len(bins)):
            pa = np.array(bins[b_i]) if bins[b_i] else np.empty((0, 6))
            cands = _gen_candidates(pa, W, H, D, w, h, d)
            if cands:
                best_bin = b_i
                best_pos = cands[0]
                break
        
        if best_bin >= 0:
            bins[best_bin].append((best_pos[0], best_pos[1], best_pos[2], w, h, d))
            result[idx] = (best_bin, best_pos[0], best_pos[1], best_pos[2], w, h, d)
        else:
            new_bin = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            result[idx] = (new_bin, 0, 0, 0, w, h, d)
    
    return result


def _score(result, boxes, container):
    if not result or any(r is None for r in result):
        return float('inf')
    W, H, D = container
    cv = W * H * D
    max_bin = max(r[0] for r in result)
    lb_vol = sum(r[4] * r[5] * r[6] for r in result if r[0] == max_bin)
    return max_bin + 1 + 0.1 * (1 - lb_vol / cv)


def _local_search(result, boxes, container, deadline, seed=0):
    """
    Local search: try to relocate boxes from higher bins to lower bins.
    Returns improved result.
    """
    W, H, D = container
    n = len(boxes)
    rng = random.Random(seed)
    
    # Build bin structure
    bins = {}
    for i, r in enumerate(result):
        b = r[0]
        if b not in bins:
            bins[b] = []
        bins[b].append(i)
    
    cur_score = _score(result, boxes, container)
    no_improve = 0
    max_no_improve = 300
    
    while time.time() < deadline - 2 and no_improve < max_no_improve:
        bin_ids = sorted(bins.keys())
        if len(bin_ids) <= 1:
            break
        
        # Pick box from higher bin (biased toward last)
        src_bin = rng.choice(bin_ids[1:])
        if src_bin not in bins or not bins[src_bin]:
            no_improve += 1
            continue
        
        box_idx = rng.choice(bins[src_bin])
        w, h, d = boxes[box_idx]
        old_placement = result[box_idx]
        improved = False
        
        # Try moving to each lower bin
        for dst_bin in bin_ids:
            if dst_bin >= src_bin:
                continue
            
            dst_boxes = [result[i] for i in bins[dst_bin]]
            pa = np.array([[r[1], r[2], r[3], r[4], r[5], r[6]] for r in dst_boxes]) if dst_boxes else np.empty((0, 6))
            cands = _gen_candidates(pa, W, H, D, w, h, d)
            
            if cands:
                pos = cands[0]
                # Perform move
                result[box_idx] = (dst_bin, pos[0], pos[1], pos[2], w, h, d)
                bins[src_bin].remove(box_idx)
                if not bins[src_bin]:
                    del bins[src_bin]
                bins[dst_bin].append(box_idx)
                
                new_score = _score(result, boxes, container)
                if new_score < cur_score:
                    cur_score = new_score
                    improved = True
                    no_improve = 0
                else:
                    # Revert
                    result[box_idx] = old_placement
                    bins[dst_bin].remove(box_idx)
                    if not bins[dst_bin]:
                        del bins[dst_bin]
                    bins[src_bin].append(box_idx)
                    no_improve += 1
                break  # Only try one destination per box
        
        if not improved:
            no_improve += 1
    
    return result


def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    if n == 0:
        return []
    
    t0 = time.time()
    DEADLINE = t0 + 145.0  # 5s safety margin from 150s
    
    indices = list(range(n))
    
    # Define diverse sorting orders
    orders = []
    
    # 1. Volume descending
    orders.append(sorted(indices, key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True))
    
    # 2. Max dimension descending
    orders.append(sorted(indices, key=lambda i: max(boxes[i]), reverse=True))
    
    # 3. Surface area descending
    orders.append(sorted(indices, key=lambda i: 2 * (boxes[i][0] * boxes[i][1] + boxes[i][1] * boxes[i][2] + boxes[i][0] * boxes[i][2]), reverse=True))
    
    # 4. Min dimension descending
    orders.append(sorted(indices, key=lambda i: min(boxes[i]), reverse=True))
    
    # 5. Height (y) descending
    orders.append(sorted(indices, key=lambda i: boxes[i][1], reverse=True))
    
    # 6. Width (x) descending
    orders.append(sorted(indices, key=lambda i: boxes[i][0], reverse=True))
    
    # 7. Depth (z) descending
    orders.append(sorted(indices, key=lambda i: boxes[i][2], reverse=True))
    
    # 8. Aspect ratio (max/min) descending
    orders.append(sorted(indices, key=lambda i: max(boxes[i]) / max(min(boxes[i]), 1), reverse=True))
    
    best_result = None
    best_score = float('inf')
    
    # Run deterministic orders
    for order in orders:
        if time.time() > DEADLINE - 30:
            break
        res = _greedy_pack(boxes, container, order)
        sc = _score(res, boxes, container)
        if sc < best_score:
            best_score = sc
            best_result = res
    
    # Randomized restarts
    rng = random.Random(42)
    restarts_done = 0
    max_restarts = 200
    
    while restarts_done < max_restarts and time.time() < DEADLINE - 30:
        restarts_done += 1
        # Perturbed volume order
        vols = np.array([boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)], dtype=float)
        noise = rng.random(n) * 0.3 * vols.max()
        perturbed = vols + noise
        rand_order = sorted(indices, key=lambda i: perturbed[i], reverse=True)
        
        res = _greedy_pack(boxes, container, rand_order)
        sc = _score(res, boxes, container)
        if sc < best_score:
            best_score = sc
            best_result = res
    
    # Local search on best result
    if best_result is not None and time.time() < DEADLINE - 5:
        polished = _local_search(best_result, boxes, container, DEADLINE, seed=123)
        psc = _score(polished, boxes, container)
        if psc < best_score:
            best_result = polished
            best_score = psc
        
        # Second local search pass with different seed
        if time.time() < DEADLINE - 3:
            polished2 = _local_search(best_result, boxes, container, DEADLINE, seed=456)
            psc2 = _score(polished2, boxes, container)
            if psc2 < best_score:
                best_result = polished2
                best_score = psc2
    
    return best_result

# EVOLVE-BLOCK-END
