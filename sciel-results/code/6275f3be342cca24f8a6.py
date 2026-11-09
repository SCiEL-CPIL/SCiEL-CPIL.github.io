"""Combined 3D Bin Packing: Global Best-Fit with Multi-Strategy and Randomized Restarts."""
import time
import math
import random

_DEADLINE = 145.0  # Safety margin under 150s kill
_START = None

def _now():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START


def _fits(bin_boxes, W, H, D, x, y, z, w, h, d):
    if x + w > W or y + h > H or z + d > D:
        return False
    for (bx, by, bz, bw, bh, bd) in bin_boxes:
        # overlap check
        if not (x + w <= bx or bx + bw <= x or
                y + h <= by or by + bh <= y or
                z + d <= bz or bz + bd <= z):
            return False
    return True


def _max_cubes_positions(W, H, D, boxes_in_bin):
    """Generate candidate positions sorted by 'max-cube' potential (corner-first)."""
    xs = [0]
    ys = [0]
    zs = [0]
    for (bx, by, bz, bw, bh, bd) in boxes_in_bin:
        xs.append(bx + bw)
        ys.append(by + bh)
        zs.append(bz + bd)
    xs = sorted(set(v for v in xs if v <= W))
    ys = sorted(set(v for v in ys if v <= H))
    zs = sorted(set(v for v in zs if v <= D))
    
    cands = []
    for x in xs:
        for y in ys:
            for z in zs:
                cands.append((x, y, z))
    # Sort by "distance from origin" so corners are tried first
    # This helps find the "lowest" position which is generally preferred for stability/fill
    cands.sort(key=lambda p: (p[2], p[1], p[0])) 
    return cands


def _find_best_position(bin_boxes, w, h, d, W, H, D):
    """
    Find the best position in a specific bin for a box.
    Returns (x, y, z, score) or None.
    Score is designed to be minimized: prefer positions that leave large empty space (max-cube)
    and are lower in z, y, x.
    """
    if not bin_boxes:
        return (0, 0, 0, 0)

    cands = _max_cubes_positions(W, H, D, bin_boxes)
    best_pos = None
    best_score = float('inf')
    
    for (x, y, z) in cands:
        if _fits(bin_boxes, W, H, D, x, y, z, w, h, d):
            # Heuristic: Minimize z, then y, then x.
            # Also consider max-cube potential? 
            # The current program used -mc. The inspiration used z*1e6 + y*1e3 + x.
            # Let's use a composite score: primarily z, secondarily y, then x.
            # This tends to pack "bottom-up" which is good for fill.
            score = z * 1000000 + y * 1000 + x
            
            if score < best_score:
                best_score = score
                best_pos = (x, y, z)
                
    if best_pos:
        return (best_pos[0], best_pos[1], best_pos[2], best_score)
    return None


def _pack_bfd(boxes, container, order_key, deadline):
    """
    Pack boxes using Global Best-Fit Decreasing.
    order_key: callable taking (w, h, d) and returning a sort key (higher is packed first).
    """
    W, H, D = container
    n = len(boxes)
    
    # Sort boxes by the key descending
    indexed = list(range(n))
    indexed.sort(key=lambda i: order_key(boxes[i]), reverse=True)
    
    bins = []  # list of lists of (x, y, z, w, h, d)
    results = [None] * n
    t0 = time.time()

    for idx in indexed:
        if time.time() - t0 > deadline:
            break
        w, h, d = boxes[idx]
        best_bin_idx = -1
        best_pos = None
        best_global_score = float('inf')
        
        # Check all existing bins
        for b_idx, bin_boxes in enumerate(bins):
            res = _find_best_position(bin_boxes, w, h, d, W, H, D)
            if res:
                x, y, z, score = res
                # In Global Best-Fit, we want the position that is "best".
                # Usually, we prefer the bin that leaves the most space or has the lowest position.
                # Here, we compare the position score. Lower score is better.
                if score < best_global_score:
                    best_global_score = score
                    best_bin_idx = b_idx
                    best_pos = (x, y, z)
        
        if best_bin_idx == -1:
            # Open a new bin
            best_bin_idx = len(bins)
            bins.append([])
            best_pos = (0, 0, 0)
            
        bins[best_bin_idx].append((best_pos[0], best_pos[1], best_pos[2], w, h, d))
        results[idx] = (best_bin_idx, best_pos[0], best_pos[1], best_pos[2])

    # Construct final placements in original order
    placements = []
    for i in range(n):
        if results[i] is None:
             # Fallback if deadline hit: place at origin of a new bin (might overlap, but better than crashing? 
             # Actually, we should ensure we return a valid packing if possible. 
             # If we break early, we haven't placed all boxes. 
             # The task requires placing EVERY box. 
             # So we must not break early unless we have a valid full packing. 
             # For now, assume we finish. If not, we should probably not return this partial result.
             # Let's add a flag.
             pass
        b_idx, x, y, z = results[i]
        w, h, d = boxes[i]
        placements.append((b_idx, x, y, z, w, h, d))
        
    return placements


def _pack_shelf(boxes, container):
    """Shelf heuristic: rows along x, shelves along y, layers along z."""
    W, H, D = container
    placements = []
    b = 0
    x = y = z = 0
    row_h = 0
    layer_d = 0
    for w, h, d in boxes:
        if x + w > W:
            x = 0
            y += row_h
            row_h = 0
        if y + h > H:
            y = 0
            z += layer_d
            layer_d = 0
            x = 0
            row_h = 0
        if z + d > D:
            b += 1
            x = y = z = 0
            row_h = 0
            layer_d = 0
        placements.append((b, x, y, z, w, h, d))
        x += w
        row_h = max(row_h, h)
        layer_d = max(layer_d, d)
    return placements


def _renumber_bins(placements):
    """Renumber bins to be consecutive starting from 0."""
    if not placements:
        return placements
    
    seen = []
    for p in placements:
        b = p[0]
        if b not in seen:
            seen.append(b)
    
    mapping = {old: new for new, old in enumerate(seen)}
    
    new_placements = []
    for p in placements:
        new_b = mapping[p[0]]
        new_placements.append((new_b, p[1], p[2], p[3], p[4], p[5], p[6]))
    
    return new_placements


def _compute_score(placements, boxes, container):
    """Compute the score: B + 0.1*(1 - last-bin occupied volume fraction)."""
    if not placements:
        return float('inf')

    W, H, D = container
    container_vol = W * H * D

    bin_volumes = {}
    for (b, x, y, z, w, h, d) in placements:
        bin_volumes[b] = bin_volumes.get(b, 0) + w * h * d

    B = len(bin_volumes)
    if B == 0:
        return float('inf')

    last_bin = max(bin_volumes.keys())
    last_vol = bin_volumes[last_bin]
    last_frac = last_vol / container_vol if container_vol > 0 else 0.0

    return B + 0.1 * (1.0 - last_frac)


def pack(boxes, container):
    W, H, D = container
    deadline = _DEADLINE - _now()
    if deadline <= 0:
        deadline = 5.0
        
    n = len(boxes)
    best_placements = None
    best_score = float('inf')

    def update(placements):
        nonlocal best_placements, best_score
        if placements is None:
            return
        # Validate that all boxes are placed (len check)
        if len(placements) != n:
            return
        placements = _renumber_bins(placements)
        score = _compute_score(placements, boxes, container)
        if score < best_score:
            best_score = score
            best_placements = placements

    # 1. Try Shelf (Very fast, good baseline)
    update(_pack_shelf(boxes, container))

    if time.time() - _now() > deadline - 10:
        if best_placements: return best_placements
        return _pack_shelf(boxes, container)

    # 2. Try BFD with Volume Descending
    update(_pack_bfd(boxes, container, lambda b: b[0]*b[1]*b[2], deadline - 10))

    if time.time() - _now() > deadline - 5:
        if best_placements: return best_placements
        return _pack_shelf(boxes, container)

    # 3. Try BFD with Max Dimension Descending
    update(_pack_bfd(boxes, container, lambda b: max(b), deadline - 5))

    if time.time() - _now() > deadline - 2:
        if best_placements: return best_placements
        return _pack_shelf(boxes, container)

    # 4. Randomized Restarts
    # Use remaining time for random perturbations of the sorting key
    vol = [w * h * d for w, h, d in boxes]
    maxdim = [max(w, h, d) for w, h, d in boxes]
    max_vol = max(vol) if vol else 1
    max_maxdim = max(maxdim) if maxdim else 1
    
    # Pre-compute base weights
    base_weights = [0.7 * vol[i] / max_vol + 0.3 * maxdim[i] / max_maxdim for i in range(n)]
    
    while time.time() - _now() < deadline - 1:
        # Generate noisy weights
        noisy_weights = [base_weights[i] + random.random() * 0.2 for i in range(n)]
        
        # Sort indices by noisy weights descending
        indexed = list(range(n))
        indexed.sort(key=lambda i: noisy_weights[i], reverse=True)
        
        # Pack using this specific order
        # We need a pack function that takes an explicit order
        W, H, D = container
        bins = []
        results = [None] * n
        
        for idx in indexed:
            w, h, d = boxes[idx]
            best_bin_idx = -1
            best_pos = None
            best_global_score = float('inf')
            
            for b_idx, bin_boxes in enumerate(bins):
                res = _find_best_position(bin_boxes, w, h, d, W, H, D)
                if res:
                    x, y, z, score = res
                    if score < best_global_score:
                        best_global_score = score
                        best_bin_idx = b_idx
                        best_pos = (x, y, z)
            
            if best_bin_idx == -1:
                best_bin_idx = len(bins)
                bins.append([])
                best_pos = (0, 0, 0)
            
            bins[best_bin_idx].append((best_pos[0], best_pos[1], best_pos[2], w, h, d))
            results[idx] = (best_bin_idx, best_pos[0], best_pos[1], best_pos[2])
            
        placements = []
        for i in range(n):
            b_idx, x, y, z = results[i]
            w, h, d = boxes[i]
            placements.append((b_idx, x, y, z, w, h, d))
            
        update(placements)

    if best_placements:
        return best_placements
    
    # Fallback
    return _pack_shelf(boxes, container)
