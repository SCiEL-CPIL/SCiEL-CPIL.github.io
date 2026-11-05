"""Search-based 3D bin packing: multiple heuristics, keep best by score."""
import time
import random
import sys

# Safety margin: kill before 150s wall-clock
DEADLINE = time.time() + 135.0


def _score(placements, boxes, container):
    """Compute the score: mean B + 0.1*(1 - last-bin occupied volume fraction)."""
    W, H, D = container
    container_vol = W * H * D
    if not placements:
        return 0.0
    # B = number of occupied bins
    bins_used = set(p[0] for p in placements)
    B = len(bins_used)
    last_bin = max(bins_used)
    # Last-bin occupied volume
    last_vol = sum(p[4] * p[5] * p[6] for p in placements if p[0] == last_bin)
    return B + 0.1 * (1.0 - last_vol / container_vol)


def _fits_bin(placements_in_bin, cx, cy, cz, w, h, d, W, H, D):
    """Check if a box fits at (cx,cy,cz) in a bin with given placements, no overlap."""
    if cx < 0 or cy < 0 or cz < 0:
        return False
    if cx + w > W or cy + h > H or cz + d > D:
        return False
    for (px, py, pz, pw, ph, pd) in placements_in_bin:
        if (cx < px + pw and px < cx + w and
                cy < py + ph and py < cy + h and
                cz < pz + pd and pz < cz + d):
            return False
    return True


def _shelf_pack(boxes, container, order=None):
    """Shelf heuristic with optional box ordering."""
    W, H, D = container
    if order is None:
        order = list(range(len(boxes)))
    placements = []
    b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
    for idx in order:
        w, h, d = boxes[idx]
        if x + w > W:
            x, y, row_h = 0, y + row_h, 0
        if y + h > H:
            y, z, layer_d = 0, z + layer_d, 0
            x, row_h = 0, 0
        if z + d > D:
            b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
        placements.append((b, x, y, z, w, h, d))
        x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
    return placements


def _best_fit_pack(boxes, container, order=None, max_candidates=12):
    """Best-fit decreasing style: try to place each box in the best position."""
    W, H, D = container
    if order is None:
        order = list(range(len(boxes)))
    n = len(boxes)
    # bins: list of list of (x,y,z,w,h,d)
    bins = []
    placements = []
    for idx in order:
        w, h, d = boxes[idx]
        best_score = None
        best_bin = None
        best_pos = None
        # Try existing bins
        for bi in range(len(bins)):
            bin_placements = bins[bi]
            # Generate candidate positions
            candidates = [(0, 0, 0)]
            for (px, py, pz, pw, ph, pd) in bin_placements:
                candidates.append((px + pw, py, pz))
                candidates.append((px, py + ph, pz))
                candidates.append((px, py, pz + pd))
            # Limit candidates for speed
            if len(candidates) > max_candidates:
                # Keep only unique, in-bounds
                seen = set()
                filtered = []
                for c in candidates:
                    if c not in seen and c[0] + w <= W and c[1] + h <= H and c[2] + d <= D:
                        seen.add(c)
                        filtered.append(c)
                candidates = filtered[:max_candidates]
            for (cx, cy, cz) in candidates:
                if _fits_bin(bin_placements, cx, cy, cz, w, h, d, W, H, D):
                    # Score: prefer lower z, then y, then x (bottom-left-back)
                    s = (cz, cy, cx)
                    if best_score is None or s < best_score:
                        best_score = s
                        best_bin = bi
                        best_pos = (cx, cy, cz)
        if best_bin is not None:
            bins[best_bin].append((best_pos[0], best_pos[1], best_pos[2], w, h, d))
            placements.append((best_bin, best_pos[0], best_pos[1], best_pos[2], w, h, d))
        else:
            # New bin
            bi = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            placements.append((bi, 0, 0, 0, w, h, d))
    return placements


def _random_order(n, rng):
    """Generate a random permutation."""
    order = list(range(n))
    rng.shuffle(order)
    return order


def _sort_by_volume_desc(boxes):
    """Sort indices by volume descending."""
    return sorted(range(len(boxes)), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)


def _sort_by_max_dim_desc(boxes):
    """Sort indices by max dimension descending."""
    return sorted(range(len(boxes)), key=lambda i: max(boxes[i]), reverse=True)


def _sort_by_volume_asc(boxes):
    """Sort indices by volume ascending."""
    return sorted(range(len(boxes)), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2])


def _sort_by_max_dim_asc(boxes):
    """Sort indices by max dimension ascending."""
    return sorted(range(len(boxes)), key=lambda i: max(boxes[i]))


def pack(boxes, container):
    """Main entry point. Returns list of (bin, x, y, z, w, h, d) per box."""
    n = len(boxes)
    W, H, D = container
    
    # Generate diverse orderings
    orderings = []
    orderings.append(_sort_by_volume_desc(boxes))
    orderings.append(_sort_by_max_dim_desc(boxes))
    orderings.append(_sort_by_volume_asc(boxes))
    orderings.append(_sort_by_max_dim_asc(boxes))
    orderings.append(list(range(n)))  # original order
    orderings.append(list(reversed(range(n))))  # reverse original
    
    rng = random.Random(42)
    # Add some random orderings
    for _ in range(6):
        orderings.append(_random_order(n, rng))
    
    best_placements = None
    best_score_val = float('inf')
    
    start_time = time.time()
    
    # Try shelf packing with various orderings
    for order in orderings:
        if time.time() > DEADLINE:
            break
        placements = _shelf_pack(boxes, container, order)
        s = _score(placements, boxes, container)
        if s < best_score_val:
            best_score_val = s
            best_placements = placements
    
    # Try best-fit packing with various orderings
    for order in orderings:
        if time.time() > DEADLINE:
            break
        placements = _best_fit_pack(boxes, container, order)
        s = _score(placements, boxes, container)
        if s < best_score_val:
            best_score_val = s
            best_placements = placements
    
    # If we have time, try more random orderings with both methods
    attempts = 0
    while time.time() < DEADLINE - 5.0 and attempts < 200:
        order = _random_order(n, rng)
        if attempts % 2 == 0:
            placements = _shelf_pack(boxes, container, order)
        else:
            placements = _best_fit_pack(boxes, container, order)
        s = _score(placements, boxes, container)
        if s < best_score_val:
            best_score_val = s
            best_placements = placements
        attempts += 1
    
    if best_placements is None:
        # Fallback: simple shelf pack
        best_placements = _shelf_pack(boxes, container)
    
    return best_placements
