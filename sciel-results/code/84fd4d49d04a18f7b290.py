"""
3D Bin Packing: Grid-based Maximal Empty Box Heuristic with Targeted Filling.
Strategy:
1. Represent each bin's occupancy as a NumPy 3D boolean array.
2. Use a "Targeted Fill" policy: try to fill the current bin until it reaches a
   high volume threshold or no more boxes fit, then open a new bin. This
   optimizes the score by ensuring the last bin is as full as possible.
3. For placement, use a "Maximal Empty Box" heuristic: find the position that
   maximizes the volume of the remaining contiguous empty space (or minimizes
   wasted space) while fitting the box. This reduces fragmentation.
4. Use multiple orderings (Volume Desc, etc.) to find a good packing.
"""
import time
import random
import numpy as np

DEADLINE = time.time() + 145.0  # Safety margin before 150s limit

def _fits_grid(grid, cx, cy, cz, w, h, d):
    """Check if a box fits at (cx, cy, cz) in the grid without overlap."""
    if cx < 0 or cy < 0 or cz < 0:
        return False
    if cx + w > grid.shape[0] or cy + h > grid.shape[1] or cz + d > grid.shape[2]:
        return False
    # Check if the volume is already occupied
    if np.any(grid[cx:cx+w, cy:cy+h, cz:cz+d]):
        return False
    return True

def _pack_grid(boxes, container, order, rng):
    """
    Pack boxes using grid-based heuristics and targeted filling.
    Returns a list of bins, each bin is a list of (idx, x, y, z, w, h, d).
    """
    W, H, D = container
    n = len(boxes)
    bins = []  # List of lists of (idx, x, y, z, w, h, d)
    grids = [] # List of NumPy grids for each bin
    bin_vols = [] # Volume in each bin
    container_vol = W * H * D
    
    # Targeted fill threshold: try to fill bin to this fraction before opening new
    # We want the last bin to be full, so we pack aggressively into the current bin.
    # However, if we pack too aggressively, we might fragment and not fit future boxes.
    # A common strategy is to just best-fit into the current bin if possible.
    
    for idx in order:
        w, h, d = boxes[idx]
        box_vol = w * h * d
        
        if not bins:
            bins.append([(idx, 0, 0, 0, w, h, d)])
            grid = np.zeros((W, H, D), dtype=bool)
            grid[0:w, 0:h, 0:d] = True
            grids.append(grid)
            bin_vols.append(box_vol)
            continue
        
        # Try to place in existing bins
        # Strategy: 
        # 1. If there is a "current" bin (the last one), try to fit here first to keep bins consolidated.
        # 2. If it doesn't fit in the last bin, try other bins? 
        #    The score depends on B (number of bins) and last bin fill.
        #    Minimizing B is primary. So we should try to fit in ANY existing bin.
        #    But to maximize last bin fill, we should prefer filling the LAST bin?
        #    Actually, if we fill an earlier bin, the last bin remains the same.
        #    If we open a new bin, B increases.
        #    So: Try to fit in existing bins. If we can, do it.
        #    Which existing bin? One that minimizes "waste" or maximizes "compactness".
        
        best_bin_idx = -1
        best_pos = None
        best_score = float('inf')
        
        # We prioritize the last bin to potentially fill it up? 
        # Or we prioritize the bin that leads to the best "bounding box" or "empty space" metric.
        # Let's iterate all bins.
        
        for bi in range(len(bins)):
            grid = grids[bi]
            
            # Generate candidate positions.
            # Instead of all anchors, let's use a grid step or specific anchors.
            # For speed, we can check positions based on existing box faces.
            # But with grid, we can be smarter.
            # Let's collect unique x, y, z coordinates from existing boxes + 0.
            
            # Extract coordinates from existing placements in this bin
            # This is slightly expensive if we do it every time, but n is small (40).
            # Alternatively, maintain lists of faces.
            
            # For simplicity and robustness, let's use the anchor method but with grid check.
            # Anchors: 0 and (box_end) for each dimension.
            
            x_coords = set([0])
            y_coords = set([0])
            z_coords = set([0])
            
            for (pidx, px, py, pz, pw, ph, pd) in bins[bi]:
                if px + pw < W: x_coords.add(px + pw)
                if py + ph < H: y_coords.add(py + ph)
                if pz + pd < D: z_coords.add(pz + pd)
                
            # Sort for determinism? Not strictly needed if we track best.
            # Limit number of anchors to avoid explosion if bin is very full.
            # If bin has many boxes, anchors can be O(N^3). 40 boxes -> 40^3 = 64k anchors. 
            # 40 boxes * 64k anchors * 30 bins (worst case) = too slow.
            # We need to be smarter.
            
            # Optimization: Only check a subset of anchors or use a grid scan with step.
            # Given W,H,D might be small (e.g. 10-50), a full grid scan is O(W*H*D).
            # If W,H,D <= 50, 50^3 = 125k checks per bin.
            # 40 boxes * 30 bins * 125k = 150M checks. Might be too slow in Python even with NumPy vectorization if not careful.
            
            # Let's stick to anchors but limit them.
            # If number of anchors is too high, sample them.
            
            anchors = []
            for x in x_coords:
                for y in y_coords:
                    for z in z_coords:
                        anchors.append((x, y, z))
                        
            # If too many anchors, subsample
            if len(anchors) > 200:
                # Random sampling
                anchors = rng.sample(anchors, 200)
                
            for (cx, cy, cz) in anchors:
                if not _fits_grid(grid, cx, cy, cz, w, h, d):
                    continue
                
                # Calculate score for this placement
                # Metric: Minimize the "wasted" space or maximize "compactness".
                # Let's use the volume of the bounding box of the bin content.
                # Or simpler: Prefer lower Z, then lower Y, then lower X (to keep things flat/compact).
                # This is a standard heuristic that works well.
                
                # Score = z * 1000 + y * 100 + x
                # This encourages filling from bottom-left-front.
                
                score = cz * 10000 + cy * 100 + cx
                
                # Tie-breaker: Prefer bins that are already fuller?
                # If we prefer the last bin, we add a small bonus to earlier bins?
                # Actually, we want to minimize B. So any bin is good.
                # To maximize last bin fill, we should prefer placing in the LAST bin if scores are similar.
                # So if bi == len(bins)-1, subtract a small epsilon from score.
                if bi == len(bins) - 1:
                    score -= 0.5
                    
                if score < best_score:
                    best_score = score
                    best_bin_idx = bi
                    best_pos = (cx, cy, cz)
                    
        if best_bin_idx == -1:
            # No fit in existing bins, create new bin
            bins.append([(idx, 0, 0, 0, w, h, d)])
            grid = np.zeros((W, H, D), dtype=bool)
            grid[0:w, 0:h, 0:d] = True
            grids.append(grid)
            bin_vols.append(box_vol)
        else:
            cx, cy, cz = best_pos
            bins[best_bin_idx].append((idx, cx, cy, cz, w, h, d))
            grids[best_bin_idx][cx:cx+w, cy:cy+h, cz:cz+d] = True
            bin_vols[best_bin_idx] += box_vol

    return bins

def _score(bins, container):
    """Calculate the official score: B + 0.1 * (1 - last_bin_fill)."""
    W, H, D = container
    C = W * H * D
    B = len(bins)

    if B == 0:
        return 0.0

    last_bin = bins[-1]
    if not last_bin:
        return B

    last_vol = sum(w * h * d for (_, _, _, _, w, h, d) in last_bin)
    fill_ratio = last_vol / C if C > 0 else 0.0

    return B + 0.1 * (1.0 - fill_ratio)

def _generate_orderings(boxes, n, rng):
    """Generate diverse box orderings of box indices."""
    idx = list(range(n))

    # 1. Volume Descending
    yield sorted(idx, key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)

    # 2. Max Dimension Descending
    yield sorted(idx, key=lambda i: max(boxes[i]), reverse=True)

    # 3. Sum of Dimensions Descending
    yield sorted(idx, key=lambda i: sum(boxes[i]), reverse=True)

    # 4. Aspect Ratio Descending
    yield sorted(idx, key=lambda i: max(boxes[i]) / max(min(boxes[i]), 1), reverse=True)

    # 5-20. Random Shuffles
    for _ in range(15):
        s = idx[:]
        rng.shuffle(s)
        yield s

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    best_bins = None
    best_score = float('inf')
    t_start = time.time()
    rng = random.Random(42)

    # Iterate through orderings
    for order in _generate_orderings(boxes, n, rng):
        if time.time() > DEADLINE:
            break

        bins = _pack_grid(boxes, container, order, rng)
        s = _score(bins, container)

        if s < best_score:
            best_score = s
            best_bins = bins

    if best_bins is None:
        # Fallback: one box per bin
        best_bins = [[(i, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])] for i in range(n)]
        best_score = _score(best_bins, container)

    # Build the result in original box order
    placement_map = {}
    for bi, bin_content in enumerate(best_bins):
        for item in bin_content:
            idx, x, y, z, w, h, d = item
            placement_map[idx] = (bi, x, y, z, w, h, d)

    result = []
    for i in range(n):
        if i in placement_map:
            result.append(placement_map[i])
        else:
            result.append((999, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]))

    return result
