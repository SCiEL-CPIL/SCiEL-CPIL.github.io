"""3D bin packer: fast best-fit with efficient corner search, multi-strategy
sorting, and time-bounded consolidation to minimize bin count B and maximize
last-bin fill. Uses numpy for vectorized overlap checks."""
import time
import random
import numpy as np

_DEADLINE = 140.0  # safety margin before the 150s kill


def _overlap_fast(box, items_arr, W, H, D):
    """Check if box at position overlaps any item. items_arr is Nx6 array."""
    x, y, z, w, h, d = box
    if items_arr.shape[0] == 0:
        return False
    # Vectorized overlap check
    ax, ay, az = x, y, z
    bx = items_arr[:, 0]
    by = items_arr[:, 1]
    bz = items_arr[:, 2]
    bw = items_arr[:, 3]
    bh = items_arr[:, 4]
    bd = items_arr[:, 5]
    
    # Overlap in all 3 dimensions
    ox = (ax < bx + bw) & (bx < ax + w)
    oy = (ay < by + bh) & (by < ay + h)
    oz = (az < bz + bd) & (bz < az + d)
    return bool(np.any(ox & oy & oz))


def _find_corner(box, items_arr, W, H, D):
    """Find the lowest (z, then y, then x) corner where box fits without overlap."""
    w, h, d = box
    n = items_arr.shape[0]
    
    if n == 0:
        return (0, 0, 0)
    
    # Candidate coordinates: 0 and right/top/back edges of existing items
    xs = sorted(set([0] + [int(it[0] + it[3]) for it in items_arr if it[0] + it[3] + w <= W]))
    ys = sorted(set([0] + [int(it[1] + it[4]) for it in items_arr if it[1] + it[4] + h <= H]))
    zs = sorted(set([0] + [int(it[2] + it[5]) for it in items_arr if it[2] + it[5] + d <= D]))
    
    for z in zs:
        for y in ys:
            for x in xs:
                if not _overlap_fast((x, y, z, w, h, d), items_arr, W, H, D):
                    return (x, y, z)
    return None


def _best_bin(box, bins_arr, bin_vols, W, H, D):
    """Return (bin_index, x, y, z) of the best fit for box, or None.
    Best-fit: prefer bin with highest current volume (fill it up),
    tie-break by lower z, then y, then x."""
    w, h, d = box
    if w > W or h > H or d > D:
        return None

    best = None
    best_key = None
    for bi in range(len(bins_arr)):
        items_arr = bins_arr[bi]
        placed = _find_corner(box, items_arr, W, H, D)
        if placed is None:
            continue
        x, y, z = placed
        vol = bin_vols[bi]
        key = (-vol, z, y, x)
        if best_key is None or key < best_key:
            best_key = key
            best = (bi, x, y, z)
    return best


def _compute_score(placements, container):
    W, H, D = container
    cvol = W * H * D
    if not placements:
        return 0.0
    bins = {}
    last = -1
    for bi, x, y, z, w, h, d in placements:
        bins.setdefault(bi, 0)
        bins[bi] += w * h * d
        if bi > last:
            last = bi
    B = len(bins)
    last_vol = bins.get(last, 0)
    return B + 0.1 * (1 - last_vol / cvol)


def _pack_with_order(boxes, container, order):
    """Greedy best-fit placement with a given box order."""
    W, H, D = container
    n = len(boxes)
    bins_arr = []
    bin_vols = []
    pos = [None] * n

    for idx in order:
        box = boxes[idx]
        w, h, d = box
        res = _best_bin(box, bins_arr, bin_vols, W, H, D)
        if res is not None:
            bi, x, y, z = res
            bins_arr[bi] = np.vstack([bins_arr[bi], np.array([x, y, z, w, h, d])]) if bins_arr[bi].shape[0] > 0 else np.array([[x, y, z, w, h, d]])
            bin_vols[bi] += w * h * d
            pos[idx] = (bi, x, y, z, w, h, d)
        else:
            bi = len(bins_arr)
            bins_arr.append(np.array([[0, 0, 0, w, h, d]]))
            bin_vols.append(w * h * d)
            pos[idx] = (bi, 0, 0, 0, w, h, d)
    return pos, bins_arr, bin_vols


def _consolidate(pos, bins_arr, boxes, container, t0, deadline):
    """Try to move boxes from the last bin into earlier bins to reduce B."""
    W, H, D = container
    n = len(boxes)

    def current_score():
        return _compute_score([pos[i] for i in range(n)], container)

    best_pos = [list(p) for p in pos]
    best_s = current_score()

    def try_move(box_idx, target_bin):
        bi, x, y, z, w, h, d = pos[box_idx]
        if target_bin == bi:
            return False
        
        # Find the item in source bin
        src_arr = bins_arr[bi]
        found_idx = -1
        for i in range(src_arr.shape[0]):
            if (src_arr[i, 0] == x and src_arr[i, 1] == y and src_arr[i, 2] == z and 
                src_arr[i, 3] == w and src_arr[i, 4] == h and src_arr[i, 5] == d):
                found_idx = i
                break
        if found_idx == -1:
            return False
        
        # Remove from source
        src_arr = np.delete(src_arr, found_idx, axis=0)
        bins_arr[bi] = src_arr
        bin_vols[bi] -= w * h * d
        
        # Try to place in target bin
        placed = _find_corner((w, h, d), bins_arr[target_bin], W, H, D)
        if placed is None:
            # Restore
            bins_arr[bi] = np.vstack([src_arr, np.array([x, y, z, w, h, d])]) if src_arr.shape[0] > 0 else np.array([[x, y, z, w, h, d]])
            bin_vols[bi] += w * h * d
            return False
        
        nx, ny, nz = placed
        bins_arr[target_bin] = np.vstack([bins_arr[target_bin], np.array([nx, ny, nz, w, h, d])]) if bins_arr[target_bin].shape[0] > 0 else np.array([[nx, ny, nz, w, h, d]])
        bin_vols[target_bin] += w * h * d
        pos[box_idx] = (target_bin, nx, ny, nz, w, h, d)
        return True

    attempts = 0
    while time.time() - t0 < deadline - 5.0 and attempts < 5000:
        attempts += 1
        used_bins = sorted(set(b for b, *_ in pos))
        if len(used_bins) <= 1:
            break
        last_bin = used_bins[-1]
        cand = [i for i in range(n) if pos[i][0] == last_bin]
        if not cand:
            break
        random.shuffle(cand)
        moved = False
        for i in cand:
            for tb in used_bins[:-1]:
                if try_move(i, tb):
                    s = current_score()
                    if s < best_s:
                        best_s = s
                        best_pos = [list(p) for p in pos]
                    moved = True
                    break
            if moved:
                break
        if not moved:
            break

    return best_pos, best_s


def pack(boxes, container):
    W, H, D = container
    t0 = time.time()
    n = len(boxes)

    # Try multiple sorting strategies and keep the best
    strategies = []

    # 1. Volume descending
    strategies.append(sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True))
    # 2. Max dimension descending
    strategies.append(sorted(range(n), key=lambda i: max(boxes[i]), reverse=True))
    # 3. Surface area descending
    strategies.append(sorted(range(n), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True))

    best_pos = None
    best_score = float('inf')
    best_bins_arr = None
    best_bin_vols = None

    for order in strategies:
        if time.time() - t0 > _DEADLINE - 10:
            break
        pos, bins_arr, bin_vols = _pack_with_order(boxes, container, order)
        s = _compute_score([pos[i] for i in range(n)], container)
        if s < best_score:
            best_score = s
            best_pos = [list(p) for p in pos]
            best_bins_arr = bins_arr
            best_bin_vols = bin_vols

    # Run consolidation on the best solution
    if best_pos is not None and time.time() - t0 < _DEADLINE - 5:
        improved_pos, improved_s = _consolidate(best_pos, best_bins_arr, boxes, container, t0, _DEADLINE)
        if improved_s < best_score:
            best_score = improved_s
            best_pos = improved_pos

    # Return placements in original order
    return [tuple(best_pos[i]) for i in range(n)]
