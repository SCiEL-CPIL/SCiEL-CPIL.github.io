"""
3D Bin Packing: Best-fit with diverse orderings and targeted bin management.
Strategy:
1. Generate multiple box orderings (volume desc, max-dim desc, random shuffles).
2. For each ordering, use a best-fit heuristic that tries to place each box in
   the bin/position that minimizes a composite score (height, bounding-box volume).
3. Use a "targeted" approach: prefer filling the current bin before opening a new one,
   but allow new bins when no existing bin can accommodate the box.
4. Track the best solution by the official score: B + 0.1*(1 - last_bin_fill).
"""
import time
import random
import math

DEADLINE = time.time() + 140.0


def _fits(placements, cx, cy, cz, w, h, d, W, H, D):
    """Check if a box fits at (cx, cy, cz) without overlapping existing placements."""
    if cx < 0 or cy < 0 or cz < 0:
        return False
    if cx + w > W or cy + h > H or cz + d > D:
        return False
    for (px, py, pz, pw, ph, pd) in placements:
        if (cx < px + pw and px < cx + w and
                cy < py + ph and py < cy + h and
                cz < pz + pd and pz < cz + d):
            return False
    return True


def _get_anchors(placements, W, H, D):
    """
    Generate candidate positions (anchors) for placing a new box.
    Anchors are the origin and the faces of existing boxes, plus combinations.
    """
    anchors = [(0, 0, 0)]
    for (px, py, pz, pw, ph, pd) in placements:
        # Right face
        if px + pw < W:
            anchors.append((px + pw, py, pz))
        # Top face
        if py + ph < H:
            anchors.append((px, py + ph, pz))
        # Front face
        if pz + pd < D:
            anchors.append((px, py, pz + pd))
        # Right + Top
        if px + pw < W and py + ph < H:
            anchors.append((px + pw, py + ph, pz))
        # Right + Front
        if px + pw < W and pz + pd < D:
            anchors.append((px + pw, py, pz + pd))
        # Top + Front
        if py + ph < H and pz + pd < D:
            anchors.append((px, py + ph, pz + pd))
        # Right + Top + Front
        if px + pw < W and py + ph < H and pz + pd < D:
            anchors.append((px + pw, py + ph, pz + pd))
    return anchors


def _bin_metrics(placements, W, H, D):
    """Calculate metrics for a bin."""
    if not placements:
        return {'vol': 0, 'max_z': 0, 'max_x': 0, 'max_y': 0, 'bbox_vol': 0}
    total_vol = sum(pw * ph * pd for (_, _, _, pw, ph, pd) in placements)
    max_z = max(pz + pd for (_, _, pz, _, _, pd) in placements)
    max_x = max(px + pw for (px, _, _, pw, _, _) in placements)
    max_y = max(py + ph for (_, py, _, _, ph, _) in placements)
    bbox_vol = max_x * max_y * max_z
    return {'vol': total_vol, 'max_z': max_z, 'max_x': max_x, 'max_y': max_y, 'bbox_vol': bbox_vol}


def _pack(boxes, container, order, rng, use_targeted=True):
    """
    Pack boxes using a best-fit strategy with optional targeted bin management.
    Returns a list of bins, each bin is a list of (idx, x, y, z, w, h, d).
    """
    W, H, D = container
    n = len(boxes)
    bins = []  # List of lists of (idx, x, y, z, w, h, d)
    bin_vols = []  # Volume in each bin

    for idx in order:
        w, h, d = boxes[idx]
        box_vol = w * h * d

        if not bins:
            bins.append([(idx, 0, 0, 0, w, h, d)])
            bin_vols.append(box_vol)
            continue

        best_bin_idx = -1
        best_pos = None
        best_score = float('inf')

        # Try all existing bins
        for bi in range(len(bins)):
            placements = [(x, y, z, w_, h_, d_) for (_, x, y, z, w_, h_, d_) in bins[bi]]
            anchors = _get_anchors(placements, W, H, D)

            for (cx, cy, cz) in anchors:
                if not _fits(placements, cx, cy, cz, w, h, d, W, H, D):
                    continue

                # Compute a score for this placement
                new_placements = placements + [(cx, cy, cz, w, h, d)]
                new_vol = bin_vols[bi] + box_vol
                max_z = max(pz + pd for (_, _, pz, _, _, pd) in new_placements)
                max_x = max(px + pw for (px, _, _, pw, _, _) in new_placements)
                max_y = max(py + ph for (_, py, _, _, ph, _) in new_placements)
                bbox_vol = max_x * max_y * max_z

                # Composite score: prefer lower height, then lower bbox volume, then higher fill
                fill = new_vol / bbox_vol if bbox_vol > 0 else 0
                score = max_z * 1000 + bbox_vol * 0.1 - fill * 500

                if use_targeted:
                    # Targeted: prefer bins that are already partially filled
                    # Add a small bonus for bins with higher current volume
                    score -= bin_vols[bi] * 0.01

                if score < best_score:
                    best_score = score
                    best_bin_idx = bi
                    best_pos = (cx, cy, cz)

        if best_bin_idx == -1:
            # No fit in existing bins, create new bin
            bins.append([(idx, 0, 0, 0, w, h, d)])
            bin_vols.append(box_vol)
        else:
            bins[best_bin_idx].append((idx, best_pos[0], best_pos[1], best_pos[2], w, h, d))
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
    """Generate diverse orderings of box indices."""
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
    best_order = None
    t_start = time.time()
    rng = random.Random(42)

    for order in _generate_orderings(boxes, n, rng):
        if time.time() > DEADLINE:
            break

        bins = _pack(boxes, container, order, rng, use_targeted=True)
        s = _score(bins, container)

        if s < best_score:
            best_score = s
            best_bins = bins
            best_order = order

    if best_bins is None:
        # Fallback: one box per bin
        best_bins = [[(i, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])] for i in range(n)]
        best_score = _score(best_bins, container)
        best_order = list(range(n))

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
