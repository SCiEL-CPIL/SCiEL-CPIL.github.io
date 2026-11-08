"""Improved 3D Bin Packing: fast best-fit with corner candidates and time guard."""
import time

# EVOLVE-BLOCK-START
def pack(boxes, container):
    """
    Place boxes into bins using a fast best-fit decreasing approach.
    Returns list of (bin, x, y, z, w, h, d) tuples.
    """
    start_time = time.time()
    deadline = 148.0  # safety margin before 150s kill

    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    # Indexed boxes: (orig_idx, w, h, d, vol)
    indexed_boxes = [(i, w, h, d, w * h * d) for i, (w, h, d) in enumerate(boxes)]

    def compute_score(placements):
        """B + 0.1*(1 - last-bin occupied volume / container volume)"""
        if not placements:
            return 0.0
        bins_used = set()
        last_bin = -1
        for (b, x, y, z, w, h, d) in placements:
            bins_used.add(b)
            if b > last_bin:
                last_bin = b
        last_bin_vol = 0
        for (b, x, y, z, w, h, d) in placements:
            if b == last_bin:
                last_bin_vol += w * h * d
        container_vol = W * H * D
        B = len(bins_used)
        occ_frac = last_bin_vol / container_vol if container_vol > 0 else 1.0
        return B + 0.1 * (1.0 - occ_frac)

    def find_placement(placed, w, h, d):
        """
        Find best (x, y, z) for box (w,h,d) in a bin with given placed boxes.
        placed: list of (x, y, z, w, h, d).
        Uses corner candidates from placed boxes plus origin.
        Returns (x, y, z) or None.
        """
        # Quick fit check
        if w > W or h > H or d > D:
            return None

        # Build candidate x, y, z values from box edges
        xs = {0}
        ys = {0}
        zs = {0}
        for (px, py, pz, pw, ph, pd) in placed:
            xs.add(px)
            xs.add(px + pw)
            ys.add(py)
            ys.add(py + ph)
            zs.add(pz)
            zs.add(pz + pd)

        # Filter to valid positions
        xs = [x for x in xs if x + w <= W]
        ys = [y for y in ys if y + h <= H]
        zs = [z for z in zs if z + d <= D]

        if not xs or not ys or not zs:
            return None

        best_pos = None
        # Prefer tight fits: minimize (x, y, z) lexicographically,
        # which also tends to keep boxes packed toward the origin.
        best_metric = None

        for cx in sorted(xs):
            for cy in sorted(ys):
                for cz in sorted(zs):
                    # Check overlap
                    overlap = False
                    for (px, py, pz, pw, ph, pd) in placed:
                        if not (cx + w <= px or px + pw <= cx or
                                cy + h <= py or py + ph <= cy or
                                cz + d <= pz or pz + pd <= cz):
                            overlap = True
                            break
                    if overlap:
                        continue
                    # Found a valid position
                    metric = (cx, cy, cz)
                    if best_metric is None or metric < best_metric:
                        best_metric = metric
                        best_pos = (cx, cy, cz)
                        # Since we iterate in sorted order, the first valid
                        # in each dimension is the minimum; break early if
                        # we find (0,0,0)
                        if cx == 0 and cy == 0 and cz == 0:
                            return best_pos

        return best_pos

    def solve(sorted_boxes):
        """Best-fit decreasing: try most-filled bins first."""
        bins = []  # each bin: {'placed': [...], 'vol': int}
        placements = []

        for (orig_idx, w, h, d, vol) in sorted_boxes:
            if time.time() > deadline:
                return None

            # Try existing bins, most-filled first (best-fit)
            # Sort bin indices by volume descending
            if bins:
                bin_order = sorted(range(len(bins)), key=lambda bi: -bins[bi]['vol'])
            else:
                bin_order = []

            placed_in_bin = False
            for bi in bin_order:
                pos = find_placement(bins[bi]['placed'], w, h, d)
                if pos is not None:
                    x, y, z = pos
                    bins[bi]['placed'].append((x, y, z, w, h, d))
                    bins[bi]['vol'] += vol
                    placements.append((bi, x, y, z, w, h, d))
                    placed_in_bin = True
                    break

            if not placed_in_bin:
                # Open new bin
                new_bin = {'placed': [(0, 0, 0, w, h, d)], 'vol': vol}
                bins.append(new_bin)
                placements.append((len(bins) - 1, 0, 0, 0, w, h, d))

        return placements

    # Multiple sorting strategies
    strategies = [
        lambda bl: sorted(bl, key=lambda x: -x[4]),                    # volume desc
        lambda bl: sorted(bl, key=lambda x: (-(x[4]), -(x[1]+x[2]+x[3]))),  # vol then sum
        lambda bl: sorted(bl, key=lambda x: (-(x[1]+x[2]+x[3]), -x[4])),    # sum then vol
        lambda bl: sorted(bl, key=lambda x: (-(x[1]*x[2]), -x[4])),     # area then vol
        lambda bl: sorted(bl, key=lambda x: (-(x[2]*x[3]), -x[4])),     # area then vol
        lambda bl: sorted(bl, key=lambda x: (-(x[1]*x[3]), -x[4])),     # area then vol
        lambda bl: sorted(bl, key=lambda x: (-max(x[1],x[2],x[3]), -x[4])),  # max dim then vol
    ]

    best_placements = None
    best_score = float('inf')

    for strat in strategies:
        if time.time() > deadline - 1.0:
            break
        sorted_boxes = strat(indexed_boxes)
        placements = solve(sorted_boxes)
        if placements is None:
            continue
        score = compute_score(placements)
        if score < best_score:
            best_score = score
            best_placements = placements

    if best_placements is None:
        # Fallback: simple sequential packing
        placements = []
        b = 0
        x = y = z = 0
        row_h = layer_d = 0
        for (orig_idx, w, h, d, vol) in indexed_boxes:
            if x + w > W:
                x = 0
                y += row_h
                row_h = 0
            if y + h > H:
                y = 0
                z += layer_d
                layer_d = 0
                row_h = 0
            if z + d > D:
                b += 1
                x = y = z = 0
                row_h = layer_d = 0
            placements.append((b, x, y, z, w, h, d))
            x += w
            row_h = max(row_h, h)
            layer_d = max(layer_d, d)
        best_placements = placements

    return best_placements
# EVOLVE-BLOCK-END
