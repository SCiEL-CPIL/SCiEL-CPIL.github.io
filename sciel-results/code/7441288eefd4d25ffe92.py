"""3D Container Loading with advanced shelf packing and multi-strategy search."""
import time
import itertools


# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    deadline = time.time() + 110.0  # safety margin before 120s kill

    # Precompute unique orientations for each box
    box_orients = []
    for w, h, d in boxes:
        opts = set()
        for perm in itertools.permutations((w, h, d)):
            opts.add(perm)
        box_orients.append(list(opts))

    def shelf_pack(order, orient_choice):
        """Shelf packing: rows along X, shelves along Y, layers along Z."""
        placed_vol = 0
        result = [None] * len(order)
        x, y, z = 0, 0, 0
        row_h, layer_d = 0, 0
        for idx, bi in enumerate(order):
            ow, oh, od = box_orients[bi][orient_choice[idx]]
            # Try to place
            if x + ow > W:
                x, y, row_h = 0, y + row_h, 0
            if y + oh > H:
                y, z, layer_d, x, row_h = 0, z + layer_d, 0, 0, 0
            if z + od > D:
                result[idx] = -1
                continue
            result[idx] = (x, y, z, ow, oh, od)
            placed_vol += ow * oh * od
            x, row_h, layer_d = x + ow, max(row_h, oh), max(layer_d, od)
        return placed_vol, result

    def build_result(order, orient_choice, res):
        placed_vol, result = res
        placements = [None] * len(boxes)
        for idx, bi in enumerate(order):
            w, h, d = boxes[bi]
            if result[idx] is None or result[idx] == -1:
                placements[bi] = (-1, 0, 0, 0, w, h, d)
            else:
                px, py, pz, pw, ph, pd = result[idx]
                placements[bi] = (0, px, py, pz, pw, ph, pd)
        return placed_vol, placements

    def greedy_orient(order):
        """For each box in order, pick orientation that fits best in current shelf."""
        x, y, z = 0, 0, 0
        row_h, layer_d = 0, 0
        choices = []
        for bi in order:
            best_o = 0
            best_score = -1
            for oi, (ow, oh, od) in enumerate(box_orients[bi]):
                # Simulate placement with this orientation
                nx, ny, nz = x, y, z
                nrow_h, nlayer_d = row_h, layer_d
                if nx + ow > W:
                    nx, ny, nrow_h = 0, ny + nrow_h, 0
                if ny + oh > H:
                    ny, nz, nlayer_d, nx, nrow_h = 0, nz + nlayer_d, 0, 0, 0
                if nz + od > D:
                    score = -1
                else:
                    # Prefer orientations that keep row_h and layer_d small
                    score = -(max(nrow_h, oh) * 1.0 + max(nlayer_d, od) * 1.0 + nx * 0.1)
                if score > best_score:
                    best_score = score
                    best_o = oi
            choices.append(best_o)
            # Update state with chosen orientation
            ow, oh, od = box_orients[bi][best_o]
            if x + ow > W:
                x, y, row_h = 0, y + row_h, 0
            if y + oh > H:
                y, z, layer_d, x, row_h = 0, z + layer_d, 0, 0, 0
            if z + od > D:
                pass
            else:
                x, row_h, layer_d = x + ow, max(row_h, oh), max(layer_d, od)
        return choices

    # Multiple sorting strategies
    strategies = []

    # 1. Volume descending
    vol_order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    strategies.append(vol_order)

    # 2. Largest dimension first
    maxdim_order = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    strategies.append(maxdim_order)

    # 3. Volume ascending (sometimes helps)
    vol_asc = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2])
    strategies.append(vol_asc)

    # 4. Sorted by (max_dim, volume)
    mixed_order = sorted(range(n), key=lambda i: (max(boxes[i]), boxes[i][0]*boxes[i][1]*boxes[i][2]), reverse=True)
    strategies.append(mixed_order)

    # 5. Original order
    strategies.append(list(range(n)))

    # 6. Volume descending but with smallest-dimension-first tiebreak
    vol_smallest = sorted(range(n), key=lambda i: (boxes[i][0]*boxes[i][1]*boxes[i][2], min(boxes[i])), reverse=True)
    strategies.append(vol_smallest)

    # 7. Sorted by surface area descending
    area_order = sorted(range(n), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
    strategies.append(area_order)

    # 8. Sorted by (volume, max_dim) with volume primary
    vol_maxdim = sorted(range(n), key=lambda i: (boxes[i][0]*boxes[i][1]*boxes[i][2], max(boxes[i])), reverse=True)
    strategies.append(vol_maxdim)

    best_vol = 0
    best_placements = None

    for order in strategies:
        if time.time() > deadline:
            break
        choices = greedy_orient(order)
        res = shelf_pack(order, choices)
        placed_vol, placements = build_result(order, choices, res)
        if placed_vol > best_vol:
            best_vol = placed_vol
            best_placements = placements

    # If nothing was placed, fall back to a simple approach
    if best_placements is None:
        best_placements = []
        x, y, z, row_h, layer_d = 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d, x, row_h = 0, z + layer_d, 0, 0, 0
            if z + d > D:
                best_placements.append((-1, 0, 0, 0, w, h, d))
                continue
            best_placements.append((0, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)

    return best_placements
# EVOLVE-BLOCK-END
