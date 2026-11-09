"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def pack(boxes, container):
    W, H, D = container
    container_vol = W * H * D
    n = len(boxes)
    if n == 0:
        return []

    t_start = time.time()
    DEADLINE = 115.0  # safety margin under 120s/150s kill

    # Precompute box volumes
    box_vols = [w * h * d for w, h, d in boxes]

    # ---------- 3D BSP Tree ----------
    class BSPNode:
        __slots__ = ['x', 'y', 'z', 'w', 'h', 'd', 'left', 'right']
        def __init__(self, x, y, z, w, h, d):
            self.x = x; self.y = y; self.z = z
            self.w = w; self.h = h; self.d = d
            self.left = None; self.right = None

    def bsp_insert(node, bx, by, bz, bw, bh, bd):
        """Insert box into BSP. Returns (success, node, split_axis, split_pos, leftover_nodes)."""
        # Check if box fits entirely within this node
        if (bx >= node.x and by >= node.y and bz >= node.z and
            bx + bw <= node.x + node.w and by + bh <= node.y + node.h and bz + bd <= node.z + node.d):
            if node.left is None and node.right is None:
                # Leaf node: place box here
                return True, node, -1, -1, []
            # Internal node: recurse
            # Try left first, then right
            if node.left is not None:
                ok, _, _, _, _ = bsp_insert(node.left, bx, by, bz, bw, bh, bd)
                if ok:
                    return True, node.left, -1, -1, []
            if node.right is not None:
                ok, _, _, _, _ = bsp_insert(node.right, bx, by, bz, bw, bh, bd)
                if ok:
                    return True, node.right, -1, -1, []
            # Neither child worked; this node is too fragmented
            return False, None, -1, -1, []

        # Box does not fit entirely in this node. Try to split.
        # Determine if we can split along an axis
        # We need the box to be aligned with the node's boundaries for a clean split
        # Check if box touches the near face on some axis
        # Try splitting on x-axis: box starts at node.x
        if bx == node.x:
            # Split at bx + bw
            split_pos = bx + bw
            if split_pos < node.x + node.w:
                # Left part: x in [node.x, split_pos), full y,z
                left = BSPNode(node.x, node.y, node.z, split_pos - node.x, node.h, node.d)
                # Right part: x in [split_pos, node.x+node.w), full y,z
                right = BSPNode(split_pos, node.y, node.z, node.x + node.w - split_pos, node.h, node.d)
                # Place box in left part
                if (bw <= left.w and bh <= left.h and bd <= left.d):
                    # Recursively place in left
                    ok, child_node, sa, sp, leftovers = bsp_insert(left, bx, by, bz, bw, bh, bd)
                    if ok:
                        node.left = left
                        node.right = right
                        return True, child_node, -1, -1, []
        # Try splitting on y-axis: box starts at node.y
        if by == node.y:
            split_pos = by + bh
            if split_pos < node.y + node.h:
                left = BSPNode(node.x, node.y, node.z, node.w, split_pos - node.y, node.d)
                right = BSPNode(node.x, split_pos, node.z, node.w, node.y + node.h - split_pos, node.d)
                if (bw <= left.w and bh <= left.h and bd <= left.d):
                    ok, child_node, sa, sp, leftovers = bsp_insert(left, bx, by, bz, bw, bh, bd)
                    if ok:
                        node.left = left
                        node.right = right
                        return True, child_node, -1, -1, []
        # Try splitting on z-axis: box starts at node.z
        if bz == node.z:
            split_pos = bz + bd
            if split_pos < node.z + node.d:
                left = BSPNode(node.x, node.y, node.z, node.w, node.h, split_pos - node.z)
                right = BSPNode(node.x, node.y, split_pos, node.w, node.h, node.z + node.d - split_pos)
                if (bw <= left.w and bh <= left.h and bd <= left.d):
                    ok, child_node, sa, sp, leftovers = bsp_insert(left, bx, by, bz, bw, bh, bd)
                    if ok:
                        node.left = left
                        node.right = right
                        return True, child_node, -1, -1, []

        return False, None, -1, -1, []

    def bsp_find_position(node, bw, bh, bd):
        """Find a position for box of size (bw,bh,bd) in the BSP tree. Returns (x,y,z) or None.
        Uses BLF heuristic: minimize z, then y, then x."""
        best = None
        best_key = None

        def _search(n):
            nonlocal best, best_key
            if n is None:
                return
            # Check if box fits in this node
            if bw <= n.w and bh <= n.h and bd <= n.d:
                # Candidate position is the node's origin (bottom-left-front)
                pos = (n.x, n.y, n.z)
                key = (n.z, n.y, n.x)
                if best_key is None or key < best_key:
                    best_key = key
                    best = pos
                # Don't recurse further; this is the best position in this subtree
                return
            # Recurse into children
            _search(n.left)
            _search(n.right)

        _search(node)
        return best

    # ---------- Packing Strategies ----------

    def try_packing_bsp(order, bin_order_strategy='best_fit'):
        """Pack boxes in given order using BSP per bin. Returns (placements, num_bins, last_bin_vol) or None."""
        bins_bsp = []       # list of BSPNode roots
        bins_boxes = []     # list of lists of (x,y,z,w,h,d) per bin
        bin_volumes = []    # cumulative volume per bin
        result_placements = [None] * n

        for idx in order:
            w, h, d = boxes[idx]
            placed = False

            if w > W or h > H or d > D:
                return None, 0, 0

            best_bin_idx = -1
            best_pos = None
            best_score_key = None

            for b_idx in range(len(bins_bsp)):
                pos = bsp_find_position(bins_bsp[b_idx], w, h, d)
                if pos is not None:
                    # Score: prefer most full bin (best fit)
                    vol_score = bin_volumes[b_idx]
                    # Tie-break by position (lower z,y,x is better)
                    key = (-vol_score, pos[2], pos[1], pos[0])
                    if best_score_key is None or key < best_score_key:
                        best_score_key = key
                        best_bin_idx = b_idx
                        best_pos = pos

            if best_bin_idx != -1:
                x, y, z = best_pos
                # Insert into BSP
                ok, _, _, _, _ = bsp_insert(bins_bsp[best_bin_idx], x, y, z, w, h, d)
                if not ok:
                    # Should not happen if find_position worked, but safety
                    return None, 0, 0
                bins_boxes[best_bin_idx].append((x, y, z, w, h, d))
                bin_volumes[best_bin_idx] += w * h * d
                result_placements[idx] = (best_bin_idx, x, y, z, w, h, d)
                placed = True
            else:
                # Open new bin
                b_idx = len(bins_bsp)
                root = BSPNode(0, 0, 0, W, H, D)
                ok, _, _, _, _ = bsp_insert(root, 0, 0, 0, w, h, d)
                if not ok:
                    return None, 0, 0
                bins_bsp.append(root)
                bins_boxes.append([(0, 0, 0, w, h, d)])
                bin_volumes.append(w * h * d)
                result_placements[idx] = (b_idx, 0, 0, 0, w, h, d)
                placed = True

            if not placed:
                return None, 0, 0

            if time.time() - t_start > DEADLINE:
                return None, 0, 0

        num_bins = len(bins_bsp)
        last_bin_idx = num_bins - 1
        last_bin_vol = bin_volumes[last_bin_idx]
        return result_placements, num_bins, last_bin_vol

    # ---------- Generate Orders ----------
    orders = []

    # 1. Volume descending (BFD)
    orders.append(sorted(range(n), key=lambda i: box_vols[i], reverse=True))

    # 2. Max dimension descending
    orders.append(sorted(range(n), key=lambda i: max(boxes[i][0], boxes[i][1], boxes[i][2]), reverse=True))

    # 3. Sum of dimensions descending
    orders.append(sorted(range(n), key=lambda i: sum(boxes[i]), reverse=True))

    # 4. Volume ascending
    orders.append(sorted(range(n), key=lambda i: box_vols[i]))

    # 5. Largest face area descending
    orders.append(sorted(range(n), key=lambda i: max(boxes[i][0]*boxes[i][1], boxes[i][0]*boxes[i][2], boxes[i][1]*boxes[i][2]), reverse=True))

    # 6-10. Randomized perturbations of BFD
    rng = random.Random(42)
    for _ in range(5):
        order = sorted(range(n), key=lambda i: box_vols[i], reverse=True)
        for _ in range(8):
            i, j = rng.sample(range(n), 2)
            order[i], order[j] = order[j], order[i]
        orders.append(order)

    # 11-14. More random orders with different seeds
    for seed in [1, 2, 3, 4]:
        rng2 = random.Random(seed)
        order = list(range(n))
        rng2.shuffle(order)
        orders.append(order)

    best_placements = None
    best_score = float('inf')

    for order in orders:
        if time.time() - t_start > DEADLINE:
            break

        placements, num_bins, last_bin_vol = try_packing_bsp(order)
        if placements is None:
            continue

        if num_bins == 0:
            score = float('inf')
        else:
            last_bin_frac = last_bin_vol / container_vol if container_vol > 0 else 0
            score = num_bins + 0.1 * (1 - last_bin_frac)

        if score < best_score:
            best_score = score
            best_placements = placements

    # Fallback: simple shelf packing if BSP failed entirely
    if best_placements is None:
        placements = []
        b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            placements.append((b, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        best_placements = placements

    return best_placements
# EVOLVE-BLOCK-END
