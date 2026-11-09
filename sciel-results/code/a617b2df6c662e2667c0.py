"""3D Bin Packing with BSP tree for fast empty-space queries and multi-strategy search."""
import time
import math
import random
import sys

_DEADLINE = 145.0
_START = None


def _now():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START


class BSPNode:
    __slots__ = ('x', 'y', 'z', 'w', 'h', 'd', 'left', 'right')

    def __init__(self, x, y, z, w, h, d):
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.h = h
        self.d = d
        self.left = None
        self.right = None


class BSP:
    def __init__(self, W, H, D):
        self.root = BSPNode(0, 0, 0, W, H, D)
        self.W = W
        self.H = H
        self.D = D

    def insert(self, x, y, z, w, h, d):
        """Insert a box at (x,y,z) with dimensions (w,h,d). Returns True if successful."""
        self._insert(self.root, x, y, z, w, h, d)

    def _insert(self, node, x, y, z, w, h, d):
        # Check if box fits within this node
        if x < node.x or y < node.y or z < node.z:
            return False
        if x + w > node.x + node.w or y + h > node.y + node.h or z + d > node.z + node.d:
            return False

        # If node is a leaf (no children), try to split
        if node.left is None and node.right is None:
            # Try to place at origin of this node first (best fit)
            # Actually, we want to place the box and split the remaining space
            # We'll try different split strategies
            
            # Strategy: place box at (x, y, z) within this node
            # The remaining space needs to be split into up to 6 regions, but BSP only splits in 2
            # So we need to recursively split
            
            # First, check if box fits exactly
            if x == node.x and y == node.y and z == node.z:
                # Box is at the corner; split along one axis
                # Choose the axis that leaves the largest remaining space
                remaining_w = node.w - w
                remaining_h = node.h - h
                remaining_d = node.d - d
                
                # Try splitting along x first
                if remaining_w > 0:
                    node.w = w
                    node.left = BSPNode(node.x + w, node.y, node.z, remaining_w, node.h, node.d)
                    node.right = None
                    return True
                elif remaining_h > 0:
                    node.h = h
                    node.left = BSPNode(node.x, node.y + h, node.z, node.w, remaining_h, node.d)
                    node.right = None
                    return True
                elif remaining_d > 0:
                    node.d = d
                    node.left = BSPNode(node.x, node.y, node.z + d, node.w, node.h, remaining_d)
                    node.right = None
                    return True
                else:
                    # Box fills the entire node
                    node.w = w
                    node.h = h
                    node.d = d
                    return True
            
            # Box doesn't start at node origin - need to split the node first
            # Split along x-axis at x
            if x > node.x:
                # Left part: (node.x, node.y, node.z, x-node.x, node.h, node.d)
                # Right part: (x, node.y, node.z, node.w-(x-node.x), node.h, node.d)
                left_node = BSPNode(node.x, node.y, node.z, x - node.x, node.h, node.d)
                right_node = BSPNode(x, node.y, node.z, node.w - (x - node.x), node.h, node.d)
                
                # Try inserting into right part
                saved_w, saved_h, saved_d = node.w, node.h, node.d
                node.w = x - node.x
                node.left = left_node
                node.right = right_node
                
                if self._insert(right_node, x, y, z, w, h, d):
                    return True
                
                # Try inserting into left part
                node.w = saved_w
                node.h = saved_h
                node.d = saved_d
                node.left = left_node
                node.right = right_node
                
                if self._insert(left_node, x, y, z, w, h, d):
                    return True
                
                # Backtrack
                node.w = saved_w
                node.h = saved_h
                node.d = saved_d
                node.left = None
                node.right = None
                return False
            
            # Try splitting along y-axis at y
            if y > node.y:
                left_node = BSPNode(node.x, node.y, node.z, node.w, y - node.y, node.d)
                right_node = BSPNode(node.x, y, node.z, node.w, node.h - (y - node.y), node.d)
                
                saved_w, saved_h, saved_d = node.w, node.h, node.d
                node.h = y - node.y
                node.left = left_node
                node.right = right_node
                
                if self._insert(right_node, x, y, z, w, h, d):
                    return True
                
                node.h = saved_h
                if self._insert(left_node, x, y, z, w, h, d):
                    return True
                
                node.h = saved_d
                node.left = None
                node.right = None
                return False
            
            # Try splitting along z-axis at z
            if z > node.z:
                left_node = BSPNode(node.x, node.y, node.z, node.w, node.h, z - node.z)
                right_node = BSPNode(node.x, node.y, z, node.w, node.h, node.d - (z - node.z))
                
                saved_w, saved_h, saved_d = node.w, node.h, node.d
                node.d = z - node.z
                node.left = left_node
                node.right = right_node
                
                if self._insert(right_node, x, y, z, w, h, d):
                    return True
                
                node.d = saved_d
                if self._insert(left_node, x, y, z, w, h, d):
                    return True
                
                node.d = saved_d
                node.left = None
                node.right = None
                return False
            
            return False
        
        # Node has children; try inserting into them
        if node.left is not None:
            if self._insert(node.left, x, y, z, w, h, d):
                return True
        if node.right is not None:
            if self._insert(node.right, x, y, z, w, h, d):
                return True
        
        return False

    def find_position(self, w, h, d):
        """Find a position for a box with dimensions (w,h,d). Returns (x,y,z) or None."""
        result = self._find(self.root, w, h, d)
        return result

    def _find(self, node, w, h, d):
        # Check if box fits in this node
        if w > node.w or h > node.h or d > node.d:
            return None

        # If leaf, place at node origin
        if node.left is None and node.right is None:
            return (node.x, node.y, node.z)

        # Try children
        if node.left is not None:
            result = self._find(node.left, w, h, d)
            if result is not None:
                return result
        if node.right is not None:
            result = self._find(node.right, w, h, d)
            if result is not None:
                return result
        
        return None


def _pack_bsp(boxes, W, H, D, deadline, order='volume_desc'):
    """Pack using BSP tree."""
    bins = []  # list of BSP objects
    placements = []
    t0 = time.time()

    # Sort boxes according to order
    indexed = list(enumerate(boxes))
    if order == 'volume_desc':
        indexed.sort(key=lambda t: -t[1][0] * t[1][1] * t[1][2])
    elif order == 'volume_asc':
        indexed.sort(key=lambda t: t[1][0] * t[1][1] * t[1][2])
    elif order == 'largest_dim_desc':
        indexed.sort(key=lambda t: -max(t[1]))
    elif order == 'surface_desc':
        indexed.sort(key=lambda t: -(2*(t[1][0]*t[1][1] + t[1][1]*t[1][2] + t[1][0]*t[1][2])))
    else:
        indexed.sort(key=lambda t: -t[1][0] * t[1][1] * t[1][2])

    for orig_idx, (w, h, d) in indexed:
        if time.time() - t0 > deadline:
            break

        best_bin = -1
        best_pos = None
        best_score = float('inf')

        for bi, bsp in enumerate(bins):
            pos = bsp.find_position(w, h, d)
            if pos is not None:
                x, y, z = pos
                # Score: prefer positions that leave more space (max-cube heuristic)
                mc = (W - x - w) * (H - y - h) * (D - z - d)
                score = -mc
                if score < best_score:
                    best_score = score
                    best_bin = bi
                    best_pos = pos
                # With BSP, first fit is reasonable; but let's check all bins for best fit
                # Actually, for speed, we can break after finding first fit in best-fit strategy
                # But let's check all bins to find the best position overall
                # For now, break after first bin that fits (first-fit)
                break

        if best_bin == -1:
            best_bin = len(bins)
            bins.append(BSP(W, H, D))
            best_pos = (0, 0, 0)

        bins[best_bin].insert(best_pos[0], best_pos[1], best_pos[2], w, h, d)
        placements.append((best_bin, best_pos[0], best_pos[1], best_pos[2], w, h, d))

    return _pack_bsp_v2(boxes, W, H, D, deadline, order)


def _pack_bsp_v2(boxes, W, H, D, deadline, order='volume_desc'):
    """Pack using BSP tree, preserving original box order in output."""
    bins = []  # list of BSP objects
    placements = [None] * len(boxes)
    t0 = time.time()

    indexed = list(range(len(boxes)))
    if order == 'volume_desc':
        indexed.sort(key=lambda i: -boxes[i][0] * boxes[i][1] * boxes[i][2])
    elif order == 'volume_asc':
        indexed.sort(key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2])
    elif order == 'largest_dim_desc':
        indexed.sort(key=lambda i: -max(boxes[i]))
    elif order == 'surface_desc':
        indexed.sort(key=lambda i: -(2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2])))
    else:
        indexed.sort(key=lambda i: -boxes[i][0] * boxes[i][1] * boxes[i][2])

    for orig_idx in indexed:
        if time.time() - t0 > deadline:
            break

        w, h, d = boxes[orig_idx]
        best_bin = -1
        best_pos = None
        best_score = float('inf')

        for bi, bsp in enumerate(bins):
            pos = bsp.find_position(w, h, d)
            if pos is not None:
                x, y, z = pos
                mc = (W - x - w) * (H - y - h) * (D - z - d)
                score = -mc
                if score < best_score:
                    best_score = score
                    best_bin = bi
                    best_pos = pos
                break  # first-fit within bins

        if best_bin == -1:
            best_bin = len(bins)
            bins.append(BSP(W, H, D))
            best_pos = (0, 0, 0)

        bins[best_bin].insert(best_pos[0], best_pos[1], best_pos[2], w, h, d)
        placements[orig_idx] = (best_bin, best_pos[0], best_pos[1], best_pos[2], w, h, d)

    # Fill any remaining unplaced boxes with a simple fallback
    for i in range(len(boxes)):
        if placements[i] is None:
            w, h, d = boxes[i]
            best_bin = len(bins)
            bins.append(BSP(W, H, D))
            placements[i] = (best_bin, 0, 0, 0, w, h, d)
            bins[best_bin].insert(0, 0, 0, w, h, d)

    return placements


def _pack_best_fit(boxes, W, H, D, deadline):
    """Classic best-fit decreasing with overlap check."""
    bins = []  # each: list of (x,y,z,w,h,d)
    placements = [None] * len(boxes)
    t0 = time.time()

    indexed = list(range(len(boxes)))
    indexed.sort(key=lambda i: -boxes[i][0] * boxes[i][1] * boxes[i][2])

    def fits(bin_boxes, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            if not (x + w <= bx or bx + bw <= x or
                    y + h <= by or by + bh <= y or
                    z + d <= bz or bz + bd <= z):
                return False
        return True

    def candidate_positions(bin_boxes):
        xs = [0]
        ys = [0]
        zs = [0]
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
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
        cands.sort(key=lambda p: (p[0], p[1], p[2]))
        return cands

    for orig_idx in indexed:
        if time.time() - t0 > deadline:
            break

        w, h, d = boxes[orig_idx]
        best_bin = -1
        best_pos = None
        best_score = float('inf')

        for bi, bin_boxes in enumerate(bins):
            cands = candidate_positions(bin_boxes)
            for (x, y, z) in cands:
                if fits(bin_boxes, x, y, z, w, h, d):
                    mc = (W - x - w) * (H - y - h) * (D - z - d)
                    score = -mc
                    if score < best_score:
                        best_score = score
                        best_bin = bi
                        best_pos = (x, y, z)
                    break  # first valid position in this bin

        if best_bin == -1:
            best_bin = len(bins)
            bins.append([])
            best_pos = (0, 0, 0)

        bins[best_bin].append((best_pos[0], best_pos[1], best_pos[2], w, h, d))
        placements[orig_idx] = (best_bin, best_pos[0], best_pos[1], best_pos[2], w, h, d)

    for i in range(len(boxes)):
        if placements[i] is None:
            w, h, d = boxes[i]
            best_bin = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            placements[i] = (best_bin, 0, 0, 0, w, h, d)

    return placements


def _compute_score(placements, boxes, container):
    """Compute the score: B + 0.1*(1 - last-bin occupied volume fraction)."""
    if not placements:
        return 0.0

    W, H, D = container
    container_vol = W * H * D

    bin_volumes = {}
    for (b, x, y, z, w, h, d) in placements:
        if b is not None:
            bin_volumes[b] = bin_volumes.get(b, 0) + w * h * d

    B = len(bin_volumes)
    if B == 0:
        return 0.0

    last_bin = max(bin_volumes.keys())
    last_vol = bin_volumes[last_bin]
    last_frac = last_vol / container_vol if container_vol > 0 else 0.0

    return B + 0.1 * (1.0 - last_frac)


def _validate(placements, boxes, container):
    """Check that placement is valid: no overlaps, within bounds, all boxes placed."""
    W, H, D = container
    if len(placements) != len(boxes):
        return False

    bins = {}
    for i, (b, x, y, z, w, h, d) in enumerate(placements):
        if b is None:
            return False
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        if (w, h, d) != boxes[i]:
            return False
        bins.setdefault(b, []).append((x, y, z, w, h, d))

    for b, bin_boxes in bins.items():
        for i in range(len(bin_boxes)):
            for j in range(i + 1, len(bin_boxes)):
                x1, y1, z1, w1, h1, d1 = bin_boxes[i]
                x2, y2, z2, w2, h2, d2 = bin_boxes[j]
                if not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                        y1 + h1 <= y2 or y2 + h2 <= y1 or
                        z1 + d1 <= z2 or z2 + d2 <= z1):
                    return False

    return True


def pack(boxes, container):
    W, H, D = container
    deadline = _DEADLINE - _now()
    if deadline <= 0:
        deadline = 2.0

    orders = ['volume_desc', 'largest_dim_desc', 'surface_desc', 'volume_asc']
    best_placements = None
    best_score = float('inf')

    for order in orders:
        if _now() > _DEADLINE - 5.0:
            break
        sub_deadline = min(deadline, 30.0)
        try:
            placements = _pack_bsp_v2(boxes, W, H, D, sub_deadline, order)
            if _validate(placements, boxes, container):
                score = _compute_score(placements, boxes, container)
                if score < best_score:
                    best_score = score
                    best_placements = placements
        except Exception:
            pass

    if best_placements is None:
        # Fallback to best-fit
        try:
            best_placements = _pack_best_fit(boxes, W, H, D, max(5.0, deadline))
        except Exception:
            # Last resort: place all in one bin at origin (may overlap, but avoids crash)
            best_placements = [(0, 0, 0, 0, w, h, d) for (w, h, d) in boxes]

    return best_placements
