"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START

class BSPNode:
    """Binary Space Partition node representing an axis-aligned rectangular region."""
    __slots__ = ['x', 'y', 'z', 'w', 'h', 'd', 'left', 'right', 'box_idx']
    
    def __init__(self, x, y, z, w, h, d):
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.h = h
        self.d = d
        self.left = None
        self.right = None
        self.box_idx = -1  # -1 means free space

    def volume(self):
        return self.w * self.h * self.d

    def contains_point(self, px, py, pz):
        return (self.x <= px < self.x + self.w and
                self.y <= py < self.y + self.h and
                self.z <= pz < self.z + self.d)

    def split(self, axis, offset):
        """Split this node into two along the given axis at the given offset.
        Returns (left_node, right_node) where left is the smaller part."""
        if axis == 0:  # split along x
            left = BSPNode(self.x, self.y, self.z, offset, self.h, self.d)
            right = BSPNode(self.x + offset, self.y, self.z, self.w - offset, self.h, self.d)
        elif axis == 1:  # split along y
            left = BSPNode(self.x, self.y, self.z, self.w, offset, self.d)
            right = BSPNode(self.x, self.y + offset, self.z, self.w, self.h - offset, self.d)
        else:  # split along z
            left = BSPNode(self.x, self.y, self.z, self.w, self.h, offset)
            right = BSPNode(self.x, self.y, self.z + offset, self.w, self.h, self.d - offset)
        return left, right


class BinPacker:
    """Manages a single bin using BSP tree for space tracking."""
    
    def __init__(self, W, H, D):
        self.W = W
        self.H = H
        self.D = D
        self.root = BSPNode(0, 0, 0, W, H, D)
        self.placements = []  # list of (x, y, z, w, h, d, box_idx)
        self.free_volume = W * H * D
    
    def find_best_position(self, w, h, d):
        """Find the best-fit position for a box of dimensions (w, h, d).
        Returns (x, y, z) or None if no position fits."""
        best_pos = None
        best_score = float('inf')
        self._search_for_position(self.root, w, h, d, best_pos, best_score)
        # Re-search properly
        result = self._find_best(self.root, w, h, d)
        return result
    
    def _find_best(self, node, w, h, d):
        """Recursively find best-fit position. Returns (x,y,z,score) or None."""
        if node.box_idx >= 0:
            return None  # occupied
        
        # Check if box fits in this region
        if node.w < w or node.h < h or node.d < d:
            return None
        
        # If this is a leaf (no children), it's a candidate
        if node.left is None and node.right is None:
            # Score: prefer positions closer to origin (minimize x+y+z)
            score = node.x + node.y + node.z
            return (node.x, node.y, node.z, score)
        
        # Search children
        best = None
        if node.left:
            res = self._find_best(node.left, w, h, d)
            if res and (best is None or res[3] < best[3]):
                best = res
        if node.right:
            res = self._find_best(node.right, w, h, d)
            if res and (best is None or res[3] < best[3]):
                best = res
        return best
    
    def place_box(self, x, y, z, w, h, d, box_idx):
        """Place a box at the given position, updating the BSP tree."""
        self._place_in_node(self.root, x, y, z, w, h, d, box_idx)
        self.placements.append((x, y, z, w, h, d, box_idx))
        self.free_volume -= w * h * d
    
    def _place_in_node(self, node, x, y, z, w, h, d, box_idx):
        """Recursively place box in the BSP tree."""
        # Check if box is fully contained in this node
        if (x >= node.x and y >= node.y and z >= node.z and
            x + w <= node.x + node.w and
            y + h <= node.y + node.h and
            z + d <= node.z + node.d):
            
            # If this node is a leaf, mark it occupied
            if node.left is None and node.right is None:
                node.box_idx = box_idx
                return
            
            # If box exactly matches this node, mark occupied
            if (x == node.x and y == node.y and z == node.z and
                w == node.w and h == node.h and d == node.d):
                node.box_idx = box_idx
                node.left = None
                node.right = None
                return
            
            # Split and recurse
            self._split_and_place(node, x, y, z, w, h, d, box_idx)
            return
        
        # Box spans multiple regions - need to split
        self._split_and_place(node, x, y, z, w, h, d, box_idx)
    
    def _split_and_place(self, node, x, y, z, w, h, d, box_idx):
        """Split node as needed and place box."""
        # Try splitting along x first
        if x > node.x and x + w < node.x + node.w:
            # Split along x
            left, right = node.split(0, x - node.x)
            node.left = left
            node.right = right
            node.box_idx = -1
            self._place_in_node(left, x, y, z, w, h, d, box_idx)
            self._place_in_node(right, x, y, z, w, h, d, box_idx)
            return
        elif x + w < node.x + node.w:
            # Box is in left part
            left, right = node.split(0, x + w - node.x)
            node.left = left
            node.right = right
            node.box_idx = -1
            self._place_in_node(left, x, y, z, w, h, d, box_idx)
            return
        elif x > node.x:
            # Box is in right part
            left, right = node.split(0, x - node.x)
            node.left = left
            node.right = right
            node.box_idx = -1
            self._place_in_node(right, x, y, z, w, h, d, box_idx)
            return
        
        # Try splitting along y
        if y > node.y and y + h < node.y + node.h:
            left, right = node.split(1, y - node.y)
            node.left = left
            node.right = right
            node.box_idx = -1
            self._place_in_node(left, x, y, z, w, h, d, box_idx)
            self._place_in_node(right, x, y, z, w, h, d, box_idx)
            return
        elif y + h < node.y + node.h:
            left, right = node.split(1, y + h - node.y)
            node.left = left
            node.right = right
            node.box_idx = -1
            self._place_in_node(left, x, y, z, w, h, d, box_idx)
            return
        elif y > node.y:
            left, right = node.split(1, y - node.y)
            node.left = left
            node.right = right
            node.box_idx = -1
            self._place_in_node(right, x, y, z, w, h, d, box_idx)
            return
        
        # Try splitting along z
        if z > node.z and z + d < node.z + node.d:
            left, right = node.split(2, z - node.z)
            node.left = left
            node.right = right
            node.box_idx = -1
            self._place_in_node(left, x, y, z, w, h, d, box_idx)
            self._place_in_node(right, x, y, z, w, h, d, box_idx)
            return
        elif z + d < node.z + node.d:
            left, right = node.split(2, z + d - node.z)
            node.left = left
            node.right = right
            node.box_idx = -1
            self._place_in_node(left, x, y, z, w, h, d, box_idx)
            return
        elif z > node.z:
            left, right = node.split(2, z - node.z)
            node.left = left
            node.right = right
            node.box_idx = -1
            self._place_in_node(right, x, y, z, w, h, d, box_idx)
            return
        
        # If we get here, the box should exactly fill this node or there's an issue
        node.box_idx = box_idx
        node.left = None
        node.right = None


def check_overlap(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
    """Check if two boxes overlap."""
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                y1 + h1 <= y2 or y2 + h2 <= y1 or
                z1 + d1 <= z2 or z2 + d2 <= z1)


def try_place_box_simple(packed, x, y, z, w, h, d):
    """Check if a box can be placed at (x,y,z) without overlapping existing boxes."""
    for px, py, pz, pw, ph, pd in packed:
        if check_overlap(x, y, z, w, h, d, px, py, pz, pw, ph, pd):
            return False
    return True


def find_position_simple(packed, W, H, D, w, h, d):
    """Find a valid position using edge-aligned coordinates from existing boxes."""
    # Generate candidate x, y, z coordinates
    xs = [0]
    ys = [0]
    zs = [0]
    for px, py, pz, pw, ph, pd in packed:
        xs.append(px)
        xs.append(px + pw)
        ys.append(py)
        ys.append(py + ph)
        zs.append(pz)
        zs.append(pz + pd)
    
    # Filter valid positions
    xs = sorted(set(x for x in xs if x + w <= W))
    ys = sorted(set(y for y in ys if y + h <= H))
    zs = sorted(set(z for z in zs if z + d <= D))
    
    # Try all combinations, prefer positions closer to origin
    candidates = []
    for x in xs:
        for y in ys:
            for z in zs:
                if try_place_box_simple(packed, x, y, z, w, h, d):
                    score = x * W * H + y * W + z  # Prioritize filling from origin
                    candidates.append((score, x, y, z))
    
    if not candidates:
        return None
    
    candidates.sort()
    return (candidates[0][1], candidates[0][2], candidates[0][3])


def pack_single_order(boxes, container, order):
    """Pack boxes in the given order. Returns placements or None if failed."""
    W, H, D = container
    placements = []  # (bin, x, y, z, w, h, d)
    bin_packed = []  # list of lists, each containing (x,y,z,w,h,d)
    
    for idx in order:
        w, h, d = boxes[idx]
        placed = False
        
        # Try existing bins
        for b in range(len(bin_packed)):
            pos = find_position_simple(bin_packed[b], W, H, D, w, h, d)
            if pos is not None:
                x, y, z = pos
                bin_packed[b].append((x, y, z, w, h, d))
                placements.append((b, x, y, z, w, h, d))
                placed = True
                break
        
        if not placed:
            # Try new bin
            if w <= W and h <= H and d <= D:
                bin_packed.append([(0, 0, 0, w, h, d)])
                placements.append((len(bin_packed) - 1, 0, 0, 0, w, h, d))
            else:
                return None  # Box doesn't fit in any bin
    
    return placements


def compute_score(placements, container):
    """Compute the score: mean B + 0.1 * (1 - last-bin occupied volume / container volume)."""
    W, H, D = container
    container_vol = W * H * D
    
    if not placements:
        return 0.0
    
    bins = set(p[0] for p in placements)
    B = len(bins)
    
    # Find last bin (largest label)
    last_bin = max(bins)
    last_bin_vol = sum(p[4] * p[5] * p[6] for p in placements if p[0] == last_bin)
    
    if container_vol == 0:
        return float(B)
    
    return B + 0.1 * (1 - last_bin_vol / container_vol)


def generate_sort_orders(boxes):
    """Generate multiple sorting orders for the boxes."""
    n = len(boxes)
    orders = []
    
    # 1. Volume descending
    orders.append(sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True))
    
    # 2. Max dimension descending
    orders.append(sorted(range(n), key=lambda i: max(boxes[i][0], boxes[i][1], boxes[i][2]), reverse=True))
    
    # 3. Surface area descending
    orders.append(sorted(range(n), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True))
    
    # 4. Aspect ratio (max/min) descending
    orders.append(sorted(range(n), key=lambda i: max(boxes[i][0], boxes[i][1], boxes[i][2]) / max(min(boxes[i][0], boxes[i][1], boxes[i][2]), 1), reverse=True))
    
    # 5. Width descending
    orders.append(sorted(range(n), key=lambda i: boxes[i][0], reverse=True))
    
    # 6. Height descending
    orders.append(sorted(range(n), key=lambda i: boxes[i][1], reverse=True))
    
    # 7. Depth descending
    orders.append(sorted(range(n), key=lambda i: boxes[i][2], reverse=True))
    
    # 8. Volume ascending (smallest first)
    orders.append(sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2]))
    
    return orders


def pack(boxes, container):
    """Main packing function."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    W, H, D = container
    n = len(boxes)
    
    if n == 0:
        return []
    
    # Generate multiple sorting orders
    orders = generate_sort_orders(boxes)
    
    best_placements = None
    best_score = float('inf')
    
    for order in orders:
        if time.time() > deadline:
            break
        
        try:
            placements = pack_single_order(boxes, container, order)
            if placements is not None:
                score = compute_score(placements, container)
                if score < best_score:
                    best_score = score
                    best_placements = placements
        except Exception:
            continue
    
    # If no valid placement found, use a simple fallback
    if best_placements is None:
        # Fallback: place each box in its own bin at origin
        best_placements = [(i, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    
    return best_placements

# EVOLVE-BLOCK-END
