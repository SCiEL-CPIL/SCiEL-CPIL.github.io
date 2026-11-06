"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import permutations

def _get_rotations(w, h, d):
    """Get all unique rotations of a box, sorted by preferred orientation."""
    rots = set()
    for perm in permutations((w, h, d)):
        rots.add(perm)
    # Sort: prefer lower z-extent (last dim) to stack better, then lower y, then lower x
    rots = sorted(rots, key=lambda r: (r[2], r[1], r[0]))
    return list(rots)

class _BSPNode:
    """A node in the Binary Space Partitioning tree."""
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
        self.box_idx = -1  # -1 means empty leaf
    
    def volume(self):
        return self.w * self.h * self.d
    
    def fits(self, rw, rh, rd):
        return self.w >= rw and self.h >= rh and self.d >= rd

def _bsp_place(boxes, container, sort_key='volume', deadline=None):
    """
    Place boxes using BSP tree approach.
    Each box is placed into the smallest available leaf that fits,
    then the leaf is split around the box.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    if deadline is None:
        start_time = time.time()
        deadline = start_time + 110.0
    
    # Precompute rotations for each box
    box_rotations = []
    for w, h, d in boxes:
        rots = _get_rotations(w, h, d)
        box_rotations.append(rots)
    
    # Sort boxes based on sort_key
    box_indices = list(range(n))
    if sort_key == 'volume':
        box_volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
        box_indices.sort(key=lambda i: box_volumes[i], reverse=True)
    elif sort_key == 'max_dim':
        max_dims = [max(boxes[i]) for i in range(n)]
        box_indices.sort(key=lambda i: max_dims[i], reverse=True)
    elif sort_key == 'aspect_ratio':
        def aspect_score(i):
            w, h, d = boxes[i]
            sides = sorted([w, h, d])
            return sides[0] / sides[2] if sides[2] > 0 else 1.0
        box_indices.sort(key=aspect_score, reverse=True)
    
    # Initialize BSP tree with the container as root
    root = _BSPNode(0, 0, 0, W, H, D)
    
    # Result: for each original box index, store placement or None
    result = [None] * n
    
    # Maintain a list of free leaves for quick access
    # Each entry is (node, depth) where depth helps prioritize shallower/smaller nodes
    free_leaves = [(root, 0)]
    
    for idx in box_indices:
        if time.time() > deadline:
            break
        
        w_orig, h_orig, d_orig = boxes[idx]
        rots = box_rotations[idx]
        
        best_node = None
        best_rot = None
        best_score = float('inf')
        
        # Try each rotation
        for rw, rh, rd in rots:
            # Find the smallest free leaf that fits this rotation
            # We want to minimize the wasted space in the leaf
            for node, depth in free_leaves:
                if not node.fits(rw, rh, rd):
                    continue
                
                # Score: prefer smaller leaves (less waste), then shallower depth
                waste = node.volume() - rw * rh * rd
                score = waste + depth * 0.001
                
                if score < best_score:
                    best_score = score
                    best_node = node
                    best_rot = (rw, rh, rd)
        
        if best_node is not None:
            bw, bh, bd = best_rot
            
            # Remove this node from free_leaves
            free_leaves.remove((best_node, [n2 for n2, d2 in free_leaves if n2 is best_node][0] if any(n2 is best_node for n2, d2 in free_leaves) else 0))
            
            # Actually, let's rebuild free_leaves properly
            # This is inefficient; let's use a different approach
            # For now, just mark this node as occupied and create sub-nodes
            best_node.box_idx = idx
            bx, by, bz = best_node.x, best_node.y, best_node.z
            bw, bh, bd = best_rot
            
            result[idx] = (0, bx, by, bz, bw, bh, bd)
            
            # Create sub-nodes around the placed box
            # The remaining space in the original node is split into up to 6 regions
            nx, ny, nz = best_node.x, best_node.y, best_node.z
            nw, nh, nd = best_node.w, best_node.h, best_node.d
            
            # Region above the box (z direction)
            if nz + bd < nz + nd:
                new_node = _BSPNode(nx, ny, nz + bd, nw, nh, nd - bd)
                free_leaves.append((new_node, 1))
            
            # Region to the right of the box (x direction)
            if nx + bw < nx + nw:
                new_node = _BSPNode(nx + bw, ny, nz, nw - bw, nh, bd)
                free_leaves.append((new_node, 1))
            
            # Region in front of the box (y direction)
            if ny + bh < ny + nh:
                new_node = _BSPNode(nx, ny + bh, nz, bw, nh - bh, bd)
                free_leaves.append((new_node, 1))
            
            # Corner regions
            # Top-right
            if nx + bw < nx + nw and nz + bd < nz + nd:
                new_node = _BSPNode(nx + bw, ny, nz + bd, nw - bw, nh, nd - bd)
                free_leaves.append((new_node, 1))
            
            # Top-front
            if ny + bh < ny + nh and nz + bd < nz + nd:
                new_node = _BSPNode(nx, ny + bh, nz + bd, bw, nh - bh, nd - bd)
                free_leaves.append((new_node, 1))
            
            # Right-front
            if nx + bw < nx + nw and ny + bh < ny + nh:
                new_node = _BSPNode(nx + bw, ny + bh, nz, nw - bw, nh - bh, bd)
                free_leaves.append((new_node, 1))
            
            # Top-right-front
            if nx + bw < nx + nw and ny + bh < ny + nh and nz + bd < nz + nd:
                new_node = _BSPNode(nx + bw, ny + bh, nz + bd, nw - bw, nh - bh, nd - bd)
                free_leaves.append((new_node, 1))
        else:
            result[idx] = (-1, 0, 0, 0, w_orig, h_orig, d_orig)
    
    # Fill any remaining None entries
    for i in range(n):
        if result[i] is None:
            w_i, h_i, d_i = boxes[i]
            result[i] = (-1, 0, 0, 0, w_i, h_i, d_i)
    
    return result

def _extreme_point_place(boxes, container, sort_key='volume', deadline=None):
    """
    Place boxes using extreme point / skyline heuristic.
    This is the original approach, kept as a fallback.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    if deadline is None:
        start_time = time.time()
        deadline = start_time + 110.0
    
    # Precompute rotations for each box
    box_rotations = []
    for w, h, d in boxes:
        rots = _get_rotations(w, h, d)
        box_rotations.append(rots)
    
    # Sort boxes based on sort_key
    box_indices = list(range(n))
    if sort_key == 'volume':
        box_volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
        box_indices.sort(key=lambda i: box_volumes[i], reverse=True)
    elif sort_key == 'max_dim':
        max_dims = [max(boxes[i]) for i in range(n)]
        box_indices.sort(key=lambda i: max_dims[i], reverse=True)
    elif sort_key == 'aspect_ratio':
        def aspect_score(i):
            w, h, d = boxes[i]
            sides = sorted([w, h, d])
            return sides[0] / sides[2] if sides[2] > 0 else 1.0
        box_indices.sort(key=aspect_score, reverse=True)
    
    placed = []
    result = [None] * n
    
    for idx in box_indices:
        if time.time() > deadline:
            break
        
        w_orig, h_orig, d_orig = boxes[idx]
        rots = box_rotations[idx]
        
        best_pos = None
        best_rot = None
        best_score = float('inf')
        
        # Generate candidate positions from placed boxes
        candidates = set()
        candidates.add((0, 0, 0))
        
        for px, py, pz, pw, ph, pd in placed:
            if px + pw <= W:
                candidates.add((px + pw, py, pz))
                candidates.add((px + pw, py + ph, pz))
                candidates.add((px + pw, py, pz + pd))
                candidates.add((px + pw, py + ph, pz + pd))
            if py + ph <= H:
                candidates.add((px, py + ph, pz))
                candidates.add((px + pw, py + ph, pz))
                candidates.add((px, py + ph, pz + pd))
                candidates.add((px + pw, py + ph, pz + pd))
            if pz + pd <= D:
                candidates.add((px, py, pz + pd))
                candidates.add((px + pw, py, pz + pd))
                candidates.add((px, py + ph, pz + pd))
                candidates.add((px + pw, py + ph, pz + pd))
        
        for rw, rh, rd in rots:
            for cx, cy, cz in candidates:
                if cx + rw > W or cy + rh > H or cz + rd > D:
                    continue
                
                # Check overlap
                overlap = False
                for px, py, pz, pw, ph, pd in placed:
                    if not (cx + rw <= px or px + pw <= cx or
                            cy + rh <= py or py + ph <= cy or
                            cz + rd <= pz or pz + pd <= cz):
                        overlap = True
                        break
                if overlap:
                    continue
                
                score = cz * 1.0 + cx * 0.5 + cy * 0.5
                if score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                    best_rot = (rw, rh, rd)
        
        if best_pos is not None:
            bx, by, bz = best_pos
            bw, bh, bd = best_rot
            placed.append((bx, by, bz, bw, bh, bd))
            result[idx] = (0, bx, by, bz, bw, bh, bd)
        else:
            result[idx] = (-1, 0, 0, 0, w_orig, h_orig, d_orig)
    
    for i in range(n):
        if result[i] is None:
            w_i, h_i, d_i = boxes[i]
            result[i] = (-1, 0, 0, 0, w_i, h_i, d_i)
    
    return result

def _try_place_multi(boxes, container, deadline=None):
    """
    Try multiple strategies and return the best result.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    if deadline is None:
        start_time = time.time()
        deadline = start_time + 110.0
    
    best_result = None
    best_volume = -1
    
    # Try BSP approach with different sorting
    for sort_key in ['volume', 'max_dim', 'aspect_ratio']:
        if time.time() > deadline - 5:
            break
        
        result = _bsp_place(boxes, container, sort_key=sort_key, deadline=deadline)
        packed_vol = sum(r[3] * r[4] * r[5] for r in result if r[0] == 0)
        
        if packed_vol > best_volume:
            best_volume = packed_vol
            best_result = result
    
    # Try extreme point approach as well
    for sort_key in ['volume', 'max_dim']:
        if time.time() > deadline - 5:
            break
        
        result = _extreme_point_place(boxes, container, sort_key=sort_key, deadline=deadline)
        packed_vol = sum(r[3] * r[4] * r[5] for r in result if r[0] == 0)
        
        if packed_vol > best_volume:
            best_volume = packed_vol
            best_result = result
    
    if best_result is None:
        best_result = _extreme_point_place(boxes, container, sort_key='volume', deadline=deadline)
    
    return best_result

def load(boxes, container):
    """
    Place boxes into container. Returns list of (bin, x, y, z, w, h, d) for each box.
    bin=0 for loaded, bin=-1 for left out.
    """
    W, H, D = container
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    placements = _try_place_multi(boxes, container, deadline=deadline)
    
    # Final safety check
    if time.time() > deadline:
        for i, p in enumerate(placements):
            if p[0] == 0:
                if p[1] < 0 or p[2] < 0 or p[3] < 0:
                    w, h, d = boxes[i]
                    placements[i] = (-1, 0, 0, 0, w, h, d)
    
    return placements
# EVOLVE-BLOCK-END
