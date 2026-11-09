<<<<<<< SEARCH
import time
import random
import itertools

def _rotations(w, h, d):
    """Return all 6 unique orientations of a box."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _fits(box_dims, pos, container):
    W, H, D = container
    x, y, z = pos
    w, h, d = box_dims
    return x + w <= W and y + h <= H and z + d <= D

def _no_overlap(box_dims, pos, placed):
    x, y, z = pos
    w, h, d = box_dims
    for (px, py, pz, pw, ph, pd) in placed:
        if (x < px + pw and x + w > px and
            y < py + ph and y + h > py and
            z < pz + pd and z + d > pz):
            return False
    return True

def _extreme_points(placed, container):
    W, H, D = container
    candidates = set()
    candidates.add((0, 0, 0))
    for (px, py, pz, pw, ph, pd) in placed:
        points = [
            (px + pw, py, pz),
            (px, py + ph, pz),
            (px, py, pz + pd),
            (px + pw, py + ph, pz),
            (px, py + ph, pz + pd),
            (px + pw, py, pz + pd),
            (px + pw, py + ph, pz + pd),
        ]
        for (cx, cy, cz) in points:
            if cx <= W and cy <= H and cz <= D:
                candidates.add((cx, cy, cz))
    return list(candidates)

def _try_place_eap(boxes, container, order_idx, deadline, scoring_func=None, rng=None):
    """Extreme Points Approach with optional custom scoring and randomization."""
    W, H, D = container
    placed = []
    placements = [None] * len(boxes)
    
    if scoring_func is None:
        def scoring_func(pos, dim):
            return (pos[2], pos[1], pos[0]) # Bottom, Left, Far
    
    for orig_idx in order_idx:
        if time.time() > deadline:
            break
        w, h, d = boxes[orig_idx]
        rots = _rotations(w, h, d)
        candidates = _extreme_points(placed, container)
        
        best_pos = None
        best_dim = None
        best_score = None
        best_candidates = [] # For random tie-breaking
        
        for (cx, cy, cz) in candidates:
            for (rw, rh, rd) in rots:
                if not _fits((rw, rh, rd), (cx, cy, cz), container):
                    continue
                if not _no_overlap((rw, rh, rd), (cx, cy, cz), placed):
                    continue
                
                score = scoring_func((cx, cy, cz), (rw, rh, rd))
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                    best_dim = (rw, rh, rd)
                    best_candidates = [(best_pos, best_dim)]
                elif score == best_score:
                    best_candidates.append(((cx, cy, cz), (rw, rh, rd)))
        
        if best_pos is not None:
            # If randomization is enabled, pick from best candidates
            if rng and len(best_candidates) > 1:
                best_pos, best_dim = rng.choice(best_candidates)
            
            placed.append((best_pos[0], best_pos[1], best_pos[2],
                          best_dim[0], best_dim[1], best_dim[2]))
            placements[orig_idx] = (0, best_pos[0], best_pos[1], best_pos[2],
                                   best_dim[0], best_dim[1], best_dim[2])
        else:
            placements[orig_idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(len(boxes)):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    volume_filled = sum(p[4] * p[5] * p[6] for p in placements if p[0] == 0)
    return placements, volume_filled
=======
import time
import random
import itertools

def _rotations(w, h, d):
    """Return all 6 unique orientations of a box."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _fits(box_dims, pos, container):
    W, H, D = container
    x, y, z = pos
    w, h, d = box_dims
    return x + w <= W and y + h <= H and z + d <= D

def _no_overlap(box_dims, pos, placed):
    x, y, z = pos
    w, h, d = box_dims
    for (px, py, pz, pw, ph, pd) in placed:
        if (x < px + pw and x + w > px and
            y < py + ph and y + h > py and
            z < pz + pd and z + d > pz):
            return False
    return True

def _extreme_points(placed, container):
    W, H, D = container
    candidates = set()
    candidates.add((0, 0, 0))
    for (px, py, pz, pw, ph, pd) in placed:
        points = [
            (px + pw, py, pz),
            (px, py + ph, pz),
            (px, py, pz + pd),
            (px + pw, py + ph, pz),
            (px, py + ph, pz + pd),
            (px + pw, py, pz + pd),
            (px + pw, py + ph, pz + pd),
        ]
        for (cx, cy, cz) in points:
            if cx <= W and cy <= H and cz <= D:
                candidates.add((cx, cy, cz))
    return list(candidates)

class Heightmap:
    """3D heightmap for fast floor queries."""
    def __init__(self, W, H, D):
        self.W = W
        self.H = H
        self.D = D
        # heightmap[x][y] = max z occupied
        self.hm = [[0] * (D + 1) for _ in range(W + 1)]
    
    def get_floor(self, x, y, w, d):
        """Get the minimum floor height for a box at (x,y) with footprint wxd."""
        min_h = self.H
        for xi in range(x, min(x + w, self.W + 1)):
            for yi in range(y, min(y + d, self.D + 1)):
                if xi < self.W + 1 and yi < self.D + 1:
                    h = self.hm[xi][yi]
                    if h < min_h:
                        min_h = h
        return min_h
    
    def can_place(self, x, y, z, w, h, d):
        """Check if a box can be placed at (x,y,z) with dims (w,h,d)."""
        if x + w > self.W or y + d > self.D or z + h > self.H:
            return False
        for xi in range(x, x + w):
            for yi in range(y, y + d):
                if self.hm[xi][yi] > z:
                    return False
        return True
    
    def place(self, x, y, z, w, h, d):
        """Update heightmap after placing a box."""
        for xi in range(x, x + w):
            for yi in range(y, y + d):
                if self.hm[xi][yi] < z + h:
                    self.hm[xi][yi] = z + h
    
    def get_candidate_positions(self, w, h, d):
        """Get all valid (x, y, z) positions for a box with given dims."""
        candidates = []
        for x in range(self.W - w + 1):
            for y in range(self.D - d + 1):
                floor_z = self.get_floor(x, y, w, d)
                if floor_z + h <= self.H:
                    candidates.append((x, y, floor_z))
        return candidates

def _try_place_eap(boxes, container, order_idx, deadline, scoring_func=None, rng=None):
    """Extreme Points Approach with optional custom scoring and randomization."""
    W, H, D = container
    placed = []
    placements = [None] * len(boxes)
    
    if scoring_func is None:
        def scoring_func(pos, dim):
            return (pos[2], pos[1], pos[0]) # Bottom, Left, Far
    
    for orig_idx in order_idx:
        if time.time() > deadline:
            break
        w, h, d = boxes[orig_idx]
        rots = _rotations(w, h, d)
        candidates = _extreme_points(placed, container)
        
        best_pos = None
        best_dim = None
        best_score = None
        best_candidates = [] # For random tie-breaking
        
        for (cx, cy, cz) in candidates:
            for (rw, rh, rd) in rots:
                if not _fits((rw, rh, rd), (cx, cy, cz), container):
                    continue
                if not _no_overlap((rw, rh, rd), (cx, cy, cz), placed):
                    continue
                
                score = scoring_func((cx, cy, cz), (rw, rh, rd))
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                    best_dim = (rw, rh, rd)
                    best_candidates = [(best_pos, best_dim)]
                elif score == best_score:
                    best_candidates.append(((cx, cy, cz), (rw, rh, rd)))
        
        if best_pos is not None:
            # If randomization is enabled, pick from best candidates
            if rng and len(best_candidates) > 1:
                best_pos, best_dim = rng.choice(best_candidates)
            
            placed.append((best_pos[0], best_pos[1], best_pos[2],
                          best_dim[0], best_dim[1], best_dim[2]))
            placements[orig_idx] = (0, best_pos[0], best_pos[1], best_pos[2],
                                   best_dim[0], best_dim[1], best_dim[2])
        else:
            placements[orig_idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(len(boxes)):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    volume_filled = sum(p[4] * p[5] * p[6] for p in placements if p[0] == 0)
    return placements, volume_filled

def _try_place_heightmap(boxes, container, order_idx, deadline, scoring_func=None, rng=None):
    """Heightmap-based packing with O(1) floor queries."""
    W, H, D = container
    hm = Heightmap(W, H, D)
    placed = []
    placements = [None] * len(boxes)
    
    if scoring_func is None:
        def scoring_func(pos, dim):
            return (pos[2], pos[1], pos[0])
    
    for orig_idx in order_idx:
        if time.time() > deadline:
            break
        w, h, d = boxes[orig_idx]
        rots = _rotations(w, h, d)
        
        best_pos = None
        best_dim = None
        best_score = None
        best_candidates = []
        
        for (rw, rh, rd) in rots:
            candidates = hm.get_candidate_positions(rw, rh, rd)
            for (cx, cy, cz) in candidates:
                score = scoring_func((cx, cy, cz), (rw, rh, rd))
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                    best_dim = (rw, rh, rd)
                    best_candidates = [(best_pos, best_dim)]
                elif score == best_score:
                    best_candidates.append(((cx, cy, cz), (rw, rh, rd)))
        
        if best_pos is not None:
            if rng and len(best_candidates) > 1:
                best_pos, best_dim = rng.choice(best_candidates)
            
            hm.place(best_pos[0], best_pos[1], best_pos[2],
                     best_dim[0], best_dim[1], best_dim[2])
            placed.append((best_pos[0], best_pos[1], best_pos[2],
                          best_dim[0], best_dim[1], best_dim[2]))
            placements[orig_idx] = (0, best_pos[0], best_pos[1], best_pos[2],
                                   best_dim[0], best_dim[1], best_dim[2])
        else:
            placements[orig_idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(len(boxes)):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    volume_filled = sum(p[4] * p[5] * p[6] for p in placements if p[0] == 0)
    return placements, volume_filled
>>>>>>> REPLACE

