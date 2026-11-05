"""3D Container Loading with grid-accelerated search and multi-strategy optimization."""
# EVOLVE-BLOCK-START
import time
import itertools
import sys
import random

def _orientations(w, h, d):
    """Generate all unique orientations of a box."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

class _SpatialGrid:
    """3D grid-based spatial index for fast overlap checking."""
    def __init__(self, W, H, D, cell_size=None):
        self.W = W
        self.H = H
        self.D = D
        # Choose cell size to balance granularity vs. memory
        if cell_size is None:
            # Aim for roughly 4-8 cells per dimension
            self.cell_size = max(1, min(W, H, D) // 4)
        else:
            self.cell_size = max(1, cell_size)
        self.grid = {}  # (cx, cy, cz) -> list of box indices
    
    def _cell_range(self, x, y, z, w, h, d):
        """Get the range of cells a box occupies."""
        cx0 = int(x // self.cell_size)
        cy0 = int(y // self.cell_size)
        cz0 = int(z // self.cell_size)
        cx1 = int((x + w - 1) // self.cell_size) if w > 0 else cx0
        cy1 = int((y + h - 1) // self.cell_size) if h > 0 else cy0
        cz1 = int((z + d - 1) // self.cell_size) if d > 0 else cz0
        return cx0, cy0, cz0, cx1, cy1, cz1
    
    def insert(self, idx, x, y, z, w, h, d):
        """Insert a box into the grid."""
        cx0, cy0, cz0, cx1, cy1, cz1 = self._cell_range(x, y, z, w, h, d)
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                for cz in range(cz0, cz1 + 1):
                    key = (cx, cy, cz)
                    if key not in self.grid:
                        self.grid[key] = []
                    self.grid[key].append(idx)
    
    def query_potential_overlaps(self, x, y, z, w, h, d):
        """Get indices of boxes that might overlap with the given position."""
        cx0, cy0, cz0, cx1, cy1, cz1 = self._cell_range(x, y, z, w, h, d)
        candidates = set()
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                for cz in range(cz0, cz1 + 1):
                    key = (cx, cy, cz)
                    if key in self.grid:
                        for idx in self.grid[key]:
                            candidates.add(idx)
        return candidates

def _fits_fast(box, pos, placed, grid, W, H, D):
    """Check if box at pos doesn't overlap with placed boxes and fits in container."""
    x, y, z = pos
    bw, bh, bd = box
    if x + bw > W or y + bh > H or z + bd > D:
        return False
    # Query grid for potential overlaps
    candidate_indices = grid.query_potential_overlaps(x, y, z, bw, bh, bd)
    for idx in candidate_indices:
        px, py, pz, pw, ph, pd = placed[idx]
        # Check overlap: boxes overlap if they intersect in all 3 axes
        if not (x + bw <= px or px + pw <= x or
                y + bh <= py or py + ph <= y or
                z + bd <= pz or pz + pd <= z):
            return False
    return True

def _candidate_positions(placed, W, H, D):
    """Generate candidate positions based on corners of placed boxes and container floor."""
    candidates = set()
    # Bottom corners of container
    candidates.add((0, 0, 0))
    # For each placed box, generate positions at its 8 corners (offsets)
    for (px, py, pz, pw, ph, pd) in placed:
        # Right face
        candidates.add((px + pw, py, pz))
        candidates.add((px + pw, py + ph, pz))
        candidates.add((px + pw, py, pz + pd))
        candidates.add((px + pw, py + ph, pz + pd))
        # Back face
        candidates.add((px, py + ph, pz))
        candidates.add((px, py + ph, pz + pd))
        # Top face
        candidates.add((px, py, pz + pd))
        candidates.add((px + pw, py, pz + pd))
    # Filter to only valid positions within container
    valid = []
    for (x, y, z) in candidates:
        if 0 <= x < W and 0 <= y < H and 0 <= z < D:
            valid.append((x, y, z))
    return valid

def _select_candidates(placed, W, H, D, max_candidates=500):
    """Select candidate positions, prioritizing floor-level (low z) positions."""
    candidates = _candidate_positions(placed, W, H, D)
    if len(candidates) <= max_candidates:
        return candidates
    # Sort: primary by z (floor first), secondary by x+y
    candidates.sort(key=lambda p: (p[2], p[0] + p[1]))
    return candidates[:max_candidates]

def _score_position(pos, placed, W, H, D, strategy='sum'):
    """Score a position. Lower is better."""
    x, y, z = pos
    if strategy == 'sum':
        return x + y + z
    elif strategy == 'max_extent':
        # Prefer positions that don't extend the bounding box
        max_x = max((p[0] + p[3] for p in placed), default=0)
        max_y = max((p[1] + p[4] for p in placed), default=0)
        max_z = max((p[2] + p[5] for p in placed), default=0)
        new_max_x = max(max_x, x + 0)  # approximate
        return max(new_max_x, max_y, max_z)
    elif strategy == 'compact':
        # Prefer positions close to existing boxes
        if not placed:
            return x + y + z
        min_dist = min(abs(x - p[0]) + abs(y - p[1]) + abs(z - p[2]) for p in placed)
        return min_dist + 0.01 * (x + y + z)
    else:
        return x + y + z

def _best_fit_decreasing(boxes, container, deadline, max_candidates=500, score_strategy='sum', random_seed=None):
    """Best-Fit-Decreasing with grid-accelerated overlap checking."""
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    if random_seed is not None:
        random.seed(random_seed)
    
    # Sort boxes by volume descending
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    placed = []  # list of (x, y, z, w, h, d)
    grid = _SpatialGrid(W, H, D)
    placement_map = {}  # box_index -> (x, y, z, w, h, d)
    
    for i, w, h, d in indexed_boxes:
        if time.time() > deadline:
            break
        
        # Try all orientations
        orients = _orientations(w, h, d)
        best_pos = None
        best_score = float('inf')
        best_orient = None
        
        # Generate and select candidate positions
        candidates = _select_candidates(placed, W, H, D, max_candidates)
        
        for orient in orients:
            bw, bh, bd = orient
            for pos in candidates:
                if _fits_fast(orient, pos, placed, grid, W, H, D):
                    score = _score_position(pos, placed, W, H, D, score_strategy)
                    if score < best_score:
                        best_score = score
                        best_pos = pos
                        best_orient = orient
        
        if best_pos is not None:
            x, y, z = best_pos
            bw, bh, bd = best_orient
            placed.append((x, y, z, bw, bh, bd))
            grid.insert(len(placed) - 1, x, y, z, bw, bh, bd)
            placement_map[i] = (x, y, z, bw, bh, bd)
    
    # Build result
    result = []
    for i, (w, h, d) in enumerate(boxes):
        if i in placement_map:
            x, y, z, bw, bh, bd = placement_map[i]
            result.append((0, x, y, z, bw, bh, bd))
        else:
            result.append((-1, 0, 0, 0, w, h, d))
    return result

def _sort_by_max_dim(boxes, container):
    """Sort boxes by maximum dimension descending."""
    W, H, D = container
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: max(x[1], x[2], x[3]), reverse=True)
    return indexed_boxes

def _sort_by_surface(boxes, container):
    """Sort boxes by surface area descending."""
    W, H, D = container
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: 2*(x[1]*x[2] + x[2]*x[3] + x[1]*x[3]), reverse=True)
    return indexed_boxes

def _best_fit_with_sort(boxes, container, deadline, max_candidates=500, score_strategy='sum', sort_func=None, random_seed=None):
    """Best-Fit-Decreasing with custom sorting."""
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    if random_seed is not None:
        random.seed(random_seed)
    
    if sort_func is None:
        indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
        indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    else:
        indexed_boxes = sort_func(boxes, container)
    
    placed = []
    grid = _SpatialGrid(W, H, D)
    placement_map = {}
    
    for i, w, h, d in indexed_boxes:
        if time.time() > deadline:
            break
        
        orients = _orientations(w, h, d)
        best_pos = None
        best_score = float('inf')
        best_orient = None
        
        candidates = _select_candidates(placed, W, H, D, max_candidates)
        
        for orient in orients:
            bw, bh, bd = orient
            for pos in candidates:
                if _fits_fast(orient, pos, placed, grid, W, H, D):
                    score = _score_position(pos, placed, W, H, D, score_strategy)
                    if score < best_score:
                        best_score = score
                        best_pos = pos
                        best_orient = orient
        
        if best_pos is not None:
            x, y, z = best_pos
            bw, bh, bd = best_orient
            placed.append((x, y, z, bw, bh, bd))
            grid.insert(len(placed) - 1, x, y, z, bw, bh, bd)
            placement_map[i] = (x, y, z, bw, bh, bd)
    
    result = []
    for i, (w, h, d) in enumerate(boxes):
        if i in placement_map:
            x, y, z, bw, bh, bd = placement_map[i]
            result.append((0, x, y, z, bw, bh, bd))
        else:
            result.append((-1, 0, 0, 0, w, h, d))
    return result

def _local_improvement(boxes, container, initial_result, deadline):
    """Try to improve placement by shifting boxes to fill gaps."""
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return initial_result
    
    # Parse initial result
    placed = []
    placement_map = {}
    for i, (bin_id, x, y, z, w, h, d) in enumerate(initial_result):
        if bin_id == 0:
            placed.append((i, x, y, z, w, h, d))
            placement_map[i] = (x, y, z, w, h, d)
    
    if not placed:
        return initial_result
    
    grid = _SpatialGrid(W, H, D)
    for idx, (i, x, y, z, w, h, d) in enumerate(placed):
        grid.insert(idx, x, y, z, w, h, d)
    
    # Try to shift each box to a better position
    improved = False
    for attempt in range(3):  # Multiple passes
        if time.time() > deadline:
            break
        for idx in range(len(placed)):
            if time.time() > deadline:
                break
            i, x, y, z, w, h, d = placed[idx]
            # Temporarily remove
            # Find best alternative position
            candidates = _candidate_positions(
                [p for j, p in enumerate(placed) if j != idx],
                W, H, D
            )
            candidates = _select_candidates(
                [p for j, p in enumerate(placed) if j != idx],
                W, H, D, 300
            )
            
            best_pos = None
            best_score = x + y + z  # Current position score
            
            for pos in candidates:
                px, py, pz = pos
                # Check if this position is different from current
                if (px, py, pz) == (x, y, z):
                    continue
                # Check fit
                box_orient = (w, h, d)
                if _fits_fast(box_orient, pos, 
                             [p for j, p in enumerate(placed) if j != idx],
                             grid, W, H, D):
                    score = px + py + pz
                    if score < best_score:
                        best_score = score
                        best_pos = pos
            
            if best_pos is not None:
                # Move box to new position
                new_x, new_y, new_z = best_pos
                placed[idx] = (i, new_x, new_y, new_z, w, h, d)
                placement_map[i] = (new_x, new_y, new_z, w, h, d)
                improved = True
    
    # Build result
    result = []
    for i, (w, h, d) in enumerate(boxes):
        if i in placement_map:
            x, y, z, bw, bh, bd = placement_map[i]
            result.append((0, x, y, z, bw, bh, bd))
        else:
            result.append((-1, 0, 0, 0, w, h, d))
    return result

def _shelf_packing(boxes, container):
    """Simple shelf packing as fallback."""
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[2], reverse=True)
    
    placements = []
    placement_map = {}
    x, y, z, row_h, layer_d = 0, 0, 0, 0, 0
    
    for i, w, h, d in indexed_boxes:
        if x + w > W:
            x, y, row_h = 0, y + row_h, 0
        if y + h > H:
            y, z, layer_d, x, row_h = 0, z + layer_d, 0, 0, 0
        if z + d > D:
            continue
        placements.append((x, y, z, w, h, d))
        placement_map[i] = (x, y, z, w, h, d)
        x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
    
    result = []
    for i, (w, h, d) in enumerate(boxes):
        if i in placement_map:
            x, y, z, bw, bh, bd = placement_map[i]
            result.append((0, x, y, z, bw, bh, bd))
        else:
            result.append((-1, 0, 0, 0, w, h, d))
    return result

def _volume_filling(result, container):
    """Calculate volume fraction filled."""
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return 0.0
    container_vol = W * H * D
    if container_vol == 0:
        return 0.0
    filled = 0
    for (bin_id, x, y, z, w, h, d) in result:
        if bin_id == 0:
            filled += w * h * d
    return filled / container_vol

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second margin before 150s limit
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    results = []
    scores = []
    
    # Strategy 1: Best-Fit-Decreasing with sum scoring
    r1 = _best_fit_decreasing(boxes, container, deadline, max_candidates=500, score_strategy='sum')
    s1 = _volume_filling(r1, container)
    results.append(r1)
    scores.append(s1)
    
    # Strategy 2: Best-Fit-Decreasing with compact scoring
    if time.time() < deadline - 5.0:
        r2 = _best_fit_decreasing(boxes, container, deadline, max_candidates=500, score_strategy='compact')
        s2 = _volume_filling(r2, container)
        results.append(r2)
        scores.append(s2)
    
    # Strategy 3: Sort by max dimension
    if time.time() < deadline - 5.0:
        r3 = _best_fit_with_sort(boxes, container, deadline, max_candidates=500, score_strategy='sum', sort_func=_sort_by_max_dim)
        s3 = _volume_filling(r3, container)
        results.append(r3)
        scores.append(s3)
    
    # Strategy 4: Sort by surface area
    if time.time() < deadline - 5.0:
        r4 = _best_fit_with_sort(boxes, container, deadline, max_candidates=500, score_strategy='sum', sort_func=_sort_by_surface)
        s4 = _volume_filling(r4, container)
        results.append(r4)
        scores.append(s4)
    
    # Strategy 5: Shelf packing (fast fallback)
    r5 = _shelf_packing(boxes, container)
    s5 = _volume_filling(r5, container)
    results.append(r5)
    scores.append(s5)
    
    # Pick the best result so far
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_result = results[best_idx]
    best_score = scores[best_idx]
    
    # Local improvement on the best result
    if time.time() < deadline - 10.0:
        improved = _local_improvement(boxes, container, best_result, deadline - 5.0)
        imp_score = _volume_filling(improved, container)
        if imp_score > best_score:
            best_result = improved
            best_score = imp_score
    
    return best_result
# EVOLVE-BLOCK-END
