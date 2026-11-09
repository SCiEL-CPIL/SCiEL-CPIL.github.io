"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import permutations

def _rotations(w, h, d):
    """Return unique rotations of a box as (w, h, d) tuples, sorted by volume footprint."""
    seen = set()
    result = []
    for perm in permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    # Sort by footprint area (w*h) ascending to prefer compact orientations
    result.sort(key=lambda p: p[0] * p[1])
    return result

class PackingSolver:
    def __init__(self, boxes, container):
        self.boxes = boxes
        self.W, self.H, self.D = container
        self.n = len(boxes)
        self.container_vol = self.W * self.H * self.D
        
        # Determine grid resolution
        # Aim for max dimension ~ 100-200 to keep grid size manageable (< 4M cells)
        max_dim = max(self.W, self.H, self.D)
        if max_dim <= 100:
            self.res = 1
        elif max_dim <= 500:
            self.res = 5
        else:
            self.res = 10
            
        # Grid dimensions (integer units)
        self.gW = int(np.ceil(self.W / self.res))
        self.gH = int(np.ceil(self.H / self.res))
        self.gD = int(np.ceil(self.D / self.res))
        
        # Ensure grid is not too large (memory limit)
        if self.gW * self.gH * self.gD > 5_000_000:
            self.res *= 2
            self.gW = int(np.ceil(self.W / self.res))
            self.gH = int(np.ceil(self.H / self.res))
            self.gD = int(np.ceil(self.D / self.res))

        self.grid = np.zeros((self.gW, self.gH, self.gD), dtype=np.int32)
        self.placed_boxes = [] # List of (idx, x, y, z, w, h, d)

    def _grid_coords(self, x, y, z, w, h, d):
        """Convert physical coords to grid slice indices."""
        # Floor for start, ceil for end to ensure coverage
        sx = int(np.floor(x / self.res))
        sy = int(np.floor(y / self.res))
        sz = int(np.floor(z / self.res))
        
        ex = int(np.ceil((x + w) / self.res))
        ey = int(np.ceil((y + h) / self.res))
        ez = int(np.ceil((z + d) / self.res))
        
        # Clamp to grid bounds
        sx = max(0, min(sx, self.gW))
        sy = max(0, min(sy, self.gH))
        sz = max(0, min(sz, self.gD))
        ex = max(0, min(ex, self.gW))
        ey = max(0, min(ey, self.gH))
        ez = max(0, min(ez, self.gD))
        
        return (sx, sy, sz, ex, ey, ez)

    def _is_valid_in_grid(self, x, y, z, w, h, d):
        """Check if box fits in container and grid is empty."""
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > self.W or y + h > self.H or z + d > self.D:
            return False
            
        sx, sy, sz, ex, ey, ez = self._grid_coords(x, y, z, w, h, d)
        if ex <= sx or ey <= sy or ez <= sz:
            return False
            
        # Check if any cell in the sub-box is occupied
        if np.any(self.grid[sx:ex, sy:ey, sz:ez] > 0):
            return False
        return True

    def _occupy(self, x, y, z, w, h, d, idx):
        sx, sy, sz, ex, ey, ez = self._grid_coords(x, y, z, w, h, d)
        self.grid[sx:ex, sy:ey, sz:ez] = idx + 1 # +1 to distinguish from empty (0)

    def _vacate(self, x, y, z, w, h, d):
        sx, sy, sz, ex, ey, ez = self._grid_coords(x, y, z, w, h, d)
        self.grid[sx:ex, sy:ey, sz:ez] = 0

    def _find_best_position(self, w, h, d, exclude_idx=None):
        """
        Find the best position for a box of size (w,h,d).
        Uses a heuristic: Scan from bottom-left-front, prefer positions with more contact.
        """
        # To speed up, we don't scan the entire grid. We scan candidate "anchors".
        # A good anchor is the bottom-left-front-most empty cell in the bounding box of interest.
        # However, for simplicity and robustness, we will generate candidates from existing box faces
        # and the origin, then check them.
        
        candidates = set()
        candidates.add((0, 0, 0))
        
        # Add face points of currently placed boxes
        for (pid, px, py, pz, pw, ph, pd) in self.placed_boxes:
            if pid == exclude_idx:
                continue
            # Corners and face extensions
            for dx in [0, pw]:
                for dy in [0, ph]:
                    for dz in [0, pd]:
                        candidates.add((px + dx, py + dy, pz + dz))
            # Just past faces
            if px + pw < self.W:
                candidates.add((px + pw, py, pz))
                candidates.add((px + pw, py + ph, pz))
            if py + ph < self.H:
                candidates.add((px, py + ph, pz))
                candidates.add((px + pw, py + ph, pz))
            if pz + pd < self.D:
                candidates.add((px, py, pz + pd))
                candidates.add((px + pw, py, pz + pd))

        best_pos = None
        best_score = -1
        best_rot = None
        
        rots = _rotations(w, h, d)
        
        for rot in rots:
            bw, bh, bd = rot
            for (cx, cy, cz) in candidates:
                # Optimization: Skip if clearly out of bounds
                if cx + bw > self.W or cy + bh > self.H or cz + bd > self.D:
                    continue
                if cx < 0 or cy < 0 or cz < 0:
                    continue
                
                if self._is_valid_in_grid(cx, cy, cz, bw, bh, bd):
                    # Score: Prefer lower z, then y, then x.
                    # Contact bonus: Check if neighbors are occupied or walls.
                    # Simple proxy: Minimize (cx, cy, cz) lexicographically, 
                    # but add a small bonus if it touches existing mass.
                    
                    # Calculate "contact" by checking 6 neighbors in grid
                    # This is expensive, so we approximate with position closeness to origin
                    # and a tie-breaker that prefers "tight" packing.
                    
                    # Let's use a composite score:
                    # 1. Primary: Minimize z (gravity)
                    # 2. Secondary: Minimize y
                    # 3. Tertiary: Minimize x
                    # 4. Tie-breaker: Maximize "filled" volume in immediate neighborhood (hard to calc fast)
                    
                    # Instead, we'll just pick the one with minimum (z, y, x) among valid candidates.
                    # This is a standard "Bottom-Left-Back" heuristic.
                    key = (cz, cy, cx)
                    
                    if best_score == -1 or key < best_score:
                        best_score = key
                        best_pos = (cx, cy, cz)
                        best_rot = (bw, bh, bd)
                        
        if best_pos is not None:
            return best_pos, best_rot
        return None, None

    def _greedy_solve(self, order, deadline):
        """
        Perform greedy placement based on the given order.
        Returns a list of placements.
        """
        self.grid = np.zeros((self.gW, self.gH, self.gD), dtype=np.int32)
        self.placed_boxes = []
        placements = [None] * self.n
        
        for idx in order:
            if time.time() > deadline:
                break
            w, h, d = self.boxes[idx]
            pos, rot = self._find_best_position(w, h, d)
            if pos is not None:
                x, y, z = pos
                bw, bh, bd = rot
                self._occupy(x, y, z, bw, bh, bd, idx)
                self.placed_boxes.append((idx, x, y, z, bw, bh, bd))
                placements[idx] = (0, x, y, z, bw, bh, bd)
            else:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
                
        # Fill remaining
        for i in range(self.n):
            if placements[i] is None:
                w, h, d = self.boxes[i]
                placements[i] = (-1, 0, 0, 0, w, h, d)
                
        return placements

    def _local_search(self, placements, deadline):
        """
        Improve placements using single moves and 2-box swaps.
        """
        # Build initial state from placements
        self.grid = np.zeros((self.gW, self.gH, self.gD), dtype=np.int32)
        self.placed_boxes = []
        for i in range(self.n):
            if placements[i][0] == 0:
                _, x, y, z, w, h, d = placements[i]
                self._occupy(x, y, z, w, h, d, i)
                self.placed_boxes.append((i, x, y, z, w, h, d))
        
        placed_indices = [i for i in range(self.n) if placements[i][0] == 0]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Single Box Moves
            for idx in placed_indices:
                if time.time() > deadline:
                    break
                old_x, old_y, old_z, old_w, old_h, old_d = placements[idx][1:7]
                
                # Remove from grid
                self._vacate(old_x, old_y, old_z, old_w, old_h, old_d)
                # Remove from list (O(N) but N is small)
                self.placed_boxes = [p for p in self.placed_boxes if p[0] != idx]
                
                # Find new best position
                w, h, d = self.boxes[idx]
                new_pos, new_rot = self._find_best_position(w, h, d)
                
                if new_pos is not None:
                    nx, ny, nz = new_pos
                    nw, nh, nd = new_rot
                    # Check if it's actually better (lower z, y, x)
                    old_key = (old_z, old_y, old_x)
                    new_key = (nz, ny, nx)
                    if new_key < old_key:
                        self._occupy(nx, ny, nz, nw, nh, nd, idx)
                        self.placed_boxes.append((idx, nx, ny, nz, nw, nh, nd))
                        placements[idx] = (0, nx, ny, nz, nw, nh, nd)
                        improved = True
                    else:
                        # Revert
                        self._occupy(old_x, old_y, old_z, old_w, old_h, old_d, idx)
                        self.placed_boxes.append((idx, old_x, old_y, old_z, old_w, old_h, old_d))
                else:
                    # Could not place, revert
                    self._occupy(old_x, old_y, old_z, old_w, old_h, old_d, idx)
                    self.placed_boxes.append((idx, old_x, old_y, old_z, old_w, old_h, old_d))
            
            # 2. Two-Box Swaps (Limited to avoid O(N^2) explosion)
            if time.time() > deadline:
                break
                
            # Pick random pairs
            if len(placed_indices) >= 2:
                for _ in range(50): # Limit swaps per pass
                    if time.time() > deadline:
                        break
                    i, j = random.sample(placed_indices, 2)
                    
                    # Get old placements
                    pi = placements[i]
                    pj = placements[j]
                    
                    # Remove both
                    self._vacate(pi[1], pi[2], pi[3], pi[4], pi[5], pi[6])
                    self._vacate(pj[1], pj[2], pj[3], pj[4], pj[5], pj[6])
                    self.placed_boxes = [p for p in self.placed_boxes if p[0] not in (i, j)]
                    
                    wi, hi, di = self.boxes[i]
                    wj, hj, dj = self.boxes[j]
                    
                    # Try placing i then j
                    pos_i, rot_i = self._find_best_position(wi, hi, di)
                    if pos_i:
                        # Temporarily place i
                        self._occupy(pos_i[0], pos_i[1], pos_i[2], rot_i[0], rot_i[1], rot_i[2], i)
                        self.placed_boxes.append((i, pos_i[0], pos_i[1], pos_i[2], rot_i[0], rot_i[1], rot_i[2]))
                        
                        pos_j, rot_j = self._find_best_position(wj, hj, dj)
                        if pos_j:
                            # Calculate score improvement
                            # Simple heuristic: Sum of (z, y, x) for both boxes
                            old_score = (pi[3], pi[2], pi[1]) + (pj[3], pj[2], pj[1])
                            new_score = (pos_j[2], pos_j[1], pos_j[0]) + (pos_i[2], pos_i[1], pos_i[0])
                            
                            if new_score < old_score:
                                self._occupy(pos_j[0], pos_j[1], pos_j[2], rot_j[0], rot_j[1], rot_j[2], j)
                                self.placed_boxes.append((j, pos_j[0], pos_j[1], pos_j[2], rot_j[0], rot_j[1], rot_j[2]))
                                placements[i] = (0, pos_i[0], pos_i[1], pos_i[2], rot_i[0], rot_i[1], rot_i[2])
                                placements[j] = (0, pos_j[0], pos_j[1], pos_j[2], rot_j[0], rot_j[1], rot_j[2])
                                improved = True
                                continue # Skip revert
                            # If not improved, revert j
                            self._vacate(pos_j[0], pos_j[1], pos_j[2], rot_j[0], rot_j[1], rot_j[2])
                            self.placed_boxes = [p for p in self.placed_boxes if p[0] != j]
                            # Revert i
                            self._vacate(pos_i[0], pos_i[1], pos_i[2], rot_i[0], rot_i[1], rot_i[2])
                            self.placed_boxes = [p for p in self.placed_boxes if p[0] != i]
                            # Re-place originals
                            self._occupy(pi[1], pi[2], pi[3], pi[4], pi[5], pi[6], i)
                            self._occupy(pj[1], pj[2], pj[3], pj[4], pj[5], pj[6], j)
                            self.placed_boxes.append((i, pi[1], pi[2], pi[3], pi[4], pi[5], pi[6]))
                            self.placed_boxes.append((j, pj[1], pj[2], pj[3], pj[4], pj[5], pj[6]))
                        else:
                            # j couldn't be placed, revert i
                            self._vacate(pos_i[0], pos_i[1], pos_i[2], rot_i[0], rot_i[1], rot_i[2])
                            self.placed_boxes = [p for p in self.placed_boxes if p[0] != i]
                            # Re-place originals
                            self._occupy(pi[1], pi[2], pi[3], pi[4], pi[5], pi[6], i)
                            self._occupy(pj[1], pj[2], pj[3], pj[4], pj[5], pj[6], j)
                            self.placed_boxes.append((i, pi[1], pi[2], pi[3], pi[4], pi[5], pi[6]))
                            self.placed_boxes.append((j, pj[1], pj[2], pj[3], pj[4], pj[5], pj[6]))
                    else:
                        # i couldn't be placed, re-place originals
                        self._occupy(pi[1], pi[2], pi[3], pi[4], pi[5], pi[6], i)
                        self._occupy(pj[1], pj[2], pj[3], pj[4], pj[5], pj[6], j)
                        self.placed_boxes.append((i, pi[1], pi[2], pi[3], pi[4], pi[5], pi[6]))
                        self.placed_boxes.append((j, pj[1], pj[2], pj[3], pj[4], pj[5], pj[6]))

        return placements

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    W, H, D = container
    n = len(boxes)
    
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, 0, 0, 0) for _ in boxes]

    solver = PackingSolver(boxes, container)
    
    best_placements = None
    best_vol = -1
    
    # Multi-start
    # Order 1: Volume Descending
    order1 = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    # Order 2: Random (Seed 0)
    random.seed(0)
    order2 = list(range(n))
    random.shuffle(order2)
    
    # Order 3: Random (Seed 1)
    random.seed(1)
    order3 = list(range(n))
    random.shuffle(order3)
    
    orders = [order1, order2, order3]
    
    for order in orders:
        if time.time() > deadline - 5.0:
            break
            
        placements = solver._greedy_solve(order, deadline)
        
        # Calculate volume
        vol = 0
        for p in placements:
            if p[0] == 0:
                vol += p[4]*p[5]*p[6]
                
        if vol > best_vol:
            best_vol = vol
            best_placements = placements.copy()
            
        # Local Search on this placement
        if time.time() < deadline - 10.0:
            placements_ls = solver._local_search(placements, deadline)
            vol_ls = 0
            for p in placements_ls:
                if p[0] == 0:
                    vol_ls += p[4]*p[5]*p[6]
                    
            if vol_ls > best_vol:
                best_vol = vol_ls
                best_placements = placements_ls.copy()
                
    if best_placements is None:
        # Fallback: No boxes placed
        return [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        
    return best_placements
# EVOLVE-BLOCK-END
