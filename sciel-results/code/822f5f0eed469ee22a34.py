"""3D Container Loading - Voxel-grid + Skyline hybrid with local search."""
import time
import random
import numpy as np
from itertools import permutations


# EVOLVE-BLOCK-START

def _rotations(w, h, d):
    """Return unique rotations of a box as (w, h, d) tuples."""
    seen = set()
    result = []
    for perm in permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result


def _rotations_sorted(w, h, d):
    """Return unique rotations sorted by max-dim ascending (more compact first)."""
    rots = _rotations(w, h, d)
    rots.sort(key=lambda r: (max(r), r[1], r[2]))
    return rots


class VoxelGrid:
    """
    3D Voxel grid using NumPy for fast occupancy checks.
    Uses a unit-resolution grid for fast vectorized placement checks.
    Falls back to list-based check for large boxes.
    """
    def __init__(self, W, H, D):
        self.W = W
        self.H = H
        self.D = D
        # Use integer grid; if dimensions are too large for memory, use list-based
        # For typical container sizes (<= 200^3), a boolean numpy array is fine
        max_dim = max(W, H, D)
        if max_dim <= 300:
            self.occupied = np.zeros((W, H, D), dtype=np.uint8)
            self.use_grid = True
        else:
            self.occupied = None
            self.use_grid = False
        self.placed_boxes = []  # list of (x, y, z, w, h, d)

    def can_place(self, x, y, z, bw, bh, bd):
        """Check if a box of size (bw, bh, bd) fits at (x, y, z)."""
        if x < 0 or y < 0 or z < 0:
            return False
        if x + bw > self.W or y + bh > self.H or z + bd > self.D:
            return False
        if self.use_grid:
            # Vectorized check: any occupied cell in the box region?
            if np.any(self.occupied[x:x+bw, y:y+bh, z:z+bd]):
                return False
        else:
            # List-based fallback
            for (px, py, pz, pw, ph, pd) in self.placed_boxes:
                if not (x + bw <= px or px + pw <= x):
                    if not (y + bh <= py or py + ph <= y):
                        if not (z + bd <= pz or pz + pd <= z):
                            return False
        return True

    def place(self, x, y, z, bw, bh, bd):
        """Place a box and update the grid."""
        self.placed_boxes.append((x, y, z, bw, bh, bd))
        if self.use_grid:
            self.occupied[x:x+bw, y:y+bh, z:z+bd] = 1

    def find_lowest_position(self, bw, bh, bd):
        """
        Find the lowest (min z, then min y, then min x) feasible position
        for a box of size (bw, bh, bd). Uses vectorized scan.
        Returns (x, y, z) or None.
        """
        if self.use_grid:
            # Scan z from 0 up
            for z in range(self.D - bd + 1):
                # For this z, find lowest y, then x
                # Check slice at this z: occupied[x:x+bw, y:y+bh, z:z+bd]
                # We need to find (x, y) such that the entire block is free
                # Vectorized: for each y, check if any x works
                for y in range(self.H - bh + 1):
                    # Check which x positions are free at this (y, z)
                    # occupied[x:x+bw, y:y+bh, z:z+bd] must be all zeros
                    # Use sliding window sum for speed
                    slice_2d = self.occupied[:, y:y+bh, z:z+bd]
                    # Sum over y and z dimensions: gives occupancy per x
                    # Actually we need: for a given x, is sum over [x:x+bw, y:y+bh, z:z+bd] == 0?
                    # Use cumulative sum approach
                    if bw == 1:
                        col_sums = slice_2d.sum(axis=(1, 2))  # sum over y, z for each x
                        # Find first x where col_sums[x] == 0
                        free_xs = np.where(col_sums == 0)[0]
                        if len(free_xs) > 0:
                            x = free_xs[0]
                            if x + bw <= self.W:
                                return (int(x), int(y), int(z))
                    else:
                        # Need sum over window of size bw in x
                        # Use cumulative sum
                        cumsum = np.zeros(self.W + 1)
                        for i in range(self.W):
                            cumsum[i+1] = slice_2d[i].sum()
                        # Window sum: cumsum[x+bw] - cumsum[x]
                        window_sums = cumsum[bw:] - cumsum[:-bw]
                        free_xs = np.where(window_sums == 0)[0]
                        if len(free_xs) > 0:
                            x = free_xs[0]
                            return (int(x), int(y), int(z))
            return None
        else:
            # Fallback: sequential scan
            for z in range(self.D - bd + 1):
                for y in range(self.H - bh + 1):
                    for x in range(self.W - bw + 1):
                        if self.can_place(x, y, z, bw, bh, bd):
                            return (x, y, z)
            return None

    def get_candidate_positions(self, bw, bh, bd):
        """Generate candidate (x, y, z) positions based on existing box boundaries."""
        candidates = set()
        candidates.add((0, 0, 0))

        if not self.placed_boxes:
            return [(0, 0, 0)]

        # Generate positions based on existing box boundaries (corners)
        for (px, py, pz, pw, ph, pd) in self.placed_boxes:
            for dx in (0, pw):
                for dy in (0, ph):
                    for dz in (0, pd):
                        x, y, z = px + dx, py + dy, pz + dz
                        if 0 <= x <= self.W - bw and 0 <= y <= self.H - bh and 0 <= z <= self.D - bd:
                            candidates.add((x, y, z))
                        # Also try negative offsets (box to the left/behind/below)
                        x2, y2, z2 = px - bw, py - bh, pz - bd
                        if 0 <= x2 and 0 <= y2 and 0 <= z2 and x2 + bw <= self.W and y2 + bh <= self.H and z2 + bd <= self.D:
                            candidates.add((x2, y2, z2))
                        # Mixed offsets
                        x3, y3, z3 = px + pw, py - bh, pz
                        if 0 <= x3 and 0 <= y3 and 0 <= z3 and x3 + bw <= self.W and y3 + bh <= self.H and z3 + bd <= self.D:
                            candidates.add((x3, y3, z3))
                        x4, y4, z4 = px - bw, py + ph, pz
                        if 0 <= x4 and 0 <= y4 and 0 <= z4 and x4 + bw <= self.W and y4 + bh <= self.H and z4 + bd <= self.D:
                            candidates.add((x4, y4, z4))

        return list(candidates)

    def best_position(self, bw, bh, bd, scoring_fn=None):
        """Find the best position for a box. Returns (x, y, z) or None."""
        # First try the lowest position (fast vectorized)
        lowest = self.find_lowest_position(bw, bh, bd)

        # Also try corner-based candidates
        candidates = self.get_candidate_positions(bw, bh, bd)

        best = None
        best_score = None

        # Check lowest position first
        if lowest is not None and self.can_place(lowest[0], lowest[1], lowest[2], bw, bh, bd):
            cx, cy, cz = lowest
            if scoring_fn:
                s = scoring_fn(cx, cy, cz, bw, bh, bd)
            else:
                s = (cz, cy, cx)
            best_score = s
            best = lowest

        # Check other candidates
        for (cx, cy, cz) in candidates:
            if (cx, cy, cz) == lowest:
                continue
            if self.can_place(cx, cy, cz, bw, bh, bd):
                if scoring_fn:
                    s = scoring_fn(cx, cy, cz, bw, bh, bd)
                else:
                    s = (cz, cy, cx)
                if best_score is None or s < best_score:
                    best_score = s
                    best = (cx, cy, cz)

        return best


def _score_position(x, y, z, bw, bh, bd, placed_boxes, W, H, D):
    """Score a position: prefer compact, low, and well-connected."""
    # Primary: minimize z (pack from bottom)
    # Secondary: minimize y, then x
    # Tertiary: maximize contact with existing boxes
    contact = 0
    for (px, py, pz, pw, ph, pd) in placed_boxes:
        # Check face contacts
        if abs((x + bw) - px) < 1e-9:
            oy = min(y + bh, py + ph) - max(y, py)
            oz = min(z + bd, pz + pd) - max(z, pz)
            if oy > 0 and oz > 0:
                contact += oy * oz
        if abs((y + bh) - py) < 1e-9:
            ox = min(x + bw, px + pw) - max(x, px)
            oz = min(z + bd, pz + pd) - max(z, pz)
            if ox > 0 and oz > 0:
                contact += ox * oz
        if abs((z + bd) - pz) < 1e-9:
            ox = min(x + bw, px + pw) - max(x, px)
            oy = min(y + bh, py + ph) - max(y, py)
            if ox > 0 and oy > 0:
                contact += ox * oy

    return (z, y, x, -contact)


def _greedy_voxel(order, boxes, container, deadline):
    """Greedy placement using voxel grid approach."""
    W, H, D = container
    n = len(boxes)
    grid = VoxelGrid(W, H, D)
    result = [None] * n

    for idx in order:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        rots = _rotations_sorted(w, h, d)

        best_pos = None
        best_score = None
        best_rot = None

        for (bw, bh, bd) in rots:
            pos = grid.best_position(bw, bh, bd, scoring_fn=lambda x, y, z, bw, bh, bd: _score_position(x, y, z, bw, bh, bd, grid.placed_boxes, W, H, D))
            if pos is not None:
                cx, cy, cz = pos
                s = _score_position(cx, cy, cz, bw, bh, bd, grid.placed_boxes, W, H, D)
                if best_score is None or s < best_score:
                    best_score = s
                    best_pos = pos
                    best_rot = (bw, bh, bd)

        if best_pos is not None:
            x, y, z = best_pos
            bw, bh, bd = best_rot
            grid.place(x, y, z, bw, bh, bd)
            result[idx] = (0, x, y, z, bw, bh, bd)
        else:
            result[idx] = (-1, 0, 0, 0, w, h, d)

    for i in range(n):
        if result[i] is None:
            w, h, d = boxes[i]
            result[i] = (-1, 0, 0, 0, w, h, d)

    return result


def _compute_volume(result):
    """Compute total volume of placed boxes."""
    total = 0
    for r in result:
        if r[0] == 0:
            total += r[4] * r[5] * r[6]
    return total


def _heuristic_orders(boxes, n):
    """Generate multiple heuristic orderings."""
    orders = []

    vol_order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    orders.append(('vol', vol_order))

    maxdim_order = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    orders.append(('maxdim', maxdim_order))

    sa_order = sorted(range(n), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
    orders.append(('sa', sa_order))

    mindim_order = sorted(range(n), key=lambda i: min(boxes[i]), reverse=True)
    orders.append(('mindim', mindim_order))

    # Compactness: prefer more cube-like boxes first
    compact_order = sorted(range(n), key=lambda i: max(boxes[i]) / max(1, min(boxes[i])))
    orders.append(('compact', compact_order))

    # Volume-weighted by max dimension
    vol_maxdim_order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2] * max(boxes[i]), reverse=True)
    orders.append(('vol_maxdim', vol_maxdim_order))

    # Surface area to volume ratio (prefer lower ratio = more compact)
    svr_order = sorted(range(n), key=lambda i: (2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2])) / max(1, boxes[i][0]*boxes[i][1]*boxes[i][2]))
    orders.append(('svr', svr_order))

    return orders


def _local_search(result, boxes, container, deadline, max_iters=500):
    """Local search: try to place unplaced boxes and improve positions of placed boxes."""
    W, H, D = container
    n = len(boxes)
    best_result = [tuple(r) for r in result]
    best_vol = _compute_volume(best_result)
    rng = random.Random(42)

    def get_placed_list(result):
        placed = []
        for i in range(n):
            if result[i][0] == 0:
                x, y, z, bw, bh, bd = result[i][1:7]
                placed.append((x, y, z, bw, bh, bd))
        return placed

    def try_place_box(idx, placed):
        """Try to place box idx given current placed boxes."""
        w, h, d = boxes[idx]
        rots = _rotations_sorted(w, h, d)
        grid = VoxelGrid(W, H, D)
        for p in placed:
            grid.place(p[0], p[1], p[2], p[3], p[4], p[5])

        best_pos = None
        best_score = None
        best_rot = None

        for (bw, bh, bd) in rots:
            pos = grid.best_position(bw, bh, bd)
            if pos is not None:
                cx, cy, cz = pos
                s = (cz, cy, cx)
                if best_score is None or s < best_score:
                    best_score = s
                    best_pos = pos
                    best_rot = (bw, bh, bd)

        if best_pos is not None:
            return True, best_pos, best_rot
        return False, None, None

    # Phase 1: Try to place unplaced boxes
    unplaced = [i for i in range(n) if best_result[i][0] == -1]
    attempts = 0
    while unplaced and attempts < max_iters // 3 and time.time() < deadline:
        attempts += 1
        idx = rng.choice(unplaced)
        placed = get_placed_list(best_result)
        success, pos, rot = try_place_box(idx, placed)
        if success:
            x, y, z = pos
            bw, bh, bd = rot
            temp = list(best_result)
            temp[idx] = (0, x, y, z, bw, bh, bd)
            new_vol = _compute_volume(temp)
            if new_vol > best_vol:
                best_result = [tuple(r) for r in temp]
                best_vol = new_vol
                unplaced.remove(idx)

    # Phase 2: Move placed boxes to find better positions
    for _ in range(max_iters):
        if time.time() > deadline:
            break

        placed_indices = [i for i in range(n) if best_result[i][0] == 0]
        if not placed_indices:
            break

        idx = rng.choice(placed_indices)
        temp = list(best_result)
        temp[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])

        placed = []
        for i in range(n):
            if i != idx and temp[i][0] == 0:
                x, y, z, bw, bh, bd = temp[i][1:7]
                placed.append((x, y, z, bw, bh, bd))

        success, pos, rot = try_place_box(idx, placed)
        if success:
            x, y, z = pos
            bw, bh, bd = rot
            temp[idx] = (0, x, y, z, bw, bh, bd)
            new_vol = _compute_volume(temp)
            if new_vol > best_vol:
                best_result = [tuple(r) for r in temp]
                best_vol = new_vol

    # Phase 3: 2-box swap moves
    placed_indices = [i for i in range(n) if best_result[i][0] == 0]
    for _ in range(max_iters // 3):
        if time.time() > deadline:
            break
        if len(placed_indices) < 2:
            break

        i1, i2 = rng.sample(placed_indices, 2)
        temp = list(best_result)
        temp[i1] = (-1, 0, 0, 0, boxes[i1][0], boxes[i1][1], boxes[i1][2])
        temp[i2] = (-1, 0, 0, 0, boxes[i2][0], boxes[i2][1], boxes[i2][2])

        placed = []
        for i in range(n):
            if i not in (i1, i2) and temp[i][0] == 0:
                x, y, z, bw, bh, bd = temp[i][1:7]
                placed.append((x, y, z, bw, bh, bd))

        success1, pos1, rot1 = try_place_box(i1, placed)
        if success1:
            placed.append(pos1 + rot1)
            success2, pos2, rot2 = try_place_box(i2, placed)
            if success2:
                temp[i1] = (0, pos1[0], pos1[1], pos1[2], rot1[0], rot1[1], rot1[2])
                temp[i2] = (0, pos2[0], pos2[1], pos2[2], rot2[0], rot2[1], rot2[2])
                new_vol = _compute_volume(temp)
                if new_vol > best_vol:
                    best_result = [tuple(r) for r in temp]
                    best_vol = new_vol

    return best_result


def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    deadline = time.time() + 140  # Safety margin before 150s kill

    if n == 0:
        return []

    heuristic_orders = _heuristic_orders(boxes, n)

    best_result = None
    best_vol = -1

    # Run voxel greedy with different orderings
    for name, order in heuristic_orders:
        if time.time() > deadline - 30:
            break
        result = _greedy_voxel(order, boxes, container, deadline)
        vol = _compute_volume(result)
        if vol > best_vol:
            best_vol = vol
            best_result = result

    # Random restarts with voxel
    if time.time() < deadline - 15 and best_result is not None:
        rng_restart = random.Random(123)
        for _ in range(8):
            if time.time() > deadline - 10:
                break
            random_order = list(range(n))
            rng_restart.shuffle(random_order)
            result_rand = _greedy_voxel(random_order, boxes, container, deadline)
            vol_rand = _compute_volume(result_rand)
            if vol_rand > best_vol:
                best_vol = vol_rand
                best_result = result_rand

    # Local search to improve
    remaining_time = deadline - time.time()
    if remaining_time > 5 and best_result is not None:
        max_iters = int(remaining_time * 40)
        max_iters = max(100, min(max_iters, 2000))
        best_result = _local_search(best_result, boxes, container, deadline, max_iters=max_iters)

    if best_result is None:
        best_result = []
        for w, h, d in boxes:
            best_result.append((-1, 0, 0, 0, w, h, d))

    return [tuple(r) for r in best_result]

# EVOLVE-BLOCK-END
