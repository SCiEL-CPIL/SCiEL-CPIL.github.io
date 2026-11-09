"""3D Container Loading - Improved Skyline + Shelf with local search."""
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


class Skyline3D:
    """3D Skyline representation for fast box placement."""
    def __init__(self, W, H, D):
        self.W = W
        self.H = H
        self.D = D
        self.placed_boxes = []
    
    def can_place(self, x, y, z, bw, bh, bd):
        """Check if a box of size (bw, bh, bd) fits at (x, y, z)."""
        if x < 0 or y < 0 or z < 0:
            return False
        if x + bw > self.W or y + bh > self.H or z + bd > self.D:
            return False
        for (px, py, pz, pw, ph, pd) in self.placed_boxes:
            if x + bw <= px or px + pw <= x:
                continue
            if y + bh <= py or py + ph <= y:
                continue
            if z + bd <= pz or pz + pd <= z:
                continue
            return False
        return True
    
    def place(self, x, y, z, bw, bh, bd):
        """Place a box."""
        self.placed_boxes.append((x, y, z, bw, bh, bd))
    
    def get_candidate_positions(self, bw, bh, bd):
        """Generate candidate (x, y, z) positions for a box of given size."""
        candidates = set()
        candidates.add((0, 0, 0))

        if not self.placed_boxes:
            return [(0, 0, 0)]

        for (px, py, pz, pw, ph, pd) in self.placed_boxes:
            # Right of box
            if px + pw + bw <= self.W:
                candidates.add((px + pw, py, pz))
                candidates.add((px + pw, py + ph, pz))
                candidates.add((px + pw, py, pz + pd))
                candidates.add((px + pw, py + ph, pz + pd))
            # Left of box
            if px >= bw:
                candidates.add((px - bw, py, pz))
                candidates.add((px - bw, py + ph, pz))
            # Above box
            if pz + pd + bd <= self.D:
                candidates.add((px, py, pz + pd))
                candidates.add((px + pw, py, pz + pd))
                candidates.add((px, py + ph, pz + pd))
                candidates.add((px + pw, py + ph, pz + pd))
            # In front/behind in y
            if py + ph + bh <= self.H:
                candidates.add((px, py + ph, pz))
                candidates.add((px + pw, py + ph, pz))
            if py >= bh:
                candidates.add((px, py - bh, pz))
                candidates.add((px + pw, py - bh, pz))

        # Grid-based positions
        xs = {0}
        ys = {0}
        for (px, py, pz, pw, ph, pd) in self.placed_boxes:
            xs.add(px)
            xs.add(px + pw)
            ys.add(py)
            ys.add(py + ph)
        xs = sorted(set(x for x in xs if 0 <= x <= self.W - bw))
        ys = sorted(set(y for y in ys if 0 <= y <= self.H - bh))

        if len(xs) <= 25 and len(ys) <= 25:
            for x in xs[:15]:
                for y in ys[:15]:
                    min_z = 0
                    for (px, py, pz, pw, ph, pd) in self.placed_boxes:
                        if not (x + bw <= px or px + pw <= x):
                            if not (y + bh <= py or py + ph <= y):
                                if pz + pd > min_z:
                                    min_z = pz + pd
                    if min_z + bd <= self.D:
                        candidates.add((x, y, min_z))

        valid = []
        for (cx, cy, cz) in candidates:
            if 0 <= cx and cx + bw <= self.W and 0 <= cy and cy + bh <= self.H and 0 <= cz and cz + bd <= self.D:
                valid.append((cx, cy, cz))
        return valid
    
    def best_position(self, bw, bh, bd, scoring_fn=None):
        """Find the best position for a box. Returns (x, y, z) or None."""
        candidates = self.get_candidate_positions(bw, bh, bd)
        best = None
        best_score = None
        for (cx, cy, cz) in candidates:
            if self.can_place(cx, cy, cz, bw, bh, bd):
                if scoring_fn:
                    s = scoring_fn(cx, cy, cz, bw, bh, bd)
                else:
                    s = (cz, cy, cx)
                if best_score is None or s < best_score:
                    best_score = s
                    best = (cx, cy, cz)
        return best

    def best_position_multi_rot(self, rots, scoring_fn=None):
        """Try all rotations, return (x, y, z, bw, bh, bd) or None."""
        best = None
        best_score = None
        for (bw, bh, bd) in rots:
            pos = self.best_position(bw, bh, bd, scoring_fn)
            if pos is not None:
                cx, cy, cz = pos
                if scoring_fn:
                    s = scoring_fn(cx, cy, cz, bw, bh, bd)
                else:
                    s = (cz, cy, cx)
                if best_score is None or s < best_score:
                    best_score = s
                    best = (cx, cy, cz, bw, bh, bd)
        return best


def _score_position(x, y, z, bw, bh, bd, placed_boxes, W, H, D):
    """Score a position: prefer compact, low, and well-connected."""
    contact = 0
    for (px, py, pz, pw, ph, pd) in placed_boxes:
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


def _greedy_skyline(order, boxes, container, deadline):
    """Greedy placement using skyline approach."""
    W, H, D = container
    n = len(boxes)
    skyline = Skyline3D(W, H, D)
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
            pos = skyline.best_position(bw, bh, bd, scoring_fn=lambda x, y, z, bw, bh, bd: _score_position(x, y, z, bw, bh, bd, skyline.placed_boxes, W, H, D))
            if pos is not None:
                cx, cy, cz = pos
                s = _score_position(cx, cy, cz, bw, bh, bd, skyline.placed_boxes, W, H, D)
                if best_score is None or s < best_score:
                    best_score = s
                    best_pos = pos
                    best_rot = (bw, bh, bd)
        
        if best_pos is not None:
            x, y, z = best_pos
            bw, bh, bd = best_rot
            skyline.place(x, y, z, bw, bh, bd)
            result[idx] = (0, x, y, z, bw, bh, bd)
        else:
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(n):
        if result[i] is None:
            w, h, d = boxes[i]
            result[i] = (-1, 0, 0, 0, w, h, d)
    
    return result


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
    compact_order = sorted(range(n), key=lambda i: max(boxes[i]) / max(1, min(boxes[i])))
    orders.append(('compact', compact_order))
    vol_maxdim_order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2] * max(boxes[i]), reverse=True)
    orders.append(('vol_maxdim', vol_maxdim_order))
    svr_order = sorted(range(n), key=lambda i: (2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2])) / max(1, boxes[i][0]*boxes[i][1]*boxes[i][2]))
    orders.append(('svr', svr_order))
    return orders


def _compute_volume(result):
    """Compute total volume of placed boxes."""
    total = 0
    for r in result:
        if r[0] == 0:
            total += r[4] * r[5] * r[6]
    return total


def _local_search(result, boxes, container, deadline, max_iters=1000):
    """Local search: try to place unplaced boxes and improve positions of placed boxes."""
    W, H, D = container
    n = len(boxes)
    best_result = [tuple(r) for r in result]
    best_vol = _compute_volume(best_result)
    rng = random.Random(42)

    def build_skyline_from_result(res):
        sk = Skyline3D(W, H, D)
        for i in range(n):
            if res[i][0] == 0:
                x, y, z, bw, bh, bd = res[i][1:7]
                sk.place(x, y, z, bw, bh, bd)
        return sk

    def try_place_box(idx, skyline):
        """Try to place box idx in the given skyline."""
        w, h, d = boxes[idx]
        rots = _rotations_sorted(w, h, d)
        best = skyline.best_position_multi_rot(rots)
        if best is not None:
            return best
        return None

    # Phase 1: Try to place unplaced boxes
    unplaced = [i for i in range(n) if best_result[i][0] == -1]
    attempts = 0
    while unplaced and attempts < max_iters // 3 and time.time() < deadline:
        attempts += 1
        idx = rng.choice(unplaced)
        skyline = build_skyline_from_result(best_result)
        placed = try_place_box(idx, skyline)
        if placed is not None:
            x, y, z, bw, bh, bd = placed
            temp = list(best_result)
            temp[idx] = (0, x, y, z, bw, bh, bd)
            new_vol = _compute_volume(temp)
            if new_vol > best_vol:
                best_result = [tuple(r) for r in temp]
                best_vol = new_vol
                unplaced.remove(idx)

    # Phase 2: Move placed boxes
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        placed_indices = [i for i in range(n) if best_result[i][0] == 0]
        if not placed_indices:
            break
        idx = rng.choice(placed_indices)
        temp = list(best_result)
        temp[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
        skyline = build_skyline_from_result(temp)
        placed = try_place_box(idx, skyline)
        if placed is not None:
            x, y, z, bw, bh, bd = placed
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
        skyline = build_skyline_from_result(temp)
        p1 = try_place_box(i1, skyline)
        if p1 is not None:
            skyline.place(p1[0], p1[1], p1[2], p1[3], p1[4], p1[5])
            p2 = try_place_box(i2, skyline)
            if p2 is not None:
                temp[i1] = (0, p1[0], p1[1], p1[2], p1[3], p1[4], p1[5])
                temp[i2] = (0, p2[0], p2[1], p2[2], p2[3], p2[4], p2[5])
                new_vol = _compute_volume(temp)
                if new_vol > best_vol:
                    best_result = [tuple(r) for r in temp]
                    best_vol = new_vol

    return best_result


def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    deadline = time.time() + 115
    if n == 0:
        return []
    
    heuristic_orders = _heuristic_orders(boxes, n)
    best_result = None
    best_vol = -1
    
    for name, order in heuristic_orders:
        if time.time() > deadline - 35:
            break
        result = _greedy_skyline(order, boxes, container, deadline)
        vol = _compute_volume(result)
        if vol > best_vol:
            best_vol = vol
            best_result = result
    
    # Random restarts
    if time.time() < deadline - 20 and best_result is not None:
        rng_restart = random.Random(123)
        for _ in range(8):
            if time.time() > deadline - 12:
                break
            random_order = list(range(n))
            rng_restart.shuffle(random_order)
            result_rand = _greedy_skyline(random_order, boxes, container, deadline)
            vol_rand = _compute_volume(result_rand)
            if vol_rand > best_vol:
                best_vol = vol_rand
                best_result = result_rand
    
    # Local search
    remaining_time = deadline - time.time()
    if remaining_time > 5 and best_result is not None:
        max_iters = int(remaining_time * 60)
        max_iters = max(300, min(max_iters, 4000))
        best_result = _local_search(best_result, boxes, container, deadline, max_iters=max_iters)
    
    if best_result is None:
        best_result = []
        for w, h, d in boxes:
            best_result.append((-1, 0, 0, 0, w, h, d))
    
    return [tuple(r) for r in best_result]

# EVOLVE-BLOCK-END
