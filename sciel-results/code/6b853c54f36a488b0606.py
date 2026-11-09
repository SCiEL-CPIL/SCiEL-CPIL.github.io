"""3D Container Loading - Skyline-optimized greedy + enhanced local search."""
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


class SkylineState:
    """Maintains placed boxes and provides fast candidate generation."""
    def __init__(self, W, H, D):
        self.W = W
        self.H = H
        self.D = D
        self.placed = []  # list of (x, y, z, w, h, d)
        self._x_bounds = set([0])
        self._y_bounds = set([0])
        self._z_bounds = set([0])

    def place(self, x, y, z, bw, bh, bd):
        self.placed.append((x, y, z, bw, bh, bd))
        self._x_bounds.add(x)
        self._x_bounds.add(x + bw)
        self._y_bounds.add(y)
        self._y_bounds.add(y + bh)
        self._z_bounds.add(z)
        self._z_bounds.add(z + bd)

    def can_place(self, x, y, z, bw, bh, bd):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + bw > self.W or y + bh > self.H or z + bd > self.D:
            return False
        for (px, py, pz, pw, ph, pd) in self.placed:
            if not (x + bw <= px or px + pw <= x):
                if not (y + bh <= py or py + ph <= y):
                    if not (z + bd <= pz or pz + pd <= z):
                        return False
        return True

    def min_z_at(self, x, y, bw, bh):
        """Find minimum z where a box of footprint (bw, bh) at (x,y) can be placed."""
        min_z = 0
        for (px, py, pz, pw, ph, pd) in self.placed:
            if not (x + bw <= px or px + pw <= x):
                if not (y + bh <= py or py + ph <= y):
                    if pz + pd > min_z:
                        min_z = pz + pd
        return min_z

    def candidates_for(self, bw, bh, bd):
        """Generate candidate positions for a box of size (bw, bh, bd)."""
        cands = set()
        cands.add((0, 0, 0))

        if not self.placed:
            return [(0, 0, 0)]

        # Adjacent positions to each placed box
        for (px, py, pz, pw, ph, pd) in self.placed:
            # +x side
            if px + pw + bw <= self.W:
                for dy in (0, ph):
                    for dz in (0, pd):
                        cands.add((px + pw, py + dy, pz + dz))
            # -x side
            if px >= bw:
                for dy in (0, ph):
                    for dz in (0, pd):
                        cands.add((px - bw, py + dy, pz + dz))
            # +y side
            if py + ph + bh <= self.H:
                for dx in (0, pw):
                    for dz in (0, pd):
                        cands.add((px + dx, py + ph, pz + dz))
            # -y side
            if py >= bh:
                for dx in (0, pw):
                    for dz in (0, pd):
                        cands.add((px + dx, py - bh, pz + dz))
            # +z side
            if pz + pd + bd <= self.D:
                for dx in (0, pw):
                    for dy in (0, ph):
                        cands.add((px + dx, py + dy, pz + pd))
            # -z side
            if pz >= bd:
                for dx in (0, pw):
                    for dy in (0, ph):
                        cands.add((px + dx, py + dy, pz - bd))

        # Grid-based: for each (x,y) from bounds, compute min_z
        xs = sorted(b for b in self._x_bounds if 0 <= b <= self.W - bw)
        ys = sorted(b for b in self._y_bounds if 0 <= b <= self.H - bh)

        # Limit grid to avoid explosion
        if len(xs) <= 15 and len(ys) <= 15:
            for x in xs[:12]:
                for y in ys[:12]:
                    mz = self.min_z_at(x, y, bw, bh)
                    if mz + bd <= self.D:
                        cands.add((x, y, mz))

        # Filter valid
        valid = []
        for (cx, cy, cz) in cands:
            if 0 <= cx and cx + bw <= self.W and 0 <= cy and cy + bh <= self.H and 0 <= cz and cz + bd <= self.D:
                valid.append((cx, cy, cz))
        return valid

    def best_position(self, bw, bh, bd, score_fn=None):
        """Find best position for box. Returns (x,y,z) or None."""
        cands = self.candidates_for(bw, bh, bd)
        best = None
        best_s = None
        for (cx, cy, cz) in cands:
            if self.can_place(cx, cy, cz, bw, bh, bd):
                if score_fn:
                    s = score_fn(cx, cy, cz, bw, bh, bd, self)
                else:
                    s = (cz, cy, cx)
                if best_s is None or s < best_s:
                    best_s = s
                    best = (cx, cy, cz)
        return best


def _score_pos(x, y, z, bw, bh, bd, state):
    """Score: prefer low z, then low y, then low x, then max contact."""
    contact = 0
    for (px, py, pz, pw, ph, pd) in state.placed:
        # x-face contact
        if x + bw == px:
            oy = min(y + bh, py + ph) - max(y, py)
            oz = min(z + bd, pz + pd) - max(z, pz)
            if oy > 0 and oz > 0:
                contact += oy * oz
        if x == px + pw:
            oy = min(y + bh, py + ph) - max(y, py)
            oz = min(z + bd, pz + pd) - max(z, pz)
            if oy > 0 and oz > 0:
                contact += oy * oz
        # y-face contact
        if y + bh == py:
            ox = min(x + bw, px + pw) - max(x, px)
            oz = min(z + bd, pz + pd) - max(z, pz)
            if ox > 0 and oz > 0:
                contact += ox * oz
        if y == py + ph:
            ox = min(x + bw, px + pw) - max(x, px)
            oz = min(z + bd, pz + pd) - max(z, pz)
            if ox > 0 and oz > 0:
                contact += ox * oz
        # z-face contact
        if z + bd == pz:
            ox = min(x + bw, px + pw) - max(x, px)
            oy = min(y + bh, py + ph) - max(y, py)
            if ox > 0 and oy > 0:
                contact += ox * oy
        if z == pz + pd:
            ox = min(x + bw, px + pw) - max(x, px)
            oy = min(y + bh, py + ph) - max(y, py)
            if ox > 0 and oy > 0:
                contact += ox * oy
    return (z, y, x, -contact)


def _greedy(order, boxes, container, deadline, use_score=True):
    """Greedy placement with given box order."""
    W, H, D = container
    n = len(boxes)
    state = SkylineState(W, H, D)
    result = [None] * n
    sf = _score_pos if use_score else None

    for idx in order:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        rots = _rotations_sorted(w, h, d)
        best = None
        best_s = None
        best_rot = None
        for (bw, bh, bd) in rots:
            pos = state.best_position(bw, bh, bd, sf)
            if pos is not None:
                s = (pos[2], pos[1], pos[0])
                if best_s is None or s < best_s:
                    best_s = s
                    best = pos
                    best_rot = (bw, bh, bd)
        if best is not None:
            x, y, z = best
            bw, bh, bd = best_rot
            state.place(x, y, z, bw, bh, bd)
            result[idx] = (0, x, y, z, bw, bh, bd)
        else:
            result[idx] = (-1, 0, 0, 0, w, h, d)

    for i in range(n):
        if result[i] is None:
            w, h, d = boxes[i]
            result[i] = (-1, 0, 0, 0, w, h, d)
    return result


def _vol(result):
    return sum(r[4]*r[5]*r[6] for r in result if r[0] == 0)


def _orders(boxes, n):
    """Generate multiple heuristic orderings."""
    vols = [boxes[i][0]*boxes[i][1]*boxes[i][2] for i in range(n)]
    maxdims = [max(boxes[i]) for i in range(n)]
    mindims = [min(boxes[i]) for i in range(n)]
    sas = [2*(boxes[i][0]*boxes[i][1]+boxes[i][1]*boxes[i][2]+boxes[i][0]*boxes[i][2]) for i in range(n)]

    o1 = sorted(range(n), key=lambda i: vols[i], reverse=True)
    o2 = sorted(range(n), key=lambda i: maxdims[i], reverse=True)
    o3 = sorted(range(n), key=lambda i: sas[i], reverse=True)
    o4 = sorted(range(n), key=lambda i: mindims[i], reverse=True)
    o5 = sorted(range(n), key=lambda i: maxdims[i]/max(1, mindims[i]))  # compact first
    o6 = sorted(range(n), key=lambda i: vols[i]*maxdims[i], reverse=True)
    o7 = sorted(range(n), key=lambda i: sas[i]/max(1, vols[i]))  # low SVR first
    o8 = sorted(range(n), key=lambda i: vols[i]*(1+0.5*(1-maxdims[i]/max(1,mindims[i]))), reverse=True)

    return [o1, o2, o3, o4, o5, o6, o7, o8]


def _local_search(result, boxes, container, deadline, max_iters=800):
    """Enhanced local search with 1-box, 2-box, and 3-box moves."""
    W, H, D = container
    n = len(boxes)
    best = [tuple(r) for r in result]
    best_v = _vol(best)
    rng = random.Random(42)

    def build_state(exclude):
        st = SkylineState(W, H, D)
        for i in range(n):
            if i not in exclude and best[i][0] == 0:
                x, y, z, bw, bh, bd = best[i][1:7]
                st.place(x, y, z, bw, bh, bd)
        return st

    def try_place(idx, st):
        w, h, d = boxes[idx]
        rots = _rotations_sorted(w, h, d)
        best_pos = None
        best_s = None
        best_rot = None
        for (bw, bh, bd) in rots:
            pos = st.best_position(bw, bh, bd)
            if pos is not None:
                s = (pos[2], pos[1], pos[0])
                if best_s is None or s < best_s:
                    best_s = s
                    best_pos = pos
                    best_rot = (bw, bh, bd)
        if best_pos is not None:
            return True, best_pos, best_rot
        return False, None, None

    # Phase 1: Place unplaced boxes
    unplaced = [i for i in range(n) if best[i][0] == -1]
    for _ in range(max(20, len(unplaced) * 3)):
        if time.time() > deadline or not unplaced:
            break
        idx = rng.choice(unplaced)
        st = build_state(set())
        ok, pos, rot = try_place(idx, st)
        if ok:
            x, y, z = pos
            bw, bh, bd = rot
            temp = list(best)
            temp[idx] = (0, x, y, z, bw, bh, bd)
            nv = _vol(temp)
            if nv > best_v:
                best = [tuple(r) for r in temp]
                best_v = nv
                unplaced.remove(idx)

    # Phase 2: 1-box relocation
    placed_idx = [i for i in range(n) if best[i][0] == 0]
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        if not placed_idx:
            break
        idx = rng.choice(placed_idx)
        st = build_state({idx})
        ok, pos, rot = try_place(idx, st)
        if ok:
            x, y, z = pos
            bw, bh, bd = rot
            temp = list(best)
            temp[idx] = (0, x, y, z, bw, bh, bd)
            nv = _vol(temp)
            if nv > best_v:
                best = [tuple(r) for r in temp]
                best_v = nv

    # Phase 3: 2-box swap
    for _ in range(max_iters // 2):
        if time.time() > deadline:
            break
        if len(placed_idx) < 2:
            break
        i1, i2 = rng.sample(placed_idx, 2)
        st = build_state({i1, i2})
        ok1, pos1, rot1 = try_place(i1, st)
        if ok1:
            st.place(pos1[0], pos1[1], pos1[2], rot1[0], rot1[1], rot1[2])
            ok2, pos2, rot2 = try_place(i2, st)
            if ok2:
                temp = list(best)
                temp[i1] = (0, pos1[0], pos1[1], pos1[2], rot1[0], rot1[1], rot1[2])
                temp[i2] = (0, pos2[0], pos2[1], pos2[2], rot2[0], rot2[1], rot2[2])
                nv = _vol(temp)
                if nv > best_v:
                    best = [tuple(r) for r in temp]
                    best_v = nv

    # Phase 4: 3-box swap (less frequent)
    for _ in range(max_iters // 4):
        if time.time() > deadline:
            break
        if len(placed_idx) < 3:
            break
        i1, i2, i3 = rng.sample(placed_idx, 3)
        st = build_state({i1, i2, i3})
        ok1, pos1, rot1 = try_place(i1, st)
        if ok1:
            st.place(pos1[0], pos1[1], pos1[2], rot1[0], rot1[1], rot1[2])
            ok2, pos2, rot2 = try_place(i2, st)
            if ok2:
                st.place(pos2[0], pos2[1], pos2[2], rot2[0], rot2[1], rot2[2])
                ok3, pos3, rot3 = try_place(i3, st)
                if ok3:
                    temp = list(best)
                    temp[i1] = (0, pos1[0], pos1[1], pos1[2], rot1[0], rot1[1], rot1[2])
                    temp[i2] = (0, pos2[0], pos2[1], pos2[2], rot2[0], rot2[1], rot2[2])
                    temp[i3] = (0, pos3[0], pos3[1], pos3[2], rot3[0], rot3[1], rot3[2])
                    nv = _vol(temp)
                    if nv > best_v:
                        best = [tuple(r) for r in temp]
                        best_v = nv

    return best


def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    deadline = time.time() + 115

    if n == 0:
        return []

    orders = _orders(boxes, n)
    best_result = None
    best_vol = -1

    # Greedy with different orderings
    for order in orders:
        if time.time() > deadline - 25:
            break
        res = _greedy(order, boxes, container, deadline)
        v = _vol(res)
        if v > best_vol:
            best_vol = v
            best_result = res

    # A few random restarts
    if time.time() < deadline - 12 and best_result is not None:
        rng_r = random.Random(99)
        for _ in range(5):
            if time.time() > deadline - 8:
                break
            ro = list(range(n))
            rng_r.shuffle(ro)
            res = _greedy(ro, boxes, container, deadline)
            v = _vol(res)
            if v > best_vol:
                best_vol = v
                best_result = res

    # Local search
    remaining = deadline - time.time()
    if remaining > 4 and best_result is not None:
        mi = max(200, int(remaining * 50))
        mi = min(mi, 3000)
        best_result = _local_search(best_result, boxes, container, deadline, max_iters=mi)

    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    return [tuple(r) for r in best_result]

# EVOLVE-BLOCK-END
