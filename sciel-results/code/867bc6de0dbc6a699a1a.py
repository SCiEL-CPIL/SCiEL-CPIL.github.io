import time
import sys
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START
def pack(boxes, container):
    """
    3D Bin Packing using Best-Fit Decreasing with grid-based spatial index
    for fast overlap detection and comprehensive candidate position generation.
    """
    W, H, D = container
    container_vol = W * H * D

    start_time = time.time()
    timeout = 130  # Safety margin before 150s kill

    # Sort boxes by volume descending, tie-break by max dimension
    indexed_boxes = sorted(
        enumerate(boxes),
        key=lambda x: (x[1][0] * x[1][1] * x[1][2], max(x[1])),
        reverse=True
    )

    # Grid resolution for spatial index
    grid_res = max(1, min(W, H, D) // 16)
    gx, gy, gz = max(1, W // grid_res), max(1, H // grid_res), max(1, D // grid_res)

    class BinState:
        __slots__ = ['boxes', 'grid', 'fill_vol', 'max_x', 'max_y', 'max_z']

        def __init__(self):
            self.boxes = []  # list of (x, y, z, w, h, d)
            self.grid = defaultdict(list)  # (ix, iy, iz) -> list of box indices
            self.fill_vol = 0
            self.max_x = 0
            self.max_y = 0
            self.max_z = 0

        def add_box(self, x, y, z, w, h, d):
            idx = len(self.boxes)
            self.boxes.append((x, y, z, w, h, d))
            self.fill_vol += w * h * d
            self.max_x = max(self.max_x, x + w)
            self.max_y = max(self.max_y, y + h)
            self.max_z = max(self.max_z, z + d)
            # Insert into grid cells
            ix_min = x // grid_res
            ix_max = min(gx - 1, (x + w - 1) // grid_res)
            iy_min = y // grid_res
            iy_max = min(gy - 1, (y + h - 1) // grid_res)
            iz_min = z // grid_res
            iz_max = min(gz - 1, (z + d - 1) // grid_res)
            for ix in range(ix_min, ix_max + 1):
                for iy in range(iy_min, iy_max + 1):
                    for iz in range(iz_min, iz_max + 1):
                        self.grid[(ix, iy, iz)].append(idx)

        def can_place(self, x, y, z, w, h, d):
            if x + w > W or y + h > H or z + d > D:
                return False
            if not self.boxes:
                return True
            # Check grid cells that overlap with the new box
            ix_min = x // grid_res
            ix_max = min(gx - 1, (x + w - 1) // grid_res)
            iy_min = y // grid_res
            iy_max = min(gy - 1, (y + h - 1) // grid_res)
            iz_min = z // grid_res
            iz_max = min(gz - 1, (z + d - 1) // grid_res)
            checked = set()
            for ix in range(ix_min, ix_max + 1):
                for iy in range(iy_min, iy_max + 1):
                    for iz in range(iz_min, iz_max + 1):
                        for bidx in self.grid.get((ix, iy, iz), []):
                            if bidx in checked:
                                continue
                            checked.add(bidx)
                            bx, by, bz, bw, bh, bd = self.boxes[bidx]
                            if (x < bx + bw and bx < x + w and
                                y < by + bh and by < y + h and
                                z < bz + bd and bz < z + d):
                                return False
            return True

        def get_candidates(self, w, h, d):
            """Generate candidate positions for placing a box."""
            candidates = set()
            candidates.add((0, 0, 0))

            if not self.boxes:
                return candidates

            for bx, by, bz, bw, bh, bd in self.boxes:
                # Adjacent faces
                candidates.add((bx + bw, by, bz))
                candidates.add((bx, by + bh, bz))
                candidates.add((bx, by, bz + bd))

                # Corners combining two offsets
                candidates.add((bx + bw, by + bh, bz))
                candidates.add((bx + bw, by, bz + bd))
                candidates.add((bx, by + bh, bz + bd))

                # Floor-aligned variants
                candidates.add((bx + bw, by, 0))
                candidates.add((bx, by + bh, 0))
                candidates.add((bx + bw, 0, bz))
                candidates.add((0, by, bz + bd))
                candidates.add((bx, 0, bz + bd))
                candidates.add((bx + bw, 0, 0))
                candidates.add((0, by + bh, 0))
                candidates.add((0, 0, bz + bd))

                # Additional structural points
                candidates.add((bx, by + bh, 0))
                candidates.add((bx + bw, by, 0))
                candidates.add((0, by, bz))
                candidates.add((bx, by, 0))

            # Add some grid-aligned positions
            for ix in range(gx):
                cx = min(ix * grid_res, W - w) if W - w >= 0 else 0
                for iy in range(gy):
                    cy = min(iy * grid_res, H - h) if H - h >= 0 else 0
                    candidates.add((cx, cy, 0))
            for iz in range(gz):
                cz = min(iz * grid_res, D - d) if D - d >= 0 else 0
                candidates.add((0, 0, cz))

            return candidates

        def find_best_position(self, w, h, d):
            """Find the best position for a box in this bin, or None."""
            if not self.boxes:
                if self.can_place(0, 0, 0, w, h, d):
                    return (0, 0, 0)
                return None

            candidates = self.get_candidates(w, h, d)

            valid = []
            for (cx, cy, cz) in candidates:
                if 0 <= cx < W and 0 <= cy < H and 0 <= cz < D:
                    if self.can_place(cx, cy, cz, w, h, d):
                        dist = cx + cy + cz
                        valid.append((dist, cx, cy, cz))

            if not valid:
                return None

            valid.sort()
            return (valid[0][1], valid[0][2], valid[0][3])

    bins = []
    results = [None] * len(boxes)

    for idx, (w, h, d) in indexed_boxes:
        elapsed = time.time() - start_time
        if elapsed > timeout - 2:
            b_idx = len(bins)
            bs = BinState()
            bs.add_box(0, 0, 0, w, h, d)
            bins.append(bs)
            results[idx] = (b_idx, 0, 0, 0, w, h, d)
            continue

        best_bin_idx = -1
        best_pos = None
        best_score = float('inf')

        for b_idx, bs in enumerate(bins):
            pos = bs.find_best_position(w, h, d)
            if pos is not None:
                x, y, z = pos
                fill = bs.fill_vol / container_vol
                # Prefer bins with higher fill to consolidate
                score = -(x + y + z) - 100.0 * fill
                if score < best_score:
                    best_score = score
                    best_bin_idx = b_idx
                    best_pos = pos

        if best_bin_idx >= 0:
            x, y, z = best_pos
            bins[best_bin_idx].add_box(x, y, z, w, h, d)
            results[idx] = (best_bin_idx, x, y, z, w, h, d)
        else:
            b_idx = len(bins)
            bs = BinState()
            bs.add_box(0, 0, 0, w, h, d)
            bins.append(bs)
            results[idx] = (b_idx, 0, 0, 0, w, h, d)

    return results

# EVOLVE-BLOCK-END
