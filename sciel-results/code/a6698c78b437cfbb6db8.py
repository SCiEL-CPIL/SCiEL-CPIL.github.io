"""Program using 3D occupancy grid (voxel map) for fast placement checks."""
import numpy as np

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for (w, h, d) in boxes]

    # Choose a voxel resolution that keeps the grid manageable.
    # We want at least a few voxels per unit so that small boxes fit.
    # Cap total voxels to avoid memory blow-up.
    max_voxels = 2_000_000  # ~16 MB for a uint8 array
    # Estimate resolution
    vol = W * H * D
    if vol <= 0:
        return [(-1, 0, 0, 0, w, h, d) for (w, h, d) in boxes]

    # We'll use a resolution such that the grid is at most ~100 in each dim
    # but at least 1.
    def _res(dim, target_max=120, min_res=1):
        if dim <= 0:
            return 1
        # number of voxels in this dim
        n = max(min_res, int(np.ceil(dim)))
        # cap
        if n > target_max:
            n = target_max
        return n

    nx = _res(W)
    ny = _res(H)
    nz = _res(D)

    # If total voxels too big, reduce the largest dimension
    while nx * ny * nz > max_voxels and max(nx, ny, nz) > 1:
        if nx >= ny and nx >= nz:
            nx = max(1, nx - 1)
        elif ny >= nx and ny >= nz:
            ny = max(1, ny - 1)
        else:
            nz = max(1, nz - 1)

    # Scale factors: how many units per voxel
    sx = W / nx if nx > 0 else 1.0
    sy = H / ny if ny > 0 else 1.0
    sz = D / nz if nz > 0 else 1.0

    # Occupancy grid
    occ = np.zeros((nx, ny, nz), dtype=np.uint8)

    def _box_to_vox(x, y, z, w, h, d):
        """Convert a box at (x,y,z) with size (w,h,d) to voxel indices."""
        x0 = int(np.floor(x / sx))
        y0 = int(np.floor(y / sy))
        z0 = int(np.floor(z / sz))
        # End positions (exclusive in voxel space)
        x1 = int(np.ceil((x + w) / sx))
        y1 = int(np.ceil((y + h) / sy))
        z1 = int(np.ceil((z + d) / sz))
        # Clamp
        x0 = max(0, min(nx - 1, x0))
        y0 = max(0, min(ny - 1, y0))
        z0 = max(0, min(nz - 1, z0))
        x1 = max(x0 + 1, min(nx, x1))
        y1 = max(y0 + 1, min(ny, y1))
        z1 = max(z0 + 1, min(nz, z1))
        return x0, y0, z0, x1, y1, z1

    def _can_place(x, y, z, w, h, d):
        """Check if box fits in free space using the voxel grid."""
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W + 1e-9 or y + h > H + 1e-9 or z + d > D + 1e-9:
            return False
        x0, y0, z0, x1, y1, z1 = _box_to_vox(x, y, z, w, h, d)
        if x1 <= x0 or y1 <= y0 or z1 <= z0:
            return False
        # Check if any voxel in the region is occupied
        region = occ[x0:x1, y0:y1, z0:z1]
        return not np.any(region)

    def _place(x, y, z, w, h, d):
        """Mark the box's voxels as occupied."""
        x0, y0, z0, x1, y1, z1 = _box_to_vox(x, y, z, w, h, d)
        occ[x0:x1, y0:y1, z0:z1] = 1

    # Sort boxes: largest first to fill big gaps early
    # Use volume as primary key, then largest dimension
    indexed = list(enumerate(boxes))
    indexed.sort(key=lambda t: (-(t[1][0] * t[1][1] * t[1][2]), -max(t[1])))

    placements = [None] * len(boxes)
    placed_count = 0

    # Track placed boxes for generating better candidate positions
    placed_boxes = []  # list of (x, y, z, w, h, d)

    def _candidate_positions(w, h, d, max_candidates=500):
        """Generate candidate (x,y,z) positions for a box of size (w,h,d)."""
        candidates = set()
        
        # Basic corner positions
        xs = [0.0]
        ys = [0.0]
        zs = [0.0]
        
        # Add positions at container edges
        if W - w > 1e-9:
            xs.append(W - w)
        if H - h > 1e-9:
            ys.append(H - h)
        if D - d > 1e-9:
            zs.append(D - d)
        
        # Generate positions based on placed boxes: try to place adjacent to existing boxes
        for (px, py, pz, pw, ph, pd) in placed_boxes:
            # Try placing next to this box in each direction
            # X-axis directions
            if px + pw + w <= W + 1e-9:
                candidates.add((round(px + pw, 6), py, pz))
            if px - w >= 0:
                candidates.add((round(px - w, 6), py, pz))
            # Y-axis directions
            if py + ph + h <= H + 1e-9:
                candidates.add((px, round(py + ph, 6), pz))
            if py - h >= 0:
                candidates.add((px, round(py - h, 6), pz))
            # Z-axis directions
            if pz + pd + d <= D + 1e-9:
                candidates.add((px, py, round(pz + pd, 6)))
            if pz - d >= 0:
                candidates.add((px, py, round(pz - d, 6)))
        
        # Add grid-based candidates for better coverage
        # Use a coarse grid in each dimension
        n_grid = 4
        for i in range(n_grid + 1):
            x_pos = i * W / n_grid
            if x_pos + w <= W + 1e-9:
                for j in range(n_grid + 1):
                    y_pos = j * H / n_grid
                    if y_pos + h <= H + 1e-9:
                        for k in range(n_grid + 1):
                            z_pos = k * D / n_grid
                            if z_pos + d <= D + 1e-9:
                                candidates.add((round(x_pos, 6), round(y_pos, 6), round(z_pos, 6)))
        
        # Also try to fill gaps: for each placed box, try to fill the space above it
        for (px, py, pz, pw, ph, pd) in placed_boxes:
            # Try placing on top
            if pz + pd + d <= D + 1e-9:
                candidates.add((px, py, round(pz + pd, 6)))
            # Try filling side gaps
            gap_x = W - (px + pw)
            if gap_x >= w - 1e-9 and px >= 0:
                candidates.add((round(px + pw, 6), py, pz))
            gap_y = H - (py + ph)
            if gap_y >= h - 1e-9 and py >= 0:
                candidates.add((px, round(py + ph, 6), pz))
        
        # Filter valid candidates and limit count
        valid_candidates = []
        for (cx, cy, cz) in candidates:
            if 0 <= cx <= W - w + 1e-9 and 0 <= cy <= H - h + 1e-9 and 0 <= cz <= D - d + 1e-9:
                valid_candidates.append((cx, cy, cz))
        
        # Sort by priority: lower z first (fill from bottom), then lower y, then lower x
        valid_candidates.sort(key=lambda p: (p[2], p[1], p[0]))
        
        if len(valid_candidates) > max_candidates:
            valid_candidates = valid_candidates[:max_candidates]
        
        return valid_candidates

    # For each box in sorted order, try to place it
    for idx, (w, h, d) in indexed:
        # All 6 rotations (permutations of dimensions)
        rotations = [
            (w, h, d),
            (w, d, h),
            (h, w, d),
            (h, d, w),
            (d, w, h),
            (d, h, w),
        ]
        # Remove duplicates
        seen = set()
        unique_rotations = []
        for r in rotations:
            if r not in seen:
                seen.add(r)
                unique_rotations.append(r)

        best_pos = None
        best_rot = None
        best_score = -1

        for rw, rh, rd in unique_rotations:
            # Try candidate positions
            cands = _candidate_positions(rw, rh, rd, max_candidates=200)
            for (cx, cy, cz) in cands:
                if _can_place(cx, cy, cz, rw, rh, rd):
                    # Score: prefer lower z, then lower y, then lower x (fill from bottom)
                    # Also prefer positions that minimize the "wasted" space
                    score = -(cz * 10000 + cy * 100 + cx)
                    if score > best_score:
                        best_score = score
                        best_pos = (cx, cy, cz)
                        best_rot = (rw, rh, rd)

        if best_pos is not None:
            bx, by, bz = best_pos
            bw, bh, bd = best_rot
            placements[idx] = (0, bx, by, bz, bw, bh, bd)
            _place(bx, by, bz, bw, bh, bd)
            placed_boxes.append((bx, by, bz, bw, bh, bd))
            placed_count += 1
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)

    return placements
# EVOLVE-BLOCK-END
