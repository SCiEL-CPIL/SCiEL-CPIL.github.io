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

    def _unplace(x, y, z, w, h, d):
        """Mark the box's voxels as free (for backtracking if needed)."""
        x0, y0, z0, x1, y1, z1 = _box_to_vox(x, y, z, w, h, d)
        occ[x0:x1, y0:y1, z0:z1] = 0

    # Sort boxes: largest first to fill big gaps early
    # Use volume as primary key, then largest dimension
    indexed = list(enumerate(boxes))
    indexed.sort(key=lambda t: (-(t[1][0] * t[1][1] * t[1][2]), -max(t[1])))

    placements = [None] * len(boxes)
    placed_count = 0

    # Track placed box extents for generating better candidate positions
    placed_extents_x = []  # list of (x_min, x_max)
    placed_extents_y = []  # list of (y_min, y_max)
    placed_extents_z = []  # list of (z_min, z_max)

    def _candidate_positions(w, h, d, max_candidates=500):
        """Generate candidate (x,y,z) positions for a box of size (w,h,d)."""
        candidates = set()
        
        # Base positions: corners
        xs = [0.0, max(0.0, W - w)]
        ys = [0.0, max(0.0, H - h)]
        zs = [0.0, max(0.0, D - d)]
        
        # Add positions aligned with placed boxes
        for xmin, xmax in placed_extents_x:
            # Try placing just after this box in x
            x_after = xmax
            if 0 <= x_after <= W - w + 1e-9:
                xs.append(x_after)
            # Try placing just before this box in x
            x_before = xmin - w
            if 0 <= x_before <= W - w + 1e-9:
                xs.append(x_before)
        
        for ymin, ymax in placed_extents_y:
            y_after = ymax
            if 0 <= y_after <= H - h + 1e-9:
                ys.append(y_after)
            y_before = ymin - h
            if 0 <= y_before <= H - h + 1e-9:
                ys.append(y_before)
        
        for zmin, zmax in placed_extents_z:
            z_after = zmax
            if 0 <= z_after <= D - d + 1e-9:
                zs.append(z_after)
            z_before = zmin - d
            if 0 <= z_before <= D - d + 1e-9:
                zs.append(z_before)
        
        # Add some intermediate positions
        if W > w * 2:
            xs.append(W / 2.0 - w / 2.0)
            xs.append(W / 4.0)
            xs.append(3.0 * W / 4.0 - w / 4.0)
        if H > h * 2:
            ys.append(H / 2.0 - h / 2.0)
            ys.append(H / 4.0)
            ys.append(3.0 * H / 4.0 - h / 4.0)
        if D > d * 2:
            zs.append(D / 2.0 - d / 2.0)
            zs.append(D / 4.0)
            zs.append(3.0 * D / 4.0 - d / 4.0)

        # Deduplicate and clamp
        xs = sorted(set([max(0.0, min(W - w, x)) for x in xs if 0 <= x <= W - w + 1e-9]))
        ys = sorted(set([max(0.0, min(H - h, y)) for y in ys if 0 <= y <= H - h + 1e-9]))
        zs = sorted(set([max(0.0, min(D - d, z)) for z in zs if 0 <= z <= D - d + 1e-9]))
        
        # Generate combinations, prioritizing lower z (bottom-up filling)
        # Sort zs in ascending order so we try bottom positions first
        for z in zs:
            for y in ys:
                for x in xs:
                    candidates.add((round(x, 6), round(y, 6), round(z, 6)))
        
        # Limit candidates
        cand_list = list(candidates)
        if len(cand_list) > max_candidates:
            # Sort by z, then y, then x to prioritize bottom-up
            cand_list.sort(key=lambda c: (c[2], c[1], c[0]))
            cand_list = cand_list[:max_candidates]
        return cand_list

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
            cands = _candidate_positions(rw, rh, rd, max_candidates=300)
            for (cx, cy, cz) in cands:
                if _can_place(cx, cy, cz, rw, rh, rd):
                    # Score: prefer lower z, then lower y, then lower x (fill from bottom)
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
            # Track extents for future candidate generation
            placed_extents_x.append((bx, bx + bw))
            placed_extents_y.append((by, by + bh))
            placed_extents_z.append((bz, bz + bd))
            placed_count += 1
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)

    return placements
# EVOLVE-BLOCK-END
