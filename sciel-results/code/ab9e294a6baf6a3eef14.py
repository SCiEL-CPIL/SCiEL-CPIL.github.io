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
    vol = W * H * D
    if vol <= 0:
        return [(-1, 0, 0, 0, w, h, d) for (w, h, d) in boxes]

    # We'll use a resolution such that the grid is at most ~100 in each dim
    # but at least 1.
    def _res(dim, target_max=120, min_res=1):
        if dim <= 0:
            return 1
        n = max(min_res, int(np.ceil(dim)))
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
        x1 = int(np.ceil((x + w) / sx))
        y1 = int(np.ceil((y + h) / sy))
        z1 = int(np.ceil((z + d) / sz))
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
        region = occ[x0:x1, y0:y1, z0:z1]
        return not np.any(region)

    def _place(x, y, z, w, h, d):
        """Mark the box's voxels as occupied."""
        x0, y0, z0, x1, y1, z1 = _box_to_vox(x, y, z, w, h, d)
        occ[x0:x1, y0:y1, z0:z1] = 1

    # Sort boxes: largest first to fill big gaps early
    indexed = list(enumerate(boxes))
    indexed.sort(key=lambda t: (-(t[1][0] * t[1][1] * t[1][2]), -max(t[1])))

    placements = [None] * len(boxes)
    placed_count = 0

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
            # Generate candidate positions based on placed boxes' boundaries
            # Use a grid of candidate positions derived from existing placements
            # and container boundaries
            
            # Collect boundary coordinates from placed boxes
            x_bounds = [0.0, max(0.0, W - rw)]
            y_bounds = [0.0, max(0.0, H - rh)]
            z_bounds = [0.0, max(0.0, D - rd)]
            
            # Add boundaries from placed boxes
            for p in placements:
                if p is not None and p[0] == 0:
                    _, px, py, pz, pw, ph, pd = p
                    # Right face of placed box
                    if px + pw <= W - rw + 1e-9:
                        x_bounds.append(px + pw)
                    # Left face
                    if px >= rw + 1e-9:
                        x_bounds.append(px - rw)
                    # Back face
                    if py + ph <= H - rh + 1e-9:
                        y_bounds.append(py + ph)
                    # Front face
                    if py >= rh + 1e-9:
                        y_bounds.append(py - rh)
                    # Top face
                    if pz + pd <= D - rd + 1e-9:
                        z_bounds.append(pz + pd)
                    # Bottom face
                    if pz >= rd + 1e-9:
                        z_bounds.append(pz - rd)
            
            # Also add some intermediate positions
            if W > rw * 2:
                x_bounds.append(W / 2.0 - rw / 2.0)
            if H > rh * 2:
                y_bounds.append(H / 2.0 - rh / 2.0)
            if D > rd * 2:
                z_bounds.append(D / 2.0 - rd / 2.0)
            
            # Filter valid boundaries
            x_bounds = [x for x in x_bounds if 0 <= x <= W - rw + 1e-9]
            y_bounds = [y for y in y_bounds if 0 <= y <= H - rh + 1e-9]
            z_bounds = [z for z in z_bounds if 0 <= z <= D - rd + 1e-9]
            
            # Limit to avoid too many candidates
            if len(x_bounds) > 10:
                x_bounds = x_bounds[:10]
            if len(y_bounds) > 10:
                y_bounds = y_bounds[:10]
            if len(z_bounds) > 10:
                z_bounds = z_bounds[:10]
            
            # Try all combinations
            for cx in x_bounds:
                for cy in y_bounds:
                    for cz in z_bounds:
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
            placed_count += 1
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)

    return placements
# EVOLVE-BLOCK-END
