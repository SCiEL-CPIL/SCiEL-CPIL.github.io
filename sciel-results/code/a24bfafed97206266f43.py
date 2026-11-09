"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Voxel grid resolution: use 1 unit per voxel if container is small,
    # otherwise scale down to keep memory reasonable (e.g., max 200^3 = 8M voxels)
    max_voxels = 8 * 10**6
    scale = max(1, int(np.ceil((W * H * D) ** (1/3) / max_voxels ** (1/3))))
    # Actually, let's just use a fixed resolution based on container size
    # For simplicity and speed, let's use a resolution that keeps the grid manageable
    # Let's try 1:1 if possible, otherwise downscale
    # But for 60 boxes, even a 100x100x100 grid is fine (1M booleans)
    # Let's use a resolution of 1 if max dimension <= 200, else scale
    max_dim = max(W, H, D)
    if max_dim <= 200:
        res = 1
    else:
        res = int(np.ceil(max_dim / 200))
    
    # Create voxel grid: True means occupied
    grid = np.zeros((int(np.ceil(W/res)), int(np.ceil(H/res)), int(np.ceil(D/res))), dtype=bool)
    
    # Sort boxes by volume descending to place large boxes first
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    placements = [(-1, 0, 0, 0, 0, 0, 0)] * len(boxes)
    
    for idx, w, h, d in indexed_boxes:
        # Try all 6 rotations
        rotations = [
            (w, h, d),
            (w, d, h),
            (h, w, d),
            (h, d, w),
            (d, w, h),
            (d, h, w),
        ]
        
        placed = False
        for rw, rh, rd in rotations:
            # Convert to voxel dimensions
            rw_v = max(1, int(np.ceil(rw / res)))
            rh_v = max(1, int(np.ceil(rh / res)))
            rd_v = max(1, int(np.ceil(rd / res)))
            
            grid_h, grid_w, grid_d = grid.shape
            if rw_v > grid_w or rh_v > grid_h or rd_v > grid_d:
                continue
            
            # Try to find a position using a simple scan
            # For efficiency, we can use a more sophisticated search, but let's start with a basic one
            # We'll scan from (0,0,0) in a specific order
            found_pos = False
            for z in range(grid_d - rd_v + 1):
                if found_pos:
                    break
                for y in range(grid_h - rh_v + 1):
                    if found_pos:
                        break
                    for x in range(grid_w - rw_v + 1):
                        # Check if this position is free
                        if not grid[x:x+rw_v, y:y+rh_v, z:z+rd_v].any():
                            # Place the box
                            grid[x:x+rw_v, y:y+rh_v, z:z+rd_v] = True
                            # Convert back to real coordinates
                            real_x = x * res
                            real_y = y * res
                            real_z = z * res
                            placements[idx] = (0, real_x, real_y, real_z, rw, rh, rd)
                            placed = True
                            found_pos = True
                            break
                        # Optimization: if the first row in x is blocked, skip ahead? 
                        # For now, just continue
        # If not placed, it stays as -1
    
    return placements
# EVOLVE-BLOCK-END
