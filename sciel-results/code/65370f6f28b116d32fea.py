"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import permutations

# EVOLVE-BLOCK-START
import math
import random

def _all_rotations(w, h, d):
    """Return all unique (w,h,d) permutations."""
    seen = set()
    res = []
    for perm in permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            res.append(perm)
    return res

def _max_cuboid_in_grid(grid, max_w, max_h, max_d):
    """
    Find the largest volume cuboid that fits in the given boolean grid (True = empty).
    Uses a simplified approach: try all possible sizes from largest to smallest.
    This is O(W*H*D*W*H*D) in worst case but with early termination.
    For better performance, we use a heuristic: scan for the largest empty box.
    """
    W, H, D = grid.shape
    # We'll try to find the best placement by scanning
    best_vol = 0
    best_pos = None
    best_size = None
    
    # For efficiency, we limit the search space
    # Try sizes in descending order of volume
    # But that's too slow. Instead, use a heuristic:
    # Find the largest empty sub-box using a 3D prefix-sum-like approach
    
    # Simpler heuristic: try a few candidate positions and sizes
    # This is a compromise between quality and speed
    
    # Actually, let me implement a proper but bounded search
    # We'll scan all positions but limit the size search
    
    # For each position, find the maximum extent in each direction
    # This gives us the "maximal empty box" at that position
    
    # To speed up, we'll only check a subset of positions
    # Or use the fact that we want large boxes
    
    # Let's do a full scan but with early termination
    # For each (x,y,z), compute how far we can extend in +x, +y, +z directions
    # while staying within bounds and within empty space
    
    # Precompute for efficiency: for each x, the maximum y and z extent
    # This is complex. Let's use a simpler approach:
    
    # Scan all positions, but for each, try a limited set of sizes
    # Start with the largest possible size and work down
    
    # Actually, for 60 boxes, the container is probably not huge.
    # Let's just do a reasonable scan.
    
    # For each candidate position (x, y, z), find the max empty box starting there
    # We can do this by checking consecutive empty voxels
    
    # To make this tractable, let's limit the search
    # We'll check all positions but use a greedy approach for size
    
    # For each position, the max box size is limited by:
    # - Distance to container boundary
    # - Distance to next occupied voxel in each direction
    
    # Let's precompute the "next occupied" distances
    # This is getting complex. Let me use a simpler heuristic:
    
    # Try placing boxes at corner positions and a few other strategic positions
    
    # For now, let's do a full scan but with a time limit
    # Actually, let me implement a proper algorithm:
    
    # For each (x, y, z) position:
    #   max_w = min(max_w, W - x)
    #   max_h = min(max_h, H - y)  
    #   max_d = min(max_d, D - z)
    #   Then find the largest (w,h,d) <= (max_w, max_h, max_d) such that
    #   grid[x:x+w, y:y+h, z:z+d] is all True (empty)
    
    # To check if a box fits, we need grid[x:x+w, y:y+h, z:z+d].all()
    # This is O(w*h*d) per check, which is too slow.
    
    # Better: use 3D prefix sums for O(1) box sum queries
    # But the grid is boolean, so we need to check if all are True
    # i.e., the sum of False values in the box is 0
    
    # Let me compute a 3D prefix sum of the "occupied" (False) cells
    
    occupied = (~grid).astype(np.int32)
    # 3D prefix sum
    ps = np.zeros((W+1, H+1, D+1), dtype=np.int32)
    ps[1:, 1:, 1:] = occupied
    # Cumulative sums
    ps[1:, 1:, 1:] = np.cumsum(ps[1:, 1:, 1:], axis=0)
    ps[1:, 1:, 1:] = np.cumsum(ps[1:, 1:, 1:], axis=1)
    ps[1:, 1:, 1:] = np.cumsum(ps[1:, 1:, 1:], axis=2)
    
    # Now, sum of occupied in box [x:x+w, y:y+h, z:z+d] is:
    # ps[x+w, y+h, z+d] - ps[x, y+h, z+d] - ps[x+w, y, z+d] - ps[x+w, y+h, z]
    # + ps[x, y, z+d] + ps[x, y+h, z] + ps[x+w, y, z] - ps[x, y, z]
    
    # But this is still O(1) per box check, and we have O(W*H*D*W*H*D) checks
    # which is too many.
    
    # Let me use a different approach: for each position, find the max extent
    # in each direction by scanning
    
    # For position (x, y, z), the max w is the largest w such that
    # for all y' in [y, y+max_h) and z' in [z, z+max_d), grid[x+w, y', z'] is True
    # This is hard to compute quickly.
    
    # Let me just do a heuristic: try a few sizes at each position
    # Or use a greedy approach: start with the largest possible box and shrink
    
    # Given the time constraint, let me use a simpler approach:
    # Just try to place the largest possible box at each corner and a few positions
    
    # Actually, let me step back. The grid approach is good for validation,
    # but for finding the best placement, I need a different strategy.
    
    # Let me use a different approach: instead of finding the max box in the grid,
    # I'll use a heuristic packing algorithm.
    
    # For now, let me return None and use a simpler placement strategy
    return None

def load(boxes, container):
    W, H, D = container
    container_vol = W * H * D
    
    # Sort boxes by volume descending
    box_list = [(w, h, d, i) for i, (w, h, d) in enumerate(boxes)]
    box_list.sort(key=lambda x: x[0]*x[1]*x[2], reverse=True)
    
    # Use a grid for occupancy
    # Grid resolution: 1 unit per voxel
    grid = np.zeros((W, H, D), dtype=bool)  # False = empty, True = occupied
    
    placements = [(-1, 0, 0, 0, 0, 0, 0)] * len(boxes)
    
    def can_place(x, y, z, w, h, d):
        """Check if a box of size (w,h,d) can be placed at (x,y,z)."""
        if x + w > W or y + h > H or z + d > D:
            return False
        # Check if the region is empty
        region = grid[x:x+w, y:y+h, z:z+d]
        return not region.any()
    
    def place(x, y, z, w, h, d, box_idx):
        """Place a box at (x,y,z) with size (w,h,d)."""
        grid[x:x+w, y:y+h, z:z+d] = True
        placements[box_idx] = (0, x, y, z, w, h, d)
    
    def find_placement(w, h, d):
        """Find a position to place a box of size (w,h,d)."""
        # Try all rotations
        for rw, rh, rd in _all_rotations(w, h, d):
            # Try to place at each position
            # Use a heuristic: try positions in a specific order
            # Start from corners and edges
            for x in range(0, W - rw + 1):
                for y in range(0, H - rh + 1):
                    for z in range(0, D - rd + 1):
                        if can_place(x, y, z, rw, rh, rd):
                            return (x, y, z, rw, rh, rd)
        return None
    
    # Place boxes one by one
    for w, h, d, idx in box_list:
        pos = find_placement(w, h, d)
        if pos is not None:
            x, y, z, rw, rh, rd = pos
            place(x, y, z, rw, rh, rd, idx)
    
    return placements
# EVOLVE-BLOCK-END
