"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def pack(boxes, container):
    """
    Place boxes into bins using a 3D occupancy grid for fast collision checks.
    Returns list of (bin_id, x, y, z, w, h, d) for each box in original order.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    deadline = time.time() + 110.0  # safety margin before 120s kill

    # Sort boxes by volume descending for BFD, but keep original indices
    indexed_boxes = [(i, w * h * d, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1], reverse=True)

    # Each bin: (list of placed boxes, 3D occupancy grid)
    bins = []  # list of [placements_list, occupancy_grid]

    def make_grid():
        """Create a fresh 3D occupancy grid for a bin."""
        return np.zeros((W, H, D), dtype=np.bool_)

    def can_place_grid(grid, x, y, z, w, h, d):
        """Check if a box fits at (x,y,z) using the occupancy grid."""
        if x + w > W or y + h > H or z + d > D:
            return False
        # Check if the sub-cube is empty
        if not np.any(grid[x:x+w, y:y+h, z:z+d]):
            return True
        return False

    def place_in_grid(grid, x, y, z, w, h, d):
        """Mark a box as placed in the occupancy grid."""
        grid[x:x+w, y:y+h, z:z+d] = True

    def find_position(grid, w, h, d):
        """
        Find the best position for a box using candidate positions from box corners.
        Uses the occupancy grid for O(1) collision checks.
        Returns (x, y, z) or None if no position found.
        """
        # Get all placed boxes to generate candidate positions
        # We need to track placed boxes separately for candidate generation
        # For now, use a simpler approach: try positions on a grid
        # But better: generate candidates from existing box boundaries
        
        # First, try the most compact position (0,0,0)
        if can_place_grid(grid, 0, 0, 0, w, h, d):
            return (0, 0, 0)
        
        # Generate candidate positions from the grid's occupied boundaries
        # Find all unique x, y, z coordinates where boxes end
        # This is expensive, so use a heuristic: try positions along each axis
        
        best_pos = None
        best_score = float('inf')
        
        # Candidate x positions: 0 and all x where a box ends
        # We can find these by looking at the grid
        # For efficiency, just try a set of candidate positions
        
        # Try positions at (0, y, z), (x, 0, z), (x, y, 0) and combinations
        # Use the fact that good positions are near existing boxes
        
        # Get all occupied voxels to find boundaries
        # This is O(W*H*D) which might be too slow, so use a different approach
        
        # Instead, maintain a list of placed boxes for candidate generation
        # But we don't have that in this function signature
        
        # For now, use a simple strategy: try all possible positions where
        # the box could fit, prioritizing compactness
        
        # Actually, let's generate candidates from the grid more efficiently
        # Find all x positions where there's a transition from empty to occupied
        # and vice versa, and similarly for y and z
        
        # For a 40-box problem, the container is likely not too large
        # Let's just try a reasonable set of candidate positions
        
        # Candidate positions: try (x, y, z) where x, y, z are in {0, w_box, h_box, d_box}
        # for all placed boxes, plus 0
        
        # Since we don't have the placed boxes list here, let's use a different approach:
        # Scan the grid to find candidate positions
        
        # Find all unique x coordinates where the grid has changes
        # This is still expensive. Let's use a simpler heuristic.
        
        # Try positions: for each combination of x in [0, W-w], y in [0, H-h], z in [0, D-d]
        # But that's too many. Instead, use a grid of candidate positions
        
        # A better approach: use the fact that in BFD, we want to fill the bin compactly
        # Try positions that minimize (x, y, z) lexicographically
        
        # For efficiency, let's try a limited set of candidate positions
        # based on the current occupancy pattern
        
        # Find the maximum extent of occupied space in each dimension
        # and try positions just beyond those extents
        
        # Get the bounding box of all occupied voxels
        if not np.any(grid):
            return (0, 0, 0)
        
        # Find the min and max coordinates of occupied voxels
        # This is O(W*H*D) but for small containers it's fine
        # For larger containers, we need a better approach
        
        # Let's use a different strategy: maintain placed boxes separately
        # But that changes the function signature
        
        # For now, use a simple but effective heuristic:
        # Try positions in a spiral or grid pattern, prioritizing compactness
        
        # Actually, the simplest effective approach: try all positions where
        # at least one face of the new box is flush with an existing box face
        # or the container wall
        
        # Let's generate candidates from the grid's structure
        # Find all x where grid[x, :, :] has any True and grid[x-1, :, :] doesn't
        # (or x=0), and similarly for other dimensions
        
        # This is getting complex. Let's use a practical approach:
        # For 40 boxes, the container is likely manageable in size.
        # Try a set of candidate positions and pick the best.
        
        # Candidate positions: 
        # 1. (0, 0, 0) - already tried
        # 2. Positions where x=0 or x=W-w, y=0 or y=H-h, z=0 or z=D-d
        # 3. Positions adjacent to existing boxes
        
        # For simplicity, let's try a grid of candidate positions
        # with step size 1, but limit the search
        
        # Actually, let's just try all possible positions if the container is small enough
        # Otherwise, use a sampling approach
        
        # For now, use a simple approach: try positions in order of compactness
        # (x+y+z), and return the first valid one
        
        # This is O(W*H*D) in the worst case, which might be too slow
        # Let's limit the search
        
        # A better approach: use the corner-based method but with grid checks
        # We need to know the placed boxes' dimensions
        
        # Since we're refactoring, let's change the approach:
        # Keep both the grid and the list of placed boxes
        
        return None  # Placeholder - will be replaced

    # Let's redesign: keep both grid and placed boxes list
    bins = []  # list of [placements_list, occupancy_grid]

    def can_place_full(bin_data, x, y, z, w, h, d):
        """Check placement using both grid and bounds."""
        grid = bin_data[1]
        if x + w > W or y + h > H or z + d > D:
            return False
        return not np.any(grid[x:x+w, y:y+h, z:z+d])

    def find_position_full(bin_data, w, h, d):
        """
        Find the best position using corner candidates and grid checks.
        """
        placements, grid = bin_data
        
        # Generate candidate positions from corners of existing boxes
        candidates = set()
        candidates.add((0, 0, 0))
        for (px, py, pz, pw, ph, pd) in placements:
            candidates.add((px + pw, py, pz))
            candidates.add((px, py + ph, pz))
            candidates.add((px, py, pz + pd))
            candidates.add((px + pw, py + ph, pz))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px, py + ph, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
        
        best_pos = None
        best_score = None
        
        for (xc, yc, zc) in candidates:
            if xc + w > W or yc + h > H or zc + d > D:
                continue
            if can_place_full(bin_data, xc, yc, zc, w, h, d):
                score = xc + yc + zc
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = (xc, yc, zc)
        
        return best_pos

    # Main packing loop
    result = [None] * n
    
    for idx, vol, w, h, d in indexed_boxes:
        if time.time() > deadline:
            break
        
        best_bin = -1
        best_pos = None
        best_fit_score = None
        
        for b_idx, bin_data in enumerate(bins):
            pos = find_position_full(bin_data, w, h, d)
            if pos is not None:
                x, y, z = pos
                fit_score = (x + w, y + h, z + d)
                if best_fit_score is None or fit_score < best_fit_score:
                    best_fit_score = fit_score
                    best_bin = b_idx
                    best_pos = pos
        
        if best_bin >= 0:
            x, y, z = best_pos
            bins[best_bin][0].append((x, y, z, w, h, d))
            place_in_grid(bins[best_bin][1], x, y, z, w, h, d)
            result[idx] = (best_bin, x, y, z, w, h, d)
        else:
            # Create a new bin
            new_grid = make_grid()
            new_bin = [[(0, 0, 0, w, h, d)], new_grid]
            place_in_grid(new_grid, 0, 0, 0, w, h, d)
            bins.append(new_bin)
            result[idx] = (len(bins) - 1, 0, 0, 0, w, h, d)

    # Fallback for any unplaced boxes
    for idx, vol, w, h, d in indexed_boxes:
        if result[idx] is not None:
            continue
        placed = False
        for b_idx, bin_data in enumerate(bins):
            pos = find_position_full(bin_data, w, h, d)
            if pos is not None:
                x, y, z = pos
                bins[b_idx][0].append((x, y, z, w, h, d))
                place_in_grid(bins[b_idx][1], x, y, z, w, h, d)
                result[idx] = (b_idx, x, y, z, w, h, d)
                placed = True
                break
        if not placed:
            new_grid = make_grid()
            new_bin = [[(0, 0, 0, w, h, d)], new_grid]
            place_in_grid(new_grid, 0, 0, 0, w, h, d)
            bins.append(new_bin)
            result[idx] = (len(bins) - 1, 0, 0, 0, w, h, d)

    return result
# EVOLVE-BLOCK-END
