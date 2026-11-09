import numpy as np

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    n_boxes = len(boxes)
    if n_boxes == 0:
        return []
    
    # Sort boxes by volume descending to place larger boxes first (better for bin packing)
    # But we must return placements in original order
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    # Container volume for scoring
    container_vol = W * H * D
    
    # Bins will be lists of (x, y, z, w, h, d)
    bins = []
    # For each bin, maintain a 3D boolean occupancy grid
    bin_grids = []
    
    # We'll track for each bin:
    # - grid: np.ndarray of shape (W, H, D) boolean
    # - occupied_volume: int
    # - placements: list of (x, y, z, w, h, d)
    
    # Helper: try to place a box (w, h, d) in a bin with grid, return (x, y, z) or None
    # Use "best fit" by minimizing distance to origin (x+y+z) as a proxy for compactness
    def find_placement(grid, w, h, d):
        # Find all positions where the box fits
        # We want the position that minimizes (x + y + z) to keep things compact
        # Vectorized approach:
        # A position (x, y, z) is valid if grid[x:x+w, y:y+h, z:z+d] has no True values
        
        # We can compute a "validity" array of shape (W-w+1, H-h+1, D-d+1)
        # where validity[x, y, z] = not np.any(grid[x:x+w, y:y+h, z:z+d])
        
        # For efficiency, we can use a sliding window sum approach
        # But for small grids, direct checking might be fine
        
        if w > W or h > H or d > D:
            return None
        
        # Create a valid array
        # valid[x, y, z] = 1 if the box fits at (x, y, z)
        # We can compute this by checking if the sum of occupied cells in the box region is 0
        
        # For performance, let's use a cumulative sum approach
        # But given the grid sizes might be small, let's try a direct approach first
        # Actually, let's use the cumulative sum for efficiency
        
        # Compute 3D prefix sum
        # prefix[i, j, k] = sum of grid[0:i, 0:j, 0:k]
        # Then sum of grid[x:x+w, y:y+h, z:z+d] = 
        #   prefix[x+w, y+h, z+d] - prefix[x, y+h, z+d] - prefix[x+w, y, z+d] - prefix[x, y, z+d]
        #   + prefix[x, y, z+d] + prefix[x+w, y, z+d] + prefix[x, y+h, z+d] - prefix[x+w, y+h, z+d]
        
        # Let's build the prefix sum
        # grid has shape (W, H, D)
        # prefix has shape (W+1, H+1, D+1)
        prefix = np.zeros((W+1, H+1, D+1), dtype=np.int32)
        # Compute prefix sum
        for i in range(W):
            for j in range(H):
                for k in range(D):
                    prefix[i+1, j+1, k+1] = grid[i, j, k] + prefix[i, j+1, k+1] + prefix[i+1, j, k+1] + prefix[i, j, k+1] - prefix[i, j, k+1] - prefix[i+1, j+1, k] - prefix[i+1, j, k] + prefix[i, j, k]
        
        # Now check all valid positions
        # Position (x, y, z) is valid if x+w <= W, y+h <= H, z+d <= D
        # and the sum in the region is 0
        
        best_pos = None
        best_score = float('inf')
        
        for x in range(W - w + 1):
            for y in range(H - h + 1):
                for z in range(D - d + 1):
                    # Sum of occupied cells in [x, x+w) x [y, y+h) x [z, z+d)
                    s = (prefix[x+w, y+h, z+d] - prefix[x, y+h, z+d] - prefix[x+w, y, z+d] - prefix[x, y, z+d]
                         + prefix[x, y+h, z+d] + prefix[x+w, y, z+d] + prefix[x, y, z+d] - prefix[x+w, y+h, z+d])
                    if s == 0:
                        score = x + y + z  # Minimize distance to origin
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        
        return best_pos
    
    # More efficient version using vectorized operations
    def find_placement_fast(grid, w, h, d):
        if w > W or h > H or d > D:
            return None
        
        # Use cumulative sum for 3D
        # grid shape: (W, H, D)
        # We want to find all (x, y, z) such that the box fits
        
        # Compute 3D prefix sum using numpy
        # prefix[i, j, k] = sum of grid[0:i, 0:j, 0:k]
        prefix = np.zeros((W+1, H+1, D+1), dtype=np.int32)
        # This is O(W*H*D) which might be slow for large containers
        # Let's use a different approach for small grids
        
        # For now, use the direct approach but with numpy vectorization
        # Create a valid array of shape (W-w+1, H-h+1, D-d+1)
        
        # We can use a trick: for each offset, check if the region is all False
        # This is equivalent to checking if the max in the region is 0 (if we treat False as 0)
        
        # Let's just do it the simple way for now
        best_pos = None
        best_score = float('inf')
        
        # Precompute the grid as int for faster operations
        g = grid.astype(np.int32)
        
        for x in range(W - w + 1):
            for y in range(H - h + 1):
                for z in range(D - d + 1):
                    if g[x:x+w, y:y+h, z:z+d].sum() == 0:
                        score = x + y + z
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        
        return best_pos
    
    # Even faster: use 2D sliding window sums iterated over one dimension
    def find_placement_v2(grid, w, h, d):
        if w > W or h > H or d > D:
            return None
        
        # grid shape: (W, H, D)
        # We want to find (x, y, z) minimizing x+y+z such that the box fits
        
        # First, compute for each (x, y) the valid z range
        # For a fixed (x, y), the box fits in z if grid[x:x+w, y:y+h, z:z+d] is all False
        
        # Let's compute a 2D array valid_xy[x, y] = whether there exists any z such that the box fits
        # And for each valid (x, y), find the minimum z
        
        # For efficiency, compute the sum over the x and y dimensions for each z
        # sum_z[x, y, z] = sum of grid[x:x+w, y:y+h, z:z+w] -- no, that's not right
        
        # Let's just iterate
        best_pos = None
        best_score = float('inf')
        
        g = grid
        
        for x in range(W - w + 1):
            for y in range(H - h + 1):
                # Check if there's any z where the box fits
                # We need grid[x:x+w, y:y+h, z:z+d] to be all False
                # This is a 1D problem along z for each (x, y)
                
                # Extract the subgrid for this (x, y)
                sub = g[x:x+w, y:y+h, :]  # shape (w, h, D)
                # For each z, check if sub[:, :, z:z+d] is all False
                # sum_sub[z] = sum of sub[:, :, z:z+d]
                
                # Compute sliding window sum along z
                # For each z in range(D-d+1), sum = sub[:, :, z:z+d].sum()
                
                # Use cumulative sum
                cumsum = np.zeros((w, h, D+1), dtype=np.int32)
                for i in range(w):
                    for j in range(h):
                        for k in range(D):
                            cumsum[i, j, k+1] = sub[i, j, k] + cumsum[i, j, k]
                
                # Now for each z, the sum is cumsum[:, :, z+d] - cumsum[:, :, z], then sum over i, j
                for z in range(D - d + 1):
                    total = np.sum(cumsum[:, :, z+d] - cumsum[:, :, z])
                    if total == 0:
                        score = x + y + z
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
                        break  # Since we're iterating z in increasing order, the first valid z is the best for this (x, y)
        
        return best_pos
    
    # Let's use a simpler but effective approach: First Fit Decreasing with grid-based overlap check
    # Sort boxes by volume descending
    # For each box, try to place in existing bins (first fit), then create new bin
    
    # Placement results: indexed by original box index
    result = [None] * n_boxes
    
    for idx, w, h, d in indexed_boxes:
        placed = False
        
        # Try existing bins
        for b_idx in range(len(bins)):
            grid = bin_grids[b_idx]
            pos = find_placement_v2(grid, w, h, d)
            if pos is not None:
                x, y, z = pos
                # Mark as occupied
                grid[x:x+w, y:y+h, z:z+d] = True
                bins[b_idx].append((x, y, z, w, h, d))
                result[idx] = (b_idx, x, y, z, w, h, d)
                placed = True
                break
        
        if not placed:
            # Create new bin
            b_idx = len(bins)
            grid = np.zeros((W, H, D), dtype=bool)
            # Place at origin
            x, y, z = 0, 0, 0
            grid[x:x+w, y:y+h, z:z+d] = True
            bins.append([(x, y, z, w, h, d)])
            bin_grids.append(grid)
            result[idx] = (b_idx, x, y, z, w, h, d)
    
    return result
# EVOLVE-BLOCK-END
