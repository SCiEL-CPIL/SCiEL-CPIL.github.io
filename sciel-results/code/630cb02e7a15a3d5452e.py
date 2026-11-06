"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def pack(boxes, container):
    W, H, D = container
    n_boxes = len(boxes)
    
    # Sort boxes by volume descending, but keep track of original indices
    # Actually, the problem says "preserving its original ordered dimensions (no rotations)"
    # It does NOT say we must preserve the order of boxes in the output.
    # The output is "one placement ... per box". Usually this implies the i-th box in input
    # corresponds to the i-th tuple in output. Let's re-read carefully.
    # "one placement (bin, x, y, z, w, h, d) per box, preserving its original ordered dimensions"
    # This phrasing is slightly ambiguous. Does it mean the dimensions (w,h,d) of the box
    # must be preserved (i.e., no rotation), or that the order of the list must be preserved?
    # "preserving its original ordered dimensions" likely refers to the (w,h,d) tuple itself
    # not being rotated. The standard interpretation for these packing tasks is that
    # the output list index i corresponds to the input list index i.
    # Let's assume output[i] corresponds to input[i].
    
    # If we sort boxes for packing, we must map back to original indices.
    
    # Strategy:
    # 1. Sort boxes by some heuristic (e.g., max dimension descending, or volume descending)
    #    to place larger boxes first, which usually leads to better packing.
    # 2. Use a Best-Fit Decreasing (BFD) or First-Fit Decreasing (FFD) approach.
    #    Given the score function: mean B + 0.1*(1 - last-bin occupied volume fraction)
    #    Minimizing B is the primary goal. The secondary goal is to fill the last bin as much as possible.
    #    This suggests we should try to fit boxes into existing bins as tightly as possible.
    #    A simple "First Fit Decreasing" where we try to place each box in the first bin where it fits
    #    might not be optimal. "Best Fit Decreasing" (place in the bin where it leaves the least space)
    #    is often better.
    
    # However, implementing a true 3D Best-Fit with efficient collision checking is complex.
    # The directive suggests using a 3D occupancy grid with NumPy for O(1) or vectorized collision checks.
    
    # Let's implement a simplified version:
    # - Maintain a list of bins. Each bin has a 3D boolean array (or NumPy array) representing occupied voxels.
    # - For each box (in sorted order), try to place it in each existing bin.
    # - For a given bin and box, check if it fits at position (0,0,0). If not, we need to find a valid position.
    # - Finding the "best" position is hard. A common heuristic is to try to place it as low (z), as left (x), and as back (y) as possible.
    # - We can iterate over possible z, y, x positions? That's O(W*H*D) per box per bin, which might be too slow if dimensions are large.
    # - But the problem says "40-box instances". Dimensions are likely small enough (e.g., <= 50) that a grid search is feasible.
    # - Let's assume container dimensions are small (e.g., <= 100). W*H*D <= 1,000,000. 40 boxes * 100 bins * 1,000,000 checks is too slow in Python loops.
    # - We need vectorized operations.
    
    # Revised Plan:
    # 1. Sort boxes by volume descending.
    # 2. For each box, try to place it in each bin (starting from bin 0).
    # 3. For each bin, find the "lowest, leftmost, rearmost" position where the box fits.
    #    - We can create a "free space" array for the bin.
    #    - To find a valid position for a box of size (w, h, d), we need a (x,y,z) such that
    #      free[x:x+w, y:y+h, z:z+d] is all True.
    #    - This is a 3D convolution or sliding window sum problem.
    #    - If we compute the sum of occupied voxels in every (w,h,d) window, we can find windows with sum=0.
    #    - However, doing this for every box and every bin from scratch is expensive.
    
    # Alternative: Use a simpler heuristic that is fast.
    # "Shelf" packing is fast but not optimal.
    # Let's try a "Next Fit" or "First Fit" with a smarter placement rule.
    # Or, use a simple local search / hill climbing?
    
    # Let's stick to the directive: "Represent the bin state as a 3D occupancy grid... enable O(1) collision checks via bitwise operations or NumPy array slicing... evaluate thousands of candidate placements per bin in parallel".
    
    # Implementation details:
    # - Bins: list of dicts or objects.
    #   - `grid`: 3D NumPy array of shape (D, H, W) or (W, H, D). Let's use (D, H, W) where D is depth (z), H is height (y), W is width (x).
    #     Actually, standard NumPy indexing is [z, y, x] or [x, y, z]? Let's use [x, y, z] for clarity, so shape is (W, H, D).
    #   - `occupied_volume`: int.
    
    # For each box (w, h, d):
    #   1. For each bin in bins:
    #      a. Check if box fits in the remaining space of the bin.
    #      b. Find the best position (x, y, z).
    #         - We want to minimize z, then y, then x? Or minimize the "center of mass" shift?
    #         - Let's try to find the position with the smallest z, then smallest y, then smallest x.
    #         - How to find this efficiently?
    #         - We can iterate z from 0 to D-d.
    #           - For each z, check if there is any (x,y) such that grid[x:x+w, y:y+h, z:z+d] is all False (free).
    #           - This can be done by summing the grid slice: `np.sum(grid[x:x+w, y:y+h, z:z+d]) == 0`.
    #           - But we need to find the first such (x,y).
    #           - We can compute a 2D array `fits_xy` of shape (W-w+1, H-h+1) where `fits_xy[i,j]` is True if the box fits at (i,j,z).
    #           - `fits_xy[i,j] = np.all(grid[i:i+w, j:j+h, z:z+d] == 0)`
    #           - This is a 2D sliding window all-zero check.
    #           - We can use `np.min` or `np.sum` on the sliced grid.
    #           - Specifically, `sliding_sum = np.zeros((W-w+1, H-h+1))`
    #           - We can compute this using cumulative sums or just by slicing if the dimensions are small.
    #           - Given the small number of boxes (40) and likely small container dimensions, a simple loop over z and a vectorized check for x,y might be fast enough.
    
    # Let's estimate complexity:
    # 40 boxes * 10 bins (avg) * D (e.g., 50) * (vectorized check over W*H)
    # If W,H <= 50, W*H = 2500. 40 * 10 * 50 * 2500 = 5,000,000 operations. This is fine for NumPy.
    
    # Algorithm:
    # 1. Sort boxes by volume descending. Keep original indices.
    # 2. bins = []
    # 3. For each box in sorted order:
    #    placed = False
    #    for i, bin_data in enumerate(bins):
    #        # Try to place in bin i
    #        pos = find_best_position(bin_data.grid, w, h, d, W, H, D)
    #        if pos is not None:
    #            x, y, z = pos
    #            bin_data.grid[x:x+w, y:y+h, z:z+d] = 1
    #            bin_data.occupied_volume += w*h*d
    #            placements[orig_idx] = (i, x, y, z, w, h, d)
    #            placed = True
    #            break
    #    if not placed:
    #        # Create new bin
    #        bin_data = {'grid': np.zeros((W, H, D), dtype=np.uint8), 'occupied_volume': 0}
    #        bins.append(bin_data)
    #        x, y, z = 0, 0, 0 # Always place at origin in new bin
    #        bin_data.grid[x:x+w, y:y+h, z:z+d] = 1
    #        bin_data.occupied_volume += w*h*d
    #        placements[orig_idx] = (len(bins)-1, x, y, z, w, h, d)
    # 4. Return placements in original order.
    
    # Helper: find_best_position(grid, w, h, d, W, H, D)
    #   Returns (x, y, z) or None.
    #   Strategy: Iterate z from 0 to D-d.
    #   For each z, compute a 2D boolean array `fits` of shape (W-w+1, H-h+1).
    #   `fits[i,j]` is True if `np.all(grid[i:i+w, j:j+h, z:z+d] == 0)`.
    #   To compute this efficiently:
    #   Let `slice_z = grid[:, :, z:z+d]` (shape W, H, d)
    #   We need to check if any voxel in the (w,h,d) block is 1.
    #   We can sum along the z-axis: `sum_z = np.sum(slice_z, axis=2)` (shape W, H)
    #   Then we need to check if the sum in the (w,h) window is 0.
    #   This is a 2D sliding window sum.
    #   We can use `scipy.ndimage.uniform_filter` or manual convolution, or just simple slicing if w,h are small.
    #   Actually, since we only need to know if the sum is 0, we can just check `np.min` or `np.sum` of the window.
    #   But we need to find the FIRST (x,y) in row-major order (x then y? or y then x?).
    #   The problem doesn't specify a preference for x vs y, just "no overlap".
    #   However, for consistency and to avoid fragmentation, we should pick the one with smallest z, then smallest x, then smallest y?
    #   Let's pick smallest z, then smallest x, then smallest y.
    
    #   Efficient check for a single z:
    #   `sum_z = np.sum(grid[:, :, z:z+d], axis=2)` # shape (W, H)
    #   Now we need to find (i,j) such that `np.sum(sum_z[i:i+w, j:j+h]) == 0`.
    #   We can compute a 2D array `window_sums` of shape (W-w+1, H-h+1).
    #   `window_sums[i,j] = np.sum(sum_z[i:i+w, j:j+h])`
    #   We can compute this using cumulative sums or just by iterating if W,H are small.
    #   Given W,H <= 50, a simple double loop over i,j with a vectorized sum might be okay, but let's try to vectorize.
    #   Actually, `np.lib.stride_tricks.sliding_window_view` can create a view of the windows, then we can sum over the last two axes.
    
    #   Let's use `sliding_window_view` from `numpy.lib.stride_tricks`.
    #   `windows = np.lib.stride_tricks.sliding_window_view(sum_z, (w, h))` # shape (W-w+1, H-h+1, w, h)
    #   `window_sums = np.sum(windows, axis=(2,3))` # shape (W-w+1, H-h+1)
    #   `valid = window_sums == 0`
    #   If `np.any(valid)`:
    #       `i, j = np.argwhere(valid)[0]` # This gives the first True in row-major order (i then j).
    #       return (i, j, z)
    
    #   This is O(W*H*d) per z, and we do this for D z-values. Total O(W*H*D^2) per box per bin? No, O(W*H*d) per z, summed over D is O(W*H*D*d).
    #   If W,H,D,d are ~50, 50*50*50*50 = 6,250,000 operations per box per bin.
    #   40 boxes * 10 bins * 6,250,000 = 2.5 billion operations. This is too slow for Python/NumPy if not highly optimized.
    
    #   We need a faster way.
    #   Note that we break as soon as we find a valid z. So we don't always check all z.
    #   Also, we break as soon as we place the box in a bin. So we don't always check all bins.
    #   But worst case is still high.
    
    #   Optimization:
    #   Instead of checking every z, we can check if the box fits in the bin at all by checking total volume.
    #   Also, we can pre-compute the "free space" in a smarter way.
    
    #   Let's try a different approach: "Shelf" packing is O(N) and very fast.
    #   The current score is 1.0. We need to beat it.
    #   Maybe a simple improvement to the shelf heuristic is enough?
    #   Or a simple "First Fit Decreasing" with a simple placement rule (e.g., always place at (0,0,0) if possible, else increment x, else increment y, else increment z).
    
    #   Let's try "First Fit Decreasing" with the following placement rule:
    #   - For a given bin, try to place the box at the lowest possible z.
    #   - Within that z, try the lowest possible x.
    #   - Within that x, try the lowest possible y.
    #   - This is what the grid search does.
    
    #   To speed up the grid search:
    #   - We can limit the search space.
    #   - We can use a "bounding box" of the current contents to limit x,y,z search.
    
    #   Let's implement the grid search but with early termination and efficient NumPy ops.
    #   If it's too slow, we'll fall back to a simpler heuristic.
    
    #   Given the 120s limit, and 5 instances, we have ~24s per instance.
    #   40 boxes. If each box takes 0.1s, that's 4s. Should be fine.
    #   Let's assume the dimensions are small (<= 20-30). Then W*H*D is ~27,000.
    #   The sliding window sum is fast.
    
    #   Let's code it.
    
    # Sort boxes by volume descending
    box_volumes = [w*h*d for w,h,d in boxes]
    sorted_indices = np.argsort(box_volumes)[::-1]
    
    placements = [None] * n_boxes
    bins = [] # List of dicts: {'grid': np.array, 'vol': int}
    
    for orig_idx in sorted_indices:
        w, h, d = boxes[orig_idx]
        placed = False
        
        for bin_idx, bin_data in enumerate(bins):
            # Check if box fits in bin volume-wise
            if bin_data['vol'] + w*h*d > W*H*D:
                continue # Won't fit, skip to next bin (or break if we're doing best-fit? No, just skip)
            
            # Find best position
            grid = bin_data['grid']
            found_pos = None
            
            # Iterate z from 0 to D-d
            for z in range(D - d + 1):
                # Compute sum over z-slice
                # grid shape: (W, H, D)
                # slice: grid[:, :, z:z+d]
                slice_z = grid[:, :, z:z+d]
                sum_z = np.sum(slice_z, axis=2) # shape (W, H)
                
                # If total sum is 0, then any position in this z-slice is free? No, only if the whole slice is 0.
                # But we need a (w,h) window to be 0.
                
                # Use sliding window view
                if W >= w and H >= h:
                    # Create sliding window view
                    # Note: sliding_window_view might be slow if called repeatedly.
                    # Alternative: manual convolution or just check if min is 0?
                    # Let's try to find if there's any (i,j) where sum_z[i:i+w, j:j+h] is 0.
                    
                    # Efficient way:
                    # Compute 2D prefix sum of sum_z?
                    # Or just use np.min on the sliding windows?
                    
                    # Let's use a simple loop over x and y if W,H are small.
                    # But let's try vectorized first.
                    
                    # Create a boolean array of valid positions
                    # valid[i,j] = True if sum_z[i:i+w, j:j+h] has no 1s.
                    
                    # We can compute this by checking if the minimum of the window is 0? No, we need sum to be 0.
                    # Actually, if we just check `np.all(grid[i:i+w, j:j+h, z:z+d] == 0)`, it's the same.
                    
                    # Let's use the sliding window view on sum_z.
                    # windows = np.lib.stride_tricks.sliding_window_view(sum_z, (w, h))
                    # window_sums = np.sum(windows, axis=(2,3))
                    # valid = window_sums == 0
                    
                    # This creates a view, so it's fast.
                    try:
                        windows = np.lib.stride_tricks.sliding_window_view(sum_z, (w, h))
                        window_sums = np.sum(windows, axis=(2,3))
                        valid = window_sums == 0
                        if np.any(valid):
                            # Find the first True
                            i, j = np.argwhere(valid)[0]
                            found_pos = (i, j, z)
                            break
                    except:
                        pass # Fallback if dimensions don't match, though they should
            
            if found_pos:
                x, y, z = found_pos
                # Place box
                grid[x:x+w, y:y+h, z:z+d] = 1
                bin_data['vol'] += w*h*d
                placements[orig_idx] = (bin_idx, x, y, z, w, h, d)
                placed = True
                break
        
        if not placed:
            # Create new bin
            grid = np.zeros((W, H, D), dtype=np.uint8)
            grid[0:w, 0:h, 0:d] = 1
            bins.append({'grid': grid, 'vol': w*h*d})
            placements[orig_idx] = (len(bins)-1, 0, 0, 0, w, h, d)
    
    return placements

# EVOLVE-BLOCK-END
