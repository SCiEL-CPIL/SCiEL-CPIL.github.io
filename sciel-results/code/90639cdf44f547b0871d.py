"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def pack(boxes, container):
    """
    Place boxes into bins using vectorized NumPy collision checks.
    Returns list of (bin_id, x, y, z, w, h, d) for each box in original order.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    deadline = time.time() + 115.0  # safety margin before 120s kill
    container_vol = W * H * D

    # Sort boxes by volume descending for BFD, but keep original indices
    indexed_boxes = [(i, w * h * d, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1], reverse=True)

    # Each bin: [placements_list, occupancy_grid]
    bins = []

    def make_grid():
        return np.zeros((W, H, D), dtype=np.bool_)

    def place_in_grid(grid, x, y, z, w, h, d):
        grid[x:x+w, y:y+h, z:z+d] = True

    def find_position_vectorized(bin_data, w, h, d):
        """
        Find the best position using vectorized candidate evaluation.
        Generates candidates from box corners and evaluates them in batch.
        Returns (x, y, z) or None.
        """
        placements, grid = bin_data
        
        # Generate candidate positions from corners of existing boxes
        cand_list = [(0, 0, 0)]
        for (px, py, pz, pw, ph, pd) in placements:
            cand_list.append((px + pw, py, pz))
            cand_list.append((px, py + ph, pz))
            cand_list.append((px, py, pz + pd))
            cand_list.append((px + pw, py + ph, pz))
            cand_list.append((px + pw, py, pz + pd))
            cand_list.append((px, py + ph, pz + pd))
            cand_list.append((px + pw, py + ph, pz + pd))
        
        # Deduplicate and filter by bounds
        seen = set()
        valid_cands = []
        for (xc, yc, zc) in cand_list:
            if xc in seen:
                continue
            if xc + w > W or yc + h > H or zc + d > D:
                continue
            seen.add((xc, yc, zc))
            valid_cands.append((xc, yc, zc))
        
        if not valid_cands:
            return None
        
        # Vectorized collision check: for each candidate, check if sub-cube is empty
        # Use a sliding window approach with cumulative sums for efficiency
        # But for simplicity, check each candidate individually with NumPy slicing
        
        best_pos = None
        best_score = float('inf')
        
        for (xc, yc, zc) in valid_cands:
            if not np.any(grid[xc:xc+w, yc:yc+h, zc:zc+d]):
                score = xc * 10000 + yc * 100 + zc
                if score < best_score:
                    best_score = score
                    best_pos = (xc, yc, zc)
        
        return best_pos

    def find_position_exhaustive(bin_data, w, h, d):
        """
        Exhaustive search over all valid positions using vectorized checks.
        Used when corner-based approach fails.
        """
        placements, grid = bin_data
        
        # For small containers, do exhaustive search
        # Create a 3D array of all possible positions
        # Position (x,y,z) is valid if x+w<=W, y+h<=H, z+d<=D and sub-cube is empty
        
        # Use integral image (3D prefix sum) for O(1) sub-cube sum queries
        # grid is boolean, so sum gives count of occupied voxels in sub-cube
        
        # Build 3D prefix sum
        # grid shape: (W, H, D)
        # prefix_sum[i,j,k] = sum of grid[0:i, 0:j, 0:k]
        # To get sum of grid[x1:x2, y1:y2, z1:z2]:
        # prefix[x2,y2,z2] - prefix[x1,y2,z2] - prefix[x2,y1,z2] - prefix[x2,y2,z1]
        # + prefix[x1,y1,z2] + prefix[x1,y2,z1] + prefix[x2,y1,z1] - prefix[x1,y1,z1]
        
        # Build prefix sum (add 1 to make it 1-indexed for easier boundary handling)
        grid_int = grid.astype(np.int32)
        # Pad with zeros on the lower-left-back
        padded = np.zeros((W+1, H+1, D+1), dtype=np.int32)
        padded[1:, 1:, 1:] = grid_int
        # Compute 3D prefix sum
        prefix = np.cumsum(np.cumsum(np.cumsum(padded, axis=0), axis=1), axis=2)
        
        # Now for each possible (x, y, z), check if sub-cube [x:x+w, y:y+h, z:z+d] is empty
        # x ranges from 0 to W-w, y from 0 to H-h, z from 0 to D-d
        
        # Vectorized: create arrays of all positions
        xs = np.arange(W - w + 1)
        ys = np.arange(H - h + 1)
        zs = np.arange(D - d + 1)
        
        # For each (x,y,z), the sub-cube sum is:
        # prefix[x+w, y+h, z+d] - prefix[x, y+h, z+d] - prefix[x+w, y, z+d] - prefix[x+w, y+h, z]
        # + prefix[x, y, z+d] + prefix[x, y+h, z] + prefix[x+w, y, z] - prefix[x, y, z]
        
        # This is expensive to do fully vectorized for large containers.
        # Instead, iterate over z (typically smallest dimension) and vectorize over x,y
        
        best_pos = None
        best_score = float('inf')
        
        for zc in zs:
            # For this z, compute sub-cube sums for all (x,y)
            # sub[x,y] = prefix[x+w, y+h, zc+d] - prefix[x, y+h, zc+d] - prefix[x+w, y, zc+d] - prefix[x+w, y+h, zc]
            #            + prefix[x, y, zc+d] + prefix[x, y+h, zc] + prefix[x+w, y, zc] - prefix[x, y, zc]
            
            # Extract slices
            a = prefix[w:, h:, zc+d]  # shape (W-w+1, H-h+1)
            b = prefix[:W-w+1, h:, zc+d]
            c = prefix[w:, :H-h+1, zc+d]
            d_slice = prefix[w:, h:, zc]
            e = prefix[:W-w+1, :H-h+1, zc+d]
            f = prefix[:W-w+1, h:, zc]
            g = prefix[w:, :H-h+1, zc]
            h_slice = prefix[:W-w+1, :H-h+1, zc]
            
            sub_sums = a - b - c - d_slice + e + f + g - h_slice
            
            # Find positions where sub_sums == 0
            valid_mask = (sub_sums == 0)
            if np.any(valid_mask):
                # Get all valid (x, y) for this z
                valid_xs, valid_ys = np.where(valid_mask)
                # Score: minimize x + y (compactness)
                scores = valid_xs + valid_ys
                min_idx = np.argmin(scores)
                x_best = valid_xs[min_idx]
                y_best = valid_ys[min_idx]
                score = x_best * 10000 + y_best * 100 + zc
                if score < best_score:
                    best_score = score
                    best_pos = (int(x_best), int(y_best), int(zc))
        
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
            # First try corner-based (fast)
            pos = find_position_vectorized(bin_data, w, h, d)
            
            # If corner-based fails, try exhaustive (slower but more thorough)
            if pos is None:
                pos = find_position_exhaustive(bin_data, w, h, d)
            
            if pos is not None:
                x, y, z = pos
                # Fit score: prefer positions that minimize max extent
                fit_score = (max(x + w, y + h, z + d), x + w + y + h + z + d)
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
            pos = find_position_vectorized(bin_data, w, h, d)
            if pos is None:
                pos = find_position_exhaustive(bin_data, w, h, d)
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
