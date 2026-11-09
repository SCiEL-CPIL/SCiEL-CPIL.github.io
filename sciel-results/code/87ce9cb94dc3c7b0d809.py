"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def pack(boxes, container):
    """
    Pack boxes into bins using a 3D grid-based occupancy map approach with
    best-fit decreasing strategy. Boxes are sorted by volume (descending) for
    placement to improve packing density, then placed back in original order
    for the output. Uses vectorized prefix sums to find all valid positions
    and selects the one that minimizes x+y+z (packs towards origin).
    """
    W, H, D = container
    n_boxes = len(boxes)
    
    # Sort boxes by volume descending for better packing, but remember original indices
    box_volumes = [w * h * d for w, h, d in boxes]
    sorted_indices = sorted(range(n_boxes), key=lambda i: box_volumes[i], reverse=True)
    
    bins = []  # List of numpy arrays, each of shape (W, H, D)
    placements = [None] * n_boxes
    
    def get_bin(i):
        while len(bins) <= i:
            bins.append(np.zeros((W, H, D), dtype=bool))
        return bins[i]
    
    # Process boxes in sorted order (largest first) for better packing
    for orig_idx in sorted_indices:
        w, h, d = boxes[orig_idx]
        
        best_bin_idx = None
        best_pos = None
        
        # Check existing bins
        for b_idx in range(len(bins)):
            occ = bins[b_idx]
            
            if w > W or h > H or d > D:
                continue
                
            # Compute prefix sum for fast range sum queries
            P = np.zeros((W + 1, H + 1, D + 1), dtype=np.int32)
            P[1:, 1:, 1:] = occ.astype(np.int32)
            P = P.cumsum(axis=2).cumsum(axis=1).cumsum(axis=0)
            
            x_max = W - w
            y_max = H - h
            z_max = D - d
            
            if x_max < 0 or y_max < 0 or z_max < 0:
                continue
            
            # Compute sum of occupied cells for all possible placements
            # Using inclusion-exclusion with prefix sums
            t1 = P[w:w + x_max + 1, h:h + y_max + 1, d:d + z_max + 1]
            t2 = P[0:x_max + 1, h:h + y_max + 1, d:d + z_max + 1]
            t3 = P[w:w + x_max + 1, 0:y_max + 1, d:d + z_max + 1]
            t4 = P[0:x_max + 1, 0:y_max + 1, d:d + z_max + 1]
            t5 = P[w:w + x_max + 1, h:h + y_max + 1, 0:z_max + 1]
            t6 = P[0:x_max + 1, h:h + y_max + 1, 0:z_max + 1]
            t7 = P[w:w + x_max + 1, 0:y_max + 1, 0:z_max + 1]
            t8 = P[0:x_max + 1, 0:y_max + 1, 0:z_max + 1]
            
            sums = t1 - t2 - t3 + t4 - t5 + t6 + t7 - t8
            valid_mask = (sums == 0)
            
            if np.any(valid_mask):
                # Find position minimizing x + y + z (pack towards origin)
                valid_coords = np.argwhere(valid_mask)
                scores = valid_coords[:, 0] + valid_coords[:, 1] + valid_coords[:, 2]
                min_idx = np.argmin(scores)
                best_bin_idx = b_idx
                best_pos = valid_coords[min_idx]
                break
        
        # If no existing bin works, create new bin
        if best_bin_idx is None:
            best_bin_idx = len(bins)
            get_bin(best_bin_idx)
            best_pos = np.array([0, 0, 0])
        
        x, y, z = best_pos
        bins[best_bin_idx][x:x + w, y:y + h, z:z + d] = True
        placements[orig_idx] = (best_bin_idx, int(x), int(y), int(z), w, h, d)
    
    return placements
# EVOLVE-BLOCK-END
