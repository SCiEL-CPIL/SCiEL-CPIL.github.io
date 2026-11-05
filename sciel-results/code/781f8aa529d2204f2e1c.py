import numpy as np
import scipy.ndimage as ndi

def load(boxes, container):
    W, H, D = container
    
    # Grid resolution
    # Aim for ~40 cells per dimension to balance speed and accuracy
    target_dim = 40
    step_x = max(1, int(np.ceil(W / target_dim)))
    step_y = max(1, int(np.ceil(H / target_dim)))
    step_z = max(1, int(np.ceil(D / target_dim)))
    
    nx = int(np.ceil(W / step_x))
    ny = int(np.ceil(H / step_y))
    nz = int(np.ceil(D / step_z))
    
    occ = np.zeros((nx, ny, nz), dtype=np.uint8)
    
    def compute_dist():
        # Distance to nearest occupied cell (0) or boundary.
        # We want distance from empty cells to the nearest obstacle.
        # Invert occ: 1 is empty, 0 is occupied.
        empty_mask = 1 - occ
        # Pad with 0s (obstacles) to handle boundaries
        padded = np.pad(empty_mask, pad_width=1, mode='constant', constant_values=0)
        # distance_transform_edt computes distance to the nearest 0 in the input?
        # No, it computes distance to the nearest 0 in the *foreground*?
        # Documentation: "Compute the exact Euclidean distance transform of an image."
        # "The distance is computed from each non-zero element to the nearest zero element."
        # So if we pass `empty_mask` (where 1 is empty, 0 is occupied),
        # it will compute distance from empty cells (1) to occupied cells (0).
        # This is what we want.
        dist = ndi.distance_transform_edt(padded)
        return dist[1:-1, 1:-1, 1:-1]
    
    dist = compute_dist()
    results = [None] * len(boxes)
    
    # Sort indices by volume descending
    box_indices = sorted(range(len(boxes)), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    for idx in box_indices:
        w, h, d = boxes[idx]
        best_pos = None
        best_orient = None
        best_score = -1
        
        orientations = [
            (w, h, d), (w, d, h), (h, w, d),
            (h, d, w), (d, w, h), (d, h, w)
        ]
        
        unique_orients = []
        seen = set()
        for o in orientations:
            if o not in seen:
                unique_orients.append(o)
                seen.add(o)
        
        for ow, oh, od in unique_orients:
            gw = int(np.ceil(ow / step_x))
            gh = int(np.ceil(oh / step_y))
            gd = int(np.ceil(od / step_z))
            
            if gw > nx or gh > ny or gd > nz:
                continue
            
            # Coarse search
            coarse_step = 2
            candidates = [] # (score, x, y, z)
            
            max_x = nx - gw
            max_y = ny - gh
            max_z = nz - gd
            
            for x in range(0, max_x + 1, coarse_step):
                for y in range(0, max_y + 1, coarse_step):
                    for z in range(0, max_z + 1, coarse_step):
                        # Check if fits (min dist > 0)
                        # Actually, if min dist is 0, it overlaps.
                        # We want to maximize min dist.
                        # If min dist is 0, it's invalid.
                        # So we only consider score > 0.
                        min_d = np.min(dist[x:x+gw, y:y+gh, z:z+gd])
                        if min_d > 0:
                            candidates.append((min_d, x, y, z))
            
            if not candidates:
                continue
            
            # Sort candidates by score descending
            candidates.sort(key=lambda item: item[0], reverse=True)
            
            # Take top K candidates for fine search
            top_k = candidates[:10]
            
            for score, cx, cy, cz in top_k:
                # Fine search in neighborhood
                # Range: cx +/- coarse_step
                x_start = max(0, cx - coarse_step)
                x_end = min(max_x, cx + coarse_step)
                y_start = max(0, cy - coarse_step)
                y_end = min(max_y, cy + coarse_step)
                z_start = max(0, cz - coarse_step)
                z_end = min(max_z, cz + coarse_step)
                
                for x in range(x_start, x_end + 1):
                    for y in range(y_start, y_end + 1):
                        for z in range(z_start, z_end + 1):
                            min_d = np.min(dist[x:x+gw, y:y+gh, z:z+gd])
                            if min_d > 0 and min_d > best_score:
                                best_score = min_d
                                best_pos = (x, y, z)
                                best_orient = (ow, oh, od)
        
        if best_pos is not None:
            bx, by, bz = best_pos
            ow, oh, od = best_orient
            
            gw = int(np.ceil(ow / step_x))
            gh = int(np.ceil(oh / step_y))
            gd = int(np.ceil(od / step_z))
            
            occ[bx:bx+gw, by:by+gh, bz:bz+gd] = 1
            dist = compute_dist()
            
            real_x = bx * step_x
            real_y = by * step_y
            real_z = bz * step_z
            
            results[idx] = (0, real_x, real_y, real_z, ow, oh, od)
        else:
            results[idx] = (-1, 0, 0, 0, w, h, d)
            
    return results
