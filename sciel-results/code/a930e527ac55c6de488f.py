import numpy as np
import itertools

def szemeredi_trotter_incidences():
    best_points = None
    best_lines = None
    best_I = -1
    
    # Search over grid dimensions
    # W*H should be close to 54
    for W in range(5, 21):
        for H in range(5, 21):
            total_pts = W * H
            if total_pts < 54:
                continue
            if total_pts > 70: # Limit search space
                continue
            
            # Generate points
            xs = np.arange(1, W + 1)
            ys = np.arange(1, H + 1)
            px, py = np.meshgrid(xs, ys, indexing="ij")
            all_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
            
            # Generate candidate lines
            lines = []
            
            # Vertical
            for c in range(1, W + 1):
                lines.append((1, 0, -c))
            # Horizontal
            for c in range(1, H + 1):
                lines.append((0, 1, -c))
            # Diagonal 1 (x - y = c)
            min_c = 1 - H
            max_c = W - 1
            for c in range(min_c, max_c + 1):
                lines.append((1, -1, -c))
            # Diagonal -1 (x + y = c)
            min_c = 2
            max_c = W + H
            for c in range(min_c, max_c + 1):
                lines.append((1, 1, -c))
            
            lines = np.array(lines, dtype=np.int64)
            
            # Calculate incidences for each line
            # Incidence[i, j] is 1 if point i is on line j
            # A*x + B*y + C = 0
            # points: (N, 2), lines: (M, 3)
            # N = total_pts, M = len(lines)
            
            # Efficient computation
            # For each line, count points
            # We can do this by broadcasting or loops.
            # Given small sizes, loops are fine.
            
            line_counts = np.zeros(len(lines), dtype=np.int64)
            # incidence matrix for point removal logic
            # We need to know which points are on which lines for removal
            # But N*M might be large. 70*100 = 7000. Fine.
            
            incidences = np.zeros((total_pts, len(lines)), dtype=np.int8)
            
            for j in range(len(lines)):
                A, B, C = lines[j]
                # A*x + B*y + C == 0
                # x = all_points[:, 0], y = all_points[:, 1]
                val = A * all_points[:, 0] + B * all_points[:, 1] + C
                mask = (val == 0)
                incidences[:, j] = mask
                line_counts[j] = np.sum(mask)
            
            # Select top 27 lines
            if len(lines) < 27:
                continue
                
            # Sort lines by count desc
            sorted_indices = np.argsort(-line_counts)
            top_27_indices = sorted_indices[:27]
            
            # Get the top 27 lines
            selected_lines = lines[top_27_indices]
            
            # Now we need to remove K = total_pts - 54 points
            K = total_pts - 54
            
            # We want to remove points that minimize the loss of incidences.
            # Loss of removing point i = sum of incidences[i, selected_lines]
            # But removing a point doesn't change the lines.
            # So we just pick the K points with the lowest degree in the subgraph of selected lines.
            
            # However, if we remove a point, we lose 1 incidence for each line it's on.
            # So yes, just pick K points with lowest sum of incidences over the 27 lines.
            
            point_degrees = np.sum(incidences[:, top_27_indices], axis=1)
            
            # Sort points by degree asc
            point_order = np.argsort(point_degrees)
            points_to_remove = point_order[:K]
            
            # Create mask for points to keep
            keep_mask = np.ones(total_pts, dtype=bool)
            keep_mask[points_to_remove] = False
            
            final_points = all_points[keep_mask]
            
            # Calculate final I
            # Sum of incidences for kept points on selected lines
            # final_I = sum( incidences[keep_mask, top_27_indices] )
            final_I = np.sum(incidences[keep_mask, top_27_indices])
            
            if final_I > best_I:
                best_I = final_I
                best_points = final_points.copy()
                best_lines = selected_lines.copy()
                
    # If no grid found (should not happen), fallback to seed
    if best_points is None:
        # Fallback to original
        xs = np.arange(1, 9)
        ys = np.arange(1, 8)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        all_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        mask = np.ones(56, dtype=bool)
        mask[3] = False
        mask[52] = False
        points = all_points[mask]
        lines = []
        for c in range(1, 9):
            lines.append((1, 0, -c))
        for c in range(1, 8):
            lines.append((0, 1, -c))
        for c in [0, 1, -1, 2, -2, 3]:
            lines.append((1, -1, -c))
        for c in [8, 9, 7, 10, 6, 11]:
            lines.append((1, 1, -c))
        lines = np.array(lines, dtype=np.int64)
        return points, lines

    return best_points, best_lines
