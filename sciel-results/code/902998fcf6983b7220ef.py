# EVOLVE-BLOCK-START
import numpy as np
import time

def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for n=54 points and m=27 lines.
    Uses a grid-based construction with rows, columns, and diagonals.
    """
    start_time = time.time()
    deadline = 600  # Safety margin
    
    n_points = 54
    n_lines = 27
    
    best_I = -1
    best_points = None
    best_lines = None

    # Iterate over possible grid dimensions R x C
    # R is number of rows, C is number of columns
    # We use R horizontal lines and C vertical lines.
    # Remaining lines for diagonals: D = 27 - R - C
    
    for R in range(1, n_lines + 1):
        for C in range(1, n_lines - R + 1):
            if R * C < n_points:
                continue
            
            D = n_lines - R - C
            if D < 0:
                continue
            
            # Generate all possible diagonals in the R x C grid
            # Slope 1: y = x + k  => c - r = k
            # Slope -1: y = -x + k => c + r = k
            
            diags = []
            
            # Slope 1
            # k ranges from -(R-1) to C-1
            for k in range(-(R - 1), C):
                # Calculate length
                # r from max(0, -k) to min(R-1, C-1-k)
                r_start = max(0, -k)
                r_end = min(R - 1, C - 1 - k)
                if r_start <= r_end:
                    length = r_end - r_start + 1
                    diags.append((length, 1, k)) # (length, slope_type, k)
            
            # Slope -1
            # k ranges from 0 to R+C-2
            for k in range(R + C - 1):
                # c + r = k
                # r from max(0, k-(C-1)) to min(R-1, k)
                r_start = max(0, k - (C - 1))
                r_end = min(R - 1, k)
                if r_start <= r_end:
                    length = r_end - r_start + 1
                    diags.append((length, -1, k))
            
            # Sort by length descending, pick top D
            diags.sort(key=lambda x: x[0], reverse=True)
            selected_diags = diags[:D]
            
            # Build weight grid
            weights = np.zeros((R, C), dtype=np.int64)
            
            # Add rows
            weights += 1  # Every cell is in a row
            
            # Add cols
            weights += 1  # Every cell is in a col
            
            # Add diagonals
            for length, slope_type, k in selected_diags:
                if slope_type == 1:
                    # c - r = k => c = r + k
                    for r in range(R):
                        c = r + k
                        if 0 <= c < C:
                            weights[r, c] += 1
                else:
                    # c + r = k => c = k - r
                    for r in range(R):
                        c = k - r
                        if 0 <= c < C:
                            weights[r, c] += 1
            
            # Select top n_points cells
            # Flatten
            flat_weights = weights.flatten()
            # Get indices of top n_points
            # np.argpartition is faster for top-k
            if n_points < flat_weights.size:
                # Get indices of the largest n_points values
                # Note: argpartition doesn't guarantee order, but we just need the set
                top_indices = np.argpartition(flat_weights, -n_points)[-n_points:]
            else:
                top_indices = np.arange(flat_weights.size)
            
            # Calculate total incidences
            # Sum of weights of selected points
            I = np.sum(flat_weights[top_indices])
            
            if I > best_I:
                best_I = I
                # Map indices back to (r, c)
                # r = idx // C, c = idx % C
                idx_r = top_indices // C
                idx_c = top_indices % C
                # Points are (x, y). Let x = c, y = r? Or x=r, y=c?
                # Standard math: x horizontal, y vertical.
                # Grid: r is row index (y), c is col index (x).
                # So point is (c, r).
                points = np.stack([idx_c, idx_r], axis=1).astype(np.int64)
                
                # Construct lines
                lines_list = []
                
                # Rows: y = r => 0*x + 1*y - r = 0 => (0, 1, -r)
                for r in range(R):
                    lines_list.append((0, 1, -r))
                
                # Cols: x = c => 1*x + 0*y - c = 0 => (1, 0, -c)
                for c in range(C):
                    lines_list.append((1, 0, -c))
                
                # Diagonals
                for length, slope_type, k in selected_diags:
                    if slope_type == 1:
                        # y = x + k => x - y + k = 0 => (1, -1, k)
                        lines_list.append((1, -1, k))
                    else:
                        # y = -x + k => x + y - k = 0 => (1, 1, -k)
                        lines_list.append((1, 1, -k))
                
                lines = np.array(lines_list, dtype=np.int64)
                
                # Check time
                if time.time() > deadline - 10:
                    break
            
            if time.time() > deadline - 10:
                break
        
        if time.time() > deadline - 10:
            break

    if best_points is None:
        # Fallback to baseline
        k = 3
        xs = np.arange(1, k + 1)
        ys = np.arange(1, 2 * k * k + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        rows = []
        for a in range(1, k + 1):
            for b in range(1, k * k + 1):
                rows.append((a, -1, b))
        lines = np.array(rows, dtype=np.int64)
        return points, lines

    return best_points, best_lines

# EVOLVE-BLOCK-END
