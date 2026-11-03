import numpy as np
import time
from scipy.optimize import linear_sum_assignment
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_score = -np.inf
    best_circles = None

    def validate(circles, w, h, tol=1e-7):
        if w <= 0 or h <= 0:
            return False
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        if np.any(r < -tol):
            return False
            
        # Boundary check
        if np.any(x < r - tol) or np.any(x > w - r + tol) or np.any(y < r - tol) or np.any(y > h - r + tol):
            return False
            
        # Overlap check
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if dist < r[i] + r[j] - tol:
                    return False
        return True

    def compute_max_radii_for_grid(w, h, grid_nx, grid_ny):
        """
        For a given grid of potential positions, compute the maximum radius
        each circle can have at each position given all other positions are
        occupied. Uses a greedy approach with iterative refinement.
        """
        # Generate grid positions
        xs = np.linspace(0, w, grid_nx)
        ys = np.linspace(0, h, grid_ny)
        grid_x, grid_y = np.meshgrid(xs, ys, indexing='ij')
        grid_x = grid_x.flatten()
        grid_y = grid_y.flatten()
        n_positions = len(grid_x)
        
        # For each position, compute the minimum distance to all other positions
        # This gives an upper bound on radius if all positions were used
        pos = np.column_stack((grid_x, grid_y))
        
        # Compute pairwise distances
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
        
        # For each position, the max radius is half the min distance to any other position
        # But we need to account for boundary constraints too
        max_radius_at_pos = np.zeros(n_positions)
        for i in range(n_positions):
            # Boundary constraint
            r_boundary = min(grid_x[i], w - grid_x[i], grid_y[i], h - grid_y[i])
            
            # Overlap constraint with all other positions
            if n_positions > 1:
                min_dist_to_others = np.min(dist_matrix[i, np.arange(n_positions) != i])
                r_overlap = min_dist_to_others / 2.0
            else:
                r_overlap = r_boundary
            
            max_radius_at_pos[i] = min(r_boundary, r_overlap)
        
        return grid_x, grid_y, max_radius_at_pos, dist_matrix

    def solve_assignment(greedy_order, grid_x, grid_y, max_radius_at_pos, w, h, dist_matrix):
        """
        Assign circles to positions using a greedy + local improvement approach.
        Returns the assignment and the resulting radii.
        """
        n_positions = len(grid_x)
        
        # Start with greedy assignment based on radius potential
        assigned_positions = []
        assigned_radii = []
        used_positions = set()
        
        # Sort positions by max radius potential (descending)
        order = np.argsort(-max_radius_at_pos)
        
        for pos_idx in order:
            if len(assigned_positions) >= N:
                break
            
            # Check if this position is compatible with all already assigned positions
            compatible = True
            for j, prev_pos in enumerate(assigned_positions):
                d = dist_matrix[pos_idx, prev_pos]
                # Check if we can fit a circle of radius max_radius_at_pos[pos_idx]
                # without overlapping with the circle at prev_pos
                # The radius at prev_pos is assigned_radii[j]
                if d < max_radius_at_pos[pos_idx] + assigned_radii[j] - 1e-9:
                    compatible = False
                    break
            
            if compatible:
                assigned_positions.append(pos_idx)
                assigned_radii.append(max_radius_at_pos[pos_idx])
                used_positions.add(pos_idx)
        
        # If we didn't get N circles, try a different strategy:
        # Use a more careful greedy that considers the actual radii
        if len(assigned_positions) < N:
            assigned_positions = []
            assigned_radii = []
            used_positions = set()
            
            # Try positions in order of decreasing potential radius
            for pos_idx in order:
                if len(assigned_positions) >= N:
                    break
                
                # Try to place a circle at this position with the maximum possible radius
                # that doesn't overlap with any already placed circles
                max_r = max_radius_at_pos[pos_idx]
                
                # Check against all already placed circles
                feasible_r = max_r
                for j, prev_pos in enumerate(assigned_positions):
                    d = dist_matrix[pos_idx, prev_pos]
                    # To avoid overlap: r_new + r_prev <= d
                    # So r_new <= d - r_prev
                    feasible_r = min(feasible_r, d - assigned_radii[j])
                
                if feasible_r > 1e-9:
                    assigned_positions.append(pos_idx)
                    assigned_radii.append(feasible_r)
                    used_positions.add(pos_idx)
        
        if len(assigned_positions) < N:
            return None, None
        
        # Local improvement: try to increase radii
        # For each circle, try to increase its radius if possible
        for _ in range(100):
            if time.time() > deadline - 10:
                break
            
            improved = False
            for i, pos_idx in enumerate(assigned_positions):
                current_r = assigned_radii[i]
                
                # Try to increase radius
                new_r = current_r
                
                # Boundary constraint
                r_boundary = min(grid_x[pos_idx], w - grid_x[pos_idx], 
                                 grid_y[pos_idx], h - grid_y[pos_idx])
                new_r = min(new_r + 0.005, r_boundary)
                
                # Overlap constraint
                for j, prev_pos in enumerate(assigned_positions):
                    if i == j:
                        continue
                    d = dist_matrix[pos_idx, prev_pos]
                    # r_i + r_j <= d
                    max_r_from_overlap = d - assigned_radii[j]
                    new_r = min(new_r, max_r_from_overlap)
                
                if new_r > current_r + 1e-9:
                    assigned_radii[i] = new_r
                    improved = True
            
            if not improved:
                break
        
        # Create the circles array
        circles = np.zeros((N, 3))
        for i in range(N):
            circles[i, 0] = grid_x[assigned_positions[i]]
            circles[i, 1] = grid_y[assigned_positions[i]]
            circles[i, 2] = assigned_radii[i]
        
        return circles, np.sum(assigned_radii)

    def pack_for_aspect(w, h, grid_nx=20, grid_ny=20):
        """
        Pack circles for a given aspect ratio using grid-based assignment.
        """
        # Compute grid and max radii
        grid_x, grid_y, max_radius_at_pos, dist_matrix = compute_max_radii_for_grid(
            w, h, grid_nx, grid_ny
        )
        
        # Solve assignment
        circles, score = solve_assignment(None, grid_x, grid_y, max_radius_at_pos, w, h, dist_matrix)
        
        if circles is None:
            return None, -np.inf
        
        return circles, score

    # Try several aspect ratios with different grid resolutions
    aspect_ratios = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    grid_sizes = [15, 20, 25]
    
    for w in aspect_ratios:
        if time.time() > deadline - 15:
            break
        
        h = 2.0 - w
        
        for grid_size in grid_sizes:
            if time.time() > deadline - 10:
                break
            
            try:
                circles, score = pack_for_aspect(w, h, grid_size, grid_size)
                
                if circles is not None and score > 0:
                    # Validate
                    if validate(circles, w, h):
                        if score > best_score:
                            best_score = score
                            best_circles = circles
            except Exception:
                continue

    if best_circles is None:
        # Fallback: Simple grid in a square
        w = 1.0
        h = 1.0
        cells = []
        for i in range(5):
            for j in range(5):
                if len(cells) >= N:
                    break
                x = (i + 0.5) * 0.2
                y = (j + 0.5) * 0.2
                r = 0.08
                cells.append([x, y, r])
        best_circles = np.array(cells)
        
    return best_circles
