<<<<<<< SEARCH
def _lloyd_optimize(pts, grid, time_limit, max_iter=2000):
    """
    Optimize points using Lloyd's algorithm (Voronoi relaxation).
    """
    start_time = time.time()
    current_pts = pts.copy()
    best_pts = pts.copy()
    best_radius = _compute_covering_radius(pts, grid)
    
    alpha = 1.0
    no_improve = 0
    max_no_improve = 50
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Perform one Lloyd iteration
        new_pts = _lloyd_iteration(current_pts, grid, alpha=alpha)
        
        # Check for improvement
        new_radius = _compute_covering_radius(new_pts, grid)
        
        if new_radius < best_radius - 1e-8:
            best_radius = new_radius
            best_pts = new_pts.copy()
            no_improve = 0
            # Increase alpha slightly to speed up convergence
            alpha = min(alpha * 1.02, 1.0)
        elif new_radius < current_radius if 'current_radius' in dir() else False:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                # Try smaller alpha
                alpha *= 0.5
                if alpha < 0.01:
                    break
                no_improve = 0
        
        current_pts = new_pts
        current_radius = new_radius
    
    return best_pts, best_radius
=======
def _polish_largest_facets(pts, grid, time_limit, n_polish=50):
    """
    Polish the configuration by moving points that are centers of large
    Voronoi cells toward the circumcenters of their largest facets.
    """
    start_time = time.time()
    current_pts = pts.copy()
    best_pts = pts.copy()
    best_radius = _compute_covering_radius(pts, grid)
    
    for polish_iter in range(n_polish):
        if time.time() - start_time > time_limit:
            break
        
        # Find the point with the largest Voronoi cell (using grid)
        dots = grid @ current_pts.T
        nearest = np.argmax(dots, axis=1)
        
        # Count grid points per cell
        cell_sizes = np.bincount(nearest, minlength=len(current_pts))
        
        # Find the largest cell
        max_cell_idx = np.argmax(cell_sizes)
        
        if cell_sizes[max_cell_idx] == 0:
            break
        
        # Get the cell points
        mask = (nearest == max_cell_idx)
        cell_pts = grid[mask]
        
        # Compute the centroid of the largest cell
        vec_sum = np.sum(cell_pts, axis=0)
        norm = np.linalg.norm(vec_sum)
        if norm < 1e-10:
            break
        centroid = vec_sum / norm
        
        # Move the point toward the centroid with a small step
        step_size = 0.05
        new_pt = current_pts[max_cell_idx] + step_size * (centroid - current_pts[max_cell_idx])
        new_pt = new_pt / np.linalg.norm(new_pt)
        
        # Try this move
        trial_pts = current_pts.copy()
        trial_pts[max_cell_idx] = new_pt
        
        trial_radius = _compute_covering_radius(trial_pts, grid)
        
        if trial_radius < best_radius - 1e-10:
            best_radius = trial_radius
            best_pts = trial_pts.copy()
            current_pts = trial_pts
    
    return best_pts, best_radius
>>>>>>> REPLACE

