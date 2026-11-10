<<<<<<< SEARCH
def _local_optimize(pts, max_time=30.0):
    """Local optimization to reduce covering radius."""
    start_time = time.time()
    pts = _normalize(pts.copy())
    
    best_cr = _covering_radius(pts)
    best_pts = pts.copy()
    
    step_size = 0.05
    n_iter = 0
    
    while time.time() - start_time < max_time:
        n_iter += 1
        improved = False
        
        for i in range(N):
            if time.time() - start_time > max_time - 5.0:
                break
            
            # Try moving point i in random direction
            for trial in range(10):
                direction = np.random.randn(3)
                norm_dir = np.linalg.norm(direction)
                if norm_dir < 1e-10:
                    continue
                direction /= norm_dir
                
                new_pts = pts.copy()
                new_pt = new_pts[i] + step_size * direction
                new_pts[i] = _normalize(new_pt)
                
                cr = _covering_radius(new_pts)
                if cr < best_cr - 1e-10:
                    best_cr = cr
                    best_pts = new_pts.copy()
                    pts = new_pts.copy()
                    improved = True
                    break
        
        if not improved:
            # Reduce step size
            step_size *= 0.8
            if step_size < 1e-4:
                break
    
    return best_pts, best_cr
=======
def _voronoi_centroid(pts, i):
    """Compute the centroid of the Voronoi cell for point i using nearest-neighbor regions."""
    # For each point, find the region where it's the nearest
    # Approximate by sampling and averaging the samples that are nearest to point i
    n_samples = 5000
    j = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * j / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * j
    r = np.sqrt(1.0 - z * z)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    dots = samples @ pts.T
    nearest = np.argmax(dots, axis=1)
    
    # Get samples where point i is the nearest
    mask = (nearest == i)
    if np.sum(mask) == 0:
        return pts[i]
    
    centroid = np.mean(samples[mask], axis=0)
    norm = np.linalg.norm(centroid)
    if norm < 1e-10:
        return pts[i]
    return centroid / norm


def _local_optimize(pts, max_time=30.0):
    """Local optimization to reduce covering radius using Voronoi centroid guidance."""
    start_time = time.time()
    pts = _normalize(pts.copy())
    
    best_cr = _covering_radius(pts)
    best_pts = pts.copy()
    
    step_size = 0.05
    n_iter = 0
    
    while time.time() - start_time < max_time:
        n_iter += 1
        improved = False
        
        for i in range(N):
            if time.time() - start_time > max_time - 5.0:
                break
            
            # Compute Voronoi centroid direction
            centroid = _voronoi_centroid(pts, i)
            centroid_dir = centroid - pts[i]
            centroid_norm = np.linalg.norm(centroid_dir)
            
            # Try moving toward centroid and in random directions
            directions = []
            if centroid_norm > 1e-8:
                directions.append(centroid_dir / centroid_norm)
            
            for _ in range(5):
                directions.append(np.random.randn(3))
            
            for direction in directions:
                if time.time() - start_time > max_time - 5.0:
                    break
                norm_dir = np.linalg.norm(direction)
                if norm_dir < 1e-10:
                    continue
                direction /= norm_dir
                
                new_pts = pts.copy()
                new_pt = new_pts[i] + step_size * direction
                new_pts[i] = _normalize(new_pt)
                
                cr = _covering_radius(new_pts)
                if cr < best_cr - 1e-10:
                    best_cr = cr
                    best_pts = new_pts.copy()
                    pts = new_pts.copy()
                    improved = True
                    break
        
        if not improved:
            # Reduce step size
            step_size *= 0.8
            if step_size < 1e-4:
                break
    
    return best_pts, best_cr
>>>>>>> REPLACE

