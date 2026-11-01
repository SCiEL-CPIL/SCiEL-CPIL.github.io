<<<<<<< SEARCH
def physics_relaxation(init_centers, time_limit=60.0):
    """
    Physics-based relaxation using smooth repulsive potentials.
    Each hexagon is treated as a disk of effective radius ~1.0 (circumradius).
    We use a Lennard-Jones-like potential for repulsion and minimize
    the outer hexagon side length.
    """
    start_time = time.time()
    
    # Effective radius for repulsion: use circumradius = 1.0 for unit hexagons
    # But hexagons are not circles; use a slightly larger effective radius
    # to account for the flat sides. The minimum distance between centers
    # for non-overlapping hexagons depends on orientation.
    # Use r_eff = 1.0 as a starting point; the potential will push them apart.
    r_eff = 1.0
    
    # Parameters: x, y, angle for each hexagon
    params = np.zeros(3 * N)
    for i in range(N):
        params[3*i] = init_centers[i][0]
        params[3*i+1] = init_centers[i][1]
        params[3*i+2] = init_centers[i][2]
    
    def objective(p):
        if time.time() - start_time > time_limit:
            return 1e6
        
        # Extract centers
        xs = p[0::3]
        ys = p[1::3]
        angles = p[2::3]
        
        # Compute repulsive potential between all pairs
        # Use a smooth approximation: for distance d, if d < 2*r_eff, repel
        # Potential: sum over pairs of max(0, 2*r_eff - d)^2 * strength
        repulsion = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx*dx + dy*dy)
                # For hexagons, the minimum center distance depends on relative angle
                # Use a conservative estimate: 2*r_eff = 2.0
                # But we can be smarter: the actual minimum distance for two unit
                # hexagons at relative angle delta is between 2*apothem and 2*circumradius
                # apothem = sqrt(3)/2 ~ 0.866, circumradius = 1.0
                # So min distance is between 1.732 and 2.0
                # Use 1.8 as a compromise
                min_dist = 1.8
                if d < min_dist:
                    repulsion += (min_dist - d) ** 2
        
        # Outer side length
        centers = []
        for i in range(N):
            centers.append([xs[i], ys[i], angles[i]])
        req_side = required_outer_side(centers, (0.0, 0.0), 0.0)
        
        # Total objective: outer side + penalty for overlaps
        # Scale repulsion so it's significant
        return req_side + 10.0 * repulsion
    
    # Use L-BFGS-B with bounds
    # Bounds: positions in [-10, 10], angles in [0, 60]
    bounds = []
    for i in range(N):
        bounds.append((-10.0, 10.0))  # x
        bounds.append((-10.0, 10.0))  # y
        bounds.append((0.0, 60.0))    # angle
    
    result = minimize(
        objective, params, method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 2000, 'maxfun': 5000, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    new_centers = []
    for i in range(N):
        new_centers.append([result.x[3*i], result.x[3*i+1], result.x[3*i+2]])
    
    return new_centers
=======
def physics_relaxation(init_centers, time_limit=60.0):
    """
    Optimized relaxation using exact SAT-based repulsion penalty.
    """
    start_time = time.time()
    
    params = np.zeros(3 * N)
    for i in range(N):
        params[3*i] = init_centers[i][0]
        params[3*i+1] = init_centers[i][1]
        params[3*i+2] = init_centers[i][2]
    
    def objective(p):
        if time.time() - start_time > time_limit:
            return 1e6
        
        xs = p[0::3]
        ys = p[1::3]
        angles = p[2::3]
        
        # Compute repulsive potential using exact SAT separation
        # This is expensive, so we only calculate it if the distance is "close"
        # to avoid unnecessary SAT calls for far apart hexagons.
        repulsion = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d_sq = dx*dx + dy*dy
                # Max possible separation for non-overlap is 2.0 (circumradius*2)
                # If d > 2.0, they are definitely not overlapping (for any rotation? 
                # Actually, if d > 2.0, the circles of radius 1 don't touch, 
                # so the hexagons inside them don't touch).
                if d_sq > 4.0:
                    continue
                
                # Calculate exact min separation
                c1 = [xs[i], ys[i], angles[i]]
                c2 = [xs[j], ys[j], angles[j]]
                sep = sat_min_separation(c1, c2)
                
                # If sep < 0, they overlap. Penalty is proportional to overlap depth squared.
                if sep < 0:
                    repulsion += (sep * sep)
        
        # Outer side length
        centers = []
        for i in range(N):
            centers.append([xs[i], ys[i], angles[i]])
        
        # Fast approximation for outer side to guide gradient, 
        # but we must return the true value for the final check.
        # However, L-BFGS needs a smooth function. 
        # required_outer_side is piecewise linear (max of linear functions), so it's convex.
        req_side = required_outer_side(centers, (0.0, 0.0), 0.0)
        
        # Penalty weight: high enough to prevent overlaps
        return req_side + 50.0 * repulsion
    
    bounds = []
    for i in range(N):
        bounds.append((-10.0, 10.0))
        bounds.append((-10.0, 10.0))
        bounds.append((0.0, 60.0))
    
    result = minimize(
        objective, params, method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    new_centers = []
    for i in range(N):
        new_centers.append([result.x[3*i], result.x[3*i+1], result.x[3*i+2]])
    
    return new_centers
>>>>>>> REPLACE

