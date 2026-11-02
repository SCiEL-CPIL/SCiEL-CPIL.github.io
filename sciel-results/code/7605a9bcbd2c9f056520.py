"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Pack 26 circles in the unit square, maximizing the sum of radii.
    Uses a physics-inspired relaxation (Position Based Dynamics / gradient ascent)
    with a time limit guard.
    """
    start_time = time.time()
    deadline = 120.0  # seconds, with margin
    safety_margin = 5.0

    # Initial positions: hexagonal grid packing
    # 26 circles -> roughly 5x5 grid (25) + 1 extra, or 4x7 (28) -> use 5x5 + 1
    # Let's use a hexagonal arrangement
    # For 26 circles, a good initial guess is a hexagonal grid.
    # Rows: 5 rows of 5, plus 1? Or 4 rows of 7?
    # Let's try 5 rows: 5,5,5,5,5,1? No, let's do 5x5 + 1 in the middle.
    # Actually, a hexagonal grid with 5x5 is 25 circles. Add 1 in the center gap.
    
    # Better: use a simple grid first, then relax.
    # 5x5 grid = 25 circles. Add 1 more.
    # Let's use 4 rows of 7 = 28, take first 26.
    # Or 5 rows of 5 + 1 = 26.
    
    # Let's use a hexagonal initial layout for better packing.
    # Row 0: 5 circles
    # Row 1: 5 circles (offset)
    # Row 2: 5 circles
    # Row 3: 5 circles (offset)
    # Row 4: 5 circles
    # Row 5: 1 circle (center)
    # Total: 26
    
    # Actually, let's just use a uniform grid and let the optimizer do the work.
    # 5x5 grid = 25, plus 1 in the center.
    
    n = N
    # Initial positions
    # Use a 5x5 grid for 25 circles, plus 1 in the center
    side = 5
    positions = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            positions.append([x, y])
    
    # Add the 26th circle in the center
    positions.append([0.5, 0.5])
    
    positions = np.array(positions, dtype=np.float64)
    radii = np.ones(n, dtype=np.float64) * 0.08  # initial radius
    
    # Optimization: gradient ascent on sum of radii
    # Constraints:
    # 1. Circle i must be inside [0,1]^2: r_i <= min(x_i, y_i, 1-x_i, 1-y_i)
    # 2. Circle i and j must not overlap: r_i + r_j <= ||p_i - p_j||
    
    # We'll use a simple iterative relaxation:
    # - For each pair, if overlapping, push them apart and/or reduce radii
    # - For each circle, if outside boundary, pull it in and/or reduce radius
    
    # Actually, let's use a more sophisticated approach:
    # Maximize sum(r_i) subject to constraints
    # Use Lagrangian or simple penalty method
    
    # Let's use a simple iterative method:
    # 1. Start with equal radii
    # 2. For each iteration:
    #    a. For each pair (i,j), compute distance d_ij
    #    b. If r_i + r_j > d_ij, they overlap. Push them apart and/or reduce radii
    #    c. For each circle, if outside, pull in and/or reduce radius
    #    d. Adjust radii to be as large as possible given constraints
    
    # Better approach: use scipy.optimize
    from scipy.optimize import minimize
    
    def objective(params):
        """Negative sum of radii (we minimize negative to maximize)."""
        x, y = params[:n], params[n:2*n]
        r = params[2*n:3*n]
        return -np.sum(r)
    
    def constraints(params):
        """Return list of constraint functions (each >= 0)."""
        x, y = params[:n], params[n:2*n]
        r = params[2*n:3*n]
        
        cons = []
        
        # Boundary constraints: r_i <= min(x_i, y_i, 1-x_i, 1-y_i)
        for i in range(n):
            cons.append(lambda p, i=i: p[i] - p[2*n+i])  # x_i - r_i >= 0
            cons.append(lambda p, i=i: p[n+i] - p[2*n+i])  # y_i - r_i >= 0
            cons.append(lambda p, i=i: 1 - p[i] - p[2*n+i])  # 1-x_i - r_i >= 0
            cons.append(lambda p, i=i: 1 - p[n+i] - p[2*n+i])  # 1-y_i - r_i >= 0
        
        # Pairwise non-overlap: r_i + r_j <= ||p_i - p_j||
        for i in range(n):
            for j in range(i+1, n):
                cons.append(lambda p, i=i, j=j: np.sqrt((p[i]-p[j])**2 + (p[n+i]-p[n+j])**2) - p[2*n+i] - p[2*n+j])
        
        # Radius positivity
        for i in range(n):
            cons.append(lambda p, i=i: p[2*n+i])
        
        return cons
    
    # Initial parameters
    x0 = np.zeros(3*n)
    x0[:n] = positions[:, 0]
    x0[n:2*n] = positions[:, 1]
    x0[2*n:3*n] = radii
    
    # Bounds: 0 <= x,y <= 1, 0 <= r <= 0.5
    bounds = []
    for i in range(n):
        bounds.append((0, 1))  # x_i
        bounds.append((0, 1))  # y_i
        bounds.append((0, 0.5))  # r_i
    
    # This is a large optimization problem (78 variables, many constraints).
    # scipy.optimize.minimize with SLSQP might be slow.
    
    # Let's use a simpler iterative relaxation instead.
    
    # Iterative relaxation:
    # Repeat:
    #   For each pair (i,j):
    #     d = distance(p_i, p_j)
    #     if r_i + r_j > d:
    #       overlap = r_i + r_j - d
    #       # Reduce radii proportionally to mass (r^2)
    #       mass_i = r_i ** 2
    #       mass_j = r_j ** 2
    #       total_mass = mass_i + mass_j
    #       reduce_i = overlap * mass_j / total_mass
    #       reduce_j = overlap * mass_i / total_mass
    #       r_i -= reduce_i
    #       r_j -= reduce_j
    #       # Also push apart
    #       if d > 1e-9:
    #         dx = (positions[j, 0] - positions[i, 0]) / d
    #         dy = (positions[j, 1] - positions[i, 1]) / d
    #         push = overlap / 2
    #         positions[i, 0] -= dx * push
    #         positions[i, 1] -= dy * push
    #         positions[j, 0] += dx * push
    #         positions[j, 1] += dy * push
    #   For each circle i:
    #     # Check boundary
    #     min_dist = min(positions[i, 0], positions[i, 1], 1-positions[i, 0], 1-positions[i, 1])
    #     if r_i > min_dist:
    #       r_i = min_dist
    #     # Push inside if needed
    #     if positions[i, 0] < r_i:
    #       positions[i, 0] = r_i
    #     if positions[i, 1] < r_i:
    #       positions[i, 1] = r_i
    #     if positions[i, 0] > 1 - r_i:
    #       positions[i, 0] = 1 - r_i
    #     if positions[i, 1] > 1 - r_i:
    #       positions[i, 1] = 1 - r_i
    #   # Try to increase radii
    #   for i in range(n):
    #     # Max radius given current position and neighbors
    #     max_r = min(positions[i, 0], positions[i, 1], 1-positions[i, 0], 1-positions[i, 1])
    #     for j in range(n):
    #       if i != j:
    #         d = np.sqrt((positions[i,0]-positions[j,0])**2 + (positions[i,1]-positions[j,1])**2)
    #         max_r = min(max_r, d - radii[j])
    #     radii[i] = max(max_r, 0)
    
    # Let's implement this properly.
    
    def relax_iteration(positions, radii):
        """One iteration of relaxation."""
        n = len(radii)
        # Pairwise constraints
        for i in range(n):
            for j in range(i+1, n):
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                d = np.sqrt(dx*dx + dy*dy)
                if d < 1e-9:
                    # Random small push
                    positions[i, 0] += np.random.uniform(-0.01, 0.01)
                    positions[i, 1] += np.random.uniform(-0.01, 0.01)
                    positions[j, 0] += np.random.uniform(-0.01, 0.01)
                    positions[j, 1] += np.random.uniform(-0.01, 0.01)
                    continue
                overlap = radii[i] + radii[j] - d
                if overlap > 0:
                    # Push apart
                    dx_unit = dx / d
                    dy_unit = dy / d
                    push = overlap / 2
                    positions[i, 0] -= dx_unit * push
                    positions[i, 1] -= dy_unit * push
                    positions[j, 0] += dx_unit * push
                    positions[j, 1] += dy_unit * push
                    # Reduce radii
                    mass_i = radii[i] ** 2
                    mass_j = radii[j] ** 2
                    total_mass = mass_i + mass_j
                    if total_mass > 0:
                        reduce_i = overlap * mass_j / total_mass * 0.5
                        reduce_j = overlap * mass_i / total_mass * 0.5
                        radii[i] = max(radii[i] - reduce_i, 0)
                        radii[j] = max(radii[j] - reduce_j, 0)
        
        # Boundary constraints
        for i in range(n):
            # Clamp position inside
            positions[i, 0] = np.clip(positions[i, 0], 0, 1)
            positions[i, 1] = np.clip(positions[i, 1], 0, 1)
            # Max radius from boundary
            max_r = min(positions[i, 0], positions[i, 1], 1-positions[i, 0], 1-positions[i, 1])
            radii[i] = min(radii[i], max_r)
            # If radius is too big, push inside
            if radii[i] > positions[i, 0]:
                positions[i, 0] = radii[i]
            if radii[i] > positions[i, 1]:
                positions[i, 1] = radii[i]
            if radii[i] > 1 - positions[i, 0]:
                positions[i, 0] = 1 - radii[i]
            if radii[i] > 1 - positions[i, 1]:
                positions[i, 1] = 1 - radii[i]
        
        # Try to increase radii
        for i in range(n):
            max_r = min(positions[i, 0], positions[i, 1], 1-positions[i, 0], 1-positions[i, 1])
            for j in range(n):
                if i != j:
                    d = np.sqrt((positions[i,0]-positions[j,0])**2 + (positions[i,1]-positions[j,1])**2)
                    max_r = min(max_r, d - radii[j])
            radii[i] = max(max_r, 0)
        
        return positions, radii
    
    # Run relaxation
    max_iter = 1000
    for it in range(max_iter):
        if time.time() - start_time > deadline - safety_margin:
            break
        positions, radii = relax_iteration(positions, radii)
    
    # Final check and cleanup
    # Ensure all circles are valid
    for i in range(n):
        positions[i, 0] = np.clip(positions[i, 0], 0, 1)
        positions[i, 1] = np.clip(positions[i, 1], 0, 1)
        max_r = min(positions[i, 0], positions[i, 1], 1-positions[i, 0], 1-positions[i, 1])
        radii[i] = min(radii[i], max_r)
    
    # Build result
    result = np.zeros((n, 3))
    result[:, 0] = positions[:, 0]
    result[:, 1] = positions[:, 1]
    result[:, 2] = radii
    
    return result
# EVOLVE-BLOCK-END
