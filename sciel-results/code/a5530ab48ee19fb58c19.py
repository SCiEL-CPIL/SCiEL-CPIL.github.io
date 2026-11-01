"""Pack 200 equal circles in a unit square using simulated annealing with hexagonal seeding."""
import numpy as np
import time

N = 200

def _score(pts):
    """Compute the largest common radius for given centers."""
    # Distance to boundary
    d_bound = np.min(np.stack([
        0.5 - np.abs(pts[:, 0]),
        0.5 - np.abs(pts[:, 1])
    ]), axis=1)
    min_bound = np.min(d_bound)
    
    # Minimum pairwise distance / 2
    # Use cdist or manual broadcasting for small N
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_bound, min_pair)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 115s safety margin for 120s limit
    
    # Hexagonal seeding
    # Try to arrange in a hexagonal pattern
    # For 200 circles, a hexagonal packing has ~sqrt(3)/2 efficiency
    # Let's generate a hexagonal lattice and take the first 200 points
    # that fit in the square
    
    # Generate hexagonal lattice points
    # Hexagonal lattice: points at (i*a + j*a/2, j*a*sqrt(3)/2)
    # We want to fit as many as possible in [-0.5, 0.5]^2
    
    # First, estimate the spacing. If we have N circles in area 1,
    # each circle has area pi*r^2, total area N*pi*r^2 <= 1
    # So r <= 1/sqrt(N*pi) ≈ 0.0399
    # The best known is 0.0366, so spacing ~ 2*r ≈ 0.073
    
    # Let's create a hexagonal lattice with spacing slightly larger than needed
    # and then use simulated annealing to optimize
    
    # Generate candidate positions on a hexagonal grid
    # Hex spacing s: points at (i*s + j*s*0.5, j*s*sqrt(3)/2)
    # For the square [-0.5, 0.5]^2, we need:
    # |i*s + j*s*0.5| <= 0.5 - r
    # |j*s*sqrt(3)/2| <= 0.5 - r
    
    # Let's try a few different initial spacings
    best_pts = None
    best_score = 0.0
    
    # Simulated annealing
    rng = np.random.RandomState(42)
    
    # Initial configuration: hexagonal pattern
    # Try to place 200 circles in a hexagonal arrangement
    # The hexagonal density is pi/(2*sqrt(3)) ≈ 0.9069
    # So for N=200 in unit square, effective area per circle = 1/200 = 0.005
    # Hexagonal spacing s satisfies: s^2 * sqrt(3)/2 = 1/200
    # s = sqrt(2/(200*sqrt(3))) ≈ sqrt(0.00577) ≈ 0.076
    
    s = np.sqrt(2.0 / (N * np.sqrt(3)))
    
    # Generate hexagonal lattice
    # Points: (i*s + j*s*0.5, j*s*sqrt(3)/2)
    max_i = int(np.ceil(1.0 / s)) + 2
    max_j = int(np.ceil(1.0 / (s * np.sqrt(3) / 2))) + 2
    
    candidates = []
    for j in range(-max_j, max_j + 1):
        for i in range(-max_i, max_i + 1):
            x = i * s + j * s * 0.5
            y = j * s * np.sqrt(3) / 2
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                candidates.append([x, y])
    
    candidates = np.array(candidates)
    
    if len(candidates) < N:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        pts = grid[:N].copy()
    else:
        # Select N points from candidates
        # Try to pick a contiguous block or random sample
        # For better packing, let's try to center the selection
        # Sort by distance from center and take the N closest? No, that might not be optimal.
        # Let's just take a random subset or a structured one.
        
        # Better approach: generate a larger hexagonal grid and select N points
        # that are roughly centered
        # Let's pick points in a rectangular region of the hex grid
        # that contains exactly N or more points, then randomly select N
        
        # For simplicity, let's use a random selection from candidates
        # but bias towards the center
        dists_from_center = np.sum(candidates**2, axis=1)
        # Sample N points, with probability proportional to 1/(1+dists)
        weights = 1.0 / (1.0 + dists_from_center * 10)
        weights /= weights.sum()
        
        # Reservoir sampling or just random choice
        idx = rng.choice(len(candidates), size=N, replace=False, p=weights)
        pts = candidates[idx].copy()
    
    # Ensure all points are strictly inside the square
    margin = 0.01
    pts[:, 0] = np.clip(pts[:, 0], -0.5 + margin, 0.5 - margin)
    pts[:, 1] = np.clip(pts[:, 1], -0.5 + margin, 0.5 - margin)
    
    current_score = _score(pts)
    best_pts = pts.copy()
    best_score = current_score
    
    # Simulated annealing
    T = 0.01  # Initial temperature
    T_min = 1e-8
    cooling = 0.999
    step_size = 0.02
    
    while time.time() < deadline:
        # Generate a move: pick a random point and perturb it
        idx = rng.randint(N)
        old_x, old_y = pts[idx, 0], pts[idx, 1]
        
        # Gaussian perturbation
        delta = step_size * rng.randn(2)
        new_x = old_x + delta[0]
        new_y = old_y + delta[1]
        
        # Clamp to square
        new_x = np.clip(new_x, -0.5, 0.5)
        new_y = np.clip(new_y, -0.5, 0.5)
        
        pts[idx, 0] = new_x
        pts[idx, 1] = new_y
        
        new_score = _score(pts)
        
        if new_score > current_score or rng.rand() < np.exp((new_score - current_score) / T):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
        else:
            pts[idx, 0] = old_x
            pts[idx, 1] = old_y
        
        T *= cooling
        if T < T_min:
            T = T_min
        
        # Occasionally increase step size for exploration
        if rng.rand() < 0.01:
            step_size = 0.02 * rng.uniform(0.5, 2.0)
    
    return best_pts
