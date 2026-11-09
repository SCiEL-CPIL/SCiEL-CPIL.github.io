import numpy as np
from scipy.optimize import minimize
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    def get_score(points):
        """Calculate the exact minimum radius for a given set of points."""
        points = np.asarray(points).reshape(-1, 3)
        if points.shape != (N, 3):
            return 0.0
        if not np.all(np.isfinite(points)):
            return 0.0
        if np.any(np.abs(points) > 0.5 + 1e-9):
            return 0.0
        
        # Pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        # Wall distances
        wall_dists = 0.5 - np.abs(points)
        min_wall = np.min(np.min(wall_dists, axis=1))
        
        return min(min_pair, min_wall)

    def push_apart(points, iterations=200, margin=0.001):
        """Iteratively push apart points that are too close."""
        points = points.copy()
        for _ in range(iterations):
            if time.time() > deadline - 10:
                break
            moved = False
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            
            # Find pairs too close
            iu = np.triu_indices(N, k=1)
            close_mask = dists[iu] < 2 * (0.1113823475125 + margin)
            
            if not np.any(close_mask):
                break
            
            close_pairs = list(zip(iu[0][close_mask], iu[1][close_mask]))
            
            for i, j in close_pairs:
                d = points[i] - points[j]
                dist = np.linalg.norm(d)
                if dist < 1e-12:
                    # Random nudge
                    d = np.random.randn(3) * 0.01
                    dist = np.linalg.norm(d)
                d = d / dist
                # Push apart
                push = (2 * (0.1113823475125 + margin) - dist) / 2.0
                points[i] += push * d
                points[j] -= push * d
                moved = True
            
            # Also push points away from walls
            wall_dists = 0.5 - np.abs(points)
            min_wall_per_point = np.min(wall_dists, axis=1)
            for i in range(N):
                if min_wall_per_point[i] < 0.1113823475125 + margin:
                    # Push toward center
                    points[i] *= 0.95
                    moved = True
            
            # Clip to bounds
            points = np.clip(points, -0.5, 0.5)
            
            if not moved:
                break
        
        return points

    def soft_min_objective(x_flat, beta):
        """
        Objective function to minimize.
        We want to maximize the minimum distance.
        Using soft-min approximation.
        """
        points = x_flat.reshape(-1, 3)
        
        # Pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1e6)
        
        # Wall distances
        wall_dists = 0.5 - np.abs(points)
        min_wall_per_point = np.min(wall_dists, axis=1)
        
        # Combine
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu]
        all_dists = np.concatenate([pair_dists, min_wall_per_point])
        
        # Soft min
        max_d = np.max(all_dists)
        shifted = all_dists - max_d
        exp_terms = np.exp(-beta * shifted)
        sum_exp = np.sum(exp_terms)
        if sum_exp == 0:
            return 0.0
        soft_min = max_d - (1.0 / beta) * np.log(sum_exp)
        
        return -soft_min

    def generate_fcc_lattice():
        """Generate FCC lattice points within the cube."""
        points = []
        # FCC lattice: points at (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5) + integer translations
        a = 0.5  # lattice spacing to try
        for ix in range(-2, 3):
            for iy in range(-2, 3):
                for iz in range(-2, 3):
                    base = np.array([ix * a, iy * a, iz * a])
                    # FCC basis vectors
                    for bx, by, bz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                        pt = base + np.array([bx * a, by * a, bz * a])
                        if np.all(np.abs(pt) <= 0.5):
                            points.append(pt)
        
        points = np.array(points)
        if len(points) < N:
            # Add more points by reducing spacing
            a = 0.4
            points = []
            for ix in range(-3, 4):
                for iy in range(-3, 4):
                    for iz in range(-3, 4):
                        base = np.array([ix * a, iy * a, iz * a])
                        for bx, by, bz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                            pt = base + np.array([bx * a, by * a, bz * a])
                            if np.all(np.abs(pt) <= 0.5):
                                points.append(pt)
            points = np.array(points)
        
        # Select N points with largest minimum distance to boundary
        if len(points) > N:
            wall_dists = np.min(0.5 - np.abs(points), axis=1)
            # Sort by wall distance descending, take top N
            idx = np.argsort(-wall_dists)[:N]
            points = points[idx]
        elif len(points) < N:
            # Pad with random points
            rng = np.random.default_rng(42)
            extra = rng.uniform(-0.4, 0.4, (N - len(points), 3))
            points = np.vstack([points, extra])
        
        return points

    def generate_grid(nx, ny, nz):
        """Create a grid of points centered in the cube."""
        x = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
        y = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
        z = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
        grid = np.array([(x, y, z) for z in z for y in y for x in x])
        return grid[:N]

    # Initial configurations
    inits = []
    
    # FCC lattice
    fcc = generate_fcc_lattice()
    inits.append(fcc)
    
    # 5x5x4 Grid
    x5 = np.linspace(-0.4, 0.4, 5)
    y5 = np.linspace(-0.4, 0.4, 5)
    z4 = np.linspace(-0.3, 0.3, 4)
    grid1 = np.array([(x, y, z) for z in z4 for y in y5 for x in x5])
    inits.append(grid1)
    
    # 4x5x5 Grid
    x4 = np.linspace(-0.3, 0.3, 4)
    y5b = np.linspace(-0.4, 0.4, 5)
    z5 = np.linspace(-0.4, 0.4, 5)
    grid2 = np.array([(x, y, z) for z in z5 for y in y5b for x in x4])
    inits.append(grid2)
    
    # 5x4x5 Grid
    x5c = np.linspace(-0.4, 0.4, 5)
    y4 = np.linspace(-0.3, 0.3, 4)
    z5b = np.linspace(-0.4, 0.4, 5)
    grid3 = np.array([(x, y, z) for z in z5b for y in y4 for x in x5c])
    inits.append(grid3)
    
    # Random jittered versions
    rng = np.random.default_rng(42)
    grid4 = grid1 + rng.normal(0, 0.01, grid1.shape)
    inits.append(grid4)
    
    # Add a few more random starts
    for _ in range(2):
        rand_grid = rng.uniform(-0.4, 0.4, (N, 3))
        inits.append(rand_grid)

    best_score = 0.0
    best_points = inits[0].copy()

    # Update best score with initial grids
    for g in inits:
        s = get_score(g)
        if s > best_score:
            best_score = s
            best_points = g.copy()

    bounds = [(-0.5, 0.5)] * (N * 3)

    # Stage 1: Push-apart relaxation from best initial
    if time.time() < deadline - 30:
        relaxed = push_apart(best_points, iterations=300, margin=0.002)
        s = get_score(relaxed)
        if s > best_score:
            best_score = s
            best_points = relaxed.copy()

    # Stage 2: Soft-min optimization with moderate beta
    beta1 = 50.0
    for i, init in enumerate(inits[:4]):  # Use top 4 initializations
        if time.time() > deadline - 20:
            break
            
        x0 = init.flatten()
        
        try:
            res = minimize(
                lambda x: soft_min_objective(x, beta1),
                x0, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if time.time() > deadline - 20:
                break
                
            points = res.x.reshape(-1, 3)
            points = np.clip(points, -0.5, 0.5)
            
            # Push apart to fix any violations
            points = push_apart(points, iterations=100, margin=0.001)
            
            s = get_score(points)
            if s > best_score:
                best_score = s
                best_points = points.copy()
                
        except Exception:
            pass

    # Stage 3: Refine best with high beta
    if time.time() < deadline - 10:
        beta2 = 200.0
        x0 = best_points.flatten()
        try:
            res = minimize(
                lambda x: soft_min_objective(x, beta2),
                x0, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            points = res.x.reshape(-1, 3)
            points = np.clip(points, -0.5, 0.5)
            
            # Final push-apart polish
            points = push_apart(points, iterations=50, margin=0.0005)
            
            s = get_score(points)
            if s > best_score:
                best_score = s
                best_points = points.copy()
        except Exception:
            pass

    # Final validation and clipping
    best_points = np.clip(best_points, -0.5, 0.5)
    
    return best_points
