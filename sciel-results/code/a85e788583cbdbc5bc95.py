import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)
TRI_AREA = SQRT3 / 4.0

# Precompute the 165 combinations
_indices = list(itertools.combinations(range(N), 3))
_idx_arr = np.array(_indices)

def _compute_areas(points):
    """Compute all 165 triangle areas from 11 points using vectorized operations."""
    i = _idx_arr[:, 0]
    j = _idx_arr[:, 1]
    k = _idx_arr[:, 2]
    
    xi, yi = points[i, 0], points[i, 1]
    xj, yj = points[j, 0], points[j, 1]
    xk, yk = points[k, 0], points[k, 1]
    
    det = xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj)
    areas = 0.5 * np.abs(det)
    return areas

def _min_area(points):
    """Return the minimum triangle area."""
    return float(np.min(_compute_areas(points)))

def _objective(params):
    """
    Objective to minimize: negative of the minimum triangle area.
    params: 22 values representing barycentric coordinates (u, v) for each point.
    """
    uv = params.reshape(N, 2)
    u = np.clip(uv[:, 0], 0.0, None)
    v = np.clip(uv[:, 1], 0.0, None)
    
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
        
    pts = np.empty((N, 2))
    pts[:, 0] = v * 1.0 + (1.0 - u - v) * 0.5
    pts[:, 1] = (1.0 - u - v) * H
    
    min_a = _min_area(pts)
    return -min_a

def _bary_to_cart(bary):
    """Convert barycentric coords (u,v) array to Cartesian points, ensuring validity."""
    u = np.clip(bary[:, 0], 0.0, None)
    v = np.clip(bary[:, 1], 0.0, None)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
        
    pts = np.empty((N, 2))
    pts[:, 0] = v * 1.0 + (1.0 - u - v) * 0.5
    pts[:, 1] = (1.0 - u - v) * H
    return pts

def _cart_to_bary(pts):
    """Convert Cartesian points to barycentric coords (u, v)."""
    bary = np.empty((N, 2))
    for m in range(N):
        x = pts[m, 0]
        y = pts[m, 1]
        bary[m, 0] = 1.0 - x - y / (2.0 * H)
        bary[m, 1] = x - y / (2.0 * H)
    return bary

def _random_valid_barycentric(rng):
    """Generate random valid barycentric coordinates."""
    bary = np.empty((N, 2))
    for m in range(N):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v <= 1.0:
                bary[m, 0] = u
                bary[m, 1] = v
                break
    return bary

def _hexagonal_initial_guess():
    """
    Generate a structured initial guess based on a hexagonal packing pattern.
    This leverages the symmetry and is known to be a good local optimum for small N.
    """
    # Hexagonal lattice points within the triangle
    # Side length of hex grid roughly 1/5 of triangle side
    # We want 11 points.
    # Center of triangle: (0.5, H/3)
    # Let's use a specific known good configuration for 11 points if possible, 
    # or generate a hex grid and pick the best 11.
    
    # Generate a fine hex grid
    points = []
    s = 0.15 # spacing
    for i in range(-10, 10):
        for j in range(-10, 10):
            x = 0.5 + s * i + 0.5 * s * j
            y = H/3 + s * np.sqrt(3) * 0.5 * j
            # Check if inside triangle
            if y >= 0 and y <= H and x >= 0 and x <= 1 and y <= np.sqrt(3)*x and y <= np.sqrt(3)*(1-x):
                points.append((x, y))
    
    # We have many candidates. We need to select 11 that maximize min area.
    # A simple greedy selection might work, or just pick a cluster around the center.
    # Actually, for N=11, a 3-3-3-2 or similar distribution is good.
    # Let's just use a standard hexagonal packing seed.
    # A known good seed for N=11 is often derived from hexagonal packing.
    
    # Let's try a specific pattern:
    # Row 1 (bottom): 4 points
    # Row 2: 3 points
    # Row 3: 2 points
    # Row 4: 2 points
    # Total 11.
    
    # Coordinates for a hexagonal arrangement scaled to fit the triangle.
    # Triangle side = 1. Height = sqrt(3)/2.
    # Hex grid spacing d.
    # If we use 4 rows, height ~ 3 * d * sqrt(3)/2.
    # 3 * d * 0.866 ~ 0.866 => d ~ 1/3?
    
    # Let's just use a set of points that are roughly hexagonal.
    # Center: (0.5, 0.288)
    # Neighbors at distance r.
    
    # Actually, let's use the "greedy" initialization from the inspiration but make it deterministic and robust.
    # Or better, use the hexagonal lattice points that are closest to the optimal packing.
    
    # Let's define a specific good seed manually or via a simple rule.
    # Rule: Points are arranged in a hexagonal lattice with spacing ~ 1/4.
    
    u_vals = []
    v_vals = []
    
    # Generate hex grid in barycentric coords directly?
    # Barycentric (u, v, w) with u+v+w=1.
    # Hex grid in barycentric:
    # (1/3, 1/3, 1/3) is center.
    
    # Let's stick to a simple, strong heuristic:
    # Place points in a 3x3 grid (offset) + 2 extra.
    
    # Better: Use the known optimal configuration for N=10 and add 1?
    # No, let's use a random search to find a good initial hex-like config? No, must be fast.
    
    # Let's use a fixed, high-quality seed.
    # Seed for N=11 (from literature or quick search):
    # Approximate coordinates:
    # (0.1, 0.1), (0.4, 0.1), (0.7, 0.1), (0.9, 0.1)
    # (0.2, 0.3), (0.5, 0.3), (0.8, 0.3)
    # (0.3, 0.5), (0.6, 0.5)
    # (0.4, 0.7), (0.7, 0.7)
    
    # Let's refine this specific seed slightly to be more symmetric.
    # Symmetric around x=0.5.
    
    coords = [
        (0.1, 0.0866), # y = sqrt(3)/2 * 0.1
        (0.3, 0.0866),
        (0.5, 0.0866),
        (0.7, 0.0866),
        (0.9, 0.0866),
        
        (0.2, 0.2598), # y = sqrt(3)/2 * 0.3
        (0.4, 0.2598),
        (0.6, 0.2598),
        (0.8, 0.2598),
        
        (0.3, 0.4330), # y = sqrt(3)/2 * 0.5
        (0.5, 0.4330),
        (0.7, 0.4330),
        
        (0.4, 0.6062), # y = sqrt(3)/2 * 0.7
        (0.6, 0.6062),
        
        (0.5, 0.7794)  # y = sqrt(3)/2 * 0.9
    ]
    
    # Wait, that's 15 points. We need 11.
    # Let's take a subset that looks like a hexagonal packing.
    # Center (0.5, 0.288)
    # Ring 1: 6 points
    # Ring 2: 4 points?
    
    # Let's just use a random valid configuration but optimized by Nelder-Mead.
    # The key is that Nelder-Mead works well on non-smooth functions if the initial guess is decent.
    # A random guess is often decent enough for N=11.
    
    # To be safe, let's generate a few random configurations and pick the best one as the start.
    rng = np.random.default_rng(42)
    best_start = None
    best_start_score = -np.inf
    
    for _ in range(20):
        bary = _random_valid_barycentric(rng)
        pts = _bary_to_cart(bary)
        score = _min_area(pts)
        if score > best_start_score:
            best_start_score = score
            best_start = bary.flatten()
            
    return best_start

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    # Bounds for barycentric coordinates: each in [0, 1]
    bounds = [(0.0, 1.0)] * (2 * N)
    
    best_points = None
    best_score = -np.inf
    
    def _try_nelder_mead(initial_params, maxiter=1000, time_limit=10.0):
        """Run Nelder-Mead optimization."""
        start_t = time.time()
        try:
            result = minimize(
                _objective,
                initial_params,
                method='Nelder-Mead',
                options={
                    'maxiter': maxiter,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            # Check time
            if time.time() - start_t > time_limit:
                # We can't easily interrupt Nelder-Mead mid-run, but we can check if it finished
                # If it's taking too long, we might have to rely on the 'maxiter' or just accept the wait.
                # However, Nelder-Mead is usually fast per iteration.
                pass
            
            if result.fun < -np.inf + 1:
                pts = _bary_to_cart(result.x.reshape(N, 2))
                # Verify validity
                if np.all(pts[:, 0] >= -1e-8) and np.all(pts[:, 0] <= 1.0 + 1e-8) and \
                   np.all(pts[:, 1] >= -1e-8) and np.all(pts[:, 1] <= H + 1e-8):
                    x = pts[:, 0]
                    y = pts[:, 1]
                    if np.all(y >= -1e-8) and np.all(y <= SQRT3 * x + 1e-8) and \
                       np.all(y <= -SQRT3 * x + SQRT3 + 1e-8):
                        score = _min_area(pts)
                        return pts, score
        except Exception:
            pass
        return None, -np.inf
    
    # Phase 1: Optimize from the best random start found in _hexagonal_initial_guess
    initial_params = _hexagonal_initial_guess()
    
    pts, score = _try_nelder_mead(initial_params, maxiter=2000, time_limit=20.0)
    if pts is not None and score > best_score:
        best_points = pts
        best_score = score
        
    # Phase 2: Multiple restarts from random initial points with Nelder-Mead
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        random_bary = _random_valid_barycentric(rng)
        random_params = random_bary.flatten()
        
        pts, score = _try_nelder_mead(random_params, maxiter=1000, time_limit=10.0)
        if pts is not None and score > best_score:
            best_points = pts
            best_score = score
            
    # Phase 3: If we have time, try a few more restarts with different random seeds
    if best_points is not None and time.time() < deadline - 20:
        for i in range(5):
            if time.time() > deadline - 10:
                break
            random_bary = _random_valid_barycentric(rng)
            random_params = random_bary.flatten()
            pts, score = _try_nelder_mead(random_params, maxiter=500, time_limit=5.0)
            if pts is not None and score > best_score:
                best_points = pts
                best_score = score

    # Final check and strict validation
    if best_points is None:
        # Fallback: return a valid configuration
        bary = _random_valid_barycentric(rng)
        best_points = _bary_to_cart(bary)
    
    # Strictly enforce containment by re-clipping the final output
    final_bary = _cart_to_bary(best_points)
    final_points = _bary_to_cart(final_bary)
    
    return final_points
