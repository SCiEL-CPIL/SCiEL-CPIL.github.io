import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)
TRI_AREA = SQRT3 / 4.0

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

def _min_area_and_indices(points):
    """Return (min_area, list of indices of points involved in min triangles)."""
    areas = _compute_areas(points)
    min_area = float(np.min(areas))
    # Find all triangles with area close to min
    threshold = min_area + 1e-10
    min_indices = np.where(areas <= threshold)[0]
    # Collect unique point indices involved
    involved = set()
    for idx in min_indices:
        involved.add(int(_idx_arr[idx, 0]))
        involved.add(int(_idx_arr[idx, 1]))
        involved.add(int(_idx_arr[idx, 2]))
    return min_area, sorted(involved)

def _min_area(points):
    """Return the minimum triangle area."""
    return float(np.min(_compute_areas(points)))

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

def _project_to_triangle(points):
    """Project points that are slightly outside the triangle back inside."""
    # Triangle vertices: A=(0,0), B=(1,0), C=(0.5, H)
    # Edges: AB: y >= 0
    #        BC: y <= -sqrt(3)*(x - 1) => y + sqrt(3)*x <= sqrt(3)
    #        CA: y <= sqrt(3)*x
    eps = 1e-10
    for m in range(N):
        x, y = points[m, 0], points[m, 1]
        # Check and fix if outside
        if y < -eps:
            points[m, 1] = 0.0
        if y + SQRT3 * x > SQRT3 + eps:
            # Project onto line BC
            # Line: y = -sqrt(3)(x-1) = sqrt(3)(1-x)
            # Normal direction: (sqrt(3), 1)
            # Project point onto line
            # Distance from line: (y + sqrt(3)*x - sqrt(3)) / sqrt(3+1) = (y + sqrt(3)*x - sqrt(3))/2
            d = (y + SQRT3 * x - SQRT3) / 2.0
            if d > 0:
                points[m, 0] -= SQRT3 * d
                points[m, 1] -= d
        if y > SQRT3 * x + eps:
            # Project onto line CA
            # Line: y = sqrt(3)*x
            # Normal: (sqrt(3), -1)
            d = (SQRT3 * x - y) / 2.0
            if d < 0:
                points[m, 0] += SQRT3 * d
                points[m, 1] -= d
    # Final clip to ensure inside
    for m in range(N):
        x, y = points[m, 0], points[m, 1]
        if y < 0:
            points[m, 1] = 0
        if y > SQRT3 * x:
            # Scale y down
            if x > 0:
                points[m, 1] = SQRT3 * x
            else:
                points[m, 1] = 0
        if y > SQRT3 * (1 - x):
            if x < 1:
                points[m, 1] = SQRT3 * (1 - x)
            else:
                points[m, 1] = 0
    return points

def _sobol_points_in_triangle(n, seed=42):
    """Generate n quasi-random points in the equilateral triangle using Sobol sequence."""
    from scipy.stats import qmc
    rng = qmc.Sobol(d=2, seed=seed)
    # Generate more points than needed to allow rejection sampling
    n_gen = n * 4
    samples = rng.random(n_gen)
    # Convert to barycentric: use two independent uniform [0,1] values
    # u, v with u+v <= 1
    u = samples[:, 0]
    v = samples[:, 1]
    mask = (u >= 0) & (v >= 0) & (u + v <= 1)
    valid_u = u[mask]
    valid_v = v[mask]
    # Take first n valid points
    if len(valid_u) >= n:
        u = valid_u[:n]
        v = valid_v[:n]
    else:
        # Fallback: generate more
        while len(valid_u) < n:
            more = rng.random(n_gen)
            u2, v2 = more[:, 0], more[:, 1]
            mask2 = (u2 >= 0) & (v2 >= 0) & (u2 + v2 <= 1)
            valid_u = np.concatenate([valid_u, u2[mask2]])
            valid_v = np.concatenate([valid_v, v2[mask2]])
        u = valid_u[:n]
        v = valid_v[:n]
    bary = np.column_stack([u, v])
    return _bary_to_cart(bary)

def _local_nudge_search(points, max_iter=500, step_sizes=None, time_limit=None):
    """
    Local search that only moves points involved in the smallest triangles.
    Uses targeted perturbations to break near-degeneracies.
    """
    if step_sizes is None:
        step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    current_points = points.copy()
    current_min, _ = _min_area_and_indices(current_points)
    best_points = current_points.copy()
    best_min = current_min
    
    if time_limit is None:
        time_limit = 60.0
    start_t = time.time()
    
    for step_size in step_sizes:
        if time.time() - start_t > time_limit:
            break
        
        for _ in range(max_iter):
            if time.time() - start_t > time_limit:
                break
            
            min_area, involved = _min_area_and_indices(current_points)
            
            if min_area > best_min:
                best_min = min_area
                best_points = current_points.copy()
            
            # Try nudging each involved point in random directions
            improved = False
            for idx in involved:
                if time.time() - start_t > time_limit:
                    break
                
                # Try several random perturbations for this point
                for _ in range(20):
                    # Random direction in barycentric space
                    du = np.random.uniform(-step_size, step_size)
                    dv = np.random.uniform(-step_size, step_size)
                    
                    # Apply perturbation in barycentric space
                    bary = _cart_to_bary(current_points)
                    new_u = bary[idx, 0] + du
                    new_v = bary[idx, 1] + dv
                    
                    # Project to valid simplex
                    new_u = max(0.0, new_u)
                    new_v = max(0.0, new_v)
                    s = new_u + new_v
                    if s > 1.0:
                        new_u /= s
                        new_v /= s
                    
                    # Convert back to Cartesian
                    new_x = new_v * 1.0 + (1.0 - new_u - new_v) * 0.5
                    new_y = (1.0 - new_u - new_v) * H
                    
                    # Check if inside triangle
                    if new_y < -1e-10 or new_y > SQRT3 * new_x + 1e-10 or new_y > SQRT3 * (1 - new_x) + 1e-10:
                        continue
                    
                    # Test the new configuration
                    test_points = current_points.copy()
                    test_points[idx, 0] = new_x
                    test_points[idx, 1] = new_y
                    test_min = _min_area(test_points)
                    
                    if test_min > min_area + 1e-12:
                        current_points = test_points
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                # Try a different point set or break
                break
    
    return best_points, best_min

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    np.random.seed(42)
    
    best_points = None
    best_score = -np.inf
    
    # Phase 1: Sobol-grid initialization with multiple seeds
    for seed in range(10):
        if time.time() > deadline - 30:
            break
        
        # Generate quasi-random initial points
        init_points = _sobol_points_in_triangle(N, seed=seed)
        init_points = _project_to_triangle(init_points)
        
        # Evaluate initial configuration
        init_score = _min_area(init_points)
        
        # Refine with local nudge search
        refined_points, refined_score = _local_nudge_search(
            init_points, 
            max_iter=200,
            time_limit=min(10.0, deadline - time.time() - 5)
        )
        
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points
    
    # Phase 2: If we have time, try more diverse initializations
    if best_points is not None and time.time() < deadline - 20:
        # Try a structured grid-based initialization
        # Place points in a triangular lattice pattern
        positions = []
        # Row 0 (bottom): 3 points at y=0
        for i in range(3):
            x = 0.1 + i * 0.3
            positions.append((x, 0.05))
        # Row 1: 4 points
        for i in range(4):
            x = 0.05 + i * 0.25
            y = 0.25
            if y <= SQRT3 * x and y <= SQRT3 * (1 - x):
                positions.append((x, y))
        # Row 2: 3 points
        for i in range(3):
            x = 0.15 + i * 0.25
            y = 0.45
            if y <= SQRT3 * x and y <= SQRT3 * (1 - x):
                positions.append((x, y))
        # Row 3: 1 point near top
        positions.append((0.5, 0.6))
        
        # Take first 11 valid positions
        valid_positions = []
        for p in positions:
            x, y = p
            if y >= -1e-10 and y <= SQRT3 * x + 1e-10 and y <= SQRT3 * (1 - x) + 1e-10:
                valid_positions.append(p)
        
        if len(valid_positions) >= N:
            grid_points = np.array(valid_positions[:N])
            grid_points = _project_to_triangle(grid_points)
            grid_score = _min_area(grid_points)
            
            refined_points, refined_score = _local_nudge_search(
                grid_points,
                max_iter=300,
                time_limit=min(15.0, deadline - time.time() - 5)
            )
            
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined_points
    
    # Phase 3: Final refinement of best result
    if best_points is not None and time.time() < deadline - 10:
        final_points, final_score = _local_nudge_search(
            best_points,
            max_iter=500,
            step_sizes=[0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001],
            time_limit=min(15.0, deadline - time.time() - 3)
        )
        if final_score > best_score:
            best_score = final_score
            best_points = final_points
    
    # Fallback if no valid result found
    if best_points is None:
        best_points = _sobol_points_in_triangle(N, seed=0)
        best_points = _project_to_triangle(best_points)
    
    # Final projection to ensure validity
    best_points = _project_to_triangle(best_points)
    
    return best_points
