"""Pack 50 equal circles in a unit square, maximize common radius."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 135.0  # seconds, safety margin before 150s kill

def _radius(centers):
    """Compute the max common radius for given centers."""
    # Wall radius = min of (0.5 - |coord|)
    wall_r = np.min(0.5 - np.abs(centers))
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _objective_hard(x, penalty_factor=10.0):
    """
    Objective function for minimization.
    We want to maximize r. So we minimize -r.
    To handle constraints (circles inside square, non-overlapping),
    we use a penalty method.
    
    Constraints:
    1. Centers must be inside [-0.5, 0.5]^2.
       Violation: center outside -> penalty.
    2. Circles must not overlap.
       For a given target radius r_target (or just maximizing min dist),
       if dist < 2r, there is overlap.
       
    However, the standard way to maximize min(dist/2, wall_dist) is to minimize 
    the negative of that value. But Nelder-Mead can struggle with flat regions 
    or non-differentiable mins.
    
    Let's define the objective as:
    f(x) = -r(x) + penalty(x)
    
    Where penalty(x) is large if constraints are violated.
    Actually, the definition of r(x) already accounts for the "tightest" constraint.
    If circles overlap, min_pair < 2*wall_r potentially, or vice versa.
    The "radius" is the max r such that all constraints hold.
    So r(x) = min( min_wall_dist, min_pair_dist/2 ).
    
    Minimizing -r(x) is equivalent to maximizing r(x).
    This function is continuous but not differentiable at points where 
    min_wall_dist == min_pair_dist/2 or where multiple pairs have the same min dist.
    Nelder-Mead is derivative-free and handles this well.
    
    We don't strictly need a penalty if we just maximize the "feasible radius" defined as above,
    because if circles overlap, r becomes small (specifically, if dist < 2r, then r <= dist/2).
    Wait, if circles overlap, dist < 2r. The "radius" of the packing is defined by the 
    actual geometry. If I place two centers at distance 0.1, the max radius they can have 
    without overlapping is 0.05. The wall constraint might be 0.5. So r = 0.05.
    So maximizing r automatically finds a non-overlapping configuration (or one that is 
    "least overlapping" in a metric sense, but usually it pushes them apart to increase min dist).
    
    So f(x) = - _radius(x) is a valid objective for Nelder-Mead.
    """
    centers = x.reshape(N, 2)
    r = _radius(centers)
    return -r

def _make_hexagonal_grid(n_x, n_y, spacing, offset_x=0, offset_y=0):
    """Create a hexagonal lattice of points."""
    points = []
    # Hexagonal packing: rows are offset
    # Vertical distance between rows: spacing * sqrt(3)/2
    # Horizontal distance: spacing
    # Offset for odd rows: spacing / 2
    
    # We need to fit N=50 points.
    # Approximate number of rows: sqrt(N * 2 / sqrt(3)) ~ 7-8 rows.
    # Let's just generate a large grid and take the first N.
    
    dy = spacing * np.sqrt(3) / 2
    for j in range(n_y):
        for i in range(n_x):
            x = i * spacing + offset_x
            y = j * dy + offset_y
            if j % 2 != 0:
                x += spacing / 2
            points.append((x, y))
    points = np.array(points)
    return points[:N]

def _make_grid_rect(n_x, n_y):
    """Create a rectangular grid."""
    ticks_x = (np.arange(n_x) + 0.5) / n_x - 0.5
    ticks_y = (np.arange(n_y) + 0.5) / n_y - 0.5
    grid = np.array([(x, y) for y in ticks_y for x in ticks_x])
    return grid[:N].copy()

def _local_refine_nm(centers, time_limit=5.0, initial_step=0.01):
    """
    Use Nelder-Mead to refine the configuration.
    """
    x0 = centers.ravel()
    
    start_time = time.time()
    
    # Nelder-Mead options
    # xatol, fatol control stopping criteria
    # maxiter is a hard stop
    # We check time in callback
    
    def callback(xk):
        if time.time() - start_time > time_limit:
            raise RuntimeError("Time limit exceeded")
        return False

    try:
        res = minimize(
            _objective_hard, 
            x0, 
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'xatol': 1e-6,
                'fatol': 1e-8,
                'initial_simplex': None, # Let scipy choose, or we can set it
                'adaptive': True,
                'disp': False
            },
            callback=callback
        )
        if res.x is not None:
            return res.x.reshape(N, 2)
    except RuntimeError:
        pass
    except Exception:
        pass
    
    return centers

def construct():
    global DEADLINE
    start_time = time.time()
    
    best_centers = None
    best_r = 0.0
    
    # Generate initial configurations
    inits = []
    
    # 1. Rectangular Grid 7x8 (56 pts -> 50)
    # Spacing ~ 1/7.5
    c1 = _make_grid_rect(7, 8)
    inits.append(c1)
    
    # 2. Rectangular Grid 8x7
    c2 = _make_grid_rect(8, 7)
    inits.append(c2)
    
    # 3. Hexagonal Grid
    # Estimate spacing for 50 points in unit square.
    # Area per point ~ 1/50 = 0.02.
    # Hex area ~ spacing^2 * sqrt(3)/2.
    # spacing^2 ~ 0.02 * 2 / 1.732 ~ 0.023. spacing ~ 0.15.
    # Let's try a few spacings.
    for sp in [0.14, 0.15, 0.16]:
        # Generate enough points
        c_hex = _make_hexagonal_grid(10, 10, sp, offset_x=-0.5, offset_y=-0.5)
        # Center it roughly
        c_hex = c_hex - np.mean(c_hex, axis=0)
        # Clip to bounds? No, let optimizer handle it, but ensure they are somewhat inside
        # Actually, hex grid might extend outside. Let's scale to fit.
        max_ext = np.max(np.abs(c_hex))
        if max_ext > 0.45:
            c_hex = c_hex * (0.45 / max_ext)
        inits.append(c_hex)
        
    # 4. Random
    np.random.seed(42)
    c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
    inits.append(c_rand)
    
    # 5. "V" shape or other structured? 
    # Let's stick to grids and hex for now, plus random.
    
    # Refine each initial configuration
    # Nelder-Mead is slower per iteration than L-BFGS-B but more robust to non-smoothness.
    # We have ~130s. 6 inits. ~20s each.
    
    for i, init in enumerate(inits):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        time_left = DEADLINE - 10 - (time.time() - start_time)
        # Allocate time
        alloc_time = min(20.0, time_left / max(1, len(inits) - i))
        
        # Nelder-Mead refinement
        # Use a relatively small initial step to avoid huge jumps
        refined = _local_refine_nm(init, time_limit=alloc_time)
        r = _radius(refined)
        
        if r > best_r:
            best_r = r
            best_centers = refined.copy()
            
    # If time remains, do more random restarts with NM
    while time.time() - start_time < DEADLINE - 5:
        seed = int(time.time() * 1000) % 100000
        np.random.seed(seed)
        c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
        
        time_left = DEADLINE - 5 - (time.time() - start_time)
        if time_left < 5: break
        
        # Give random restarts less time, as they are less likely to beat structured inits
        # unless the landscape is very rugged.
        refined = _local_refine_nm(c_rand, time_limit=time_left * 0.9)
        r = _radius(refined)
        
        if r > best_r:
            best_r = r
            best_centers = refined.copy()

    return best_centers
