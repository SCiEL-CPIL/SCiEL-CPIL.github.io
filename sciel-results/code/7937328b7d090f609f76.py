"""Pack 50 equal circles in a unit square, maximize common radius."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _radius(centers):
    """Compute the max common radius for given centers (exact, for scoring)."""
    # Wall radius = min of (0.5 - |coord|)
    wall_r = np.min(0.5 - np.abs(centers))
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _soft_min_radius(centers, beta=100.0):
    """
    Compute a smooth approximation of the radius using log-sum-exp.
    r = min(wall_r, min_pair/2)
    -log(sum(exp(-beta * constraints))) / beta approximates min(constraints).
    
    Constraints are:
    1. 0.5 - centers[:, 0]
    2. 0.5 + centers[:, 0]
    3. 0.5 - centers[:, 1]
    4. 0.5 + centers[:, 1]
    5. dist(i, j) / 2 for all i < j
    """
    # Wall constraints
    c1 = 0.5 - centers[:, 0]
    c2 = 0.5 + centers[:, 0]
    c3 = 0.5 - centers[:, 1]
    c4 = 0.5 + centers[:, 1]
    
    # Pairwise constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    # Take upper triangle
    iu = np.triu_indices(N, 1)
    pair_dists = dist[iu] / 2.0
    
    # Combine all constraints
    # To avoid overflow in exp, we can shift by the max, but since we want min,
    # we look at -beta * c. If c is small, -beta*c is large negative? No.
    # min(c) = - (1/beta) * log(sum(exp(-beta * c)))
    # If c is large, exp(-beta*c) is small. If c is small, exp(-beta*c) is large.
    # So the sum is dominated by the smallest c.
    
    all_constraints = np.concatenate([c1, c2, c3, c4, pair_dists])
    
    # Numerical stability: subtract max from exponents
    max_val = np.max(all_constraints)
    # We want min, so we care about small values.
    # Let's use the identity: min(x) = -max(-x)
    # max(-x) = (1/beta) log(sum(exp(-beta * x)))
    # So min(x) = -(1/beta) log(sum(exp(-beta * x)))
    
    # To compute sum(exp(-beta * x)) stably:
    # Let M = max(-beta * x) = -beta * min(x).
    # sum(exp(-beta * x)) = exp(M) * sum(exp(-beta * x - M))
    # log(sum) = M + log(sum(exp(...)))
    # min(x) = -(1/beta) * (M + log(sum(exp(...))))
    # M = -beta * min(x) ... circular.
    
    # Better:
    # min(x) = - (1/beta) * log( sum(exp(-beta * x)) )
    # Let y = -beta * x. We want log(sum(exp(y))).
    # max_y = np.max(y)
    # log_sum_exp = max_y + log(sum(exp(y - max_y)))
    # min(x) = - (1/beta) * (max_y + log(sum(exp(y - max_y))))
    
    y = -beta * all_constraints
    max_y = np.max(y)
    # If max_y is very negative, exp(y-max_y) is fine.
    # If max_y is very positive, it's fine too.
    sum_exp = np.sum(np.exp(y - max_y))
    log_sum_exp = max_y + np.log(sum_exp)
    
    return -log_sum_exp / beta

def _objective_smooth(x, beta=100.0):
    """Objective to minimize: -soft_min_radius."""
    centers = x.reshape(N, 2)
    r = _soft_min_radius(centers, beta)
    return -r

def _make_grid_rect(n_x, n_y):
    """Create a rectangular grid."""
    ticks_x = (np.arange(n_x) + 0.5) / n_x - 0.5
    ticks_y = (np.arange(n_y) + 0.5) / n_y - 0.5
    grid = np.array([(x, y) for y in ticks_y for x in ticks_x])
    return grid[:N].copy()

def _make_hexagonal_grid(n_x, n_y, spacing, offset_x=0, offset_y=0):
    """Create a hexagonal lattice of points."""
    points = []
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

def _local_refine_smooth(centers, time_limit=5.0, beta=100.0):
    """
    Use L-BFGS-B to refine the configuration using the smooth objective.
    """
    x0 = centers.ravel()
    
    start_time = time.time()
    
    def callback(xk):
        if time.time() - start_time > time_limit:
            raise RuntimeError("Time limit exceeded")
        return False

    try:
        res = minimize(
            _objective_smooth, 
            x0, 
            args=(beta,),
            method='L-BFGS-B',
            options={
                'maxiter': 1000,
                'maxfun': 5000,
                'ftol': 1e-10,
                'gtol': 1e-8,
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
    
    # 1. Rectangular Grid 7x8
    c1 = _make_grid_rect(7, 8)
    inits.append(c1)
    
    # 2. Rectangular Grid 8x7
    c2 = _make_grid_rect(8, 7)
    inits.append(c2)
    
    # 3. Hexagonal Grids
    for sp in [0.14, 0.15, 0.16]:
        c_hex = _make_hexagonal_grid(10, 10, sp, offset_x=-0.5, offset_y=-0.5)
        c_hex = c_hex - np.mean(c_hex, axis=0)
        max_ext = np.max(np.abs(c_hex))
        if max_ext > 0.45:
            c_hex = c_hex * (0.45 / max_ext)
        inits.append(c_hex)
        
    # 4. Random
    np.random.seed(42)
    c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
    inits.append(c_rand)
    
    # Refine each initial configuration
    # L-BFGS-B is much faster per iteration than NM for this smooth landscape.
    # We can afford shorter time limits per run and more runs.
    
    for i, init in enumerate(inits):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        time_left = DEADLINE - 10 - (time.time() - start_time)
        # Allocate time: L-BFGS-B is fast, so we can try many.
        # Let's give 2s per structured init.
        alloc_time = min(2.0, time_left / max(1, len(inits) - i))
        
        # Start with a lower beta to find the basin, then increase?
        # Or just use a fixed high beta. High beta is sharper, better for precise min.
        # But might get stuck in local minima of the soft-min.
        # Let's try beta=50 for the first pass to be smoother.
        refined = _local_refine_smooth(init, time_limit=alloc_time, beta=50.0)
        r = _radius(refined)
        
        if r > best_r:
            best_r = r
            best_centers = refined.copy()
            
    # If time remains, do more random restarts with L-BFGS-B
    # This is where the speed advantage shines. We can do many more restarts.
    while time.time() - start_time < DEADLINE - 5:
        seed = int(time.time() * 1000) % 100000
        np.random.seed(seed)
        c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
        
        time_left = DEADLINE - 5 - (time.time() - start_time)
        if time_left < 1: break
        
        # Give random restarts ~1.5s each. L-BFGS-B should converge quickly.
        alloc_time = min(1.5, time_left * 0.9)
        
        # Use higher beta for random restarts to sharpen the minimum once we are close?
        # Or keep it moderate.
        refined = _local_refine_smooth(c_rand, time_limit=alloc_time, beta=100.0)
        r = _radius(refined)
        
        if r > best_r:
            best_r = r
            best_centers = refined.copy()

    return best_centers
