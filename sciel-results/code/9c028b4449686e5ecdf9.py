import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _calc_score(pts):
    """Calculate the actual score (radius) for a set of points."""
    if pts.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
        
    # Pair distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1e10)
    min_pair_dist = np.sqrt(np.min(dist_sq))
    
    # Wall distances
    min_wall_dist = np.min(0.5 - np.abs(pts))
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _project_to_box(pts):
    """Ensure points are strictly inside the box to avoid boundary issues."""
    return np.clip(pts, -0.499, 0.499)

def _local_optimize(x0, deadline, max_iter=100):
    """
    Perform a local optimization using a soft potential that approximates max-min.
    We use a high power p to make the energy landscape sharper around the minimum distance.
    """
    def objective(x_flat):
        x = x_flat.reshape(N, 3)
        # Pairwise repulsion
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        mask = ~np.eye(N, dtype=bool)
        dist_sq_masked = np.where(mask, dist_sq, 1.0)
        
        # Use p=20 for sharper focus
        p = 20
        alpha = p / 2.0
        e_pairs = np.sum(1.0 / (dist_sq_masked ** alpha))
        
        # Gradient
        factor = -2.0 * alpha / (dist_sq_masked ** (alpha + 1))
        factor[np.eye(N, dtype=bool)] = 0.0
        weighted_diff = diff * factor[:, :, np.newaxis]
        grad_pairs = np.sum(weighted_diff, axis=1)
        
        # Wall repulsion
        d_left = np.maximum(x + 0.5, 1e-6)
        d_right = np.maximum(0.5 - x, 1e-6)
        
        e_walls = np.sum(1.0 / (d_left ** p)) + np.sum(1.0 / (d_right ** p))
        grad_walls = -p / (d_left ** (p + 1)) + p / (d_right ** (p + 1))
        
        total_e = e_pairs + e_walls
        total_g = (grad_pairs + grad_walls).flatten()
        
        return total_e, total_g

    bounds = [(-0.499, 0.499)] * 3 * N
    
    try:
        res = minimize(objective, x0.flatten(), method='L-BFGS-B', jac=True,
                       bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-10, 'disp': False})
        if res.success:
            return res.x.reshape(N, 3)
    except Exception:
        pass
    return x0

def _generate_lattice_init(type, rng):
    """
    Generate lattice-based initializations.
    Type 0: BCC (Body Centered Cubic)
    Type 1: FCC (Face Centered Cubic)
    Type 2: Random
    """
    if type == 2:
        return rng.uniform(-0.4, 0.4, (N, 3))
    
    points = []
    
    if type == 0: # BCC
        # Optimal BCC packing in a cube often involves a 4x4x4 grid of BCC cells?
        # 4*4*4 = 64 cells * 2 atoms = 128 atoms. Too many.
        # 3x3x3 = 27 cells * 2 = 54. Too few.
        # Let's try a mixed approach or just a dense BCC grid and prune.
        a = 1.0 / 4.0
        for i in range(-4, 5):
            for j in range(-4, 5):
                for k in range(-4, 5):
                    # Corner
                    p1 = np.array([i*a, j*a, k*a])
                    # Center
                    p2 = p1 + np.array([a/2, a/2, a/2])
                    points.append(p1)
                    points.append(p2)
                    
    elif type == 1: # FCC
        a = 1.0 / 4.0
        for i in range(-4, 5):
            for j in range(-4, 5):
                for k in range(-4, 5):
                    # Corner
                    p1 = np.array([i*a, j*a, k*a])
                    # Face centers
                    p2 = p1 + np.array([a/2, a/2, 0])
                    p3 = p1 + np.array([a/2, 0, a/2])
                    p4 = p1 + np.array([0, a/2, a/2])
                    points.extend([p1, p2, p3, p4])

    points = np.array(points)
    # Filter to box
    mask = np.all(np.abs(points) <= 0.5, axis=1)
    points = points[mask]
    
    if len(points) > N:
        # Randomly select N
        idx = rng.choice(len(points), N, replace=False)
        points = points[idx]
    elif len(points) < N:
        # Pad with random points
        pad = rng.uniform(-0.4, 0.4, (N - len(points), 3))
        points = np.vstack([points, pad])
        
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_x = None
    best_score = 0.0
    
    # List of initialization strategies to try
    # We prioritize lattice structures known to be good packings
    inits = [
        (0, "BCC"),
        (1, "FCC"),
        (2, "Random"),
        (0, "BCC_Jittered"),
        (1, "FCC_Jittered"),
    ]
    
    # We will perform a "Simulated Annealing" style loop:
    # 1. Generate a candidate (lattice or random)
    # 2. Local optimize
    # 3. If improved, keep. If not, try a "kick" (random perturbation) on the best found so far.
    
    current_best = None
    current_score = 0.0
    
    # Phase 1: Try distinct lattice initializations
    for init_type, name in inits:
        if time.time() > deadline - 10:
            break
            
        if "Jittered" in name:
            base_type = 0 if "BCC" in name else 1
            x0 = _generate_lattice_init(base_type, rng)
            x0 = x0 + rng.normal(0, 0.05, x0.shape)
        else:
            x0 = _generate_lattice_init(init_type, rng)
            
        x_opt = _local_optimize(x0, deadline, max_iter=150)
        score = _calc_score(x_opt)
        
        if score > best_score:
            best_score = score
            best_x = x_opt.copy()
            current_best = x_opt.copy()
            current_score = score
            
    # Phase 2: Refine the best solution found so far with more iterations
    if best_x is not None and time.time() < deadline - 15:
        x_opt = _local_optimize(best_x, deadline, max_iter=300)
        score = _calc_score(x_opt)
        if score > best_score:
            best_score = score
            best_x = x_opt.copy()
            current_best = x_opt.copy()
            current_score = score

    # Phase 3: Random Restarts / Kicks if time remains
    # If we are significantly below the known best (0.11138), we might be stuck.
    # Try random kicks on the current best.
    if best_x is not None:
        while time.time() < deadline - 5:
            # Create a perturbed version of the best
            # Small kick to escape local minima
            kick_size = 0.02
            x_kick = best_x + rng.normal(0, kick_size, best_x.shape)
            x_kick = _project_to_box(x_kick)
            
            x_opt = _local_optimize(x_kick, deadline, max_iter=100)
            score = _calc_score(x_opt)
            
            if score > best_score:
                best_score = score
                best_x = x_opt.copy()
            else:
                # If we failed to improve, maybe try a larger kick or a completely new random start
                # To save time, we only do a few of these.
                if rng.random() < 0.1: # Occasionally try a fresh random start
                    x_rand = rng.uniform(-0.4, 0.4, (N, 3))
                    x_opt = _local_optimize(x_rand, deadline, max_iter=100)
                    score = _calc_score(x_opt)
                    if score > best_score:
                        best_score = score
                        best_x = x_opt.copy()

    # Final check
    if best_x is None:
        # Fallback: simple grid
        x_ticks = np.linspace(-0.4, 0.4, 5)
        grid = np.array([(x, y, z) for x in x_ticks for y in x_ticks for z in x_ticks])
        best_x = grid[:N]
        best_x = _project_to_box(best_x)
        
    return best_x
