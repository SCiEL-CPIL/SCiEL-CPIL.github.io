import numpy as np
import time
import warnings
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

warnings.filterwarnings('ignore')

N = 13

_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)

def _triangle_areas_vectorized(points):
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(points):
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    if len(points) is None or len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area

def _polar_to_cartesian(polar):
    """Convert (angles, radii) to Cartesian points.
    polar shape: (2N,) -> angles[0:N], radii[N:2N]
    """
    angles = polar[:N]
    radii = polar[N:]
    # Ensure radii are non-negative
    radii = np.abs(radii)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    points = np.column_stack([x, y])
    # Center the points to remove translation invariance
    points = points - np.mean(points, axis=0)
    return points

def _cartesian_to_polar(points):
    """Convert Cartesian points to polar representation (sorted by angle)."""
    points = points - np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1], points[:, 0])
    radii = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    # Sort by angle to maintain consistent ordering for optimization
    order = np.argsort(angles)
    angles_sorted = angles[order]
    radii_sorted = radii[order]
    # Normalize angles to [0, 2*pi)
    angles_sorted = angles_sorted % (2 * np.pi)
    return np.concatenate([angles_sorted, radii_sorted])

def _soft_min_score(points, beta):
    """
    Score using soft-min for the triangle areas.
    """
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    a_min = np.min(areas)
    if a_min < 1e-15:
        return 0.0
    shifted = -beta * (areas - a_min)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_area = a_min - log_sum / beta
    if soft_min_area < 1e-15:
        return 0.0
    return soft_min_area / hull_area

def _objective_soft_polar(polar, beta):
    """Objective for polar-parameterized optimization using soft-min."""
    points = _polar_to_cartesian(polar)
    score = _soft_min_score(points, beta)
    if score <= 0:
        return 1e10
    return -score

def _objective_hard_polar(polar):
    """Objective for polar-parameterized optimization using hard min."""
    points = _polar_to_cartesian(polar)
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score

def _coordinate_descent_polish(points, deadline, max_iter=2000):
    """
    Coordinate descent on the hard objective: move each point's x and y independently.
    This is very effective for fine-tuning after a coarse optimization.
    """
    best_pts = points.copy()
    best_s = _score(best_pts)
    
    # Adaptive step sizes
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            for coord in range(2):
                # Try decreasing step sizes
                for delta in step_sizes:
                    for sign in [1, -1]:
                        trial = best_pts.copy()
                        trial[i, coord] += sign * delta
                        trial_s = _score(trial)
                        if trial_s > best_s + 1e-12:
                            best_pts = trial
                            best_s = trial_s
                            improved = True
                            break
                    if improved:
                        break
        if not improved:
            break
    
    return best_pts, best_s

def _local_polish(polar_init, deadline):
    """
    Multi-phase local optimization from a polar starting point.
    1. L-BFGS-B on soft-min objective (smooth landscape).
    2. Nelder-Mead on hard objective (Cartesian) for robustness.
    3. Coordinate Descent on hard objective (Cartesian) for precision.
    """
    best_polar = polar_init.copy()
    best_points = _polar_to_cartesian(best_polar)
    best_score = _score(best_points)
    
    # Phase 1: Gradient-based on soft objective
    if time.time() < deadline - 15:
        try:
            # Use a moderate beta for smoothness
            beta = 100.0
            res = minimize(
                _objective_soft_polar,
                best_polar,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            if res.x is not None:
                new_points = _polar_to_cartesian(res.x)
                new_score = _score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points
                    best_polar = res.x.copy()
        except Exception:
            pass

    # Phase 2: Nelder-Mead on hard objective (Cartesian)
    # Convert best points to flat array for NM
    x_cart = best_points.flatten()
    if time.time() < deadline - 5:
        try:
            def obj_cart(x_flat):
                pts = x_flat.reshape(N, 2)
                s = _score(pts)
                return -s if s > 0 else 1e10
            
            res_nm = minimize(
                obj_cart,
                x_cart,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            if res_nm.x is not None:
                new_points = res_nm.x.reshape(N, 2)
                new_score = _score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points
                    best_polar = _cartesian_to_polar(new_points)
        except Exception:
            pass

    # Phase 3: Coordinate Descent Polish
    if time.time() < deadline - 2:
        try:
            cd_pts, cd_s = _coordinate_descent_polish(best_points, deadline)
            if cd_s > best_score:
                best_score = cd_s
                best_points = cd_pts
                best_polar = _cartesian_to_polar(best_points)
        except Exception:
            pass

    return best_points, best_score

def _simulated_annealing_polar(polar_init, deadline):
    """
    Simulated Annealing in Polar space.
    """
    current_polar = polar_init.copy()
    current_points = _polar_to_cartesian(current_polar)
    current_score = _score(current_points)
    
    best_polar = current_polar.copy()
    best_points = current_points.copy()
    best_score = current_score
    
    # Time budget
    time_budget = deadline - 3.0
    total_time = max(0.5, time_budget - time.time())
    
    # Estimate iterations
    # Each iteration involves scoring and potentially updating.
    # Let's assume ~0.5ms per iteration on average for safety.
    est_iters = int(total_time * 1000 / 0.5)
    est_iters = max(1000, min(est_iters, 50000))
    
    T0 = 0.05
    T_end = 0.0001
    alpha = (T_end / T0) ** (1.0 / max(1, est_iters - 1))
    
    T = T0
    rng = np.random.RandomState(42)
    
    for it in range(est_iters):
        if time.time() > deadline:
            break
            
        # Propose move: perturb angles and radii
        perturbation_scale = 0.1 * T / T0 + 0.01
        new_polar = current_polar + rng.normal(0, perturbation_scale, size=2*N)
        
        # Wrap angles
        new_polar[:N] = new_polar[:N] % (2 * np.pi)
        
        new_points = _polar_to_cartesian(new_polar)
        new_score = _score(new_points)
        
        if new_score > current_score:
            current_polar = new_polar
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_polar = current_polar.copy()
                best_points = current_points.copy()
        else:
            # Accept with probability exp((new_score - current_score) / T)
            # Note: We are maximizing, so delta = new - current (negative if worse)
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_polar = new_polar
                    current_points = new_points
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_polar = current_polar.copy()
                        best_points = current_points.copy()
        
        T *= alpha
        
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    best_points = None
    best_score = 0.0

    def _update_best(points, score=None):
        nonlocal best_points, best_score
        if points is None or len(points) != N:
            return
        if score is None:
            score = _score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()

    # 1. Generate diverse initial configurations in Polar space
    configs_polar = []
    
    # Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii = np.ones(N)
    configs_polar.append(np.concatenate([angles, radii]))
    
    # Ring configurations: 7 outer, 6 inner
    for r_in in [0.5, 0.6, 0.7, 0.8]:
        for delta in [0, np.pi/7, np.pi/6]:
            angles_out = np.linspace(0, 2 * np.pi, 7, endpoint=False)
            angles_in = np.linspace(delta, 2 * np.pi + delta, 6, endpoint=False)
            angles = np.concatenate([angles_out, angles_in])
            radii = np.concatenate([np.ones(7), np.full(6, r_in)])
            # Sort by angle for consistency
            order = np.argsort(angles)
            configs_polar.append(np.concatenate([angles[order] % (2*np.pi), radii[order]]))
            
    # Ring configurations: 6 outer, 7 inner
    for r_in in [0.5, 0.6, 0.7, 0.8]:
        for delta in [0, np.pi/6, np.pi/7]:
            angles_out = np.linspace(0, 2 * np.pi, 6, endpoint=False)
            angles_in = np.linspace(delta, 2 * np.pi + delta, 7, endpoint=False)
            angles = np.concatenate([angles_out, angles_in])
            radii = np.concatenate([np.ones(6), np.full(7, r_in)])
            order = np.argsort(angles)
            configs_polar.append(np.concatenate([angles[order] % (2*np.pi), radii[order]]))
            
    # Center + Rings
    center = np.array([0.0, 0.0])
    for r_in in [0.5, 0.7]:
        for r_out in [1.0, 1.2]:
            for delta_in in [0, np.pi/6]:
                for delta_out in [0, np.pi/6]:
                    pts_in = np.column_stack([r_in * np.cos(np.linspace(delta_in, 2*np.pi+delta_in, 6, endpoint=False)), 
                                              r_in * np.sin(np.linspace(delta_in, 2*np.pi+delta_in, 6, endpoint=False))])
                    pts_out = np.column_stack([r_out * np.cos(np.linspace(delta_out, 2*np.pi+delta_out, 6, endpoint=False)), 
                                               r_out * np.sin(np.linspace(delta_out, 2*np.pi+delta_out, 6, endpoint=False))])
                    pts = np.vstack([center, pts_in, pts_out])
                    polar = _cartesian_to_polar(pts)
                    configs_polar.append(polar)

    # Random inits
    rng = np.random.RandomState(42)
    for _ in range(10):
        angles = rng.uniform(0, 2*np.pi, N)
        radii = rng.uniform(0.5, 1.5, N)
        order = np.argsort(angles)
        configs_polar.append(np.concatenate([angles[order], radii[order]]))
        
    # Sort initial configs by score to prioritize good starting points
    scored_configs = []
    for cfg in configs_polar:
        pts = _polar_to_cartesian(cfg)
        s = _score(pts)
        scored_configs.append((s, cfg))
    scored_configs.sort(key=lambda x: x[0], reverse=True)
    
    # 2. Refine top candidates
    # We have ~115s. 
    # Strategy: 
    # - Try SA on the best 3 configs.
    # - Then Local Polish on the best 5 configs.
    
    top_k_sa = 3
    top_k_polish = 5
    
    # First pass: Simulated Annealing on top candidates
    for i in range(min(top_k_sa, len(scored_configs))):
        if time.time() > deadline - 30:
            break
        init_score, init_polar = scored_configs[i]
        # Run SA
        sa_pts, sa_score = _simulated_annealing_polar(init_polar, deadline - 10)
        _update_best(sa_pts, sa_score)
        print(f"SA Config {i} (init {init_score:.6f}) -> {sa_score:.6f}", flush=True)
        
    # Second pass: Local Polish on top candidates (including those improved by SA if they are in the list? 
    # Actually, let's just polish the best distinct points found so far and the top initial ones)
    
    # Create a list of unique-ish starting points for polishing
    polish_candidates = []
    # Add top initial configs
    for i in range(min(top_k_polish, len(scored_configs))):
        polish_candidates.append(scored_configs[i][1])
        
    # Add the best point found so far (converted to polar)
    if best_points is not None:
        best_polar = _cartesian_to_polar(best_points)
        polish_candidates.append(best_polar)
        
    # Deduplicate roughly? Not strictly necessary, but good for time.
    # We will just iterate and keep the best.
    
    for i, polar_init in enumerate(polish_candidates):
        if time.time() > deadline - 10:
            break
        # Run Local Polish
        pol_pts, pol_score = _local_polish(polar_init, deadline - 5)
        _update_best(pol_pts, pol_score)
        print(f"Polish Candidate {i} -> {pol_score:.6f}", flush=True)
        
    # Final Polish on the absolute best
    if best_points is not None and time.time() < deadline - 2:
        best_polar_final = _cartesian_to_polar(best_points)
        final_pts, final_score = _local_polish(best_polar_final, deadline)
        _update_best(final_pts, final_score)
        print(f"Final Polish -> {final_score:.6f}", flush=True)

    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points

if __name__ == "__main__":
    pts = construct()
    print("Score:", _score(pts))
