# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

# Precompute sofa points at module level
_SOFA_U = np.arange(-3.0, 1.5, 0.01)
_SOFA_V = np.arange(-1.5, 1.5, 0.01)
_SOFA_X, _SOFA_Y = np.meshgrid(_SOFA_U, _SOFA_V, indexing='ij')
_SOFA_PTS = np.stack([_SOFA_X.ravel(), _SOFA_Y.ravel()], axis=1)  # shape (N, 2)
_N_SOFa = _SOFA_PTS.shape[0]

# Number of waypoints to optimize
N_WAYPOINTS = 10

def _generate_placements(waypoints):
    """
    Generate the full list of placements including start (0,0,0) and end (-2,0,90).
    Each placement is (x, y, a_degrees). Linearly interpolate between consecutive
    waypoints with m = max(1, ceil(|a2 - a1|)) steps.
    """
    P0 = np.array([0.0, 0.0, 0.0])
    PN = np.array([-2.0, 0.0, 90.0])
    full = [P0] + [np.array(p, dtype=np.float64) for p in waypoints] + [PN]
    
    placements = [full[0].copy()]
    for i in range(len(full) - 1):
        p1 = full[i]
        p2 = full[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da - 1e-9)))
        for k in range(1, m + 1):
            t = k / m
            placements.append(p1 + t * (p2 - p1))
    return placements

def _point_in_base(u, v):
    """Check if point (u, v) is in the base L-shaped corridor."""
    in_h = (u >= -1e-12) & (u <= 1 + 1e-12) & (v >= -3 - 1e-12) & (v <= 1 + 1e-12)
    in_v = (u >= -3 - 1e-12) & (u <= 1 + 1e-12) & (v >= -1e-12) & (v <= 1 + 1e-12)
    return in_h | in_v

def _compute_sofa_mask(waypoints):
    """
    Compute which sofa points are inside ALL placements along the path.
    Returns a boolean mask of shape (N_SOFa,).
    """
    placements = _generate_placements(waypoints)
    
    inside_all = np.ones(_N_SOFa, dtype=bool)
    
    for (x, y, a_deg) in placements:
        a_rad = np.radians(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        dx = _SOFA_PTS[:, 0] - x
        dy = _SOFA_PTS[:, 1] - y
        
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        
        in_base = _point_in_base(u, v)
        inside_all &= in_base
        
        if not inside_all.any():
            break
    
    return inside_all

def _compute_area(waypoints):
    """Compute the area of the sofa for a given path."""
    mask = _compute_sofa_mask(waypoints)
    return mask.sum() * 0.0001

def _smooth_inside_value(waypoints, beta=50.0):
    """
    Compute a smooth surrogate for the sofa area using soft-min.
    For each sofa point, compute a smooth measure of how well it fits in all placements.
    Uses log-sum-exp to create a smooth approximation of the min over placements.
    """
    placements = _generate_placements(waypoints)
    
    # For each placement, compute a signed distance-like value for all sofa points
    # Positive = inside, negative = outside (with some margin)
    # We'll use a smooth indicator: 1 if inside, 0 if outside, smoothed
    
    n_pts = _N_SOFa
    # Accumulate log-softmin across placements for each point
    # softmin over placements of (inside_indicator) ≈ 1 if inside all, 0 if outside any
    
    # For efficiency, we track the minimum "inside-ness" per point across placements
    # Using a smooth version: -log(sum(exp(-beta * s_i))) / beta where s_i is inside score
    
    # Simpler approach: for each point, compute product of smooth indicators
    # which is equivalent to sum of log indicators, then smooth via sigmoid
    
    # Most efficient: track the worst (minimum) inside-score per point, smoothed
    # We'll compute a per-point score that's 1 if inside all, 0 otherwise, but smooth
    
    # Approach: for each placement, compute a smooth inside indicator in [0,1]
    # Then for each point, take the min across placements (smoothed)
    
    # To keep it fast, use a simpler smooth surrogate:
    # For each point, count how many placements it's inside of (smoothed)
    # Area proxy = sum over points of smooth_min(inside_across_placements)
    
    # Even simpler and faster: use the hard mask but with a penalty for near-boundary points
    # This gives a smooth-ish objective
    
    # Let's use: for each point, compute the minimum distance to boundary across all placements
    # If min distance > 0, point is inside. Smooth this with a sigmoid.
    
    # Actually, the cleanest smooth surrogate:
    # s_p = softmin over placements of (1 if inside else 0)
    # ≈ 1 / (1 + exp(beta * (1 - count_inside/n_placements)))  -- not quite right
    
    # Let me use a different approach: 
    # For each point p, define f_p = min over placements of g(p, placement)
    # where g = 1 if inside, 0 if outside
    # Smooth f_p using: -log(1/n * sum(exp(-beta * g))) / beta
    # But g is binary, so this becomes:
    # If k out of M placements have g=1: -log(k/M) / beta... no that's not right either
    
    # Simplest effective smooth surrogate:
    # For each point, compute the fraction of placements it's inside of
    # Then use a smooth step: sigma(beta * (frac - 0.5)) to get a value in (0,1)
    # Sum over all points
    
    # But this changes the objective significantly. Let me stick closer to the original.
    
    # Best approach for this problem: use the hard count but add a small smooth penalty
    # for points that are close to the boundary. This makes the objective less jagged.
    
    # Actually, let me just compute the exact area but also return a smooth version
    # for the optimizer to use. The final answer uses exact area.
    
    # For the smooth version: for each point, compute min margin across placements
    # margin > 0 means inside. Smooth with sigmoid.
    
    # This is getting complex. Let me use a simpler strategy:
    # Compute exact area, and for optimization, use a smoothed version where
    # we add a small Gaussian blur to the inside/outside indicator.
    
    # Simplest effective approach: just use the exact area computation but
    # make the optimizer smarter with more restarts and better initialization.
    
    # I'll return the exact area for now, and improve the search strategy instead.
    return _compute_area(waypoints)

def _neg_area(params, deadline):
    """Negative area for minimization."""
    if time.time() > deadline:
        return 1e6
    waypoints = params.reshape(N_WAYPOINTS, 3)
    area = _compute_area(waypoints)
    return -area

def _generate_initial_guesses():
    """
    Generate diverse initial guesses.
    """
    guesses = []
    
    # Parameterize the path by angles, with x,y following a smooth curve
    # The path goes from (0,0,0) to (-2,0,90)
    
    # Base template: angles roughly linear from ~10 to ~85
    n = N_WAYPOINTS
    
    # Template 1: Standard arc
    angles = np.linspace(10, 85, n)
    # x goes from ~-0.1 to ~-1.5, y peaks around 0.3-0.5
    t = np.linspace(0, 1, n)
    x = -0.1 - 1.4 * t
    y = 0.3 * np.sin(np.pi * t) + 0.05
    
    g1 = np.column_stack((x, y, angles))
    guesses.append(g1)
    
    # Template 2: Higher arc
    g2 = g1.copy()
    g2[:, 1] += 0.15
    guesses.append(g2)
    
    # Template 3: Lower arc  
    g3 = g1.copy()
    g3[:, 1] -= 0.1
    guesses.append(g3)
    
    # Template 4: Shifted left
    g4 = g1.copy()
    g4[:, 0] -= 0.15
    guesses.append(g4)
    
    # Template 5: Different angle distribution (more gradual at ends)
    angles5 = np.linspace(8, 87, n)
    angles5 = 45 - 42 * np.sin(np.pi * (angles5 - 45) / 42)  # S-curve
    angles5 = np.clip(angles5, 5, 88)
    # Make sure it's monotonic
    for i in range(1, n):
        if angles5[i] < angles5[i-1]:
            angles5[i] = angles5[i-1] + 1
    g5 = np.column_stack((x, y, angles5))
    guesses.append(g5)
    
    # Random perturbations of best templates
    rng = np.random.RandomState(42)
    for base_idx in [0, 1, 2]:
        for _ in range(8):
            g = guesses[base_idx].copy()
            g += rng.normal(0, 0.08, g.shape)
            # Ensure monotonic angles
            g[:, 2] = np.maximum.accumulate(g[:, 2])
            g[:, 2] = np.clip(g[:, 2], 2, 88)
            guesses.append(g)
    
    return guesses

def moving_sofa_2d() -> np.ndarray:
    """
    Find the path that maximizes the sofa area through the L-shaped corridor.
    """
    start_time = time.time()
    deadline = start_time + 570  # Safety margin
    
    initial_guesses = _generate_initial_guesses()
    
    best_params = None
    best_area = -1.0
    
    # Phase 1: Evaluate all initial guesses
    scored_guesses = []
    for g in initial_guesses:
        if time.time() > deadline - 180:
            break
        area = _compute_area(g)
        scored_guesses.append((area, g))
        if area > best_area:
            best_area = area
            best_params = g.ravel().copy()
    
    if not scored_guesses or best_params is None:
        return np.array([
            [-0.1, 0.1, 10.0],
            [-0.3, 0.3, 20.0],
            [-0.5, 0.4, 30.0],
            [-0.7, 0.4, 40.0],
            [-0.9, 0.35, 50.0],
            [-1.1, 0.3, 60.0],
            [-1.3, 0.2, 70.0],
            [-1.5, 0.1, 80.0],
            [-1.7, 0.05, 85.0],
            [-1.9, 0.0, 88.0]
        ])
    
    scored_guesses.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 2: Nelder-Mead on top candidates
    top_n = min(6, len(scored_guesses))
    for i in range(top_n):
        if time.time() > deadline - 90:
            break
        g = scored_guesses[i][1]
        
        try:
            result = minimize(
                _neg_area,
                g.ravel(),
                args=(deadline,),
                method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'maxfev': 200,
                    'xatol': 1e-3,
                    'fatol': 1e-4,
                    'adaptive': True,
                }
            )
            
            cand_area = _compute_area(result.x.reshape(N_WAYPOINTS, 3))
            if cand_area > best_area:
                best_area = cand_area
                best_params = result.x.ravel().copy()
                
        except Exception:
            continue
    
    # Phase 3: Final refinement
    if best_params is not None and time.time() < deadline - 30:
        try:
            result = minimize(
                _neg_area,
                best_params,
                args=(deadline,),
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'maxfev': 300,
                    'xatol': 1e-4,
                    'fatol': 1e-5,
                    'adaptive': True,
                }
            )
            cand_area = _compute_area(result.x.reshape(N_WAYPOINTS, 3))
            if cand_area > best_area:
                best_area = cand_area
                best_params = result.x.ravel().copy()
        except Exception:
            pass
    
    # Phase 4: Local perturbation search on best
    if best_params is not None and time.time() < deadline - 15:
        best_wpts = best_params.reshape(N_WAYPOINTS, 3)
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            # Try small random perturbations
            for scale in [0.05, 0.02, 0.01]:
                if time.time() > deadline - 10:
                    break
                for trial in range(3):
                    if time.time() > deadline - 10:
                        break
                    perturbed = best_wpts + np.random.normal(0, scale, best_wpts.shape)
                    # Ensure monotonic angles
                    perturbed[:, 2] = np.maximum.accumulate(perturbed[:, 2])
                    perturbed[:, 2] = np.clip(perturbed[:, 2], 1, 89)
                    p_area = _compute_area(perturbed)
                    if p_area > best_area:
                        best_area = p_area
                        best_params = perturbed.ravel().copy()
                        best_wpts = perturbed
    
    if best_params is None:
        return scored_guesses[0][1]
    
    return best_params.reshape(N_WAYPOINTS, 3)

# EVOLVE-BLOCK-END
