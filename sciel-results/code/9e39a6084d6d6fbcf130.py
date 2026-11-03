import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
T0 = time.time()
DEADLINE = 145.0

def _elapsed():
    return time.time() - T0

def _outer_half_width(params):
    """Smooth outer hexagon half-width D (distance from center to each side) via log-sum-exp."""
    cx, cy, beta = params[-3], params[-2], params[-1]
    pts = params[:-3].reshape(N, 2)
    normals = np.array([(math.cos(beta + math.pi/6 + k*math.pi/3),
                         math.sin(beta + math.pi/6 + k*math.pi/3)) for k in range(6)])
    centered = pts - np.array([cx, cy])
    projs = centered @ normals.T  # (N, 6)
    all_p = np.concatenate([projs.ravel(), (-projs).ravel()])
    t = 0.01
    m = np.max(all_p)
    if m > 1000:
        return 1e6
    return t * (m + math.log(np.sum(np.exp((all_p - m) / t))))

def _outer_half_width_exact(params):
    cx, cy, beta = params[-3], params[-2], params[-1]
    pts = params[:-3].reshape(N, 2)
    normals = np.array([(math.cos(beta + math.pi/6 + k*math.pi/3),
                         math.sin(beta + math.pi/6 + k*math.pi/3)) for k in range(6)])
    centered = pts - np.array([cx, cy])
    projs = centered @ normals.T
    return float(np.max(np.abs(projs)))

def _soft_overlap_penalty(params):
    """Sum of soft-pairwise-overlap penalties. 0 when all disjoint."""
    pts = params[:-3].reshape(N, 2)
    angs = params[-3]  # all same rotation (we optimize one global beta)
    # Actually each hex has its own angle in params; but for simplicity use global
    # Let's support per-hex angles: params layout = [x0,y0,a0, x1,y1,a1, ..., cx,cy,beta]
    # Re-layout: 3*N + 3
    hexes = params.reshape(N, 3)  # (x, y, angle_deg)
    cx, cy, beta = params[-3], params[-2], params[-1]
    # Wait, this layout doesn't work with 3*N+3. Let me use:
    # params = [x0,y0,a0, x1,y1,a1, ..., x10,y10,a10, cx,cy,beta]
    # That's 33 params. hexes = params[:33].reshape(11,3), outer = params[33:36]
    # But 3*N+3 = 36. Let me redo.
    
    # Actually 3*11 + 3 = 36 params.
    # hexes = params[:33].reshape(11,3), cx,cy,beta = params[33:36]
    pass

def _soft_overlap_penalty_v2(params):
    """params: [x0,y0,a0, x1,y1,a1, ..., x10,y10,a10, cx,cy,beta] (36 values)"""
    hexes = params[:3*N].reshape(N, 3)  # (x, y, angle_deg)
    cx, cy, beta = params[3*N], params[3*N+1], params[3*N+2]
    
    # Compute pairwise distances and compare to 2*apothem (touching distance for same-orientation)
    # For different orientations, the minimum separation is still >= 2*apothem
    # (since each hex has inradius = apothem, so two hexes are disjoint iff center distance > 2*apothem
    #  is NOT quite right for rotated hexes, but it's a good lower bound for the penalty)
    # Actually for regular hexagons, the minimum distance between centers for non-overlap
    # depends on relative orientation. For same orientation, it's 2*apothem = sqrt(3).
    # For 30-degree relative rotation, it can be less.
    # Use a conservative threshold: 2*apothem * 0.95 to allow slight "overlap" in penalty
    # but the smooth penalty will push them apart.
    
    dists = []
    for i in range(N):
        for j in range(i+1, N):
            dx = hexes[i,0] - hexes[j,0]
            dy = hexes[i,1] - hexes[j,1]
            d = math.sqrt(dx*dx + dy*dy)
            dists.append(d)
    dists = np.array(dists)
    
    # Target minimum distance: 2*APOTHEM = sqrt(3) ~ 1.732
    # For rotated hexes, the actual minimum can be smaller, but sqrt(3) is safe
    min_d = 2 * APOTHEM
    t = 0.1
    violations = np.maximum(0, min_d - dists)
    # Soft penalty: sum of exp(violation/t) to make it smooth
    if np.max(violations) > 10:
        return 1e6
    return float(np.sum(np.exp(violations / t)))

def _objective(params):
    """Total objective: outer_side_length + lambda * overlap_penalty"""
    D = _outer_half_width(params)
    S = (D + APOTHEM) / APOTHEM
    pen = _soft_overlap_penalty_v2(params)
    lam = 50.0
    return S + lam * pen

def _objective_exact(params):
    D = _outer_half_width_exact(params)
    S = (D + APOTHEM) / APOTHEM
    # Check actual overlap
    hexes = params[:3*N].reshape(N, 3)
    for i in range(N):
        for j in range(i+1, N):
            d = math.sqrt((hexes[i,0]-hexes[j,0])**2 + (hexes[i,1]-hexes[j,1])**2)
            if d < 2*APOTHEM - 1e-9:
                return S + 10.0
    return S

def construct():
    if _elapsed() > DEADLINE:
        # Fallback: all hexes at origin (degenerate but valid format)
        inner = [[0.0, 0.0, 0.0]] * N
        return inner, [0.0, 0.0], 1.0, 0.0
    
    np.random.seed(42)
    
    # Initial guess: hexagonal lattice arrangement
    # Place 11 hexes in a compact cluster
    hex_positions = []
    # Center + 6 neighbors + 4 more
    hex_positions.append((0, 0))
    for k in range(6):
        ang = k * math.pi / 3
        hex_positions.append((2*APOTHEM*math.cos(ang), 2*APOTHEM*math.sin(ang)))
    # 4 more at second ring
    for k in range(4):
        ang = k * math.pi / 3 + math.pi/6
        hex_positions.append((3*APOTHEM*math.cos(ang), 3*APOTHEM*math.sin(ang)))
    
    # Build initial params
    params0 = np.zeros(3*N + 3)
    for i, (x, y) in enumerate(hex_positions):
        params0[3*i] = x
        params0[3*i+1] = y
        params0[3*i+2] = 0.0  # angle
    # Center at centroid
    cx0 = np.mean([p[0] for p in hex_positions])
    cy0 = np.mean([p[1] for p in hex_positions])
    params0[3*N] = cx0
    params0[3*N+1] = cy0
    params0[3*N+2] = 0.0  # beta
    
    best_params = params0.copy()
    best_S = _objective_exact(params0)
    
    # Phase 1: Simulated Annealing
    t_start = _elapsed()
    sa_time_budget = min(40.0, DEADLINE - t_start - 20.0)
    if sa_time_budget > 5.0:
        current = params0.copy()
        current_obj = _objective(current)
        T = 0.5
        T_end = 0.001
        step = 0.15
        n_steps = 0
        max_steps = 50000
        
        while n_steps < max_steps and _elapsed() < t_start + sa_time_budget:
            n_steps += 1
            # Pick a random parameter to perturb
            idx = np.random.randint(0, len(current))
            delta = np.random.normal(0, step)
            new = current.copy()
            new[idx] += delta
            
            # Keep angles in [0, 60)
            if idx < 3*N:
                new[idx] = new[idx] % 60.0
            if idx == 3*N + 2:
                new[idx] = new[idx] % 60.0
            
            new_obj = _objective(new)
            diff = new_obj - current_obj
            if diff < 0 or np.random.random() < math.exp(-diff / T):
                current = new
                current_obj = new_obj
                # Update best
                S_exact = _objective_exact(current)
                if S_exact < best_S:
                    best_S = S_exact
                    best_params = current.copy()
            
            # Cool
            T = max(T_end, T * 0.9997)
            step = max(0.01, step * 0.9999)
    
    # Phase 2: Nelder-Mead refinement from multiple starts
    t_start2 = _elapsed()
    nm_time_budget = DEADLINE - t_start2 - 5.0
    if nm_time_budget > 3.0:
        starts = [best_params.copy()]
        # Add perturbed starts
        for trial in range(10):
            p = best_params.copy()
            p[:3*N] += np.random.normal(0, 0.05, 3*N)
            p[3*N+2] = (p[3*N+2] + np.random.uniform(-5, 5)) % 60
            starts.append(p)
        
        per_start_time = nm_time_budget / len(starts)
        for i, start in enumerate(starts):
            if _elapsed() > DEADLINE - 3.0:
                break
            try:
                res = minimize(_objective, start, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
                if res.fun < 1e5:
                    # Evaluate exact
                    S_exact = _objective_exact(res.x)
                    if S_exact < best_S:
                        best_S = S_exact
                        best_params = res.x.copy()
            except Exception:
                pass
    
    # Phase 3: Fine polish
    t_start3 = _elapsed()
    if _elapsed() < DEADLINE - 3.0:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                         options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < 1e5:
                S_exact = _objective_exact(res.x)
                if S_exact < best_S:
                    best_S = S_exact
                    best_params = res.x.copy()
        except Exception:
            pass
    
    # Extract final result
    hexes = best_params[:3*N].reshape(N, 3)
    cx, cy, beta = best_params[3*N], best_params[3*N+1], best_params[3*N+2]
    
    # Compute exact S
    D = _outer_half_width_exact(best_params)
    S = (D + APOTHEM) / APOTHEM
    
    # Build inner list
    inner = []
    for i in range(N):
        x, y, ang = hexes[i]
        inner.append([float(x), float(y), float(ang % 60.0)])
    
    center = [float(cx), float(cy)]
    side = float(S)
    rotation = float(beta % 60.0)
    
    return inner, center, side, rotation
