import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 148.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

# 6 unit-circle normals (hexagon edge outward normals)
NORMALS = np.array([
    [math.cos(math.pi/6 + k*math.pi/3), math.sin(math.pi/6 + k*math.pi/3)]
    for k in range(6)
])

# 12 unit-circle normals (hexagon edge normals + vertex normals)
NORMALS12 = np.array([
    [math.cos(math.pi/6 + k*math.pi/6), math.sin(math.pi/6 + k*math.pi/6)]
    for k in range(12)
])

def axial_to_cartesian(axial_list):
    pts = []
    for q, r in axial_list:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    return np.array(pts)

def smooth_max(vals, t):
    """Smooth approximation of max(vals) using log-sum-exp."""
    if len(vals) == 0:
        return 0.0
    m = np.max(vals)
    if m > 1000:
        return m
    shifted = (vals - m) / t
    shifted = np.clip(shifted, -50, 50)
    return m + t * np.log(np.sum(np.exp(shifted)))

def objective(params, temperature=0.05):
    """
    params: [cx, cy, beta, s, x1, y1, ..., x11, y11]
    Returns outer side length + penalty for overlaps and containment violations.
    """
    cx, cy = params[0], params[1]
    beta = params[2]
    s = params[3]
    if s < 0.5:
        return 1e10
    
    centers = params[4:].reshape(N, 2)
    
    # Outer hexagon normals (rotated by beta)
    cos_b = math.cos(beta)
    sin_b = math.sin(beta)
    R = np.array([[cos_b, -sin_b], [sin_b, cos_b]])
    n_outer = NORMALS @ R.T  # 6 x 2
    
    # Containment: each hexagon center must satisfy n_outer @ (center - outer_center) <= D_outer - D_inner
    # D_outer = s * APOTHEM (apothem of outer hexagon)
    # D_inner = APOTHEM (apothem of unit hexagon)
    D_outer = s * APOTHEM
    D_inner = APOTHEM
    margin = D_outer - D_inner
    
    if margin < 0:
        return 1e10
    
    # Containment violations (smooth)
    rel = centers - np.array([cx, cy])
    projs = rel @ n_outer.T  # N x 6
    violations = np.max(projs, axis=1) - margin  # N
    contain_pen = smooth_max(violations, temperature)
    if contain_pen < 0:
        contain_pen = 0.0
    
    # Overlap: for each pair, check 12 normals
    # Distance between centers projected on normals >= 2 * APOTHEM
    overlap_pen = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = centers[i] - centers[j]
            # Project on all 12 normals
            projs12 = np.abs(NORMALS12 @ d)
            # For no overlap: min(projs12) >= 2*APOTHEM
            # Violation: 2*APOTHEM - min(projs12)
            viol = 2*APOTHEM - np.min(projs12)
            if viol > 0:
                overlap_pen += viol * viol  # quadratic penalty
    
    overlap_pen = math.sqrt(overlap_pen) if overlap_pen > 0 else 0.0
    
    # Total: minimize s + penalties
    total = s + 10.0 * contain_pen + 50.0 * overlap_pen
    return total

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Hexagonal ring + center (axial)
    axial_ring = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    initial_configs.append(axial_ring)
    
    # 2. Compact cluster
    axial_compact = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (1, -1), (-1, -1)
    ]
    initial_configs.append(axial_compact)
    
    # 3. Row-based
    axial_rows = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    initial_configs.append(axial_rows)
    
    # 4. Diamond
    axial_diamond = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (-1, 1), (1, -1), (-1, -1),
        (2, 0), (-2, 0)
    ]
    initial_configs.append(axial_diamond)
    
    # 5. Random perturbation of ring
    rng = np.random.default_rng(42)
    ring_pts = axial_to_cartesian(axial_ring)
    for trial in range(5):
        perturbed = ring_pts + rng.normal(0, 0.3, size=(N, 2))
        initial_configs.append(perturbed)
    
    # 6. More random starts
    for trial in range(10):
        random_pts = rng.uniform(-3, 3, size=(N, 2))
        initial_configs.append(random_pts)
    
    best_result = None
    best_score = float('inf')
    
    for config in initial_configs:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        if isinstance(config, list) and len(config) > 0 and isinstance(config[0], tuple):
            pts = axial_to_cartesian(config)
        else:
            pts = np.array(config)
        
        # Estimate initial outer hexagon
        cx0 = float(np.mean(pts[:, 0]))
        cy0 = float(np.mean(pts[:, 1]))
        
        # Find min enclosing hexagon for initial config (coarse)
        best_s_init = 1e10
        best_beta_init = 0.0
        for beta_i in range(13):
            beta_test = beta_i * math.pi / 78.0
            cos_b = math.cos(beta_test)
            sin_b = math.sin(beta_test)
            R = np.array([[cos_b, -sin_b], [sin_b, cos_b]])
            n_test = NORMALS @ R.T
            rel = pts - np.array([cx0, cy0])
            projs = rel @ n_test.T
            D = np.max(np.abs(projs))
            s_test = (D + APOTHEM) / APOTHEM
            if s_test < best_s_init:
                best_s_init = s_test
                best_beta_init = beta_test
        
        # Initial params
        params0 = np.concatenate([[cx0, cy0, best_beta_init, best_s_init], pts.ravel()])
        
        # Multi-temperature optimization
        temperatures = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
        
        current_params = params0.copy()
        current_score = objective(current_params, temperatures[0])
        
        for temp in temperatures:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            # Nelder-Mead optimization
            try:
                res = minimize(
                    lambda p: objective(p, temp),
                    current_params,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 5000,
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                
                if res.fun < current_score:
                    current_params = res.x
                    current_score = res.fun
            except Exception:
                pass
            
            # Local refinement with smaller perturbations
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            for _ in range(3):
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                perturbed_params = current_params + np.random.normal(0, 0.01, size=len(current_params))
                try:
                    res2 = minimize(
                        lambda p: objective(p, temp),
                        perturbed_params,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 2000,
                            'xatol': 1e-12,
                            'fatol': 1e-12,
                            'adaptive': True
                        }
                    )
                    if res2.fun < current_score:
                        current_params = res2.x
                        current_score = res2.fun
                except Exception:
                    pass
        
        # Evaluate final result
        final_score = objective(current_params, 0.001)
        
        # Extract solution
        cx, cy = current_params[0], current_params[1]
        beta = current_params[2]
        s = current_params[3]
        centers = current_params[4:].reshape(N, 2)
        
        # Verify and compute exact outer side
        cos_b = math.cos(beta)
        sin_b = math.sin(beta)
        R = np.array([[cos_b, -sin_b], [sin_b, cos_b]])
        n_outer = NORMALS @ R.T
        
        rel = centers - np.array([cx, cy])
        projs = rel @ n_outer.T
        D = np.max(np.abs(projs))
        s_exact = (D + APOTHEM) / APOTHEM
        
        # Check overlaps
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                d = centers[i] - centers[j]
                projs12 = np.abs(NORMALS12 @ d)
                if np.min(projs12) < 2*APOTHEM - 1e-6:
                    valid = False
                    break
            if not valid:
                break
        
        if valid and s_exact < best_score:
            best_score = s_exact
            best_result = (centers, cx, cy, beta, s_exact)
    
    # Fallback: use best axial configuration
    if best_result is None:
        axial_fallback = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (-2, 0), (0, 2), (0, -2)
        ]
        pts = axial_to_cartesian(axial_fallback)
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        
        best_s = 1e10
        best_beta = 0.0
        for beta_i in range(61):
            beta_test = beta_i * (math.pi/6) / 60
            cos_b = math.cos(beta_test)
            sin_b = math.sin(beta_test)
            R = np.array([[cos_b, -sin_b], [sin_b, cos_b]])
            n_test = NORMALS @ R.T
            rel = pts - np.array([cx, cy])
            projs = rel @ n_test.T
            D = np.max(np.abs(projs))
            s_test = (D + APOTHEM) / APOTHEM
            if s_test < best_s:
                best_s = s_test
                best_beta = beta_test
        
        best_result = (pts, cx, cy, best_beta, best_s)
    
    centers, cx, cy, beta, s = best_result
    
    # Small safety scaling
    SCALE = 1.0 + 1e-12
    centers_scaled = cx + (centers - cx) * SCALE
    
    # Recompute exact outer side
    cos_b = math.cos(beta)
    sin_b = math.sin(beta)
    R = np.array([[cos_b, -sin_b], [sin_b, cos_b]])
    n_outer = NORMALS @ R.T
    rel = centers_scaled - np.array([cx, cy])
    projs = rel @ n_outer.T
    D = np.max(np.abs(projs))
    s_final = (D + APOTHEM) / APOTHEM
    
    angle_deg = math.degrees(beta)
    
    inner = []
    for x, y in centers_scaled:
        inner.append([float(x), float(y), float(angle_deg)])
    
    center = [float(cx), float(cy)]
    side = float(s_final)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
