"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11
DEADLINE = 145.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, side, rot_deg):
    """Return 6 vertices of a regular hexagon."""
    rot = np.radians(rot_deg)
    angles = rot + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _get_all_vertices(inner):
    """Get all vertices from inner hexagons. inner: list of (x, y, angle_deg)."""
    verts = []
    for x, y, a in inner:
        verts.append(_hex_vertices(x, y, 1.0, a))
    return np.vstack(verts)

def _outer_side_length(vertices, outer_cx, outer_cy, outer_rot_deg):
    """Compute minimum outer side length to contain all vertices."""
    rot = np.radians(outer_rot_deg)
    apothem_factors = []
    for k in range(6):
        phi = rot + k * np.pi / 3.0 + np.pi / 6.0
        nx = np.cos(phi)
        ny = np.sin(phi)
        projections = (vertices[:, 0] - outer_cx) * nx + (vertices[:, 1] - outer_cy) * ny
        apothem_factors.append(np.max(projections))
    max_apothem = max(apothem_factors)
    if max_apothem < 0:
        max_apothem = 0
    return 2.0 * max_apothem / np.sqrt(3.0)

def _objective(params):
    """Objective: minimize outer side length."""
    n_inner = N
    inner_params = params[:3 * n_inner]
    outer_cx = params[3 * n_inner]
    outer_cy = params[3 * n_inner + 1]
    outer_rot = params[3 * n_inner + 2]
    
    inner = []
    for i in range(n_inner):
        x = inner_params[3*i]
        y = inner_params[3*i+1]
        a = inner_params[3*i+2]
        inner.append((x, y, a))
    
    vertices = _get_all_vertices(inner)
    return _outer_side_length(vertices, outer_cx, outer_cy, outer_rot)

def _initial_honeycomb():
    """Create a honeycomb-like initial arrangement of 11 hexagons."""
    d = np.sqrt(3.0)  # distance between adjacent hex centers in honeycomb
    positions = [(0.0, 0.0)]
    for k in range(6):
        angle = k * np.pi / 3.0
        positions.append((d * np.cos(angle), d * np.sin(angle)))
    for k in range(4):
        angle = k * np.pi / 3.0
        positions.append((2 * d * np.cos(angle), 2 * d * np.sin(angle)))
    inner = [(x, y, 0.0) for x, y in positions]
    return inner

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Initial honeycomb arrangement
    inner = _initial_honeycomb()
    
    # Initial params
    inner_params = []
    for x, y, a in inner:
        inner_params.extend([x, y, a])
    params0 = inner_params + [0.0, 0.0, 0.0]
    
    best_score = _objective(params0)
    best_params = list(params0)
    
    # Phase 1: L-BFGS-B optimization
    try:
        res = minimize(
            _objective, params0, method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if res.success or res.fun < best_score:
            best_score = res.fun
            best_params = list(res.x)
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > DEADLINE - 30:
        inner_out = []
        for i in range(N):
            inner_out.append([best_params[3*i], best_params[3*i+1], best_params[3*i+2]])
        return inner_out, (best_params[3*N], best_params[3*N+1]), best_score, best_params[3*N+2]
    
    # Phase 2: Nelder-Mead polish
    try:
        res2 = minimize(
            _objective, best_params, method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        if res2.fun < best_score:
            best_score = res2.fun
            best_params = list(res2.x)
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > DEADLINE - 10:
        inner_out = []
        for i in range(N):
            inner_out.append([best_params[3*i], best_params[3*i+1], best_params[3*i+2]])
        return inner_out, (best_params[3*N], best_params[3*N+1]), best_score, best_params[3*N+2]
    
    # Phase 3: Try a few random restarts if time permits
    rng = np.random.default_rng(42)
    for restart in range(5):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        perturbation = rng.normal(0, 0.5, size=len(best_params))
        params_r = np.array(best_params) + perturbation
        
        try:
            res_r = minimize(
                _objective, params_r, method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res_r.fun < best_score:
                best_score = res_r.fun
                best_params = list(res_r.x)
        except Exception:
            pass
        
        if time.time() - start_time > DEADLINE - 15:
            break
        try:
            res_r2 = minimize(
                _objective, best_params, method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res_r2.fun < best_score:
                best_score = res_r2.fun
                best_params = list(res_r2.x)
        except Exception:
            pass
    
    inner_out = []
    for i in range(N):
        inner_out.append([best_params[3*i], best_params[3*i+1], best_params[3*i+2]])
    
    return inner_out, (best_params[3*N], best_params[3*N+1]), best_score, best_params[3*N+2]
# EVOLVE-BLOCK-END
