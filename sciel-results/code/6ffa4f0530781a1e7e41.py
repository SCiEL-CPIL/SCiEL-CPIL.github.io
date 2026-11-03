import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 11

def _hex_vertices(cx, cy, angle_deg=0.0):
    angle_rad = np.deg2rad(angle_deg)
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    angles = base_angles + angle_rad
    x = cx + np.cos(angles)
    y = cy + np.sin(angles)
    return np.column_stack((x, y))

def _compute_max_proj_vectorized(centers, angles, outer_angle_deg):
    """Vectorized max projection computation."""
    theta_rad = np.deg2rad(outer_angle_deg)
    normals = np.array([[np.cos(theta_rad + np.deg2rad(30 + k * 60)),
                         np.sin(theta_rad + np.deg2rad(30 + k * 60))] for k in range(6)])
    
    max_proj = 0.0
    for i in range(len(centers)):
        verts = _hex_vertices(centers[i][0], centers[i][1], angles[i])
        projs = verts @ normals.T
        local_max = np.max(projs)
        if local_max > max_proj:
            max_proj = local_max
    return max_proj

def _min_outer_side(centers, angles, outer_angle_deg=0.0):
    max_proj = _compute_max_proj_vectorized(centers, angles, outer_angle_deg)
    S = 2.0 * max_proj / np.sqrt(3.0)
    return S

def _objective_from_params(x, outer_angle_deg=0.0):
    """x = [x1,y1,a1, x2,y2,a2, ...] angles in radians"""
    n = len(x) // 3
    centers = [(x[3*i], x[3*i+1]) for i in range(n)]
    angles_deg = [np.rad2deg(x[3*i+2]) for i in range(n)]
    return _min_outer_side(centers, angles_deg, outer_angle_deg)

def _objective_full(x):
    """x = [x1,y1,a1, ..., outer_angle_rad]"""
    n = (len(x) - 1) // 3
    centers = [(x[3*i], x[3*i+1]) for i in range(n)]
    angles_deg = [np.rad2deg(x[3*i+2]) for i in range(n)]
    outer_ang_deg = np.rad2deg(x[-1])
    return _min_outer_side(centers, angles_deg, outer_ang_deg)

def _generate_topology_topologies():
    """Generate diverse topologies using shell-based approach."""
    A = np.array([np.sqrt(3), 0.0])
    B = np.array([np.sqrt(3)/2, 1.5])
    
    topologies = []
    
    # Core: center + 6 neighbors
    core = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    
    # Shell 2 candidates (distance 2 from center)
    shell2 = [(2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1), (0,-2), (1,-1), (2,-1), (-1,2), (-2,1), (1,-2)]
    # Filter to valid lattice points at distance ~2*sqrt(3)
    valid_shell2 = []
    for pt in shell2:
        c = pt[0] * A + pt[1] * B
        dist = np.linalg.norm(c)
        if 2.5 < dist < 3.8:  # Around 2*sqrt(3) ≈ 3.46
            valid_shell2.append(pt)
    
    # Generate combinations of 4 from shell2
    from itertools import combinations
    if len(valid_shell2) >= 4:
        for combo in list(combinations(valid_shell2, 4))[:20]:
            topo = core + list(combo)
            # Check for duplicates
            if len(set(topo)) == 11:
                topologies.append(topo)
    
    # Also try different core arrangements
    # Core variant: center + 5 neighbors + 1 from shell
    for missing_neighbor in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]:
        core_var = [p for p in core if p != missing_neighbor]
        # Add 4 from shell2
        for combo in list(combinations(valid_shell2, 4))[:5]:
            topo = core_var + list(combo)
            if len(set(topo)) == 11:
                topologies.append(topo)
    
    # Fallback: standard hexagonal arrangement
    standard = core + [(2,0), (1,1), (0,2), (-2,0)]
    topologies.append(standard)
    
    return topologies

def _topology_to_centers(topo):
    A = np.array([np.sqrt(3), 0.0])
    B = np.array([np.sqrt(3)/2, 1.5])
    centers = []
    for i, j in topo:
        c = i * A + j * B
        centers.append(tuple(c))
    return centers

def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin
    
    topologies = _generate_topology_topologies()
    
    best_S = float('inf')
    best_config = None
    best_x0 = None
    
    # Phase 1: Evaluate topologies with various rotations
    for topo in topologies:
        if time.time() > deadline - 80:
            break
        centers = _topology_to_centers(topo)
        
        for rot_deg in [0, 30, 60, 90, 120, 150]:
            if time.time() > deadline - 75:
                break
            rot_rad = np.deg2rad(rot_deg)
            cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
            rotated_centers = [(x * cos_r - y * sin_r, x * sin_r + y * cos_r) for (x, y) in centers]
            
            for inner_ang in [0, 30, 60]:
                S = _min_outer_side(rotated_centers, [inner_ang] * N, 0.0)
                if S < best_S:
                    best_S = S
                    best_config = (rotated_centers, [inner_ang] * N, 0.0)
                    x0 = []
                    for (cx, cy), ang in zip(rotated_centers, [inner_ang] * N):
                        x0.extend([cx, cy, np.deg2rad(ang)])
                    x0.append(0.0)
                    best_x0 = x0
    
    if best_config is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Phase 2: Differential Evolution for global search
    if time.time() < deadline - 60 and best_x0 is not None:
        # Define bounds: positions in [-4, 4], angles in [0, 2*pi]
        bounds = []
        for i in range(N):
            bounds.extend([(-4.0, 4.0), (-4.0, 4.0), (0.0, 2 * np.pi)])
        bounds.append((0.0, 2 * np.pi))  # outer angle
        
        try:
            de_result = differential_evolution(
                _objective_full,
                bounds,
                seed=42,
                maxiter=50,
                tol=1e-6,
                popsize=15,
                mutation=(0.3, 1.0),
                recombination=0.8,
                init='latinhypercube',
                polish=True,
                options={'maxiter': 200}
            )
            if de_result.fun < best_S:
                best_S = de_result.fun
                x_opt = de_result.x
                opt_centers = [(x_opt[3*i], x_opt[3*i+1]) for i in range(N)]
                opt_angles = [np.rad2deg(x_opt[3*i+2]) for i in range(N)]
                opt_outer = np.rad2deg(x_opt[-1])
                best_config = (opt_centers, opt_angles, opt_outer)
        except Exception:
            pass
    
    # Phase 3: L-BFGS-B local refinement
    if time.time() < deadline - 30:
        centers, angles, outer_ang = best_config
        x0_refine = []
        for (cx, cy), ang in zip(centers, angles):
            x0_refine.extend([cx, cy, np.deg2rad(ang)])
        x0_refine.append(np.deg2rad(outer_ang))
        
        try:
            lbfgs_result = minimize(
                _objective_full,
                x0_refine,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if lbfgs_result.fun < best_S:
                best_S = lbfgs_result.fun
                x_opt = lbfgs_result.x
                opt_centers = [(x_opt[3*i], x_opt[3*i+1]) for i in range(N)]
                opt_angles = [np.rad2deg(x_opt[3*i+2]) for i in range(N)]
                opt_outer = np.rad2deg(x_opt[-1])
                best_config = (opt_centers, opt_angles, opt_outer)
        except Exception:
            pass
    
    # Phase 4: Nelder-Mead final polish
    if time.time() < deadline - 10:
        centers, angles, outer_ang = best_config
        x0_final = []
        for (cx, cy), ang in zip(centers, angles):
            x0_final.extend([cx, cy, np.deg2rad(ang)])
        x0_final.append(np.deg2rad(outer_ang))
        
        try:
            nm_result = minimize(
                _objective_full,
                x0_final,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if nm_result.fun < best_S:
                best_S = nm_result.fun
                x_opt = nm_result.x
                opt_centers = [(x_opt[3*i], x_opt[3*i+1]) for i in range(N)]
                opt_angles = [np.rad2deg(x_opt[3*i+2]) for i in range(N)]
                opt_outer = np.rad2deg(x_opt[-1])
                best_config = (opt_centers, opt_angles, opt_outer)
        except Exception:
            pass
    
    # Phase 5: Multi-start from perturbed best config
    if time.time() < deadline - 5:
        centers, angles, outer_ang = best_config
        for perturb_scale in [0.1, 0.05]:
            if time.time() > deadline - 3:
                break
            np.random.seed(123)
            x_perturbed = []
            for (cx, cy), ang in zip(centers, angles):
                x_perturbed.extend([
                    cx + np.random.uniform(-perturb_scale, perturb_scale),
                    cy + np.random.uniform(-perturb_scale, perturb_scale),
                    np.deg2rad(ang) + np.random.uniform(-np.deg2rad(perturb_scale * 10), np.deg2rad(perturb_scale * 10))
                ])
            x_perturbed.append(np.deg2rad(outer_ang))
            
            try:
                nm_pert = minimize(
                    _objective_full,
                    x_perturbed,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if nm_pert.fun < best_S:
                    best_S = nm_pert.fun
                    x_opt = nm_pert.x
                    opt_centers = [(x_opt[3*i], x_opt[3*i+1]) for i in range(N)]
                    opt_angles = [np.rad2deg(x_opt[3*i+2]) for i in range(N)]
                    opt_outer = np.rad2deg(x_opt[-1])
                    best_config = (opt_centers, opt_angles, opt_outer)
            except Exception:
                pass
    
    centers, angles, outer_ang = best_config
    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    return inner, (0.0, 0.0), best_S, outer_ang

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
