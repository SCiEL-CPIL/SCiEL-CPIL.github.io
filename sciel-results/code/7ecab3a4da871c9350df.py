# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _all_vertices(pos, ang, side=1.0):
    """pos (N,2), ang (N,) deg -> (N*6, 2)"""
    n = len(pos)
    verts = np.empty((n * 6, 2))
    for i in range(n):
        verts[i*6:(i+1)*6] = _hex_vertices(pos[i,0], pos[i,1], side, ang[i])
    return verts

def _outer_side(vertices):
    """Min side s of regular hex (center 0, angle 0) containing all vertices."""
    normals = np.array([[np.cos(np.radians(30+60*k)), np.sin(np.radians(30+60*k))] for k in range(6)])
    dots = vertices @ normals.T
    return 2.0 * np.max(dots) / np.sqrt(3.0)

def _pair_overlap(v1, v2):
    """
    SAT overlap depth between two convex hexagons (6,2) each.
    Returns max overlap (positive if overlapping, 0 if separated).
    """
    max_overlap = 0.0
    for v in (v1, v2):
        for k in range(6):
            edge = v[(k+1)%6] - v[k]
            nrm = np.array([-edge[1], edge[0]])
            nn = np.linalg.norm(nrm)
            if nn < 1e-12:
                continue
            nrm /= nn
            p1 = v1 @ nrm
            p2 = v2 @ nrm
            ov = min(p1.max(), p2.max()) - max(p1.min(), p2.min())
            if ov > max_overlap:
                max_overlap = ov
    return max_overlap

def _objective_flat(x):
    """
    x: (36,) = [pos(0), pos(1), ..., pos(11), ang(0), ..., ang(11)]
    Returns max(outer_side, max_pair_overlap) as a smooth-ish objective.
    """
    n = 12
    pos = x[:2*n].reshape(n, 2)
    ang = x[2*n:3*n]
    
    verts = _all_vertices(pos, ang, 1.0)
    s = _outer_side(verts)
    
    max_ov = 0.0
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pos[i] - pos[j])
            if d > 3.5:  # quick rejection: max extent of unit hex is 2, so >3.5 means definitely separated
                continue
            ov = _pair_overlap(verts[i*6:(i+1)*6], verts[j*6:(j+1)*6])
            if ov > max_ov:
                max_ov = ov
    
    return max(s, max_ov)

def _objective_grad(x):
    """Numerical gradient of _objective_flat using central differences."""
    n = 12
    eps = 1e-6
    g = np.zeros(36)
    f0 = _objective_flat(x)
    for k in range(36):
        xp = x.copy(); xm = x.copy()
        xp[k] += eps
        xm[k] -= eps
        g[k] = (_objective_flat(xp) - _objective_flat(xm)) / (2*eps)
    return g

def _is_valid(pos, ang, side=1.0):
    """Check all validity conditions."""
    n = len(pos)
    if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(ang)):
        return False
    
    verts = _all_vertices(pos, ang, side)
    s = _outer_side(verts)
    
    # s >= sqrt(12)
    if s < np.sqrt(12) - 1e-9:
        return False
    
    # Check separation
    for i in range(n):
        for j in range(i+1, n):
            v1 = verts[i*6:(i+1)*6]
            v2 = verts[j*6:(j+1)*6]
            # SAT: must have at least one separating axis
            separated = False
            for v in (v1, v2):
                for k in range(6):
                    edge = v[(k+1)%6] - v[k]
                    nrm = np.array([-edge[1], edge[0]])
                    nn = np.linalg.norm(nrm)
                    if nn < 1e-12:
                        continue
                    nrm /= nn
                    p1 = v1 @ nrm
                    p2 = v2 @ nrm
                    if p1.max() < p2.min() - 1e-6 or p2.max() < p1.min() - 1e-6:
                        separated = True
                        break
                if separated:
                    break
            if not separated:
                return False
    
    # Check containment: all inner vertices inside outer hex
    # Outer hex: center (0,0), side s, angle 0
    outer_verts = _hex_vertices(0, 0, s, 0.0)
    for i in range(n):
        for vi in range(6):
            v = verts[i*6 + vi]
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
                if cross < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
    
    return True

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600.0
    n = 12
    
    # Multiple random initializations
    best_s = np.inf
    best_pos = None
    best_ang = None
    
    # Pre-compute some fixed initial configurations
    inits = []
    
    # 1) Hexagonal close-packed 4x3 grid
    pos_init = []
    ang_init = []
    for row in range(4):
        for col in range(3):
            x = col * 3.0
            y = row * np.sqrt(3)
            if row % 2 == 1:
                x += 1.5
            pos_init.append([x, y])
            ang_init.append(0.0)
    pos_init = np.array(pos_init) - np.mean(pos_init, axis=0)
    inits.append((pos_init, np.array(ang_init)))
    
    # 2) 3x4 grid
    pos_init = []
    ang_init = []
    for row in range(3):
        for col in range(4):
            x = col * 3.0
            y = row * np.sqrt(3)
            if row % 2 == 1:
                x += 1.5
            pos_init.append([x, y])
            ang_init.append(0.0)
    pos_init = np.array(pos_init) - np.mean(pos_init, axis=0)
    inits.append((pos_init, np.array(ang_init)))
    
    # 3) 2x6 grid
    pos_init = []
    ang_init = []
    for row in range(2):
        for col in range(6):
            x = col * 3.0
            y = row * np.sqrt(3)
            pos_init.append([x, y])
            ang_init.append(0.0)
    pos_init = np.array(pos_init) - np.mean(pos_init, axis=0)
    inits.append((pos_init, np.array(ang_init)))
    
    # 4) Random perturbations of hex grid
    rng = np.random.default_rng(42)
    for trial in range(5):
        pos_init = []
        ang_init = []
        for row in range(4):
            for col in range(3):
                x = col * 3.0 + rng.uniform(-0.3, 0.3)
                y = row * np.sqrt(3) + rng.uniform(-0.3, 0.3)
                if row % 2 == 1:
                    x += 1.5
                pos_init.append([x, y])
                ang_init.append(rng.uniform(-30, 30))
        pos_init = np.array(pos_init) - np.mean(pos_init, axis=0)
        inits.append((pos_init, np.array(ang_init)))
    
    # 5) Fully random
    for trial in range(3):
        pos_init = rng.uniform(-4, 4, size=(n, 2))
        pos_init -= np.mean(pos_init, axis=0)
        ang_init = rng.uniform(0, 60, size=n)
        inits.append((pos_init, ang_init))
    
    for init_idx, (pos0, ang0) in enumerate(inits):
        if time.time() > deadline - 30:
            break
        
        x0 = np.concatenate([pos0.flatten(), ang0])
        
        try:
            result = minimize(
                _objective_flat,
                x0,
                method='L-BFGS-B',
                jac=_objective_grad,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50,
                          'maxfun': 5000}
            )
        except Exception:
            continue
        
        x_opt = result.x
        pos_opt = x_opt[:2*n].reshape(n, 2)
        ang_opt = x_opt[2*n:3*n]
        
        # Validate
        if _is_valid(pos_opt, ang_opt, 1.0):
            s_opt = _outer_side(_all_vertices(pos_opt, ang_opt, 1.0))
            if s_opt < best_s:
                best_s = s_opt
                best_pos = pos_opt.copy()
                best_ang = ang_opt.copy()
        
        # Also try a few local perturbations from the optimum
        if time.time() > deadline - 60:
            break
        
        for pert in range(3):
            if time.time() > deadline - 30:
                break
            x_pert = x_opt + rng.normal(0, 0.1, size=36)
            try:
                result2 = minimize(
                    _objective_flat,
                    x_pert,
                    method='L-BFGS-B',
                    jac=_objective_grad,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 30,
                              'maxfun': 2000}
                )
            except Exception:
                continue
            
            x_opt2 = result2.x
            pos_opt2 = x_opt2[:2*n].reshape(n, 2)
            ang_opt2 = x_opt2[2*n:3*n]
            
            if _is_valid(pos_opt2, ang_opt2, 1.0):
                s_opt2 = _outer_side(_all_vertices(pos_opt2, ang_opt2, 1.0))
                if s_opt2 < best_s:
                    best_s = s_opt2
                    best_pos = pos_opt2.copy()
                    best_ang = ang_opt2.copy()
    
    # Fallback: if no valid solution found, return a valid grid
    if best_pos is None:
        pos0, ang0 = inits[0]
        best_pos = pos0
        best_ang = ang0
        best_s = _outer_side(_all_vertices(best_pos, best_ang, 1.0))
    
    inner_hex_data = np.column_stack([best_pos, best_ang])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, best_s

# EVOLVE-BLOCK-END
