"""Unit regular hexagons inside a regular hexagon n=12, minimize outer side."""
N = 12

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math

def _hex_verts(cx, cy, side, angle_deg):
    """Generate vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _polygon_overlap_depth(A, B):
    """SAT overlap depth between convex polygons A and B. Returns 0 if disjoint."""
    min_depth = float('inf')
    axes = []
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            edge = poly[(i + 1) % n] - poly[i]
            length = np.linalg.norm(edge)
            if length > 1e-12:
                axes.append(np.array([-edge[1], edge[0]]) / length)
    for ax in axes:
        pa = np.dot(A, ax)
        pb = np.dot(B, ax)
        if pa.max() <= pb.min() or pb.max() <= pa.min():
            return 0.0
        ov = min(pa.max(), pb.max()) - max(pa.min(), pb.min())
        if ov < min_depth:
            min_depth = ov
    return min_depth

def _containment_violation(inner_verts, outer_verts):
    """Sum of distances by which inner verts fall outside outer."""
    n_out = len(outer_verts)
    pen = 0.0
    for iv in inner_verts:
        for i in range(n_out):
            v1 = outer_verts[i]
            v2 = outer_verts[(i + 1) % n_out]
            edge = v2 - v1
            el = np.linalg.norm(edge)
            if el < 1e-12:
                continue
            nrm = np.array([-edge[1], edge[0]]) / el
            sd = np.dot(iv - v1, nrm)
            if sd < 0:
                pen += -sd
    return pen

def _check_validity(params, S, center, outer_angle, tol=1e-6):
    """Check if packing is valid: no overlaps, all contained."""
    inner_list = []
    for i in range(N):
        inner_list.append(_hex_verts(params[3*i], params[3*i+1], 1.0, params[3*i+2]))
    outer = _hex_verts(center[0], center[1], S, outer_angle)
    
    for iv in inner_list:
        if _containment_violation(iv, outer) > tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            if _polygon_overlap_depth(inner_list[i], inner_list[j]) > tol:
                return False
    return True

def _physics_refine(params, S, center, outer_angle, deadline):
    """
    Physics-based repulsion method:
    - Overlapping hexagons exert strong distance-dependent forces to push apart
    - Containment violations create inward forces
    - System evolves to local energy minimum satisfying disjointness
    """
    params = params.copy()
    step_size = 0.15
    max_iters = 300
    damping = 0.9  # Damping factor for velocity
    
    # Initialize velocity
    velocity = np.zeros(3 * N)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Build current configuration
        inner_list = []
        for i in range(N):
            inner_list.append(_hex_verts(params[3*i], params[3*i+1], 1.0, params[3*i+2]))
        outer = _hex_verts(center[0], center[1], S, outer_angle)
        
        # Calculate forces
        forces = np.zeros(3 * N)
        
        # 1. Containment forces: push inward if outside
        for i in range(N):
            iv = inner_list[i]
            for iv_pt in iv:
                for k in range(6):
                    v1 = outer[k]
                    v2 = outer[(k + 1) % 6]
                    edge = v2 - v1
                    el = np.linalg.norm(edge)
                    if el < 1e-12:
                        continue
                    nrm = np.array([-edge[1], edge[0]]) / el
                    sd = np.dot(iv_pt - v1, nrm)
                    if sd < 0:
                        # Push center in the normal direction (inward)
                        forces[3*i:3*i+2] += nrm * (-sd) * 2.0
        
        # 2. Overlap repulsion forces: strong distance-dependent
        for i in range(N):
            for j in range(i+1, N):
                d = _polygon_overlap_depth(inner_list[i], inner_list[j])
                if d > 0:
                    # Calculate separation direction
                    c_i = np.array([params[3*i], params[3*i+1]])
                    c_j = np.array([params[3*j], params[3*j+1]])
                    diff = c_i - c_j
                    dist = np.linalg.norm(diff)
                    
                    if dist < 1e-6:
                        # If centers coincide, use a random small offset
                        diff = np.array([0.01, 0.01])
                        dist = np.linalg.norm(diff)
                    
                    dir_vec = diff / dist
                    
                    # Strong repulsion: force proportional to overlap depth
                    # Also add a small rotational force to help escape local minima
                    repel_strength = d * 3.0
                    forces[3*i:3*i+2] += dir_vec * repel_strength
                    forces[3*j:3*j+2] -= dir_vec * repel_strength
                    
                    # Add small torque to help rotation escape
                    torque = 0.1 * d
                    forces[3*i+2] += torque
                    forces[3*j+2] -= torque
        
        # Normalize forces to prevent instability
        f_norm = np.linalg.norm(forces)
        if f_norm < 1e-10:
            # No significant forces - check if we're at a minimum
            if f_norm < 1e-8:
                break
            forces = forces / f_norm * 0.01
        else:
            # Scale forces to reasonable magnitude
            target_mag = 0.5
            forces = forces / f_norm * target_mag
        
        # Update velocity with damping
        velocity = velocity * damping + forces * (1 - damping)
        
        # Propose new position
        new_params = params + step_size * velocity
        
        # Clamp angles to [0, 60) for periodicity
        for i in range(N):
            new_params[3*i+2] = new_params[3*i+2] % 60.0
        
        # Check if improvement using a simple energy metric
        # Energy = sum of (containment violation)^2 + (overlap depth)^2
        def _energy(p):
            inner_list_e = []
            for i in range(N):
                inner_list_e.append(_hex_verts(p[3*i], p[3*i+1], 1.0, p[3*i+2]))
            outer_e = _hex_verts(center[0], center[1], S, outer_angle)
            
            e = 0.0
            for iv in inner_list_e:
                pen = _containment_violation(iv, outer_e)
                e += pen * pen
            for i in range(N):
                for j in range(i+1, N):
                    d = _polygon_overlap_depth(inner_list_e[i], inner_list_e[j])
                    if d > 0:
                        e += d * d
            return e
        
        old_energy = _energy(params)
        new_energy = _energy(new_params)
        
        if new_energy < old_energy:
            params = new_params
            # Increase step size if improving
            step_size = min(step_size * 1.03, 0.4)
        else:
            # Decrease step size and dampen velocity
            step_size *= 0.75
            velocity *= 0.5
        
        if step_size < 1e-6:
            break
    
    return params

def _make_grid(nrows, ncols):
    """Create a hexagonal grid layout."""
    pos = []
    for r in range(nrows):
        for c in range(ncols):
            x = c * 2.0 + (1.0 if r % 2 else 0.0)
            y = r * np.sqrt(3.0)
            pos.append([x, y])
    arr = np.array(pos)
    arr -= arr.mean(axis=0)
    return arr

def _ring_init(radius=2.0):
    """Create a ring of hexagons around center."""
    pts = []
    for i in range(6):
        a = np.radians(60*i)
        pts.append([radius*np.cos(a), radius*np.sin(a)])
    for i in range(6):
        a = np.radians(60*i + 30)
        pts.append([(radius*1.5)*np.cos(a), (radius*1.5)*np.sin(a)])
    return np.array(pts) - np.array(pts).mean(axis=0)

def _linear_init(S):
    """Create a linear arrangement."""
    xs = np.linspace(-S*0.7, S*0.7, N)
    return np.column_stack([xs, np.zeros(N)])

def _binary_search_min_S(init_params, center, outer_angle, deadline,
                         S_lo=3.0, S_hi=5.5, max_iters=8):
    """Binary search for minimum S using physics-based refinement."""
    best_S = S_hi
    best_params = None
    
    # First check if S_hi works
    p = init_params.copy()
    p = _physics_refine(p, S_hi, center, outer_angle, deadline)
    if _check_validity(p, S_hi, center, outer_angle):
        best_params = p
        best_S = S_hi
    else:
        return None, None
    
    for _ in range(max_iters):
        if time.time() > deadline - 5:
            break
        S_mid = (S_lo + S_hi) / 2.0
        p = best_params.copy() if best_params is not None else init_params.copy()
        p = _physics_refine(p, S_mid, center, outer_angle, deadline)
        if _check_validity(p, S_mid, center, outer_angle):
            best_S = S_mid
            best_params = p
            S_hi = S_mid
        else:
            S_lo = S_mid
    
    return best_S, best_params

def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0  # Safety margin before 120s limit

    best_S = float('inf')
    best_result = None

    centers = [(0.0, 0.0)]
    outer_angles = [0.0, 30.0]

    # Generate diverse initial configurations
    inits = []
    for nr, nc in [(2, 6), (3, 4), (4, 3), (6, 2)]:
        if nr * nc == N:
            inits.append(_make_grid(nr, nc))
    inits.append(_ring_init(2.0))
    inits.append(_ring_init(2.5))
    inits.append(_linear_init(4.0))
    inits.append(_linear_init(3.5))
    
    # Add random inits
    rng = np.random.default_rng(42)
    for _ in range(8):
        r = rng.uniform(-2.5, 2.5, size=(N, 2))
        inits.append(r - r.mean(axis=0))

    for oa in outer_angles:
        if time.time() > DEADLINE - 25:
            break
        for ctr in centers:
            if time.time() > DEADLINE - 20:
                break
            for init_pts in inits:
                if time.time() > DEADLINE - 10:
                    break
                
                # Convert to full params (x, y, angle)
                init_params = np.zeros(3 * N)
                for i in range(N):
                    init_params[3*i] = init_pts[i, 0]
                    init_params[3*i+1] = init_pts[i, 1]
                    init_params[3*i+2] = 0.0  # Start with angle 0
                
                S_found, params_found = _binary_search_min_S(
                    init_params, ctr, oa, DEADLINE,
                    S_lo=3.0, S_hi=5.5, max_iters=7
                )
                
                if params_found is not None and S_found < best_S:
                    # Final polish with physics
                    polished = _physics_refine(params_found, S_found, ctr, oa, DEADLINE)
                    if _check_validity(polished, S_found, ctr, oa):
                        best_S = S_found
                        best_result = (S_found, polished, ctr, oa)
                    else:
                        best_S = S_found
                        best_result = (S_found, params_found, ctr, oa)

                # Early exit if very good solution found
                if best_S < 3.3:
                    break
            if time.time() > DEADLINE - 10:
                break
        if time.time() > DEADLINE - 10:
            break

    if best_result is None:
        # Fallback: return a valid but not optimal configuration
        grid = _make_grid(3, 4)
        inner = [[float(grid[i,0]), float(grid[i,1]), 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 5.0, 0.0

    S, params, ctr, oa = best_result
    inner = [[float(params[3*i]), float(params[3*i+1]), float(params[3*i+2])] for i in range(N)]
    return inner, (float(ctr[0]), float(ctr[1])), float(S), float(oa)

# EVOLVE-BLOCK-END
