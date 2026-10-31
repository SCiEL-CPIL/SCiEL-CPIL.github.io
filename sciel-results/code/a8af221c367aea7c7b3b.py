"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math

def _hex_vertices(cx, cy, side=1.0, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon as (6,2) array."""
    a = math.radians(angle_deg)
    angles = a + math.pi / 6 + np.arange(6) * math.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _sat_gap(p1, a1, p2, a2, side=1.0):
    """Compute minimum gap between two hexagons using SAT."""
    v1 = _hex_vertices(p1[0], p1[1], side, a1)
    v2 = _hex_vertices(p2[0], p2[1], side, a2)
    
    edges1 = np.roll(v1, -1, axis=0) - v1
    edges2 = np.roll(v2, -1, axis=0) - v2
    
    min_gap = float('inf')
    for e in list(edges1) + list(edges2):
        n = np.array([-e[1], e[0]])
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n = n / norm
        proj1 = v1 @ n
        proj2 = v2 @ n
        gap = min(proj1.max() - proj2.min(), proj2.max() - proj1.min())
        if gap < min_gap:
            min_gap = gap
    return min_gap

def _check_all_disjoint(positions, angles):
    """Check all pairs are disjoint."""
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            if _sat_gap(positions[i], angles[i], positions[j], angles[j]) < -1e-10:
                return False
    return True

def _min_enclosing_hex_side(positions, angles, outer_center=(0.0, 0.0), outer_rotation_deg=0.0):
    """Compute minimum side length of outer hexagon to contain all inner hexagons."""
    verts = np.vstack([_hex_vertices(p[0], p[1], 1.0, a) for p, a in zip(positions, angles)])
    verts = verts - np.array(outer_center)
    
    phi = math.radians(outer_rotation_deg)
    thetas = phi + np.array([(2*k+1)*math.pi/6 for k in range(6)])
    
    max_proj = 0.0
    for theta in thetas:
        proj = verts[:, 0] * math.cos(theta) + verts[:, 1] * math.sin(theta)
        mp = proj.max()
        if mp > max_proj:
            max_proj = mp
    
    return max_proj / (math.sqrt(3) / 2)

def _force_directed_energy(params, adjacency, outer_rot_rad, lj_sigma=2.5, lj_epsilon=50.0, spring_k=2.0, spring_rest=2.8):
    """Compute total energy: LJ repulsion + spring constraints + outer containment term.
    params: flat array [x0, y0, ..., xN-1, yN-1]
    Returns (energy, gradient_estimate_unused)
    """
    n = len(adjacency[0])
    positions = np.array([(params[2*i], params[2*i+1]) for i in range(n)])
    
    # LJ repulsion for all pairs
    energy = 0.0
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < 1e-8:
                d = 1e-8
            r_over_sigma = d / lj_sigma
            energy += lj_epsilon * (r_over_sigma**12 - r_over_sigma**6)
    
    # Spring constraints for adjacent pairs
    for i in range(n):
        for j in adjacency[i]:
            if j > i:  # avoid double counting
                d = np.linalg.norm(positions[i] - positions[j])
                energy += 0.5 * spring_k * (d - spring_rest)**2
    
    # Outer containment: penalize points outside a hexagon of side S
    # We want to minimize the required S, so add a soft penalty for extent
    phi = outer_rot_rad
    thetas = phi + np.array([(2*k+1)*math.pi/6 for k in range(6)])
    for theta in thetas:
        proj = positions[:, 0] * math.cos(theta) + positions[:, 1] * math.sin(theta)
        # Penalize max projection (soft)
        max_p = proj.max()
        energy += 0.1 * max_p**2
    
    return energy

def _force_directed_energy_and_grad(params, adjacency, outer_rot_rad, lj_sigma=2.5, lj_epsilon=50.0, spring_k=2.0, spring_rest=2.8):
    """Compute total energy and its gradient via finite differences or analytical."""
    n = len(adjacency[0])
    positions = np.array([(params[2*i], params[2*i+1]) for i in range(n)])
    
    energy = 0.0
    grad = np.zeros_like(params)
    
    # LJ repulsion
    for i in range(n):
        for j in range(i+1, n):
            diff = positions[i] - positions[j]
            d = np.linalg.norm(diff)
            if d < 1e-8:
                d = 1e-8
                diff = np.array([1e-8, 0.0])
            r_over_sigma = d / lj_sigma
            # d/dr of epsilon*(r/s)^12 - epsilon*(r/s)^6
            # = epsilon * (12*r^11/s^12 - 6*r^5/s^6) / r ... let's use chain rule
            # f(r) = eps * (r/s)^12 - eps * (r/s)^6
            # df/dr = eps * 12*(r/s)^11/s - eps * 6*(r/s)^5/s
            # = eps/s * (12*(r/s)^11 - 6*(r/s)^5)
            df_dr = lj_epsilon / lj_sigma * (12.0 * r_over_sigma**11 - 6.0 * r_over_sigma**5)
            # grad wrt position i: df/dr * (pos_i - pos_j)/d
            force_dir = diff / d
            grad[2*i] += df_dr * force_dir[0]
            grad[2*i+1] += df_dr * force_dir[1]
            grad[2*j] -= df_dr * force_dir[0]
            grad[2*j+1] -= df_dr * force_dir[1]
            energy += lj_epsilon * (r_over_sigma**12 - r_over_sigma**6)
    
    # Spring constraints
    for i in range(n):
        for j in adjacency[i]:
            if j > i:
                diff = positions[i] - positions[j]
                d = np.linalg.norm(diff)
                if d < 1e-8:
                    d = 1e-8
                    force_dir = np.array([1e-8, 0.0])
                else:
                    force_dir = diff / d
                force_mag = spring_k * (d - spring_rest)
                grad[2*i] += force_mag * force_dir[0]
                grad[2*i+1] += force_mag * force_dir[1]
                grad[2*j] -= force_mag * force_dir[0]
                grad[2*j+1] -= force_mag * force_dir[1]
                energy += 0.5 * spring_k * (d - spring_rest)**2
    
    # Outer containment soft term
    phi = outer_rot_rad
    thetas = phi + np.array([(2*k+1)*math.pi/6 for k in range(6)])
    for theta in thetas:
        proj = positions[:, 0] * math.cos(theta) + positions[:, 1] * math.sin(theta)
        max_idx = proj.argmax()
        max_p = proj[max_idx]
        energy += 0.1 * max_p**2
        # Gradient only for the max point
        grad[2*max_idx] += 0.2 * max_p * math.cos(theta)
        grad[2*max_idx+1] += 0.2 * max_p * math.sin(theta)
    
    return energy, grad

def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # 8s safety margin
    
    sqrt3 = math.sqrt(3)
    
    # Define hexagonal lattice adjacency for compactness
    # Center at 0, ring1 at 6 neighbors, ring2 at 12 neighbors
    # Node 0: center
    # Nodes 1-6: ring 1
    # Nodes 7-10: selected ring 2 (we'll pick 4 from 12)
    
    # Adjacency: center connected to all ring1, ring1 connected to center and adjacent ring1
    # and selected ring2 nodes
    adjacency = [[0]*N for _ in range(N)]
    # Center (0) connected to ring1 (1-6)
    for i in range(1, 7):
        adjacency[0][i] = 1
        adjacency[i][0] = 1
    # Ring1 neighbors
    for i in range(1, 7):
        for j in range(1, 7):
            if i != j and (i - j) % 6 in (1, 5):
                adjacency[i][j] = 1
                adjacency[j][i] = 1
    
    # Initial positions: hexagonal lattice
    def axial_to_xy(q, r):
        return (1.5 * q, sqrt3 * (r + q / 2.0))
    
    ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    ring2 = [(2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2)]
    
    # Start with center + ring1 + 4 from ring2 (opposite pairs for symmetry)
    best_side = float('inf')
    best_positions = None
    best_angles = [0.0] * N
    best_outer_rotation = 0.0
    
    # Try multiple initial configurations
    initial_configs = []
    
    # Config 1: opposite pairs from ring2
    for i in range(6):
        for j in range(i+1, 6):
            subset = [ring2[i], ring2[(i+6)%12], ring2[j], ring2[(j+6)%12]]
            axial_coords = [(0,0)] + ring1 + subset
            positions = [axial_to_xy(q, r) for q, r in axial_coords]
            initial_configs.append(positions)
    
    # Config 2: consecutive 4 from ring2
    for start in range(0, 12, 2):
        subset = [ring2[(start + k) % 12] for k in range(4)]
        axial_coords = [(0,0)] + ring1 + subset
        positions = [axial_to_xy(q, r) for q, r in axial_coords]
        initial_configs.append(positions)
    
    # Config 3: symmetric cross
    subset = [ring2[0], ring2[6], ring2[3], ring2[9]]
    axial_coords = [(0,0)] + ring1 + subset
    positions = [axial_to_xy(q, r) for q, r in axial_coords]
    initial_configs.append(positions)
    
    # Deduplicate
    unique_configs = []
    seen = set()
    for cfg in initial_configs:
        key = tuple((round(p[0], 4), round(p[1], 4)) for p in cfg)
        if key not in seen:
            seen.add(key)
            unique_configs.append(cfg)
    
    # For each initial config, run force-directed optimization
    for cfg_idx, initial_positions in enumerate(unique_configs):
        if time.time() > deadline - 20.0:
            break
        
        initial_angles = [0.0] * N
        if not _check_all_disjoint(initial_positions, initial_angles):
            continue
        
        # Evaluate initial
        for outer_rot in [0.0, 30.0]:
            side = _min_enclosing_hex_side(initial_positions, initial_angles, (0.0, 0.0), outer_rot)
            if side < best_side:
                best_side = side
                best_positions = initial_positions[:]
                best_angles = initial_angles[:]
                best_outer_rotation = outer_rot
        
        # Force-directed optimization
        params0 = np.array([initial_positions[i][0], initial_positions[i][1]] for i in range(N)).flatten()
        
        def fd_objective(params):
            e, g = _force_directed_energy_and_grad(params, adjacency, 0.0)
            return e
        
        def fd_grad(params):
            e, g = _force_directed_energy_and_grad(params, adjacency, 0.0)
            return g
        
        try:
            result = minimize(
                fd_objective,
                params0,
                jac=fd_grad,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if time.time() > deadline - 15.0:
                continue
            
            opt_positions = [(result.x[2*i], result.x[2*i+1]) for i in range(N)]
            opt_angles = [0.0] * N
            
            if _check_all_disjoint(opt_positions, opt_angles):
                for outer_rot in [0.0, 15.0, 30.0]:
                    side = _min_enclosing_hex_side(opt_positions, opt_angles, (0.0, 0.0), outer_rot)
                    if side < best_side:
                        best_side = side
                        best_positions = opt_positions[:]
                        best_angles = opt_angles[:]
                        best_outer_rotation = outer_rot
        except Exception:
            pass
    
    if best_positions is None:
        # Fallback: spread out
        best_positions = [(4.0 * k, 0.0) for k in range(-(N//2), N - N//2)]
        best_angles = [0.0] * N
        best_side = _min_enclosing_hex_side(best_positions, best_angles)
        best_outer_rotation = 0.0
    
    # Nelder-Mead polish on positions
    if time.time() < deadline - 15.0:
        def nm_objective(params):
            positions = [(params[2*i], params[2*i+1]) for i in range(N)]
            angles = best_angles[:]
            if not _check_all_disjoint(positions, angles):
                return 1e6
            side = _min_enclosing_hex_side(positions, angles, (0.0, 0.0), best_outer_rotation)
            return side
        
        x0 = np.array([best_positions[i][0], best_positions[i][1]] for i in range(N)).flatten()
        
        try:
            result = minimize(
                nm_objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if time.time() > deadline - 10.0:
                pass
            else:
                opt_positions = [(result.x[2*i], result.x[2*i+1]) for i in range(N)]
                if _check_all_disjoint(opt_positions, best_angles):
                    side = _min_enclosing_hex_side(opt_positions, best_angles, (0.0, 0.0), best_outer_rotation)
                    if side < best_side:
                        best_side = side
                        best_positions = opt_positions
        except Exception:
            pass
    
    # Try different outer rotations
    if time.time() < deadline - 5.0:
        for outer_rot in [0.0, 10.0, 15.0, 20.0, 30.0, 45.0]:
            side = _min_enclosing_hex_side(best_positions, best_angles, (0.0, 0.0), outer_rot)
            if side < best_side:
                best_side = side
                best_outer_rotation = outer_rot
    
    # Add tiny epsilon for strict containment
    best_side *= (1.0 + 1e-9)
    
    inner = [[best_positions[i][0], best_positions[i][1], best_angles[i]] for i in range(N)]
    return inner, (0.0, 0.0), best_side, best_outer_rotation
# EVOLVE-BLOCK-END
