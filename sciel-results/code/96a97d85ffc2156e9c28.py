"""Hexagon packing: 11 unit hexagons inside a regular hexagon.
Structural change: use SAT-based distance computation for both containment
and overlap, with a penalty-based objective that smoothly handles constraints.
Use multiple optimization strategies including simulated annealing.
"""
import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar

N = 11
DEADLINE = 135.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, angle_deg=0.0):
    """Return 6 vertices of a unit regular hexagon centered at (cx,cy) rotated by angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append([cx + math.cos(theta), cy + math.sin(theta)])
    return np.array(verts)

def _hex_support(centers, angles, direction):
    """Compute support function of all inner hexagons in a given direction.
    Returns max projection of any vertex of any inner hexagon onto direction."""
    # direction is a unit vector
    max_proj = -float('inf')
    for i in range(N):
        a = math.radians(angles[i])
        # For a unit hexagon, the support in direction d is:
        # max over 6 vertices of dot(vert, d)
        # vert_i = [cos(a + i*pi/3), sin(a + i*pi/3)]
        # dot(vert_i, d) = cos(a + i*pi/3 - atan2(dy, dx))
        dx, dy = direction
        d_angle = math.atan2(dy, dx)
        for k in range(6):
            theta = a + k * math.pi / 3.0
            proj = math.cos(theta - d_angle)
            total = centers[i][0] * dx + centers[i][1] * dy + proj
            if total > max_proj:
                max_proj = total
    return max_proj

def _outer_hex_normals(angle_deg=0.0):
    """Return the 6 outward normals of the outer hexagon."""
    a = math.radians(angle_deg)
    normals = []
    for k in range(6):
        theta = a + k * math.pi / 3.0
        normals.append([math.cos(theta), math.sin(theta)])
    return np.array(normals)

def _compute_min_side(centers, angles, outer_angle_deg=0.0):
    """Compute minimum side length of outer hexagon (at given rotation) containing all inner hexagons.
    
    For a regular hexagon with side S and center at origin, the distance from center to each edge
    (apothem) is S * sqrt(3)/2. The hexagon is the intersection of 6 half-planes.
    For each normal direction n_k, we need: max_i dot(v_i, n_k) <= S * sqrt(3)/2
    where v_i are vertices of inner hexagons.
    So S >= max_k (max_i dot(v_i, n_k)) / (sqrt(3)/2)
    """
    normals = _outer_hex_normals(outer_angle_deg)
    apothem_factor = math.sqrt(3) / 2.0
    max_extent = 0.0
    for k in range(6):
        d = normals[k]
        support = _hex_support(centers, angles, d)
        # Also need to check -d direction (the opposite side)
        support_neg = _hex_support(centers, angles, -d)
        # The extent in this direction is max(support, support_neg)
        extent = max(support, -support_neg)  # support_neg is max of dot(v, -d) = -min(dot(v,d))
        # Actually: the constraint is both dot(v, d) <= apothem AND dot(v, -d) <= apothem
        # i.e., -apothem <= dot(v, d) <= apothem
        # So we need max(dot(v,d)) <= apothem AND min(dot(v,d)) >= -apothem
        # min(dot(v,d)) = -max(dot(v,-d)) = -support_neg
        # So: max(dot(v,d)) <= apothem AND support_neg <= apothem
        # i.e., max(support, support_neg) <= apothem
        max_extent = max(max_extent, support, support_neg)
    
    S = max_extent / apothem_factor
    return S

def _sat_distance_hex_hex(center1, angle1_deg, center2, angle2_deg):
    """Compute minimum distance between two unit hexagons using SAT.
    Returns 0 if overlapping (interior), negative if one contains a vertex of other."""
    c1 = np.array(center1)
    c2 = np.array(center2)
    a1 = math.radians(angle1_deg)
    a2 = math.radians(angle2_deg)
    
    # Get vertices
    v1 = []
    for i in range(6):
        theta = a1 + i * math.pi / 3.0
        v1.append(c1 + np.array([math.cos(theta), math.sin(theta)]))
    v1 = np.array(v1)
    
    v2 = []
    for i in range(6):
        theta = a2 + i * math.pi / 3.0
        v2.append(c2 + np.array([math.cos(theta), math.sin(theta)]))
    v2 = np.array(v2)
    
    # Check if any vertex of one is inside the other
    def _point_in_hex(pt, verts):
        """Ray casting for point in convex hexagon."""
        x, y = pt
        inside = False
        n = len(verts)
        j = n - 1
        for i in range(n):
            xi, yi = verts[i]
            xj, yj = verts[j]
            if ((yi > y) != (yj > y)):
                x_intersect = (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi
                if x < x_intersect:
                    inside = not inside
            j = i
        return inside
    
    for v in v1:
        if _point_in_hex(v, v2):
            return -0.01  # overlapping
    for v in v2:
        if _point_in_hex(v, v1):
            return -0.01  # overlapping
    
    # SAT: check all edge normals of both hexagons
    min_dist = float('inf')
    
    def _edge_normal(v1, v2):
        """Outward normal of edge v1->v2 for a convex polygon (assuming CCW)."""
        d = v2 - v1
        n = np.array([-d[1], d[0]])
        norm = np.linalg.norm(n)
        if norm < 1e-10:
            return np.array([0.0, 0.0])
        n /= norm
        # Ensure outward: check midpoint is on the correct side
        # For our hexagons generated CCW, this should be outward
        return n
    
    # Get edge normals for hex1
    normals1 = []
    for i in range(6):
        n = _edge_normal(v1[i], v1[(i+1)%6])
        normals1.append(n)
    
    normals2 = []
    for i in range(6):
        n = _edge_normal(v2[i], v2[(i+1)%6])
        normals2.append(n)
    
    all_normals = normals1 + normals2
    
    for n in all_normals:
        # Project all vertices onto n
        proj1 = v1 @ n
        proj2 = v2 @ n
        
        min1 = np.min(proj1)
        max1 = np.max(proj1)
        min2 = np.min(proj2)
        max2 = np.max(proj2)
        
        # Gap between the two projections
        if max1 < min2:
            gap = min2 - max1
        elif max2 < min1:
            gap = min1 - max2
        else:
            gap = 0.0  # overlap in this direction
        
        if gap < min_dist:
            min_dist = gap
    
    return min_dist

def _total_overlap_penalty(centers, angles):
    """Compute total overlap penalty between all pairs of inner hexagons."""
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = _sat_distance_hex_hex(centers[i], angles[i], centers[j], angles[j])
            if d < 0.005:
                # Penalty for being too close or overlapping
                penalty += (0.005 - d) ** 2
    return penalty

def _objective(params, outer_angle_deg=0.0):
    """Objective function: minimize outer side length + overlap penalty."""
    centers = params[:2*N].reshape(N, 2)
    angles = params[2*N:]
    
    S = _compute_min_side(centers, angles, outer_angle_deg)
    pen = _total_overlap_penalty(centers, angles)
    
    return S + 100.0 * pen

def _is_valid(centers, angles, outer_side, outer_angle_deg=0.0):
    """Check validity: no overlaps and all inside outer hexagon."""
    # Check overlaps
    for i in range(N):
        for j in range(i+1, N):
            d = _sat_distance_hex_hex(centers[i], angles[i], centers[j], angles[j])
            if d < 1e-7:
                return False
    
    # Check containment
    normals = _outer_hex_normals(outer_angle_deg)
    apothem = outer_side * math.sqrt(3) / 2.0
    for k in range(6):
        d = normals[k]
        support = _hex_support(centers, angles, d)
        if support > apothem + 1e-7:
            return False
        support_neg = _hex_support(centers, angles, -d)
        if support_neg > apothem + 1e-7:
            return False
    
    return True

def _make_seed_lattice(spacing_factor=1.0, rot_pattern=None):
    """Create a hexagonal lattice seed configuration."""
    d = 2 * math.sqrt(3) * spacing_factor
    centers = []
    angles = []
    
    # Central hexagon
    centers.append([0.0, 0.0])
    angles.append(0.0)
    
    # 6 surrounding
    for i in range(6):
        theta = i * math.pi / 3.0
        centers.append([d * math.cos(theta), d * math.sin(theta)])
        if rot_pattern and i < len(rot_pattern):
            angles.append(rot_pattern[i])
        else:
            angles.append(0.0)
    
    # 4 more from second ring
    ring2_positions = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = d * (q + r / 2.0)
                y = d * (math.sqrt(3) / 2.0) * r
                ring2_positions.append([x, y])
    
    # Pick 4 well-separated positions
    if len(ring2_positions) >= 4:
        selected_indices = [0, len(ring2_positions)//3, 2*len(ring2_positions)//3, 3*len(ring2_positions)//4]
        for idx in selected_indices:
            centers.append(ring2_positions[idx])
            angles.append(0.0)
    else:
        # Fallback: place at 2*d in 4 directions
        for i in range(4):
            theta = (2*i+1) * math.pi / 12.0
            centers.append([2*d * math.cos(theta), 2*d * math.sin(theta)])
            angles.append(0.0)
    
    return np.array(centers), np.array(angles)

def _seed_solution():
    """Generate a valid seed solution."""
    centers, angles = _make_seed_lattice(1.0)
    S = _compute_min_side(centers, angles) * 1.02
    return centers, angles, S

def _seed_solution2():
    """Alternative seed with rotated outer hexagons."""
    rot_pattern = [30.0, 0.0, 30.0, 0.0, 30.0, 0.0]
    centers, angles = _make_seed_lattice(1.0, rot_pattern)
    S = _compute_min_side(centers, angles) * 1.02
    return centers, angles, S

def _seed_solution3():
    """Seed with different spacing."""
    centers, angles = _make_seed_lattice(0.95)
    S = _compute_min_side(centers, angles) * 1.03
    return centers, angles, S

def construct():
    start_time = time.time()
    
    # Generate multiple seeds
    seeds = []
    for seed_fn in [_seed_solution, _seed_solution2, _seed_solution3]:
        try:
            c, a, S = seed_fn()
            if _is_valid(c, a, S):
                seeds.append((c, a, S))
        except Exception:
            pass
    
    if not seeds:
        # Fallback: spread out more
        d = 2 * math.sqrt(3) * 1.2
        c_fb = []
        a_fb = []
        c_fb.append([0.0, 0.0])
        a_fb.append(0.0)
        for i in range(6):
            theta = i * math.pi / 3.0
            c_fb.append([d * math.cos(theta), d * math.sin(theta)])
            a_fb.append(0.0)
        for i in range(4):
            theta = (2*i+1) * math.pi / 12.0
            c_fb.append([2*d * math.cos(theta), 2*d * math.sin(theta)])
            a_fb.append(0.0)
        c_fb = np.array(c_fb)
        a_fb = np.array(a_fb)
        S_fb = _compute_min_side(c_fb, a_fb) * 1.05
        seeds.append((c_fb, a_fb, S_fb))
    
    best_S = min(s[2] for s in seeds)
    best_centers = seeds[0][0].copy()
    best_angles = seeds[0][1].copy()
    for s in seeds:
        if s[2] < best_S:
            best_S = s[2]
            best_centers = s[0].copy()
            best_angles = s[1].copy()
    
    # Define bounds
    max_extent = 10.0
    bounds = []
    for i in range(N):
        bounds.append((-max_extent, max_extent))
        bounds.append((-max_extent, max_extent))
    for i in range(N):
        bounds.append((0.0, 180.0))
    
    def objective(params):
        if time.time() - start_time > DEADLINE:
            return 1e10
        return _objective(params)
    
    # Strategy 1: Differential Evolution
    try:
        init_pop = []
        for c, a, S in seeds:
            init_pop.append(c.flatten().tolist() + a.tolist())
        
        # Add some random perturbations
        np.random.seed(42)
        for _ in range(12):
            base_c, base_a = best_centers, best_angles
            c = base_c + np.random.normal(0, 0.5, size=(N, 2))
            a = np.mod(base_a + np.random.normal(0, 15, size=N), 180)
            init_pop.append(c.flatten().tolist() + a.tolist())
        
        res = differential_evolution(
            objective, bounds,
            seed=42,
            maxiter=80,
            popsize=15,
            tol=1e-6,
            mutation=(0.4, 1.2),
            recombination=0.7,
            polish=False,
            init=init_pop,
            updating='deferred'
        )
        
        if res.fun < best_S:
            params = res.x
            c = params[:2*N].reshape(N, 2)
            a = params[2*N:]
            S_val = _compute_min_side(c, a)
            pen_val = _total_overlap_penalty(c, a)
            if pen_val < 1e-5 and S_val < best_S:
                best_S = S_val
                best_centers = c
                best_angles = a
    except Exception:
        pass
    
    # Strategy 2: Nelder-Mead local refinement
    if time.time() - start_time < DEADLINE - 20:
        def nm_obj(params):
            if time.time() - start_time > DEADLINE:
                return 1e10
            centers = params[:2*N].reshape(N, 2)
            angles = params[2*N:]
            S = _compute_min_side(centers, angles)
            pen = _total_overlap_penalty(centers, angles)
            return S + 500.0 * pen
        
        x0 = best_centers.flatten().tolist() + best_angles.tolist()
        try:
            res = minimize(nm_obj, x0, method='Nelder-Mead',
                           options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 3000, 'maxfev': 8000})
            if res.fun < best_S:
                params = res.x
                c = params[:2*N].reshape(N, 2)
                a = params[2*N:]
                S_val = _compute_min_side(c, a)
                pen_val = _total_overlap_penalty(c, a)
                if pen_val < 1e-6 and S_val < best_S:
                    best_S = S_val
                    best_centers = c
                    best_angles = a
        except Exception:
            pass
    
    # Strategy 3: Simulated Annealing
    if time.time() - start_time < DEADLINE - 25:
        def sa_objective(centers, angles):
            S = _compute_min_side(centers, angles)
            pen = _total_overlap_penalty(centers, angles)
            return S + 200.0 * pen
        
        sa_centers = best_centers.copy()
        sa_angles = best_angles.copy()
        sa_best_S = best_S
        sa_best_c = best_centers.copy()
        sa_best_a = best_angles.copy()
        
        current_obj = sa_objective(sa_centers, sa_angles)
        T = 0.5
        cooling = 0.995
        n_iter = 5000
        
        np.random.seed(99)
        for it in range(n_iter):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Random perturbation
            idx = np.random.randint(N)
            delta_c = np.random.normal(0, T * 0.5, size=2)
            delta_a = np.random.normal(0, T * 10)
            
            new_centers = sa_centers.copy()
            new_angles = sa_angles.copy()
            new_centers[idx] += delta_c
            new_angles[idx] = (new_angles[idx] + delta_a) % 180.0
            
            new_obj = sa_objective(new_centers, new_angles)
            delta = new_obj - current_obj
            
            if delta < 0 or np.random.random() < math.exp(-delta / max(T, 1e-10)):
                sa_centers = new_centers
                sa_angles = new_angles
                current_obj = new_obj
                
                if current_obj < sa_best_S:
                    sa_best_S = current_obj
                    sa_best_c = sa_centers.copy()
                    sa_best_a = sa_angles.copy()
            
            T *= cooling
        
        if sa_best_S < best_S:
            best_S = sa_best_S
            best_centers = sa_best_c
            best_angles = sa_best_a
    
    # Strategy 4: Second DE round from improved solution
    if time.time() - start_time < DEADLINE - 20:
        try:
            init_pop2 = [best_centers.flatten().tolist() + best_angles.tolist()]
            np.random.seed(123)
            for _ in range(10):
                scale = np.random.uniform(0.1, 0.4)
                c = best_centers + np.random.normal(0, scale, size=(N, 2))
                a = np.mod(best_angles + np.random.normal(0, 10, size=N), 180)
                init_pop2.append(c.flatten().tolist() + a.tolist())
            
            res = differential_evolution(
                objective, bounds,
                seed=123,
                maxiter=50,
                popsize=12,
                tol=1e-6,
                mutation=(0.3, 0.9),
                recombination=0.7,
                polish=False,
                init=init_pop2,
                updating='deferred'
            )
            
            if res.fun < best_S:
                params = res.x
                c = params[:2*N].reshape(N, 2)
                a = params[2*N:]
                S_val = _compute_min_side(c, a)
                pen_val = _total_overlap_penalty(c, a)
                if pen_val < 1e-5 and S_val < best_S:
                    best_S = S_val
                    best_centers = c
                    best_angles = a
        except Exception:
            pass
    
    # Final refinement
    if time.time() - start_time < DEADLINE - 10:
        def final_obj(params):
            if time.time() - start_time > DEADLINE:
                return 1e10
            centers = params[:2*N].reshape(N, 2)
            angles = params[2*N:]
            S = _compute_min_side(centers, angles)
            pen = _total_overlap_penalty(centers, angles)
            return S + 1000.0 * pen
        
        x0 = best_centers.flatten().tolist() + best_angles.tolist()
        try:
            res = minimize(final_obj, x0, method='Nelder-Mead',
                           options={'xatol': 1e-9, 'fatol': 1e-9, 'maxiter': 2000, 'maxfev': 6000})
            if res.fun < best_S:
                params = res.x
                c = params[:2*N].reshape(N, 2)
                a = params[2*N:]
                S_val = _compute_min_side(c, a)
                pen_val = _total_overlap_penalty(c, a)
                if pen_val < 1e-6 and S_val < best_S:
                    best_S = S_val
                    best_centers = c
                    best_angles = a
        except Exception:
            pass
    
    # Try optimizing outer hexagon rotation
    if time.time() - start_time < DEADLINE - 10:
        def outer_angle_obj(theta_deg):
            S = _compute_min_side(best_centers, best_angles, theta_deg)
            return S
        
        try:
            res = minimize_scalar(outer_angle_obj, bounds=(0, 60), method='bounded',
                                  options={'xatol': 1e-6})
            if res.fun < best_S:
                best_S = res.fun
        except Exception:
            pass
    
    # Final verification
    if not _is_valid(best_centers, best_angles, best_S):
        S_val = _compute_min_side(best_centers, best_angles)
        best_S = S_val * 1.02
    
    # Format output
    inner = []
    for i in range(N):
        inner.append([float(best_centers[i][0]), float(best_centers[i][1]), float(best_angles[i])])
    
    return inner, (0.0, 0.0), float(best_S), 0.0

# EVOLVE-BLOCK-END
