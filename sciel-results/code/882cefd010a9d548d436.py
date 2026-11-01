"""Improved program for unit hexagon packing in a regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def get_hex_vertices_array(cx, cy, side, angle_rad):
    """Return 6 vertices as a numpy array for vectorized operations."""
    angles = angle_rad + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + side * np.cos(angles), 
                           cy + side * np.sin(angles)])

def compute_required_side_fast(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    Vectorized version for speed.
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    # Normals for a hexagon with angle 0: outward normals at 30, 90, 150, 210, 270, 330 degrees
    normals = np.array([(math.cos(math.radians(30 + i * 60)), 
                        math.sin(math.radians(30 + i * 60))) for i in range(6)])
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    max_projection = 0.0
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices_array(ix, iy, 1.0, math.radians(iangle))
        # Translate to outer center
        dx = verts[:, 0] - ox
        dy = verts[:, 1] - oy
        # Rotate points by -outer_angle
        lx = dx * cos_a + dy * sin_a
        ly = -dx * sin_a + dy * cos_a
        
        # Project onto all 6 normals: shape (6, 6) -> (6,)
        projections = lx @ normals[:, 0] + ly @ normals[:, 1]
        current_max = np.max(projections)
        if current_max > max_projection:
            max_projection = current_max
    
    return 2.0 * max_projection / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=80.0):
    """
    Smooth approximation using log-sum-exp for the max projection.
    Vectorized for speed.
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    normals = np.array([(math.cos(math.radians(30 + i * 60)), 
                        math.sin(math.radians(30 + i * 60))) for i in range(6)])
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    all_projections = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices_array(ix, iy, 1.0, math.radians(iangle))
        dx = verts[:, 0] - ox
        dy = verts[:, 1] - oy
        lx = dx * cos_a + dy * sin_a
        ly = -dx * sin_a + dy * cos_a
        projections = lx @ normals[:, 0] + ly @ normals[:, 1]
        all_projections.append(projections)
    
    if not all_projections:
        return 0.0
    
    proj_array = np.concatenate(all_projections)
    
    # Center for numerical stability
    proj_mean = np.mean(proj_array)
    centered = proj_array - proj_mean
    
    max_centered = np.max(centered)
    scaled = beta * (centered - max_centered)
    scaled = np.clip(scaled, -700, 700)
    sum_exp = np.sum(np.exp(scaled))
    smooth_max_centered = max_centered + math.log(sum_exp) / beta
    smooth_max = smooth_max_centered + proj_mean
    
    return 2.0 * smooth_max / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons overlap using SAT.
    Returns True if they overlap (including touching).
    """
    axes = []
    
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    
    return True

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    verts_list = [get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2])) 
                  for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap_check(verts_list[i], verts_list[j], eps):
                return False
    return True

def generate_polyhexes(n_cells):
    """
    Generate all distinct n-cell polyhexes using BFS on axial grid.
    Returns list of sets of axial coordinates (normalized to origin).
    """
    def normalize(cells):
        """Normalize a set of axial coords so min q=0, min r=0 (among those with q=0)."""
        cells = list(cells)
        min_q = min(c[0] for c in cells)
        min_r = min(c[1] for c in cells)
        normed = {(c[0] - min_q, c[1] - min_r) for c in cells}
        return frozenset(normed)
    
    def canonical(cells):
        """Get canonical form by trying all 6 rotations and 2 reflections."""
        cells = list(cells)
        # Center of mass
        avg_q = sum(c[0] for c in cells) / len(cells)
        avg_r = sum(c[1] for c in cells) / len(cells)
        
        forms = set()
        for rot in range(6):
            for reflect in range(2):
                rotated = []
                for q, r in cells:
                    dq = q - avg_q
                    dr = r - avg_r
                    if reflect:
                        dq, dr = dr, dq  # reflection
                    # Rotate by 60 degrees * rot
                    for _ in range(rot):
                        new_dq = dq - dr
                        new_dr = dr
                        dq, dr = new_dq, new_dr
                    rotated.append((round(dq + avg_q, 6), round(dr + avg_r, 6)))
                # Normalize
                min_q = min(c[0] for c in rotated)
                min_r = min(c[1] for c in rotated)
                normed = frozenset((c[0] - min_q, c[1] - min_r) for c in rotated)
                forms.add(normed)
        return min(forms)
    
    # BFS: start with single hexagon, grow by adding adjacent hexagons
    # Axial neighbors: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    neighbors = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    
    # Start with one hexagon
    current_level = [frozenset([(0, 0)])]
    seen = set()
    all_polyhexes = []
    
    for size in range(1, n_cells + 1):
        next_level = []
        for poly in current_level:
            if size == n_cells:
                all_polyhexes.append(poly)
                continue
            
            # Try adding each neighbor to each cell
            for (q, r) in poly:
                for (dq, dr) in neighbors:
                    new_cell = (q + dq, r + dr)
                    if new_cell not in poly:
                        new_poly = frozenset(list(poly) + [new_cell])
                        if new_poly not in seen:
                            seen.add(new_poly)
                            next_level.append(new_poly)
        
        current_level = next_level
    
    # Deduplicate using canonical form
    unique = set()
    results = []
    for poly in all_polyhexes:
        canon = canonical(list(poly))
        if canon not in unique:
            unique.add(canon)
            # Find the original poly that matches this canonical form
            for p in all_polyhexes:
                if canonical(list(p)) == canon:
                    results.append(list(p))
                    break
    
    return results

def axial_to_cartesian(q, r, scale=1.0):
    """Convert axial coordinates to Cartesian."""
    x = 1.5 * scale * q
    y = SQRT3 * scale * (r + q / 2.0)
    return x, y

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Generate polyhex topologies
    polyhexes = generate_polyhexes(N)
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Process each polyhex topology
    for poly in polyhexes:
        if time.time() > deadline - 20:
            break
        
        # Convert to Cartesian with tight packing (scale = 1 + epsilon)
        epsilon = 1e-4
        scale = 1.0 + epsilon
        centers = [axial_to_cartesian(q, r, scale) for (q, r) in poly]
        
        # Initial layout: all hexagons aligned
        initial_inner = [[x, y, 0.0] for x, y in centers]
        
        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)
        
        # Optimize outer hexagon
        def objective_outer_smooth(params):
            cx, cy, angle_deg = params
            return compute_required_side_smooth(initial_inner, (cx, cy), angle_deg, beta=60.0)
        
        def objective_outer_exact(params):
            cx, cy, angle_deg = params
            return compute_required_side(initial_inner, (cx, cy), angle_deg)
        
        best_val_local = float('inf')
        best_result_local = None
        
        # Try multiple initial angles
        init_angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        
        for init_angle in init_angles:
            if time.time() > deadline - 15:
                break
            
            init_params = [cx_avg, cy_avg, float(init_angle)]
            
            try:
                # Smooth optimization first
                res_smooth = minimize(objective_outer_smooth, init_params, method='Nelder-Mead', 
                                      options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 500})
                
                if res_smooth.fun < best_val_local:
                    # Exact optimization from smooth result
                    res = minimize(objective_outer_exact, res_smooth.x, method='Nelder-Mead', 
                                  options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 1000})
                    if res.fun < best_val_local:
                        best_val_local = res.fun
                        best_result_local = res.x
            except Exception:
                pass
        
        if best_result_local is None:
            continue
        
        local_outer_center = (best_result_local[0], best_result_local[1])
        local_outer_angle = best_result_local[2]
        local_side = best_val_local
        
        # Optimize inner positions using simulated annealing
        def objective_inner(params):
            inner_list = []
            for i in range(N):
                x = params[i*3]
                y = params[i*3+1]
                a = params[i*3+2]
                inner_list.append([x, y, a])
            
            side = compute_required_side(inner_list, local_outer_center, local_outer_angle)
            
            # Penalty for overlaps
            penalty = 0.0
            min_dist = SQRT3 - 0.01
            for i in range(N):
                for j in range(i+1, N):
                    xi, yi, _ = inner_list[i]
                    xj, yj, _ = inner_list[j]
                    dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
                    if dist < min_dist:
                        penalty += (min_dist - dist) ** 2
            
            return side + 500.0 * penalty
        
        current_params = np.array([v for inner in initial_inner for v in inner])
        current_val = objective_inner(current_params)
        best_params_local = current_params.copy()
        best_val_inner_local = current_val
        
        T = 0.5
        T_min = 0.001
        cooling = 0.999
        step_size = 0.3
        no_improve_count = 0
        max_no_improve = 150
        
        for i in range(5000):
            if time.time() > deadline - 10:
                break
            
            neighbor = current_params.copy()
            idx = np.random.randint(0, N)
            dim = np.random.randint(0, 3)
            delta = np.random.normal(0, step_size)
            neighbor[idx*3 + dim] += delta
            
            neighbor_val = objective_inner(neighbor)
            
            if neighbor_val < current_val or np.random.random() < math.exp(-(neighbor_val - current_val) / max(T, 1e-10)):
                current_params = neighbor
                current_val = neighbor_val
                
                if current_val < best_val_inner_local:
                    best_params_local = current_params.copy()
                    best_val_inner_local = current_val
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                if no_improve_count > max_no_improve:
                    step_size *= 0.85
                    no_improve_count = 0
            else:
                no_improve_count += 1
            
            T *= cooling
            if T < T_min:
                break
        
        # Reconstruct best inner
        best_inner_local = []
        for i in range(N):
            x = best_params_local[i*3]
            y = best_params_local[i*3+1]
            a = best_params_local[i*3+2]
            best_inner_local.append([x, y, a])
        
        # Verify with SAT
        if check_all_disjoint(best_inner_local, eps=1e-5):
            local_side_final = compute_required_side(best_inner_local, local_outer_center, local_outer_angle)
            
            if local_side_final < best_side:
                best_side = local_side_final
                best_inner = best_inner_local
                best_outer_center = local_outer_center
                best_outer_angle = local_outer_angle
        
        # Final polish if this is the best so far
        if best_inner is not None and best_inner == best_inner_local and time.time() < deadline - 5:
            try:
                def polish_objective(params):
                    cx, cy, angle_deg = params
                    return compute_required_side(best_inner, (cx, cy), angle_deg)
                
                polish_result = minimize(polish_objective, [best_outer_center[0], best_outer_center[1], best_outer_angle],
                                        method='Nelder-Mead',
                                        options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 3000})
                
                if polish_result.fun < best_side:
                    best_outer_center = (polish_result.x[0], polish_result.x[1])
                    best_outer_angle = polish_result.x[2]
                    best_side = polish_result.fun
            except Exception:
                pass
    
    # Fallback if no valid solution found
    if best_inner is None:
        # Use the first polyhex with basic optimization
        poly = polyhexes[0] if polyhexes else [(0,0)]
        epsilon = 1e-4
        scale = 1.0 + epsilon
        centers = [axial_to_cartesian(q, r, scale) for (q, r) in poly]
        best_inner = [[x, y, 0.0] for x, y in centers]
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)
    
    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
