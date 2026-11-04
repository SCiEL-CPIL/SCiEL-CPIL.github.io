import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Precompute unit hexagon vertices for angle 0
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_outer_side(pos, ang, outer_center, outer_angle):
        """Calculate outer hexagon side length given inner positions, angles, and outer params."""
        theta_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            ang_n = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang_n), math.sin(ang_n)])
        normals = np.array(normals)
        
        oc = np.array(outer_center)
        max_proj = 0.0
        for i in range(N):
            verts = get_hex_verts(pos[i][0], pos[i][1], 1.0, np.degrees(ang[i]))
            verts_rel = verts - oc
            projs = verts_rel @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / SQRT3

    def sat_gap(p1, p2):
        """Returns minimum separation between two hexagons. Negative if overlapping."""
        min_sep = float('inf')
        axes = []
        for poly in [p1, p2]:
            for k in range(6):
                edge = poly[(k+1)%6] - poly[k]
                n = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(n)
                if norm > 1e-9:
                    axes.append(n/norm)
        
        for axis in axes:
            proj1 = np.dot(p1, axis)
            proj2 = np.dot(p2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            sep = max(min1, min2) - min(max1, max2)
            if sep < min_sep:
                min_sep = sep
            if min_sep > 0:
                break
        
        return min_sep

    def is_valid(pos, ang, tol=1e-6):
        """Check if configuration is valid (no overlaps)."""
        polys = [get_hex_verts(pos[i][0], pos[i][1], 1.0, np.degrees(ang[i])) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap(polys[i], polys[j]) < -tol:
                    return False
        return True

    def objective(pos, ang, outer_center, outer_angle):
        """Total objective: outer side length + overlap penalty."""
        side = calc_outer_side(pos, ang, outer_center, outer_angle)
        
        # Overlap penalty
        polys = [get_hex_verts(pos[i][0], pos[i][1], 1.0, np.degrees(ang[i])) for i in range(N)]
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                gap = sat_gap(polys[i], polys[j])
                if gap < 0:
                    penalty += (-gap) ** 2 * 100.0
        
        return side + penalty

    def optimize_outer(pos, ang):
        """Find optimal outer center and angle for given inner configuration."""
        def outer_obj(outer_params):
            oc = outer_params[:2]
            oa = outer_params[2]
            return calc_outer_side(pos, ang, oc, oa)
        
        centroid = np.mean(pos, axis=0)
        x0 = np.array([centroid[0], centroid[1], 0.0])
        
        try:
            res = minimize(outer_obj, x0, method='Nelder-Mead', 
                          options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.success:
                return res.x[:2], res.x[2]
        except:
            pass
        return centroid, 0.0

    def local_refine(pos, ang, outer_center, outer_angle, max_iters=200):
        """Local refinement using coordinate descent."""
        best_pos = pos.copy()
        best_ang = ang.copy()
        best_oc = outer_center.copy() if isinstance(outer_center, np.ndarray) else np.array(outer_center)
        best_oa = outer_angle
        best_obj = objective(best_pos, best_ang, best_oc, best_oa)
        
        step = 0.05
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            improved = False
            # Try moving each hexagon
            for i in range(N):
                for dx in [step, -step]:
                    trial_pos = best_pos.copy()
                    trial_pos[i][0] += dx
                    obj = objective(trial_pos, best_ang, best_oc, best_oa)
                    if obj < best_obj:
                        best_pos = trial_pos
                        best_obj = obj
                        improved = True
                
                for dy in [step, -step]:
                    trial_pos = best_pos.copy()
                    trial_pos[i][1] += dy
                    obj = objective(trial_pos, best_ang, best_oc, best_oa)
                    if obj < best_obj:
                        best_pos = trial_pos
                        best_obj = obj
                        improved = True
                
                for da in [step, -step]:
                    trial_ang = best_ang.copy()
                    trial_ang[i] += da
                    obj = objective(best_pos, trial_ang, best_oc, best_oa)
                    if obj < best_obj:
                        best_ang = trial_ang
                        best_obj = obj
                        improved = True
            
            # Try adjusting outer params
            for dx in [step, -step]:
                trial_oc = best_oc.copy()
                trial_oc[0] += dx
                obj = objective(best_pos, best_ang, trial_oc, best_oa)
                if obj < best_obj:
                    best_oc = trial_oc
                    best_obj = obj
                    improved = True
            
            for dy in [step, -step]:
                trial_oc = best_oc.copy()
                trial_oc[1] += dy
                obj = objective(best_pos, best_ang, trial_oc, best_oa)
                if obj < best_obj:
                    best_oc = trial_oc
                    best_obj = obj
                    improved = True
            
            for da in [step, -step]:
                trial_oa = best_oa + da
                obj = objective(best_pos, best_ang, best_oc, trial_oa)
                if obj < best_obj:
                    best_oa = trial_oa
                    best_obj = obj
                    improved = True
            
            if not improved:
                step *= 0.7
                if step < 1e-5:
                    break
        
        # Final outer optimization
        best_oc, best_oa = optimize_outer(best_pos, best_ang)
        best_obj = objective(best_pos, best_ang, best_oc, best_oa)
        
        return best_pos, best_ang, best_oc, best_oa, best_obj

    def simulated_annealing(pos, ang, outer_center, outer_angle, T0=1.0, cooling=0.97, max_iters=3000):
        """Simulated annealing for inner positions and angles."""
        current_pos = pos.copy()
        current_ang = ang.copy()
        current_oc = outer_center.copy() if isinstance(outer_center, np.ndarray) else np.array(outer_center)
        current_oa = outer_angle
        current_obj = objective(current_pos, current_ang, current_oc, current_oa)
        
        best_pos = current_pos.copy()
        best_ang = current_ang.copy()
        best_oc = current_oc.copy()
        best_oa = current_oa
        best_obj = current_obj
        
        T = T0
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Perturb a random hexagon
            i = np.random.randint(N)
            trial_pos = current_pos.copy()
            trial_ang = current_ang.copy()
            
            trial_pos[i][0] += np.random.normal(0, 0.15)
            trial_pos[i][1] += np.random.normal(0, 0.15)
            trial_ang[i] += np.random.normal(0, 0.15)
            
            trial_obj = objective(trial_pos, trial_ang, current_oc, current_oa)
            
            delta = trial_obj - current_obj
            if delta < 0 or np.random.rand() < math.exp(-delta / max(T, 1e-10)):
                current_pos = trial_pos
                current_ang = trial_ang
                current_obj = trial_obj
                
                if current_obj < best_obj:
                    best_pos = current_pos.copy()
                    best_ang = current_ang.copy()
                    best_oc = current_oc.copy()
                    best_oa = current_oa
                    best_obj = current_obj
            
            T *= cooling
            if T < 0.01:
                break
        
        return best_pos, best_ang, best_oc, best_oa, best_obj

    # Generate initial configurations using lattice-based approach
    inits = []
    
    # 1. Standard honeycomb lattice
    lattice_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * 1.5
            y = i * SQRT3 / 2.0 + j * SQRT3
            lattice_points.append((x, y))
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    init_pos = np.array(lattice_points[:12])
    init_pos -= np.mean(init_pos, axis=0)
    init_ang = np.zeros(N)
    inits.append((init_pos, init_ang))
    
    # 2. Ring configuration
    ring_pos = []
    ring_ang = []
    r = 2.2
    for i in range(N):
        a = 2 * math.pi * i / N
        ring_pos.append([r * math.cos(a), r * math.sin(a)])
        ring_ang.append(a)
    ring_pos = np.array(ring_pos)
    ring_pos -= np.mean(ring_pos, axis=0)
    inits.append((ring_pos, np.array(ring_ang)))
    
    # 3. Compact cluster
    cluster_pos = []
    cluster_ang = []
    positions = [
        (0, 0), (1.5, SQRT3/2), (1.5, -SQRT3/2),
        (3, 0), (0, SQRT3), (0, -SQRT3),
        (3, SQRT3), (3, -SQRT3), (4.5, SQRT3/2),
        (4.5, -SQRT3/2), (1.5, 3*SQRT3/2), (1.5, -3*SQRT3/2)
    ]
    for p in positions:
        cluster_pos.append(p)
        cluster_ang.append(0)
    cluster_pos = np.array(cluster_pos)
    cluster_pos -= np.mean(cluster_pos, axis=0)
    inits.append((cluster_pos, np.array(cluster_ang)))
    
    # 4-8. Random starts with lattice bias
    np.random.seed(42)
    for _ in range(5):
        pos = np.random.uniform(-3, 3, (N, 2))
        # Add lattice bias
        for i in range(N):
            nearest = min(range(len(lattice_points)), key=lambda k: (pos[i][0]-lattice_points[k][0])**2 + (pos[i][1]-lattice_points[k][1])**2)
            pos[i] = 0.5 * pos[i] + 0.5 * np.array(lattice_points[nearest])
        ang = np.random.uniform(0, 2*math.pi, N)
        inits.append((pos, ang))
    
    best_side = float('inf')
    best_pos = None
    best_ang = None
    best_oc = np.array([0.0, 0.0])
    best_oa = 0.0
    
    # Process each initial configuration
    for idx, (init_pos, init_ang) in enumerate(inits):
        if time.time() > deadline - 30:
            break
        
        # Check if initial is valid
        if not is_valid(init_pos, init_ang):
            continue
        
        # Get initial outer params
        oc, oa = optimize_outer(init_pos, init_ang)
        init_obj = objective(init_pos, init_ang, oc, oa)
        
        # Run SA
        sa_pos, sa_ang, sa_oc, sa_oa, sa_obj = simulated_annealing(
            init_pos, init_ang, oc, oa, T0=0.5, cooling=0.98, max_iters=2000)
        
        # Local refinement
        lr_pos, lr_ang, lr_oc, lr_oa, lr_obj = local_refine(
            sa_pos, sa_ang, sa_oc, sa_oa, max_iters=100)
        
        # Check validity and update best
        if is_valid(lr_pos, lr_ang) and lr_obj < best_side:
            final_oc, final_oa = optimize_outer(lr_pos, lr_ang)
            final_side = calc_outer_side(lr_pos, lr_ang, final_oc, final_oa)
            
            if final_side < best_side:
                best_side = final_side
                best_pos = lr_pos.copy()
                best_ang = lr_ang.copy()
                best_oc = final_oc.copy()
                best_oa = final_oa
    
    # Fallback if no valid configuration found
    if best_pos is None:
        # Use the honeycomb lattice as fallback
        lattice_points = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i * 1.5
                y = i * SQRT3 / 2.0 + j * SQRT3
                lattice_points.append((x, y))
        lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
        best_pos = np.array(lattice_points[:12])
        best_pos -= np.mean(best_pos, axis=0)
        best_ang = np.zeros(N)
        best_oc, best_oa = optimize_outer(best_pos, best_ang)
        best_side = calc_outer_side(best_pos, best_ang, best_oc, best_oa)
    
    # Construct return values
    inner = []
    for i in range(N):
        inner.append([best_pos[i][0], best_pos[i][1], np.degrees(best_ang[i])])
    
    return inner, tuple(best_oc), best_side, best_oa
