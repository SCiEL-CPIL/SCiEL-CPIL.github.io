"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree


def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _soft_min(values, beta):
    """Smooth approximation of min using log-sum-exp."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    v_min = np.min(values)
    shifted = values - v_min
    shifted = np.clip(shifted, 0, 100.0 / beta)
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return v_min - log_sum / beta


def _smooth_objective(params, beta):
    """Smooth objective: minimize -soft_min(radii_components)."""
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    n = c.shape[0]
    
    boundary_dists = 0.5 - np.abs(c)
    
    if n < 2:
        return -np.min(boundary_dists)
    
    dists = pdist(c, metric='euclidean')
    pair_dists_half = dists / 2.0
    
    all_dists = np.concatenate([pair_dists_half, boundary_dists.ravel()])
    r_soft = _soft_min(all_dists, beta)
    
    if r_soft <= 0:
        return 1.0
    return -r_soft


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    best_c = None
    best_r = 0
    
    for d in np.linspace(0.055, 0.075, 30):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        points = np.array(points)
        if len(points) >= N:
            c = points[:N].copy()
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is not None:
        return best_c
    
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _local_optimize_smooth(c0, max_time=30.0):
    """Local optimization using smooth objective with annealing."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    betas = [15.0, 30.0, 60.0, 120.0, 250.0, 500.0, 1000.0, 2000.0]
    
    for beta in betas:
        if time.time() > start_time + max_time - 3:
            break
        
        try:
            def obj(params):
                return _smooth_objective(params, beta)
            
            maxiter = 80 if beta < 100 else 150
            
            result = minimize(
                obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
    
    return best_c, best_r


def _simulated_annealing_rigid(c0, r0, max_time=40.0):
    """
    Simulated annealing with rigid-body cluster moves to escape local minima.
    Moves clusters of nearby circles simultaneously to preserve local structure.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    current_c = c0.copy()
    current_r = r0
    
    T_start = 0.001
    T_end = 1e-8
    n_steps = 5000
    
    def _build_clusters(c, r_target):
        """Group circles into clusters based on proximity."""
        n = c.shape[0]
        tree = cKDTree(c)
        dist_limit = 2.5 * r_target
        
        parent = list(range(n))
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        for i in range(n):
            neighbors = tree.query_ball_point(c[i], dist_limit)
            for j in neighbors:
                if i < j:
                    union(i, j)
        
        clusters = {}
        for i in range(n):
            root = find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(i)
        
        return list(clusters.values())
    
    clusters = _build_clusters(current_c, current_r)
    cluster_rebuild_interval = 500
    
    for step in range(n_steps):
        if time.time() > start_time + max_time:
            break
        
        T = T_start * (T_end / T_start) ** (step / n_steps)
        
        if step % cluster_rebuild_interval == 0 and step > 0:
            clusters = _build_clusters(current_c, current_r)
        
        if np.random.random() < 0.6 and len(clusters) > 1:
            cluster = clusters[np.random.randint(len(clusters))]
            angle = np.random.random() * 2 * np.pi
            magnitude = np.random.exponential(0.003)
            dx = magnitude * np.cos(angle)
            dy = magnitude * np.sin(angle)
            
            new_c = current_c.copy()
            for idx in cluster:
                new_c[idx, 0] += dx
                new_c[idx, 1] += dy
        else:
            idx = np.random.randint(N)
            angle = np.random.random() * 2 * np.pi
            magnitude = np.random.exponential(0.002)
            dx = magnitude * np.cos(angle)
            dy = magnitude * np.sin(angle)
            
            new_c = current_c.copy()
            new_c[idx, 0] += dx
            new_c[idx, 1] += dy
        
        new_c = np.clip(new_c, -0.5 + 1e-12, 0.5 - 1e-12)
        
        new_r = _compute_radius(new_c)
        delta_r = new_r - current_r
        
        if delta_r > 0 or np.random.random() < np.exp(delta_r / max(T, 1e-12)):
            current_c = new_c
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_c = current_c.copy()
    
    return best_c, best_r


def _mechanics_polish(c0, r0, max_time=30.0):
    """
    High-precision polish using rigid-body mechanics simulation.
    Uses Velocity-Verlet integration with repulsive forces and collision damping.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    n = N
    
    # State
    x = c0.copy()
    v = np.zeros_like(x)
    
    # Parameters
    dt = 1e-4
    max_dt = 1e-3
    min_dt = 1e-6
    damping = 0.995 # Global velocity damping
    collision_damping = 0.5 # Velocity damping on collision
    spring_k = 1000.0 # Stiffness for overlap
    repulsive_k = 100.0 # Stiffness for near-miss
    wall_k = 1000.0 # Stiffness for wall repulsion
    
    tree = cKDTree(x)
    
    steps = 0
    max_steps = 50000
    
    while steps < max_steps:
        if time.time() > start_time + max_time:
            break
        
        # Rebuild tree every 100 steps to keep it updated
        if steps % 100 == 0:
            tree = cKDTree(x)
        
        # Calculate forces
        F = np.zeros_like(x)
        
        # 1. Wall repulsion
        # Distance to walls
        d_left = x[:, 0] + 0.5
        d_right = 0.5 - x[:, 0]
        d_bottom = x[:, 1] + 0.5
        d_top = 0.5 - x[:, 1]
        
        # Forces pushing away from walls if close (within 2*r)
        # We use a linear spring force for simplicity and stability
        margin = 2 * r0
        
        # Left wall
        mask_l = d_left < margin
        F[mask_l, 0] += wall_k * (margin - d_left[mask_l])
        # Right wall
        mask_r = d_right < margin
        F[mask_r, 0] -= wall_k * (margin - d_right[mask_r])
        # Bottom wall
        mask_b = d_bottom < margin
        F[mask_b, 1] += wall_k * (margin - d_bottom[mask_b])
        # Top wall
        mask_t = d_top < margin
        F[mask_t, 1] -= wall_k * (margin - d_top[mask_t])
        
        # 2. Pair repulsion
        # Find neighbors within interaction distance
        # Interaction distance is slightly larger than 2*r to allow "soft" pushing
        interaction_dist = 2 * r0 + 0.05 
        
        for i in range(n):
            neighbors = tree.query_ball_point(x[i], interaction_dist)
            for j in neighbors:
                if i >= j:
                    continue
                
                dx = x[i, 0] - x[j, 0]
                dy = x[i, 1] - x[j, 1]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                
                if dist < 2 * r0:
                    # Overlap: Strong spring force
                    force_mag = spring_k * (2 * r0 - dist)
                elif dist < interaction_dist:
                    # Near miss: Softer repulsive force
                    # Linear decay from 0 at interaction_dist to spring_k*(2r - 2r)=0? 
                    # Let's use a quadratic or linear ramp up to the overlap force
                    # Actually, simpler: just use a weaker constant force if not overlapping but close
                    # Or better: force proportional to (interaction_dist - dist)
                    force_mag = repulsive_k * (interaction_dist - dist)
                else:
                    continue
                
                # Direction from j to i
                nx = dx / dist
                ny = dy / dist
                
                F[i, 0] += force_mag * nx
                F[i, 1] += force_mag * ny
                F[j, 0] -= force_mag * nx
                F[j, 1] -= force_mag * ny
        
        # Velocity Verlet Step 1: Update position
        x_new = x + v * dt + 0.5 * F * dt * dt
        
        # Check for wall collisions and handle immediately for stability
        # Clip positions to be strictly inside, but apply bounce
        # We handle wall collisions by reflecting velocity if position goes out
        # But since we have wall forces, we might not need hard bounce if forces are strong enough.
        # However, to ensure no invalid output, we clip.
        x_new = np.clip(x_new, -0.5 + 1e-12, 0.5 - 1e-12)
        
        # Update tree if we want to use it for force calc in next step? 
        # We already built it at start of loop.
        
        # Calculate new forces at x_new
        F_new = np.zeros_like(x_new)
        
        # Wall forces
        d_left = x_new[:, 0] + 0.5
        d_right = 0.5 - x_new[:, 0]
        d_bottom = x_new[:, 1] + 0.5
        d_top = 0.5 - x_new[:, 1]
        
        margin = 2 * r0
        
        mask_l = d_left < margin
        F_new[mask_l, 0] += wall_k * (margin - d_left[mask_l])
        mask_r = d_right < margin
        F_new[mask_r, 0] -= wall_k * (margin - d_right[mask_r])
        mask_b = d_bottom < margin
        F_new[mask_b, 1] += wall_k * (margin - d_bottom[mask_b])
        mask_t = d_top < margin
        F_new[mask_t, 1] -= wall_k * (margin - d_top[mask_t])
        
        # Pair forces
        # Note: Tree is based on x, not x_new. For small dt, this is usually fine.
        # For higher precision, we'd rebuild, but that's expensive.
        for i in range(n):
            neighbors = tree.query_ball_point(x[i], interaction_dist)
            for j in neighbors:
                if i >= j:
                    continue
                
                dx = x_new[i, 0] - x_new[j, 0]
                dy = x_new[i, 1] - x_new[j, 1]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                
                if dist < 2 * r0:
                    force_mag = spring_k * (2 * r0 - dist)
                elif dist < interaction_dist:
                    force_mag = repulsive_k * (interaction_dist - dist)
                else:
                    continue
                
                nx = dx / dist
                ny = dy / dist
                
                F_new[i, 0] += force_mag * nx
                F_new[i, 1] += force_mag * ny
                F_new[j, 0] -= force_mag * nx
                F_new[j, 1] -= force_mag * ny
        
        # Velocity Verlet Step 2: Update velocity
        v_new = 0.5 * (v + v + (F + F_new) * dt) * damping
        
        # Handle collisions explicitly to damp velocity
        # This is an approximation: if two circles are close, damp their relative velocity
        # We skip explicit pairwise collision damping for speed, relying on forces + global damping.
        # However, if we detect an overlap in x_new, we can apply a small impulse.
        
        # Check for stability
        if not np.all(np.isfinite(x_new)) or not np.all(np.isfinite(v_new)):
            dt *= 0.5
            continue
            
        # Update state
        x = x_new
        v = v_new
        
        # Adapt dt based on kinetic energy / max force
        max_force = np.max(np.linalg.norm(F_new, axis=1))
        if max_force > 100.0:
            dt = max(min_dt, dt * 0.8)
        else:
            dt = min(max_dt, dt * 1.05)
            
        steps += 1
        
        # Check for improvement
        if steps % 100 == 0:
            current_r = _compute_radius(x)
            if current_r > best_r:
                best_r = current_r
                best_c = x.copy()
            
            # If we've settled (low kinetic energy and no improvement for a while), stop
            ke = 0.5 * np.sum(v**2)
            if ke < 1e-8 and steps > 1000:
                break
                
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (evaluator kills at 150s)
    
    best_radius = 0.0
    best_c = None
    
    # 1. Start with hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()
    
    # 2. Primary optimization with smooth objective
    if time.time() < deadline - 60:
        remaining = deadline - time.time() - 60
        c_opt, r_opt = _local_optimize_smooth(c_hex, max_time=min(40.0, remaining))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # 3. Simulated annealing with rigid-body moves to escape local minima
    if best_c is not None and time.time() < deadline - 50:
        remaining = max(10.0, deadline - time.time() - 50)
        c_sa, r_sa = _simulated_annealing_rigid(best_c, best_radius, max_time=min(30.0, remaining))
        if r_sa > best_radius:
            best_radius = r_sa
            best_c = c_sa.copy()
    
    # 4. Another smooth optimization pass from the SA result
    if best_c is not None and time.time() < deadline - 40:
        remaining = max(5.0, deadline - time.time() - 40)
        c_opt2, r_opt2 = _local_optimize_smooth(best_c, max_time=min(25.0, remaining))
        if r_opt2 > best_radius:
            best_radius = r_opt2
            best_c = c_opt2.copy()
    
    # 5. Final high-precision polish using Mechanics Simulation
    if best_c is not None and time.time() < deadline - 3:
        remaining = max(5.0, deadline - time.time() - 3)
        c_final, r_final = _mechanics_polish(best_c, best_radius, max_time=min(20.0, remaining))
        if r_final > best_radius:
            best_radius = r_final
            best_c = c_final
    
    if best_c is None:
        best_c = _initialize_hexagonal()
    
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
