"""
Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle.
Strategy: Force-directed graph relaxation with symplectic integration.
Points are treated as particles with repulsive forces derived from the minimum triangle area.
This explores the configuration space via physical dynamics rather than stochastic jumps.
"""

import numpy as np
import time

N = 11
H = np.sqrt(3) / 2.0
AREA_EQUI = np.sqrt(3) / 4.0

# Pre-compute the 165 index triples for triangles
combos = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            combos.append((i, j, k))
combos = np.array(combos)
idx_i = combos[:, 0]
idx_j = combos[:, 1]
idx_k = combos[:, 2]

def min_triangle_area(points):
    """
    Vectorized computation of the minimum triangle area.
    points: shape (N, 2)
    Returns the minimum area and the index of the triangle achieving it.
    """
    pi = points[idx_i]
    pj = points[idx_j]
    pk = points[idx_k]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_idx = np.argmin(areas)
    return areas[min_idx], min_idx

def is_inside_triangle(points):
    """
    Check if all points are inside the unit equilateral triangle.
    Vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    """
    y = points[:, 1]
    x = points[:, 0]
    
    # Constraints for the triangle:
    # 1. y >= 0
    # 2. y <= sqrt(3) * x
    # 3. y <= -sqrt(3) * (x - 1) = sqrt(3) * (1 - x)
    
    cond1 = y >= -1e-9
    cond2 = y <= np.sqrt(3) * x + 1e-9
    cond3 = y <= np.sqrt(3) * (1 - x) + 1e-9
    
    return np.all(cond1) and np.all(cond2) and np.all(cond3)

def project_to_triangle(points):
    """
    Project points back into the triangle if they are outside.
    Uses barycentric coordinate clamping.
    """
    pts = points.copy()
    for i in range(N):
        x, y = pts[i]
        
        # Barycentric coordinates for triangle (0,0), (1,0), (0.5, H)
        # w1 = 1 - x - y/H
        # w2 = x
        # w3 = y/H
        # Actually, let's use the standard barycentric formula for vertices A, B, C
        # A=(0,0), B=(1,0), C=(0.5, H)
        
        # Area of ABC = H/2
        # Area of PBC (opposite A) = 0.5 * | (B-P) x (C-P) |
        # Area of PCA (opposite B) = 0.5 * | (C-P) x (A-P) |
        # Area of PAB (opposite C) = 0.5 * | (A-P) x (B-P) |
        
        # Simpler:
        # x = w2 * 1 + w3 * 0.5
        # y = w3 * H
        # w1 + w2 + w3 = 1
        
        # From y = w3 * H => w3 = y / H
        # From x = w2 + 0.5 * w3 => w2 = x - 0.5 * w3 = x - 0.5 * y / H
        # w1 = 1 - w2 - w3 = 1 - (x - 0.5*y/H) - y/H = 1 - x - 0.5*y/H
        
        w3 = y / H
        w2 = x - 0.5 * w3
        w1 = 1.0 - w2 - w3
        
        # Clamp barycentric coordinates to [0, 1]
        w1 = np.clip(w1, 0.0, 1.0)
        w2 = np.clip(w2, 0.0, 1.0)
        w3 = np.clip(w3, 0.0, 1.0)
        
        # Normalize if sum is not 1 (due to clamping)
        s = w1 + w2 + w3
        if s > 0:
            w1 /= s
            w2 /= s
            w3 /= s
        
        new_x = w2 * 1.0 + w3 * 0.5
        new_y = w3 * H
        
        pts[i] = [new_x, new_y]
        
    return pts

def compute_forces(points, min_area, min_idx):
    """
    Compute repulsive forces for the force-directed graph.
    The force is designed to increase the minimum triangle area.
    We apply a force to the vertices of the smallest triangle to push them apart.
    Additionally, we apply a weak global repulsion to prevent clustering.
    """
    forces = np.zeros((N, 2))
    
    # 1. Force on the vertices of the minimum triangle
    i, j, k = combos[min_idx]
    
    pi = points[i]
    pj = points[j]
    pk = points[k]
    
    # Calculate the gradient of the area with respect to the vertices
    # Area = 0.5 * | (pj - pi) x (pk - pi) |
    # Let v1 = pj - pi, v2 = pk - pi
    # cross = v1x * v2y - v1y * v2x
    # d(cross)/dpi = - (v1 x v2) direction? 
    # Actually, it's easier to just use the normal to the edges.
    # To increase the area of triangle (i,j,k), we should move i opposite to the normal of (j,k) facing i?
    # No, we want to expand the triangle.
    # The gradient of the area w.r.t. point i is 0.5 * | (pk - pi) x (pj - pi) | ... wait.
    
    # Let's use a simpler heuristic:
    # Move i away from the midpoint of jk.
    # Move j away from the midpoint of ik.
    # Move k away from the midpoint of ji.
    
    mid_jk = (pj + pk) / 2.0
    mid_ik = (pi + pk) / 2.0
    mid_ji = (pi + pj) / 2.0
    
    # Direction from midpoint to vertex
    dir_i = pi - mid_jk
    dir_j = pj - mid_ik
    dir_k = pk - mid_ji
    
    # Normalize
    norm_i = np.linalg.norm(dir_i)
    norm_j = np.linalg.norm(dir_j)
    norm_k = np.linalg.norm(dir_k)
    
    if norm_i > 1e-6: dir_i /= norm_i
    if norm_j > 1e-6: dir_j /= norm_j
    if norm_k > 1e-6: dir_k /= norm_k
    
    # Strength of force proportional to how small the area is? 
    # Or just a constant strength to keep pushing.
    # Let's scale it by the current min area to avoid overshooting if it's already large?
    # Actually, we want to maximize, so we always push.
    # But we need to balance with other triangles.
    
    # Let's add forces from all triangles, weighted by how close they are to the minimum.
    # This is O(N^3) per step if done naively, but N=11 is small.
    
    pi_all = points[idx_i]
    pj_all = points[idx_j]
    pk_all = points[idx_k]
    
    v1 = pj_all - pi_all
    v2 = pk_all - pi_all
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # We want to maximize the minimum. 
    # A common approach: apply force proportional to (min_area - area_t) if area_t < min_area? 
    # No, that would shrink larger triangles.
    # We want to push apart the vertices of the *smallest* triangles.
    
    # Let's use a "pressure" model:
    # Each triangle exerts a repulsive force on its vertices if its area is small.
    # Force magnitude ~ max(0, target_area - area_t)
    
    # What is target_area? We don't know it. 
    # Let's just use a constant force on the smallest few triangles.
    
    # Sort triangles by area
    sorted_indices = np.argsort(areas)
    num_active = 10 # Act on the 10 smallest triangles
    
    for idx in sorted_indices[:num_active]:
        i, j, k = combos[idx]
        a = areas[idx]
        
        # Weight: stronger force for smaller areas
        # If a is very small, force is high.
        weight = 1.0 / (a + 1e-6)
        
        pi = points[i]
        pj = points[j]
        pk = points[k]
        
        mid_jk = (pj + pk) / 2.0
        mid_ik = (pi + pk) / 2.0
        mid_ji = (pi + pj) / 2.0
        
        dir_i = pi - mid_jk
        dir_j = pj - mid_ik
        dir_k = pk - mid_ji
        
        norm_i = np.linalg.norm(dir_i)
        norm_j = np.linalg.norm(dir_j)
        norm_k = np.linalg.norm(dir_k)
        
        if norm_i > 1e-6: dir_i /= norm_i
        if norm_j > 1e-6: dir_j /= norm_j
        if norm_k > 1e-6: dir_k /= norm_k
        
        # Apply force
        # Scale down to prevent explosion
        f_scale = 0.01 * weight / 100.0 # Normalization factor
        
        forces[i] += dir_i * f_scale
        forces[j] += dir_j * f_scale
        forces[k] += dir_k * f_scale
        
    return forces

def force_directed_relaxation(initial_points, time_limit=10.0):
    """
    Run force-directed relaxation using a symplectic integrator (Verlet).
    """
    start_time = time.time()
    
    # Initialize
    points = initial_points.copy()
    velocities = np.zeros((N, 2))
    
    # Parameters
    dt = 0.01
    damping = 0.99 # Velocity damping to settle
    max_iter = int(time_limit / dt) # Rough estimate
    
    best_points = points.copy()
    best_score = min_triangle_area(points)[0]
    
    for step in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Compute min area and forces
        min_area, min_idx = min_triangle_area(points)
        
        # Update best
        if min_area > best_score:
            best_score = min_area
            best_points = points.copy()
            
        forces = compute_forces(points, min_area, min_idx)
        
        # Symplectic integration (Velocity Verlet)
        # v_{n+1/2} = v_n + 0.5 * dt * F_n
        # x_{n+1} = x_n + dt * v_{n+1/2}
        # F_{n+1} = F(x_{n+1})
        # v_{n+1} = v_{n+1/2} + 0.5 * dt * F_{n+1}
        
        # For simplicity and stability with constraints, we use a simpler Euler-Cromer with damping
        # v = v * damping + dt * F
        # x = x + dt * v
        
        velocities = velocities * damping + dt * forces
        points = points + dt * velocities
        
        # Project back to triangle
        points = project_to_triangle(points)
        
        # Zero out velocity for points that hit the boundary? 
        # Or just let the projection handle it. The projection might cause energy gain.
        # To be safe, we can damp velocities slightly more.
        
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    rng = np.random.RandomState(42)
    random.seed(42)
    
    best_points = None
    best_score = -np.inf
    
    # Phase 1: Generate a few diverse initial configurations
    # 1. Grid-like
    # 2. Random
    # 3. Symmetric
    
    initial_configs = []
    
    # Config 1: Jittered Grid
    # Create a 4x4 grid, pick 11 best? Or just a structured set.
    # Let's use a 3x4 grid clipped to triangle.
    grid_x = np.linspace(0.1, 0.9, 4)
    grid_y = np.linspace(0.05, H*0.8, 3)
    pts_grid = []
    for x in grid_x:
        for y in grid_y:
            if is_inside_triangle(np.array([[x, y]])):
                pts_grid.append([x, y])
    # Add center if not there
    if len(pts_grid) < N:
        pts_grid.append([0.5, H/3])
    # Shuffle and take N
    rng.shuffle(pts_grid)
    pts_grid = np.array(pts_grid[:N])
    initial_configs.append(pts_grid)
    
    # Config 2: Random
    pts_rand = np.zeros((N, 2))
    for i in range(N):
        while True:
            u = random.random()
            v = random.random()
            if u + v <= 1.0:
                x = 0.5 - 0.5*u + 0.5*v
                y = (1 - u - v) * H
                pts_rand[i] = [x, y]
                break
    initial_configs.append(pts_rand)
    
    # Config 3: Symmetric "Star"
    pts_star = np.zeros((N, 2))
    pts_star[0] = [0.5, H/3]
    for i in range(1, N):
        angle = (i-1) * 2 * np.pi / (N-1)
        r = 0.3
        x = 0.5 + r * np.cos(angle)
        y = H/3 + r * np.sin(angle)
        # Check if inside, if not, scale r
        for _ in range(10):
            if is_inside_triangle(np.array([[x, y]])):
                break
            r *= 0.8
            x = 0.5 + r * np.cos(angle)
            y = H/3 + r * np.sin(angle)
        pts_star[i] = [x, y]
    initial_configs.append(pts_star)
    
    # Run force-directed relaxation on each config
    for i, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 20.0:
            break
            
        # Run relaxation
        # Use shorter time limits for more restarts if time permits
        # We have ~145s. 3 configs * 20s = 60s. We have room for more.
        
        p, s = force_directed_relaxation(init_pts, time_limit=20.0)
        
        if s > best_score:
            best_score = s
            best_points = p
            
        # If time allows, do a second pass on the best found so far
        if i < 2 and time.time() < deadline - 30.0:
             # Restart from the best found in this config with slightly different noise?
             # Or just run again.
             pass
             
    # If we have time left, run more random restarts
    while time.time() < deadline - 10.0:
        pts_rand = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = random.random()
                v = random.random()
                if u + v <= 1.0:
                    x = 0.5 - 0.5*u + 0.5*v
                    y = (1 - u - v) * H
                    pts_rand[i] = [x, y]
                    break
                    
        p, s = force_directed_relaxation(pts_rand, time_limit=10.0)
        
        if s > best_score:
            best_score = s
            best_points = p
            
    # Fallback
    if best_points is None:
        best_points = initial_configs[0]
        
    # Ensure validity
    if not is_inside_triangle(best_points):
        best_points = project_to_triangle(best_points)
        
    return best_points

# Keep required imports and structure
import random

if __name__ == "__main__":
    pts = construct()
    print(pts)
