"""
Heilbronn convex regions: maximize min triangle area over convex hull area for 13 points.

Strategy: Physics-based Molecular Dynamics (MD) with Simulated Annealing.
We model the points as particles in a 2D box.
The potential energy is designed to be low when the minimum triangle area is high.
Specifically, we use a "soft-min" potential for the triangle areas to make the energy landscape smooth and differentiable, allowing for efficient gradient-based dynamics.
We also add a repulsive force to prevent points from coinciding and a weak attractive force to keep them within a bounded region (though scale is invariant, this helps numerical stability).
"""

import time
import numpy as np
from itertools import combinations
from scipy.spatial import ConvexHull

N = 13

# Pre-compute triangle indices for speed
def get_triangle_indices(n):
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    return np.array(idx)

TRI_IDX = get_triangle_indices(N)
NUM_TRIS = len(TRI_IDX)

def compute_triangle_areas(points):
    """
    Compute areas of all triangles formed by the points.
    Returns an array of shape (NUM_TRIS,).
    """
    p0 = points[TRI_IDX[:, 0]]
    p1 = points[TRI_IDX[:, 1]]
    p2 = points[TRI_IDX[:, 2]]
    
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas

def soft_min(areas, temperature):
    """
    Approximate the minimum of a set of values using a soft-min (log-sum-exp trick).
    As temperature -> 0, this approaches the true minimum.
    """
    # To avoid numerical overflow/underflow, subtract max
    # But since we want min, we can use -logsumexp(-areas/T) * T
    # However, for large N, direct calculation might be unstable.
    # A simpler robust soft-min: 
    # min(a) ~ -1/b * log(sum(exp(-b*a)))
    # We want to maximize the min area, so in energy minimization, we want to minimize -min(area).
    # So Energy ~ -soft_min(areas).
    # Let's compute the soft min value.
    
    # If T is very small, this is just min.
    # If T is larger, it's a smoothed minimum.
    b = 1.0 / temperature
    # Clip areas to avoid extreme values
    a = np.clip(areas, 1e-10, None)
    # Use log-sum-exp for stability
    max_a = np.max(a)
    # sum(exp(-b*(a - max_a)))
    exp_terms = np.exp(-b * (a - max_a))
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = max_a - (1.0 / b) * log_sum
    return soft_min_val

def convex_hull_area(points):
    """
    Calculate the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def score(points):
    """
    Calculate the score: min_triangle_area / convex_hull_area
    """
    areas = compute_triangle_areas(points)
    a_min = np.min(areas)
    a_hull = convex_hull_area(points)
    if a_hull < 1e-10:
        return 0.0
    return a_min / a_hull

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    rng = np.random.default_rng(12345)
    
    best_score = -1.0
    best_points = None
    
    # Initial configurations
    inits = []
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg_points = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(reg_points)
    
    # 2. Random in unit disk
    for _ in range(5):
        r = np.sqrt(rng.random(N))
        theta = rng.uniform(0, 2 * np.pi, N)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(pts)
        
    # 3. Grid-like perturbation
    grid_x = np.linspace(-1, 1, 4)
    grid_y = np.linspace(-1, 1, 4)
    grid_pts = []
    for x in grid_x:
        for y in grid_y:
            grid_pts.append([x, y])
    # Take first 13
    inits.append(np.array(grid_pts[:13]))
    
    def run_md_simulation(init_points, time_budget):
        """
        Run Molecular Dynamics with Simulated Annealing.
        """
        current_time = time.time()
        points = init_points.copy()
        velocities = np.zeros_like(points)
        
        # Annealing parameters
        T_start = 1.0
        T_end = 1e-4
        num_steps = 20000  # Max steps, but we check time
        
        # Force calculation
        def calc_force(pts, T):
            """
            Calculate force on each point to increase min triangle area.
            Force is derived from the gradient of the soft-min of triangle areas.
            """
            areas = compute_triangle_areas(pts)
            b = 1.0 / T
            
            # Gradient of soft-min: 
            # d/dx soft_min(a_i) = sum_i (exp(-b*a_i) / sum_j exp(-b*a_j)) * (da_i/dx)
            # We want to MAXIMIZE soft_min, so force = -d(energy)/dx = +gradient(soft_min)
            
            # Compute weights for each triangle
            max_a = np.max(areas)
            exp_terms = np.exp(-b * (areas - max_a))
            weights = exp_terms / np.sum(exp_terms)
            
            # Gradient of area_i with respect to points
            # Area_i = 0.5 * |cross(p1-p0, p2-p0)|
            # Let's assume the sign of the cross product is consistent (orientation).
            # For a convex configuration, we can fix orientation or use abs.
            # Using abs makes gradient discontinuous at 0. 
            # However, if we assume points are in general position and we start from a convex config,
            # the sign of the cross product for a specific triangle (i,j,k) is fixed locally.
            # To be safe, we compute the gradient of the *signed* area and multiply by the sign of the current area.
            
            force = np.zeros_like(pts)
            
            # Vectorized gradient computation
            p0 = pts[TRI_IDX[:, 0]]
            p1 = pts[TRI_IDX[:, 1]]
            p2 = pts[TRI_IDX[:, 2]]
            
            v1 = p1 - p0
            v2 = p2 - p0
            cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            signs = np.sign(cross)
            
            # dArea/dp0 = -0.5 * sign * perp(v1 - v2) ?? 
            # Let A = 0.5 * (x1-x0)(y2-y0) - (y1-y0)(x2-x0)
            # dA/dx0 = -0.5(y2-y0) + 0.5(y1-y0) = 0.5(y1-y2)
            # dA/dy0 = 0.5(x2-x0) - 0.5(x1-x0) = 0.5(x2-x1)
            # dA/dx1 = 0.5(y2-y0)
            # dA/dy1 = -0.5(x2-x0)
            # dA/dx2 = -0.5(y1-y0)
            # dA/dy2 = 0.5(x1-x0)
            
            # We need to accumulate weighted gradients.
            # force += weights[:, None] * grad_signed_area
            # Since we want to maximize the min area, and we are using soft-min,
            # the "force" direction is the gradient of the soft-min.
            
            # Let's compute the gradient of the signed area for each triangle
            # grad_p0 = [0.5*(y1-y2), 0.5*(x2-x1)]
            # grad_p1 = [0.5*(y2-y0), -0.5*(x2-x0)]
            # grad_p2 = [-0.5*(y1-y0), 0.5*(x1-x0)]
            
            # Note: The actual area is 0.5 * |cross|. 
            # If we use signed area S, Area = 0.5 * |S|.
            # dArea/dp = 0.5 * sign(S) * dS/dp.
            # Our weights are based on Area (positive).
            # So we multiply the gradient of S by 0.5 * sign(S) and the weight.
            
            # Pre-compute gradients of S
            # S = (x1-x0)(y2-y0) - (y1-y0)(x2-x0)
            # dS/dx0 = -(y2-y0) + (y1-y0) = y1 - y2
            # dS/dy0 = (x2-x0) - (x1-x0) = x2 - x1
            # dS/dx1 = y2 - y0
            # dS/dy1 = -(x2 - x0)
            # dS/dx2 = -(y1 - y0)
            # dS/dy2 = x1 - x0
            
            # Area = 0.5 * S (assuming positive orientation for now, we handle sign via 'signs')
            # Actually, if we use weights from |S|, we should just push in the direction that increases |S|.
            # The gradient of |S| is sign(S) * grad(S).
            
            # Let's accumulate
            # We can do this with np.add.at for efficiency
            
            # grad_S_p0
            gx0 = 0.5 * (p1[:, 1] - p2[:, 1])
            gy0 = 0.5 * (p2[:, 0] - p1[:, 0])
            
            # grad_S_p1
            gx1 = 0.5 * (p2[:, 1] - p0[:, 1])
            gy1 = -0.5 * (p2[:, 0] - p0[:, 0])
            
            # grad_S_p2
            gx2 = -0.5 * (p1[:, 1] - p0[:, 1])
            gy2 = 0.5 * (p1[:, 0] - p0[:, 0])
            
            # Multiply by sign(S) to get gradient of Area (not signed)
            # Wait, if S is negative, Area = -0.5*S. dArea/dx = -0.5*dS/dx.
            # sign(S) * dS/dx * 0.5 gives the gradient of 0.5*|S|.
            
            w = weights * signs * 0.5
            
            # Accumulate forces
            # For each triangle t, add w_t * grad_p0 to point i_t
            idx0 = TRI_IDX[:, 0]
            idx1 = TRI_IDX[:, 1]
            idx2 = TRI_IDX[:, 2]
            
            # Use np.add.at
            np.add.at(force, idx0, w[:, None] * np.column_stack([gx0, gy0]))
            np.add.at(force, idx1, w[:, None] * np.column_stack([gx1, gy1]))
            np.add.at(force, idx2, w[:, None] * np.column_stack([gx2, gy2]))
            
            # Add a small repulsive force to prevent overlap (distance based)
            # This helps with numerical stability
            # F_rep = k * (1/r^2) * r_vec
            # We can approximate this or just rely on the triangle area force.
            # Triangle area force naturally prevents collinearity.
            # Let's add a very weak centering force to keep points in view
            force -= 0.001 * points
            
            return force

        # MD Loop
        dt = 0.01
        for step in range(num_steps):
            if time.time() > time_budget:
                break
            
            T = T_start * (T_end / T_start) ** (step / num_steps)
            
            f = calc_force(points, T)
            
            # Velocity Verlet or simple Euler
            # Simple Euler with damping
            velocities = 0.9 * velocities + dt * f
            points = points + dt * velocities
            
            # Normalize scale? 
            # The problem is scale invariant. 
            # If points fly apart, area increases, but hull area increases too.
            # The ratio is what matters.
            # To keep numbers stable, we can periodically rescale points to unit hull area.
            if step % 100 == 0:
                a_hull = convex_hull_area(points)
                if a_hull > 1e-10:
                    scale = 1.0 / np.sqrt(a_hull)
                    points *= scale
                    velocities *= scale
            
            # Evaluate score periodically
            if step % 100 == 0:
                s = score(points)
                # We return the score, let the caller update best
                # But we need to return the points that gave the best score in this simulation
                # For now, we just track locally
                if s > best_score_local:
                    best_score_local = s
                    best_points_local = points.copy()
                    
        return best_points_local, best_score_local

    best_score_local = -1.0
    best_points_local = None
    
    for init in inits:
        if time.time() > deadline:
            break
            
        # Run MD for this init
        # Give each init a portion of the remaining time
        remaining = deadline - time.time()
        # Don't spend too long on one init if we have many
        t_budget = min(remaining / (len(inits)), 10.0) 
        if t_budget < 1.0:
            continue
            
        pts, s = run_md_simulation(init, time.time() + t_budget)
        
        if s > best_score:
            best_score = s
            best_points = pts

    # Final polish with Nelder-Mead on the best found points
    if best_points is not None:
        # Define objective for NM
        def objective(x):
            pts = x.reshape(-1, 2)
            # We want to maximize score. Minimize -score.
            # But score involves convex hull which is not differentiable.
            # NM doesn't need gradients.
            return -score(pts)
        
        x0 = best_points.flatten()
        try:
            from scipy.optimize import minimize
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < -best_score:
                best_score = -res.fun
                best_points = res.x.reshape(-1, 2)
        except Exception:
            pass

    if best_points is None:
        # Fallback
        best_points = inits[0]
        best_score = score(best_points)

    return best_points

# EVOLVE-BLOCK-END
