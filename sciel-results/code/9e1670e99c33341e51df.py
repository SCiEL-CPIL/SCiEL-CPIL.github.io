"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Pack N equal circles in a unit square [-0.5, 0.5]^2 to maximize the minimum distance.
    Uses a soft-contact dynamics approach with symplectic Euler integration and
    adaptive time-stepping, followed by local refinement via scipy.optimize.
    """
    # Initial positions: grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Add small random perturbation to break symmetry
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.01, size=centers.shape)
    
    # Clip to square
    centers = np.clip(centers, -0.499, 0.499)
    
    # Soft-contact dynamics simulation
    # We want to maximize the minimum distance between centers and from walls.
    # Equivalent to minimizing a potential that penalizes close contacts.
    
    # Parameters
    dt = 0.001
    max_steps = 5000
    damping = 0.98
    k_contact = 100.0  # stiffness for overlap penalty
    power = 4  # high-order polynomial
    
    # We'll simulate forces that push circles apart when they are too close.
    # Target: all pairwise distances > some threshold, and wall distances > some threshold.
    # We'll use an adaptive approach: start with a small target radius and increase it.
    
    # Actually, let's use a different strategy: 
    # 1. Run dynamics to spread out the circles (minimize sum of inverse distances or similar)
    # 2. Then use scipy.optimize to refine the minimum distance
    
    # Step 1: Dynamics to spread out
    # Force: for each pair, if distance < d_target, push apart with force proportional to (d_target - d)^power
    # Also wall forces
    
    d_target = 0.037  # approximate target radius (diameter = 2*r)
    # Actually, let's think in terms of radius. The distance between centers should be >= 2*r.
    # And distance from wall should be >= r.
    
    # Let's use a potential energy approach:
    # E = sum_{i<j} max(0, 2*r - |c_i - c_j|)^power + sum_i sum_k max(0, r - (0.5 - |c_ik|))^power
    
    # But we don't know r yet. Let's use a fixed "desired" separation.
    
    # Alternative: use a repulsive force that is always active, proportional to 1/d^power
    # This will naturally spread out the circles.
    
    # Let's use: F_ij = k / d_ij^power * (c_i - c_j)/d_ij for all pairs
    # And wall forces: F_wall = k_wall / d_wall^power * normal
    
    k_repel = 0.1
    k_wall = 0.1
    power_repel = 2
    
    velocities = np.zeros_like(centers)
    
    for step in range(max_steps):
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                d_vec = centers[i] - centers[j]
                d = np.linalg.norm(d_vec)
                if d < 1e-6:
                    d = 1e-6
                    d_vec = rng.normal(0, 1e-3, size=2)
                f_mag = k_repel / (d ** power_repel)
                f_vec = f_mag * d_vec / d
                forces[i] += f_vec
                forces[j] -= f_vec
        
        # Wall repulsion
        for i in range(N):
            for k in range(2):
                # Left wall: x = -0.5
                d_left = centers[i, k] - (-0.5)
                if d_left < 0.1:
                    f_wall = k_wall / (max(d_left, 1e-6) ** power_repel)
                    forces[i, k] += f_wall
                # Right wall: x = 0.5
                d_right = 0.5 - centers[i, k]
                if d_right < 0.1:
                    f_wall = k_wall / (max(d_right, 1e-6) ** power_repel)
                    forces[i, k] -= f_wall
        
        # Symplectic Euler
        velocities = velocities * damping + forces * dt
        centers += velocities * dt
        
        # Clip to square
        centers = np.clip(centers, -0.5 + 1e-6, 0.5 - 1e-6)
        
        # Adaptive time step: reduce if velocities are large
        max_vel = np.max(np.linalg.norm(velocities, axis=1))
        if max_vel > 0.01:
            dt = max(0.0001, dt * 0.5)
        elif max_vel < 0.001 and dt < 0.001:
            dt = min(0.001, dt * 1.2)
    
    # Step 2: Local refinement using scipy.optimize
    # Maximize the minimum distance (radius)
    
    def negative_radius(x):
        """Return negative of the minimum radius (to minimize)."""
        x = x.reshape(N, 2)
        # Pairwise distances
        min_dist = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(x[i] - x[j])
                if d < min_dist:
                    min_dist = d
        # Wall distances
        for i in range(N):
            for k in range(2):
                d_wall = 0.5 - abs(x[i, k])
                if d_wall < min_dist:
                    min_dist = d_wall
        return -min_dist / 2.0
    
    # Use Nelder-Mead or Powell for local optimization
    # Flatten centers
    x0 = centers.flatten()
    
    # Use Powell method for better convergence
    result = minimize(negative_radius, x0, method='Powell', 
                      options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12})
    
    centers_opt = result.x.reshape(N, 2)
    
    # Clip to ensure validity
    centers_opt = np.clip(centers_opt, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return centers_opt
# EVOLVE-BLOCK-END
