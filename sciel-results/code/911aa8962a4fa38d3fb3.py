"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Pack 200 equal circles in the unit square [-0.5, 0.5]^2 using a soft-contact
    potential with symplectic Euler integration and adaptive time-stepping.
    """
    # Initialize with a grid plus small random perturbation to break symmetry
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Add small random perturbation to avoid exact grid symmetries
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.01, size=centers.shape)
    
    # Clip to stay within bounds initially
    centers = np.clip(centers, -0.49, 0.49)
    
    n = len(centers)
    
    # Soft-contact potential parameters
    # We use a high-order polynomial penalty for overlaps
    # V(r) = k * (d - r)^p for r < d, where d is the minimum distance
    # But since we're maximizing the minimum radius, we instead minimize
    # the overlap penalty while keeping circles inside the square.
    
    # Approach: We'll use a potential that penalizes:
    # 1. Overlaps between circles: sum over i<j of max(0, r_min - |ci-cj|)^4
    # 2. Circles outside the square: sum over i of max(0, |ci| - 0.5 + r_min)^4
    # where r_min is the current minimum radius we're trying to achieve.
    
    # Better approach: Directly optimize for the maximum minimum distance.
    # Use a potential based on inverse distances to encourage spreading.
    
    # Let's use a different strategy: 
    # Minimize a potential that is large when circles are close or outside.
    # V = sum_{i<j} 1/(|ci-cj|^2 + eps) + sum_i 1/(0.5 - |ci_k|^2 + eps)
    # This encourages large separations and keeping circles inside.
    
    # Actually, let's use the soft-contact dynamics approach as specified:
    # Force-based simulation with symplectic Euler.
    
    # Parameters
    k_repel = 100.0  # Repulsion strength
    k_boundary = 50.0  # Boundary repulsion strength
    damping = 0.995  # Velocity damping
    dt_initial = 0.001  # Initial time step
    dt_min = 1e-6
    dt_max = 0.01
    max_steps = 50000
    energy_tol = 1e-12
    
    # Convert to flat array for easier manipulation
    pos = centers.copy()
    vel = np.zeros_like(pos)
    
    def compute_forces(positions):
        """Compute forces on all circles."""
        forces = np.zeros_like(positions)
        n = len(positions)
        
        # Pairwise repulsion (soft-contact)
        for i in range(n):
            for j in range(i + 1, n):
                diff = positions[i] - positions[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # If coincident, push apart randomly
                    diff = rng.normal(0, 1e-4, size=2)
                    dist = np.linalg.norm(diff)
                # Soft-contact potential: F = k * (1/r^3) for r < 1, scaled
                # Use a high-order polynomial: F = k * (1/r - 1)^4 * (1/r^2) for r < 1
                # Simpler: F = k / r^2 in direction of diff, but capped
                # Let's use F = k * (0.5 - dist)^3 / dist for dist < 0.5
                # This gives a smooth force that vanishes at dist = 0.5
                
                # Actually, for packing, we want to maximize the minimum distance.
                # A good potential is: V(r) = 1/r^4 for small r, which gives F ~ 1/r^5
                # But this can be numerically unstable. Let's use:
                # F = k * (r_target - r)^3 / r for r < r_target, where r_target = 0.5
                # This is a spring-like force that pushes circles apart when too close.
                
                r_target = 0.5
                if dist < r_target:
                    # Force magnitude: k * (r_target - dist)^3 / dist
                    force_mag = k_repel * (r_target - dist) ** 3 / dist
                    force_dir = diff / dist
                    forces[i] += force_dir * force_mag
                    forces[j] -= force_dir * force_mag
        
        # Boundary forces: push circles back inside [-0.5, 0.5]^2
        for i in range(n):
            for axis in range(2):
                coord = positions[i, axis]
                # Distance to boundary
                dist_to_boundary = 0.5 - abs(coord)
                if dist_to_boundary < 0.1:  # Only apply force when close to boundary
                    # Force pushing inward
                    sign = np.sign(coord) if coord != 0 else 1.0
                    force_mag = k_boundary * (0.1 - dist_to_boundary) ** 3
                    forces[i, axis] -= sign * force_mag
        
        return forces
    
    # Main simulation loop
    for step in range(max_steps):
        # Adaptive time-stepping: estimate max force and adjust dt
        forces = compute_forces(pos)
        max_force = np.max(np.abs(forces))
        
        if max_force < 1e-10:
            break
        
        # Adjust dt based on force magnitude
        dt = np.clip(dt_initial / (1 + max_force * 10), dt_min, dt_max)
        
        # Symplectic Euler: update velocity first, then position
        vel = vel * damping + forces * dt
        pos = pos + vel * dt
        
        # Keep positions within bounds (hard constraint)
        pos = np.clip(pos, -0.5, 0.5)
        
        # Check convergence: if max velocity is very small
        max_vel = np.max(np.abs(vel))
        if max_vel < 1e-8:
            break
    
    # After simulation, run a local optimization to fine-tune
    # Use the final positions as initial guess for a more precise optimization
    
    def objective(flat_pos):
        """Objective function: negative of the minimum radius."""
        positions = flat_pos.reshape(-1, 2)
        
        # Compute minimum distance between any two circles
        min_dist = np.inf
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_dist:
                    min_dist = d
        
        # Compute minimum distance to boundary
        min_boundary = np.inf
        for i in range(n):
            for axis in range(2):
                d = 0.5 - abs(positions[i, axis])
                if d < min_boundary:
                    min_boundary = d
        
        # The radius is half the minimum of these
        r = 0.5 * min(min_dist, min_boundary)
        
        # Penalize if any circle is outside
        for i in range(n):
            for axis in range(2):
                if abs(positions[i, axis]) > 0.5:
                    r -= 10.0 * (abs(positions[i, axis]) - 0.5) ** 2
        
        return -r
    
    # Run local optimization
    x0 = pos.flatten()
    result = minimize(objective, x0, method='L-BFGS-B', 
                      bounds=[(-0.5, 0.5)] * (2 * n),
                      options={'maxiter': 1000, 'ftol': 1e-12})
    
    # Use the optimized positions
    final_pos = result.x.reshape(-1, 2)
    
    # Ensure all positions are within bounds
    final_pos = np.clip(final_pos, -0.5, 0.5)
    
    return final_pos
# EVOLVE-BLOCK-END
