"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # 1. Initialization: Hexagonal lattice for a good dense start
    # Estimate spacing to fit N points in unit circle
    # Area approx: N * (2r)^2 * (2/sqrt(3)) <= pi * 1^2
    # r ~ sqrt(pi / (N * 4 * 2/sqrt(3)))
    r_est = np.sqrt(np.pi / (N * 4 * (2.0/np.sqrt(3.0))))
    
    # Generate a grid of points with spacing slightly smaller than 2*r_est to ensure we have enough candidates
    spacing = 2.0 * r_est * 0.95
    k = int(1.0 / spacing) + 3
    
    # Generate hexagonal lattice points
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            points.append([x, y])
    points = np.array(points)
    
    # Filter points to be well inside the unit circle initially
    # We want a superset of N points to choose from, or just take the first N if we have enough
    # Actually, for MD, we want exactly N points. 
    # Let's select the N points closest to the origin from the lattice.
    radii = np.linalg.norm(points, axis=1)
    points = points[np.argsort(radii)[:N]]
    
    # Ensure we have exactly N points
    if len(points) < N:
        # Fallback: random if lattice generation failed (unlikely)
        angles = np.random.uniform(0, 2*np.pi, N)
        radii_rand = np.sqrt(np.random.uniform(0, 1, N))
        points = np.column_stack([radii_rand * np.cos(angles), radii_rand * np.sin(angles)])

    # 2. Molecular Dynamics Relaxation
    # We want to maximize the minimum distance between any two points, 
    # and the minimum distance from any point to the boundary (1 - |p|).
    # This is equivalent to minimizing the "energy" where energy is a function of violations.
    
    # Parameters for MD
    dt = 0.005
    friction = 0.9
    repulsion_power = 4  # Higher power makes the force steeper, helping to resolve contacts
    boundary_force_strength = 5.0
    
    velocities = np.zeros_like(points)
    
    # Pre-compute some constants
    max_iter = 20000
    
    for step in range(max_iter):
        if time.time() > deadline:
            break
            
        # Compute forces
        forces = np.zeros_like(points)
        
        # Pairwise repulsion
        # Use a vectorized approach for efficiency, but N=300 is small enough that 
        # a loop or simple broadcasting might be fine. 
        # Broadcasting: (N, 1, 2) - (1, N, 2) -> (N, N, 2)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        
        # Avoid division by zero
        np.fill_diagonal(dists_sq, 1.0)
        dists = np.sqrt(dists_sq)
        
        # Force magnitude: k * (1/d^p - 1) if d < 1, else 0? 
        # We want them to push apart until they are as far apart as possible.
        # A common potential is U = 1/d^2. Force = -grad U = 2/d^3 * r_hat.
        # To maximize min dist, we can just apply a repulsive force 1/d^p.
        # We don't need a "rest length" because we just want them to spread out.
        # However, without attraction, they will fly to the boundary.
        # The boundary constraint provides the counter-force.
        
        # Force vector: direction is diff, magnitude is 1/d^p
        # F_ij = (p_i - p_j) / |p_i - p_j|^p
        inv_dists_p = np.power(dists_sq, -repulsion_power/2.0)
        # Zero out self-interaction
        np.fill_diagonal(inv_dists_p, 0.0)
        
        # forces[i] = sum_j (p_i - p_j) * inv_dists_p[i,j]
        # diff is (N, N, 2). inv_dists_p is (N, N).
        # We need to multiply diff by inv_dists_p along the last axis.
        force_pairs = diff * inv_dists_p[:, :, np.newaxis]
        forces = np.sum(force_pairs, axis=1)
        
        # Boundary force
        # For each point, if |p| > 1 - epsilon, push it inward.
        # Actually, we want them to be as close to 1 as possible but not exceed it.
        # A smooth potential for boundary: U_b = (1 - |p|)^2 if we want to maximize distance to boundary?
        # No, we want to MAXIMIZE the radius of the circles.
        # The radius of a circle at p is limited by min( (1-|p|), min_j |p_i-p_j|/2 ).
        # To maximize the global min radius, we should push points towards the boundary 
        # and push points away from each other.
        
        # Let's use a force that pushes points towards the boundary if they are inside,
        # but strongly repels if they cross.
        # Better: Apply a force proportional to (1 - |p|) towards the center? No, that pulls in.
        # We want to push OUT to the boundary.
        # Force = (1 - |p|) * p_hat ? This pushes out.
        # But if they hit the boundary, they need to slide or stop.
        # Let's use a constraint projection method instead of pure forces for the boundary.
        
        # Update velocities
        velocities = velocities * friction + forces * dt
        points += velocities * dt
        
        # Boundary Constraint: Project points back inside the unit circle
        # If a point is outside, project it to the boundary.
        # This is a hard constraint.
        radii = np.linalg.norm(points, axis=1)
        outside_mask = radii > 1.0
        if np.any(outside_mask):
            # Project to circle
            points[outside_mask] /= radii[outside_mask, np.newaxis]
            # Zero out radial velocity component to prevent bouncing out again immediately
            # v_radial = (v . p_hat) p_hat
            # v_tangent = v - v_radial
            # This is complex to vectorize simply. Simple damping is enough.
            velocities[outside_mask] *= 0.5 # Dampen velocity upon collision
        
        # Optional: Center of mass correction to keep packing centered?
        # The forces are symmetric, so it should stay centered.
        
    # Final polish: Calculate the actual score
    # Score is min( min_{i<j} |p_i - p_j| / 2, min_i (1 - |p_i|) )
    
    # Calculate min pairwise distance
    # Re-use the dists calculation logic
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    min_pair_dist = np.sqrt(np.min(dists_sq))
    
    # Calculate min boundary distance
    radii = np.linalg.norm(points, axis=1)
    min_boundary_dist = 1.0 - np.max(radii)
    
    # The radius is the minimum of these two
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    
    # Check validity
    if not np.all(np.isfinite(points)):
        # Fallback to lattice if something went wrong
        # Re-generate lattice
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        pts = np.array(rows)
        rad = np.hypot(pts[:, 0], pts[:, 1])
        pts, rad = pts[rad <= 1.0 - spacing / 2.0], rad[rad <= 1.0 - spacing / 2.0]
        if len(pts) >= N:
            return pts[np.argsort(rad, kind="stable")[:N]]
        else:
            return np.zeros((N, 2))
            
    return points
# EVOLVE-BLOCK-END
