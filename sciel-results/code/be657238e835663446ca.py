"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # 1. Initialization: Hexagonal lattice is a good dense start
    # Estimate spacing based on area: pi*1^2 / 300 ~ 0.01047 area per circle
    # r ~ sqrt(0.01047/pi) ~ 0.057. Spacing ~ 2r ~ 0.114
    # Let's start with a slightly tighter lattice to allow expansion
    spacing = 0.11
    k = int(1.0 / spacing) + 2
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x*x + y*y < 1.0:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) < N:
        # Fallback if lattice is too sparse
        points = np.random.rand(N, 2) * 0.9 - 0.45
        # Actually, let's just ensure we have enough points by adjusting spacing
        while len(points) < N:
             spacing *= 0.95
             k = int(1.0 / spacing) + 2
             points = []
             for j in range(-k, k + 1):
                 for i in range(-k, k + 1):
                     x = (i + 0.5 * (j % 2)) * spacing
                     y = j * spacing * np.sqrt(3.0) / 2.0
                     if x*x + y*y < 1.0:
                         points.append([x, y])
             points = np.array(points)
    
    # Take the N points closest to the center (most stable positions)
    dists = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(dists)[:N]
    pts = points[idx].copy()
    
    # 2. Physics-based Relaxation
    # Forces:
    # 1. Repulsion between particles: F_rep = k_rep / dist^2 (along vector)
    # 2. Confinement: F_conf = k_conf * (1 - r) * (r_vec / r) (inward if r>1, but we want to push out to max radius? 
    #    Actually, for packing, we want to push them OUT to the boundary and away from each other.
    #    Better model: 
    #    - Strong repulsion between all pairs.
    #    - Weak attraction to the boundary or just a penalty for being inside?
    #    Standard approach for "packing in a circle": 
    #    Minimize potential energy U = sum_{i<j} 1/|ri-rj|^2 + lambda * sum_i (|ri|^2 - 1)^2 ?
    #    No, we want to MAXIMIZE the minimum distance. 
    #    Equivalent to MINIMIZING the sum of inverse powers of distances.
    #    U = sum_{i<j} 1/d_ij^p
    #    We also need to keep them inside. We can use a hard constraint or a soft potential.
    #    Let's use a soft potential for the boundary: U_bound = sum_i max(0, |ri| - 1)^2
    #    But to maximize radius, we want them to push against the boundary.
    #    Actually, if we just minimize U_rep, they will fly apart until constrained by the boundary.
    #    So: U_total = U_rep + U_boundary_penalty
    #    U_boundary = sum_i (|ri| - 1)^2 if |ri| > 1, else 0. 
    #    But this doesn't push them OUT to the boundary if they are inside. 
    #    To maximize the packing radius, we want the circles to be as far apart as possible.
    #    This is equivalent to finding the configuration that minimizes the maximum overlap.
    #    A common trick: simulate repulsion with a confining potential that is flat inside and rises sharply outside.
    #    But to ensure they fill the space, we might need a slight attraction to the boundary? 
    #    No, repulsion alone with a hard wall is sufficient. They will push outwards.
    
    # Let's use a Verlet-like or simple gradient descent step.
    # Given N=300, O(N^2) force calculation is 90k pairs. 
    # 115s / (steps * 90k * ops). 
    # If we do 1000 steps, 90M pair operations. NumPy vectorization is key.
    
    # Parameters
    k_rep = 1.0
    k_bound = 100.0
    dt = 0.01
    steps = 2000 # Will be adjusted by time
    
    # Precompute indices for vectorization? 
    # i < j pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    for step in range(steps):
        if time.time() > deadline - 5:
            break
            
        # Calculate displacements
        diff = pts[i_idx] - pts[j_idx]
        dist_sq = np.sum(diff**2, axis=1)
        
        # Avoid division by zero
        dist_sq = np.maximum(dist_sq, 1e-12)
        dist = np.sqrt(dist_sq)
        
        # Repulsive force magnitude: k_rep / dist^2
        # Force vector: magnitude * (diff / dist) = k_rep * diff / dist^3
        f_mag = k_rep / dist_sq
        f_vec = f_mag[:, np.newaxis] * diff / dist[:, np.newaxis]
        
        # Accumulate forces
        forces = np.zeros_like(pts)
        np.add.at(forces, i_idx, -f_vec)
        np.add.at(forces, j_idx, f_vec)
        
        # Boundary force
        # If point is outside (r > 1), push in.
        # If point is inside, we might want to push it out slightly to maximize radius?
        # Actually, for "packing", the optimal configuration has many circles touching the boundary.
        # Pure repulsion with a hard wall will push them to the wall.
        # Let's add a small outward force if they are well inside? No, that might cause chaos.
        # Let's stick to: Repulsion + Hard Boundary (velocity zeroing or position clamping).
        # Position clamping is simpler and stable.
        
        # Update positions
        pts += forces * dt
        
        # Clamp to unit circle
        r = np.hypot(pts[:, 0], pts[:, 1])
        outside = r > 1.0
        if np.any(outside):
            pts[outside] /= r[outside, np.newaxis]
            
    # Final check and cleanup
    # Ensure no NaNs
    pts = np.nan_to_num(pts)
    
    return pts
# EVOLVE-BLOCK-END
