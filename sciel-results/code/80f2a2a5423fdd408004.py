import numpy as np
import time
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def required_outer_side(positions):
        """
        Compute the exact side length of the smallest regular hexagon (angle 0, centered at origin)
        that contains all unit hexagons at given positions.
        """
        # Vertices of a unit hexagon at origin, angle 0
        # Angles: 0, 60, 120, 180, 240, 300
        # x = cos(0)=1, y=0
        # x = cos(60)=0.5, y=sin(60)=sqrt(3)/2
        # etc.
        base_verts = np.array([
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2],
            [-0.5, np.sqrt(3)/2],
            [-1.0, 0.0],
            [-0.5, -np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2]
        ])
        
        # positions shape (N, 2)
        # verts shape (N, 6, 2)
        verts = positions[:, np.newaxis, :] + base_verts[np.newaxis, :, :]
        verts_flat = verts.reshape(-1, 2)
        
        x = verts_flat[:, 0]
        y = verts_flat[:, 1]
        sqrt3 = np.sqrt(3.0)
        
        # Normals for hexagon with flat top/bottom (vertices at 0, 60, ...)
        # Normals are at 30, 90, 150, 210, 270, 330
        # n_0 = (cos(30), sin(30)) = (sqrt(3)/2, 1/2)
        # n_1 = (cos(90), sin(90)) = (0, 1)
        # n_2 = (cos(150), sin(150)) = (-sqrt(3)/2, 1/2)
        # n_3 = (cos(210), sin(210)) = (-sqrt(3)/2, -1/2)
        # n_4 = (cos(270), sin(270)) = (0, -1)
        # n_5 = (cos(330), sin(330)) = (sqrt(3)/2, -1/2)
        
        L1 = x * (sqrt3/2) + y * (1/2)
        L2 = y
        L3 = x * (-sqrt3/2) + y * (1/2)
        L4 = x * (-sqrt3/2) + y * (-1/2)
        L5 = -y
        L6 = x * (sqrt3/2) + y * (-1/2)
        
        all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
        max_proj = np.max(all_L)
        return (2.0 / sqrt3) * max_proj

    def check_validity_vectorized(positions, tol=1e-4):
        """
        Check if hexagons are disjoint using vectorized SAT.
        positions: (N, 2)
        Returns True if valid.
        """
        # Base vertices for unit hexagon
        base_verts = np.array([
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2],
            [-0.5, np.sqrt(3)/2],
            [-1.0, 0.0],
            [-0.5, -np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2]
        ])
        
        # Vertices for all hexagons: (N, 6, 2)
        verts = positions[:, np.newaxis, :] + base_verts[np.newaxis, :, :]
        
        # Edges: (N, 6, 2)
        edges = np.roll(verts, -1, axis=1) - verts
        
        # Normals: (N, 6, 2)
        normals = np.column_stack((-edges[:, :, 1], edges[:, :, 0]))
        norms = np.linalg.norm(normals, axis=2, keepdims=True)
        normals = normals / (norms + 1e-12)
        
        # For each pair (i, j), check separation along all normals of i and j.
        # We can vectorize over pairs.
        # Total pairs: N*(N-1)/2
        # Instead of looping pairs, we can compute projections for all pairs at once.
        
        # Let's loop over pairs but vectorize the projection calculation for speed.
        # N=12 is small, so N^2/2 = 66 pairs.
        
        for i in range(N):
            for j in range(i + 1, N):
                v1 = verts[i]
                v2 = verts[j]
                n1 = normals[i]
                n2 = normals[j]
                
                # Project v1 and v2 onto all 12 normals (6 from v1, 6 from v2)
                # Actually, due to symmetry and same orientation, normals are the same set rotated?
                # No, they are fixed in world frame since angle is 0 for all.
                # So n1 and n2 are the same set of vectors.
                # We only need to check 6 unique directions.
                # Let's just use n1 (which is n2).
                
                # Projections of v1 onto n1: (6,)
                proj1 = np.dot(v1, n1.T)
                # Projections of v2 onto n1: (6,)
                proj2 = np.dot(v2, n1.T)
                
                min1 = np.min(proj1, axis=0)
                max1 = np.max(proj1, axis=0)
                min2 = np.min(proj2, axis=0)
                max2 = np.max(proj2, axis=0)
                
                # Check if any axis separates them
                # Separation if max1 + tol < min2 OR max2 + tol < min1
                sep1 = max1 + tol < min2
                sep2 = max2 + tol < min1
                if np.any(sep1) or np.any(sep2):
                    continue
                else:
                    return False
        return True

    def generate_initial_positions():
        """
        Generate a diverse set of initial positions using a honeycomb-like structure
        but with random perturbations to break symmetry and allow PSO to explore.
        """
        positions = np.zeros((N, 2))
        # Center
        positions[0] = [0.0, 0.0]
        # Ring 1
        for i in range(6):
            theta = np.radians(60 * i)
            positions[i+1, 0] = np.sqrt(3) * np.cos(theta)
            positions[i+1, 1] = np.sqrt(3) * np.sin(theta)
        # Ring 2
        for i in range(5):
            theta = np.radians(72 * i + 30) # Offset to fit in gaps
            positions[i+7, 0] = 2 * np.sqrt(3) * np.cos(theta)
            positions[i+7, 1] = 2 * np.sqrt(3) * np.sin(theta)
            
        # Add small random noise
        noise = np.random.normal(0, 0.1, positions.shape)
        positions += noise
        return positions

    def psso_optimize(initial_pos, max_iter=2000, swarm_size=20):
        """
        Particle Swarm Optimization.
        """
        # Swarm of particles
        # Each particle is (N, 2)
        
        # Initialize swarm
        swarm = np.array([initial_pos + np.random.normal(0, 0.5, initial_pos.shape) for _ in range(swarm_size)])
        
        # Personal bests
        pbest_pos = swarm.copy()
        pbest_fit = np.zeros(swarm_size)
        
        # Global best
        gbest_pos = initial_pos.copy()
        gbest_fit = float('inf')
        
        # Velocity
        v = np.random.normal(0, 0.1, (swarm_size, N, 2))
        
        # PSO parameters
        w = 0.9
        c1 = 1.5
        c2 = 1.5
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            # Evaluate swarm
            for i in range(swarm_size):
                pos = swarm[i]
                if check_validity_vectorized(pos):
                    fit = required_outer_side(pos)
                else:
                    fit = float('inf')
                    
                # Update personal best
                if fit < pbest_fit[i]:
                    pbest_fit[i] = fit
                    pbest_pos[i] = pos.copy()
                    
                # Update global best
                if fit < gbest_fit:
                    gbest_fit = fit
                    gbest_pos = pos.copy()
            
            # Update velocities and positions
            for i in range(swarm_size):
                r1 = np.random.random((N, 2))
                r2 = np.random.random((N, 2))
                
                v[i] = w * v[i] + c1 * r1 * (pbest_pos[i] - swarm[i]) + c2 * r2 * (gbest_pos - swarm[i])
                
                # Clamp velocity
                max_v = 0.5
                v[i] = np.clip(v[i], -max_v, max_v)
                
                swarm[i] += v[i]
            
            # Adaptive parameters
            w = max(0.4, w * 0.999)
            
            # Optional: Local refinement every 100 iterations
            if it % 100 == 0 and gbest_fit < float('inf'):
                # Simple Nelder-Mead on gbest
                # We need a function that takes a flat array
                def nm_func(flat_pos):
                    pos = flat_pos.reshape(N, 2)
                    if check_validity_vectorized(pos):
                        return required_outer_side(pos)
                    return float('inf')
                
                x0 = gbest_pos.flatten()
                # Simple NM implementation
                n = len(x0)
                simplex = [x0.copy()]
                step = 0.1
                for k in range(n):
                    x = x0.copy()
                    x[k] += step
                    simplex.append(x)
                f = [nm_func(x) for x in simplex]
                
                for _ in range(50):
                    order = np.argsort(f)
                    simplex = [simplex[i] for i in order]
                    f = [f[i] for i in order]
                    if abs(f[-1] - f[0]) < 1e-6:
                        break
                    centroid = np.mean(simplex[:-1], axis=0)
                    xr = centroid + (centroid - simplex[-1])
                    fr = nm_func(xr)
                    if fr < f[0]:
                        xe = centroid + 2 * (centroid - simplex[-1])
                        fe = nm_func(xe)
                        if fe < fr:
                            simplex[-1] = xe
                            f[-1] = fe
                        else:
                            simplex[-1] = xr
                            f[-1] = fr
                    elif fr < f[-2]:
                        simplex[-1] = xr
                        f[-1] = fr
                    else:
                        xc = centroid + 0.5 * (simplex[-1] - centroid)
                        fc = nm_func(xc)
                        if fc < f[-1]:
                            simplex[-1] = xc
                            f[-1] = fc
                        else:
                            for k in range(1, n+1):
                                simplex[k] = simplex[0] + 0.5 * (simplex[k] - simplex[0])
                                f[k] = nm_func(simplex[k])
                
                best_idx = np.argmin(f)
                if f[best_idx] < gbest_fit:
                    gbest_fit = f[best_idx]
                    gbest_pos = simplex[best_idx].reshape(N, 2)
                    # Update swarm to center around new gbest
                    swarm = np.array([gbest_pos + np.random.normal(0, 0.1, gbest_pos.shape) for _ in range(swarm_size)])
                    pbest_pos = swarm.copy()
                    pbest_fit = np.array([nm_func(p.reshape(N,2).flatten()) for p in pbest_pos])
                    v = np.random.normal(0, 0.1, (swarm_size, N, 2))

        return gbest_pos, gbest_fit

    # Initial configuration
    init_pos = generate_initial_positions()
    
    # Run PSO
    best_pos, best_fit = psso_optimize(init_pos)
    
    # If PSO failed or returned inf, try a fallback
    if not np.isfinite(best_fit):
        # Fallback: Linear arrangement
        best_pos = np.array([[i * 3.0, 0.0] for i in range(N)])
        best_fit = required_outer_side(best_pos)
    
    # Final validity check
    if not check_validity_vectorized(best_pos):
        # This should not happen if PSO is working, but just in case
        # Revert to a known valid large config
        best_pos = generate_initial_positions()
        # Ensure it's valid by expanding if necessary
        while not check_validity_vectorized(best_pos):
            best_pos *= 1.01
        best_fit = required_outer_side(best_pos)

    final_side = required_outer_side(best_pos)
    
    inner = []
    for i in range(N):
        inner.append([float(best_pos[i, 0]), float(best_pos[i, 1]), 0.0])
        
    return inner, (0.0, 0.0), float(final_side), 0.0

# EVOLVE-BLOCK-END
