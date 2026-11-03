import numpy as np
import time

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Precompute vertices of a unit hexagon (side length 1)
    # Vertices are at angles 0, 60, ..., 300 degrees
    # Side length 1 -> Circumradius 1
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    unit_hex_verts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Function to get vertices of a hexagon at position (x, y) and rotation theta
    def get_hex_verts(x, y, theta):
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return unit_hex_verts @ rot.T + np.array([x, y])

    # Function to compute minimal outer side length S for a given set of inner hexagons
    # inner_pos: (N, 2) array of centers
    # inner_rot: (N,) array of rotations
    def min_outer_side(inner_pos, inner_rot):
        # Collect all vertices of all inner hexagons
        # This might be slow if done naively, but N=12 is small.
        # We can vectorize the vertex generation.
        # inner_pos: (N, 2), inner_rot: (N,)
        # We need to rotate unit_hex_verts for each hex and translate.
        
        # Expand to (N, 6, 2)
        # rot matrices: (N, 2, 2)
        c = np.cos(inner_rot)
        s = np.sin(inner_rot)
        rots = np.zeros((N, 2, 2))
        rots[:, 0, 0] = c
        rots[:, 0, 1] = -s
        rots[:, 1, 0] = s
        rots[:, 1, 1] = c
        
        # Rotate vertices: (N, 2, 2) @ (6, 2).T -> (N, 2, 6)
        # Better: unit_hex_verts (6, 2)
        # verts = np.einsum('nij,jk->nik', rots, unit_hex_verts.T) -> (N, 2, 6)
        # Then transpose to (N, 6, 2) and add centers
        verts = np.einsum('nij,jk->nik', rots, unit_hex_verts.T).transpose(0, 2, 1)
        verts += inner_pos[:, np.newaxis, :]
        
        # Flatten to (N*6, 2)
        all_verts = verts.reshape(-1, 2)
        
        # Now find min S over outer rotation theta_outer
        # S(theta) = (2/sqrt(3)) * max_k max_p n_k(theta) . p
        # n_k(theta) are normals to the 6 sides.
        # Normal angles: theta + k*pi/3
        # We can sample theta_outer.
        # To be efficient, we can note that the maximum of dot products is convex in some sense? No.
        # But we can just sample 360 angles for theta_outer.
        theta_outers = np.linspace(0, np.pi/3, 180) # Symmetry: pi/3 is enough? No, 2pi. But hexagon has 6-fold symmetry.
        # Actually, rotating outer hex by 60 deg is same. So 0 to pi/3 is enough.
        # Let's use 360 steps for safety.
        theta_outers = np.linspace(0, 2*np.pi, 360, endpoint=False)
        
        min_S = np.inf
        # We can vectorize the dot products
        # all_verts: (M, 2) where M = 72
        # n_k: (K, 2) where K = 360
        # dot: (K, M)
        # max_p: (K,)
        # S_k: (K,)
        # min_S = min(S_k)
        
        # Normals for all theta_outers
        # n_k(theta) = [cos(theta + k*pi/3), sin(theta + k*pi/3)]
        # We only need the max over k for each theta.
        # max_k n_k(theta) . p
        # This is the support function of the set of points in direction theta + k*pi/3.
        # Actually, for a fixed theta, we compute h(theta + k*pi/3) for k=0..5 and take max.
        # h(phi) = max_p p . [cos(phi), sin(phi)]
        
        # Let's compute h for all phi in a fine grid?
        # Or just compute for the 6 normals of each theta_outer.
        
        # For each theta_outer, compute 6 normals.
        # normals = np.column_stack([np.cos(theta_outers[:, np.newaxis] + np.arange(6)*np.pi/3),
        #                             np.sin(theta_outers[:, np.newaxis] + np.arange(6)*np.pi/3)])
        # normals shape: (360, 6, 2)
        # dots = np.einsum('kmn,mp->knp', normals, all_verts) # (360, 6, 72)
        # max_dots = dots.max(axis=2) # (360, 6)
        # max_over_k = max_dots.max(axis=1) # (360,)
        # S_vals = max_over_k * (2/np.sqrt(3))
        # min_S = S_vals.min()
        
        # This is 360 * 6 * 72 operations ~ 150k. Very fast.
        # Let's implement this.
        
        k_angles = np.arange(6) * np.pi / 3
        # theta_outers: (360,)
        # phi: (360, 6)
        phi = theta_outers[:, np.newaxis] + k_angles[np.newaxis, :]
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        
        # dots: (360, 6, 72)
        # all_verts: (72, 2)
        dots = all_verts[0] * cos_phi[:, :, np.newaxis] + all_verts[1] * sin_phi[:, :, np.newaxis]
        # Wait, all_verts is (72, 2).
        # dots[i, j, m] = all_verts[m, 0] * cos_phi[i, j] + all_verts[m, 1] * sin_phi[i, j]
        # Using broadcasting:
        # cos_phi: (360, 6, 1)
        # sin_phi: (360, 6, 1)
        # all_verts: (72, 2) -> need (1, 1, 72, 2)? No.
        # Let's reshape all_verts to (72, 1, 2) ? No.
        # dots = np.sum(all_verts[np.newaxis, np.newaxis, :, :] * np.stack([cos_phi, sin_phi], axis=-1), axis=-1)
        # This creates (360, 6, 72).
        
        # Optimization:
        # h(phi) = max_m (x_m cos phi + y_m sin phi)
        # We can compute this for all phi.
        
        # Let's just do it.
        dots = all_verts[0][np.newaxis, np.newaxis, :] * cos_phi + all_verts[1][np.newaxis, np.newaxis, :] * sin_phi
        max_dots = dots.max(axis=2) # (360, 6)
        max_over_k = max_dots.max(axis=1) # (360,)
        S_vals = max_over_k * (2.0 / np.sqrt(3.0))
        return S_vals.min()

    # Initial configuration: Honeycomb cluster
    # Center at (0,0)
    # Ring 1: 6 hexagons at distance 2? No, distance between centers in honeycomb is 2*inradius = sqrt(3)?
    # Side length 1 -> Inradius sqrt(3)/2. Distance between centers = 2 * sqrt(3)/2 = sqrt(3).
    # Let's place them in a hexagonal lattice.
    # Basis vectors: a1 = (sqrt(3), 0), a2 = (sqrt(3)/2, 1.5)
    # No, standard hex lattice spacing is sqrt(3) for side 1.
    # Let's use a1 = (sqrt(3), 0), a2 = (sqrt(3)/2, 3/2)
    # Distance a1 = sqrt(3). Distance a2 = sqrt(3/4 + 9/4) = sqrt(12/4) = sqrt(3).
    # Angle 60 deg. Correct.
    
    pos = np.zeros((N, 2))
    rot = np.zeros(N)
    
    # Center hex
    pos[0] = [0, 0]
    
    # 6 neighbors
    idx = 1
    for i in range(6):
        angle = i * np.pi / 3
        pos[idx] = [np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)]
        rot[idx] = 0
        idx += 1
        
    # 5 more neighbors attached to the ring
    # Pick 5 of the 12 outer positions of the H3 shell?
    # H3 shell has 12 positions.
    # Let's just add them at distance 2*sqrt(3) from center, at angles 0, 60, ...
    # Actually, let's place them adjacent to the ring hexagons.
    # For each of the 6 ring hexagons, there are 2 outer positions.
    # We need 5 more.
    # Let's pick the ones at angles 0, 60, 120, 180, 240 for the outer shell?
    # Coordinates of outer shell:
    # For a ring hex at angle alpha, its outer neighbors are at alpha +/- 30 deg? No.
    # Let's just use the lattice vectors.
    # Positions in H3 shell (distance 2*sqrt(3) from center? No, lattice coords (2,0) etc).
    # Lattice coords (i, j) -> pos = i*a1 + j*a2
    # H3 shell has 12 points: (2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1), (0,-2), (1,-1), (2,-1), (1,0), (0,1), (-1,0) ? No.
    # Let's just generate 12 points in a hexagonal pattern and pick 12.
    
    # Simpler: Start with a random or heuristic config and optimize.
    # Let's try a specific heuristic:
    # 1 center, 6 around it, 5 on the periphery.
    
    # Let's initialize with a tight cluster.
    # We'll use a simple local search.
    
    best_S = np.inf
    best_pos = pos.copy()
    best_rot = rot.copy()
    
    # Local search
    # Variables: pos (24 floats), rot (12 floats)
    # Step size decay
    
    step_size = 0.5
    max_iter = 5000
    
    # We will use a simple hill climbing with random restarts.
    # Number of restarts depends on time.
    
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        # Initialize
        if restarts == 1:
            cur_pos = best_pos.copy()
            cur_rot = best_rot.copy()
        else:
            # Perturb best or random
            if np.random.random() < 0.5:
                cur_pos = best_pos + np.random.normal(0, 0.5, (N, 2))
                cur_rot = best_rot + np.random.normal(0, 0.5, N)
            else:
                # Random honeycomb-like
                cur_pos = np.zeros((N, 2))
                cur_rot = np.zeros(N)
                # Center
                cur_pos[0] = [0,0]
                for i in range(1, 7):
                    angle = (i-1) * np.pi / 3
                    cur_pos[i] = [np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)]
                for i in range(7, 12):
                    angle = (i-7) * np.pi / 3
                    cur_pos[i] = [2*np.sqrt(3) * np.cos(angle), 2*np.sqrt(3) * np.sin(angle)]
                cur_rot = np.random.uniform(0, np.pi/3, N)
                
        cur_S = min_outer_side(cur_pos, cur_rot)
        if cur_S < best_S:
            best_S = cur_S
            best_pos = cur_pos.copy()
            best_rot = cur_rot.copy()
            
        # Local optimization
        for i in range(max_iter):
            if time.time() > deadline:
                break
            # Pick a random hexagon
            j = np.random.randint(N)
            # Perturb position
            dx = np.random.normal(0, step_size)
            dy = np.random.normal(0, step_size)
            dtheta = np.random.normal(0, step_size / 2)
            
            old_px = cur_pos[j, 0]
            old_py = cur_pos[j, 1]
            old_rot = cur_rot[j]
            
            cur_pos[j, 0] += dx
            cur_pos[j, 1] += dy
            cur_rot[j] += dtheta
            
            new_S = min_outer_side(cur_pos, cur_rot)
            
            if new_S < cur_S:
                cur_S = new_S
                if cur_S < best_S:
                    best_S = cur_S
                    best_pos = cur_pos.copy()
                    best_rot = cur_rot.copy()
            else:
                # Revert
                cur_pos[j, 0] = old_px
                cur_pos[j, 1] = old_py
                cur_rot[j] = old_rot
                
        # Decay step size
        step_size *= 0.95
        
        # If no improvement in a while, break?
        # For now, just run restarts.

    # Format output
    inner = []
    for i in range(N):
        x, y = best_pos[i]
        angle_deg = np.degrees(best_rot[i]) % 360
        inner.append([x, y, angle_deg])
        
    # Outer hexagon: centered at (0,0), side length best_S, rotation 0
    # We need to return the rotation that minimizes S.
    # Our min_outer_side function found the min S over theta, but didn't return the theta.
    # We need to find the theta that gives best_S.
    
    # Recompute theta
    theta_outers = np.linspace(0, 2*np.pi, 360, endpoint=False)
    k_angles = np.arange(6) * np.pi / 3
    phi = theta_outers[:, np.newaxis] + k_angles[np.newaxis, :]
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    
    # Reconstruct verts for best_pos, best_rot
    c = np.cos(best_rot)
    s = np.sin(best_rot)
    rots = np.zeros((N, 2, 2))
    rots[:, 0, 0] = c
    rots[:, 0, 1] = -s
    rots[:, 1, 0] = s
    rots[:, 1, 1] = c
    verts = np.einsum('nij,jk->nik', rots, unit_hex_verts.T).transpose(0, 2, 1)
    verts += best_pos[:, np.newaxis, :]
    all_verts = verts.reshape(-1, 2)
    
    dots = all_verts[0][np.newaxis, np.newaxis, :] * cos_phi + all_verts[1][np.newaxis, np.newaxis, :] * sin_phi
    max_dots = dots.max(axis=2)
    max_over_k = max_dots.max(axis=1)
    S_vals = max_over_k * (2.0 / np.sqrt(3.0))
    
    best_theta_idx = np.argmin(S_vals)
    best_theta = theta_outers[best_theta_idx]
    
    return inner, (0.0, 0.0), best_S, np.degrees(best_theta)

