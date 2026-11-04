import numpy as np
from scipy.optimize import minimize

N = 11

def construct():
    # Standard hexagonal lattice positions for 11 unit hexagons
    # Center: (0,0)
    # Ring 1: 6 hexagons at distance sqrt(3) (center-to-center for touching unit hexagons)
    # Ring 2: 4 hexagons at distance 2*sqrt(3)
    
    # Actually for unit regular hexagons (side=1), the center-to-center distance for touching is 2*apothem = sqrt(3)
    # Wait, apothem = sqrt(3)/2, so distance between centers of touching hexagons = 2*apothem = sqrt(3)
    
    # Let me verify: a regular hexagon with side 1 has inradius (apothem) = sqrt(3)/2
    # Two adjacent hexagons sharing an edge: center distance = 2 * sqrt(3)/2 = sqrt(3)
    
    lattice_positions = []
    lattice_angles = []
    
    # Center
    lattice_positions.append([0.0, 0.0])
    lattice_angles.append(0.0)
    
    # Ring 1: 6 hexagons
    for k in range(6):
        theta = k * np.pi / 3
        cx = np.sqrt(3) * np.cos(theta)
        cy = np.sqrt(3) * np.sin(theta)
        lattice_positions.append([cx, cy])
        lattice_angles.append(0.0)
    
    # Ring 2: 4 hexagons at positions that fit in the gaps
    # These should be at distance 2*sqrt(3) from center, at angles 0, pi/3, 2pi/3, pi
    # Actually, let me think: the second ring in a hexagonal packing
    # Positions at (2*sqrt(3)*cos(theta), 2*sqrt(3)*sin(theta)) for theta = 0, pi/3, 2pi/3, pi
    for k in range(4):
        theta = k * np.pi / 3
        cx = 2 * np.sqrt(3) * np.cos(theta)
        cy = 2 * np.sqrt(3) * np.sin(theta)
        lattice_positions.append([cx, cy])
        lattice_angles.append(0.0)
    
    lattice_positions = np.array(lattice_positions)
    lattice_angles = np.array(lattice_angles)
    
    def hex_vertices(cx, cy, angle, s=1.0):
        angles = angle + np.arange(6) * np.pi / 3
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])
    
    def support_function(vertices_list, ocx, ocy, oangle):
        """Compute the minimal side length S of outer hexagon to contain all vertices."""
        cos_a = np.cos(oangle)
        sin_a = np.sin(oangle)
        apothem = np.sqrt(3) / 2
        
        # Transform all vertices to container local coords
        all_verts = []
        for verts in vertices_list:
            lx = verts[:, 0] - ocx
            ly = verts[:, 1] - ocy
            rx = lx * cos_a + ly * sin_a
            ry = -lx * sin_a + ly * cos_a
            all_verts.append(np.column_stack([rx, ry]))
        
        all_verts = np.vstack(all_verts)
        
        # Project onto 6 normals
        S = 0.0
        for k in range(6):
            theta_n = np.pi / 6 + k * np.pi / 3
            nx = np.cos(theta_n)
            ny = np.sin(theta_n)
            proj = all_verts[:, 0] * nx + all_verts[:, 1] * ny
            S = max(S, np.abs(proj).max())
        
        return S / apothem
    
    def smooth_max(x, temperature=0.01):
        """Smooth approximation of max using log-sum-exp."""
        x = np.asarray(x)
        m = x.max()
        return m + temperature * np.log(np.sum(np.exp((x - m) / temperature)))
    
    def overlap_penalty(vertices_list, temperature=0.05):
        """Compute smooth overlap penalty between all pairs of hexagons."""
        n = len(vertices_list)
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                v1 = vertices_list[i]
                v2 = vertices_list[j]
                c1 = v1.mean(axis=0)
                c2 = v2.mean(axis=0)
                dist = np.linalg.norm(c1 - c2)
                if dist < 2.5:  # Only check if potentially close
                    # Compute minimum distance between two convex polygons
                    # Use SAT to find minimum penetration
                    min_pen = -np.inf
                    polys = [v1, v2]
                    for poly in polys:
                        nv = len(poly)
                        for idx in range(nv):
                            p1 = poly[idx]
                            p2 = poly[(idx + 1) % nv]
                            edge = p2 - p1
                            normal = np.array([-edge[1], edge[0]])
                            norm_normal = np.linalg.norm(normal)
                            if norm_normal < 1e-10:
                                continue
                            normal = normal / norm_normal
                            proj1 = v1 @ normal
                            proj2 = v2 @ normal
                            # Penetration depth: if intervals overlap, positive
                            pen = min(proj1.max(), proj2.max()) - max(proj1.min(), proj2.min())
                            if pen > min_pen:
                                min_pen = pen
                    # If min_pen > 0, they overlap. Smooth penalty.
                    if min_pen > 0:
                        penalty += temperature * np.log(1 + np.exp(min_pen / temperature))
        return penalty
    
    def objective(params):
        """
        params: [cx0, cy0, angle0, cx1, cy1, angle1, ..., cx10, cy10, angle10, ocx, ocy, oangle]
        """
        n_inner = N
        cx_arr = params[0:n_inner*3:3]
        cy_arr = params[1:n_inner*3:3]
        ang_arr = params[2:n_inner*3:3]
        ocx = params[n_inner*3]
        ocy = params[n_inner*3+1]
        oangle = params[n_inner*3+2]
        
        vertices_list = []
        for i in range(n_inner):
            verts = hex_vertices(cx_arr[i], cy_arr[i], ang_arr[i])
            vertices_list.append(verts)
        
        S = support_function(vertices_list, ocx, ocy, oangle)
        pen = overlap_penalty(vertices_list)
        
        return S + pen
    
    # Initial parameters: lattice positions + container at origin
    init_params = np.zeros(N*3 + 3)
    for i in range(N):
        init_params[3*i] = lattice_positions[i, 0]
        init_params[3*i+1] = lattice_positions[i, 1]
        init_params[3*i+2] = lattice_angles[i]
    init_params[N*3] = 0.0  # ocx
    init_params[N*3+1] = 0.0  # ocy
    init_params[N*3+2] = 0.0  # oangle
    
    # Optimize using L-BFGS-B
    res = minimize(objective, init_params, method='L-BFGS-B', 
                   options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6})
    
    best_params = res.x
    best_score = res.fun
    
    # Try a few random perturbations
    np.random.seed(42)
    for trial in range(5):
        p = best_params + np.random.normal(0, 0.1, size=best_params.size)
        r = minimize(objective, p, method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-6})
        if r.fun < best_score:
            best_score = r.fun
            best_params = r.x
    
    # Extract results
    inner = []
    for i in range(N):
        cx = best_params[3*i]
        cy = best_params[3*i+1]
        angle = best_params[3*i+2]
        inner.append([cx, cy, np.degrees(angle)])
    
    ocx = best_params[N*3]
    ocy = best_params[N*3+1]
    oangle = best_params[N*3+2]
    
    # Compute final S
    vertices_list = []
    for i in range(N):
        verts = hex_vertices(inner[i][0], inner[i][1], np.radians(inner[i][2]))
        vertices_list.append(verts)
    
    S = support_function(vertices_list, ocx, ocy, oangle)
    
    return inner, (ocx, ocy), S, np.degrees(oangle)
