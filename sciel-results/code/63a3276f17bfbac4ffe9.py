"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    def hex_vertices(cx, cy, angle, s=1.0):
        """Return the 6 vertices of a regular hexagon."""
        angles = angle + np.arange(6) * np.pi / 3
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def compute_S_for_container(vertices_list, ocx, ocy, oangle):
        """Compute the minimum side length S of the outer hexagon (centered at ocx, ocy, rot oangle) to contain all vertices."""
        cos_a = np.cos(oangle)
        sin_a = np.sin(oangle)
        S = 0.0
        apothem = np.sqrt(3) / 2
        for verts in vertices_list:
            for v in verts:
                lx = v[0] - ocx
                ly = v[1] - ocy
                rx = lx * cos_a + ly * sin_a
                ry = -lx * sin_a + ly * cos_a
                for k in range(6):
                    theta_n = np.pi / 6 + k * np.pi / 3
                    proj = rx * np.cos(theta_n) + ry * np.sin(theta_n)
                    S = max(S, abs(proj))
        return S / apothem

    def sat_overlap(v1, v2):
        """Check if two convex polygons overlap using Separating Axis Theorem."""
        polys = [v1, v2]
        for poly in polys:
            n = len(poly)
            for i in range(n):
                p1 = poly[i]
                p2 = poly[(i + 1) % n]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                if proj1.max() < proj2.min() or proj2.max() < proj1.min():
                    return False
        return True

    def objective(params):
        """
        params: [cx1, cy1, a1, ..., cxN, cyN, aN, ocx, ocy, oangle]
        Returns S + penalty.
        """
        n = N
        # Extract inner params
        inner_params = params[:3 * n]
        container_params = params[3 * n:]
        ocx, ocy, oangle = container_params
        
        vertices_list = []
        for i in range(n):
            cx = inner_params[3 * i]
            cy = inner_params[3 * i + 1]
            angle = inner_params[3 * i + 2]
            verts = hex_vertices(cx, cy, angle)
            vertices_list.append(verts)
        
        S = compute_S_for_container(vertices_list, ocx, ocy, oangle)
        
        # Overlap penalty
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                c1 = vertices_list[i].mean(axis=0)
                c2 = vertices_list[j].mean(axis=0)
                dist = np.linalg.norm(c1 - c2)
                if dist < 2.0:
                    if sat_overlap(vertices_list[i], vertices_list[j]):
                        penalty += 100.0 * (2.0 - dist)**2
        
        return S + penalty

    # Initial guess: hexagonal lattice centered at 0,0
    # 1 center, 6 ring 1, 4 ring 2 (symmetric selection)
    params = np.zeros(3 * N + 3)
    
    # Center
    params[0:3] = [0, 0, 0]
    
    # Ring 1 (6 hexagons)
    idx = 1
    for k in range(6):
        theta = k * np.pi / 3
        cx = np.sqrt(3) * np.cos(theta)
        cy = np.sqrt(3) * np.sin(theta)
        params[3 * idx] = cx
        params[3 * idx + 1] = cy
        params[3 * idx + 2] = 0
        idx += 1
        
    # Ring 2 (4 hexagons)
    # To maintain symmetry and minimize bounding box, pick 4 from the 12.
    # A symmetric choice is at angles 0, pi/2, pi, 3pi/2? No, lattice angles are 0, 60, 120...
    # The vertices of the "cluster" formed by 1+6+4 should be compact.
    # Let's pick the 4 that are at the "midpoints" of the arcs between the ring 1 hexagons?
    # Or simply optimize from a reasonable guess.
    # Let's guess positions at radius 2*sqrt(3) at angles 30, 150, 270, 30? 
    # Actually, the optimal packing of 11 hexagons is known to be the lattice packing.
    # Let's use lattice positions for the 4 ring 2 hexagons.
    # Lattice points at distance 2*sqrt(3) are at angles 0, 60, 120, 180, 240, 300 (6 points) and 30, 90, 150, 210, 270, 330 (6 points)?
    # No, the second ring in a hexagonal lattice has 12 points.
    # Let's just use 4 of them symmetrically. E.g., angles 0, pi/2, pi, 3pi/2 is not on the lattice.
    # Let's pick angles 0, 120, 240, 60? That's 4.
    # Let's try angles 0, pi/3, 2pi/3, pi.
    # Let's just initialize them at radius 2*sqrt(3) with angles 0, pi/2, pi, 3pi/2 and let the optimizer fix it.
    for k in range(4):
        theta = k * np.pi / 2
        cx = 2 * np.sqrt(3) * np.cos(theta)
        cy = 2 * np.sqrt(3) * np.sin(theta)
        params[3 * idx] = cx
        params[3 * idx + 1] = cy
        params[3 * idx + 2] = 0
        idx += 1

    # Container initial guess
    params[3 * N] = 0.0
    params[3 * N + 1] = 0.0
    params[3 * N + 2] = 0.0

    best_params = params
    best_score = objective(params)
    
    # Optimize
    # Use Powell for robustness with non-smooth penalties
    res = minimize(objective, best_params, method='Powell', options={'maxiter': 2000, 'maxfev': 20000})
    
    if res.fun < best_score:
        best_params = res.x
        best_score = res.fun

    # Extract results
    inner = []
    for i in range(N):
        cx = best_params[3 * i]
        cy = best_params[3 * i + 1]
        angle = best_params[3 * i + 2]
        inner.append([cx, cy, np.degrees(angle)])
        
    ocx = best_params[3 * N]
    ocy = best_params[3 * N + 1]
    oangle = best_params[3 * N + 2]
    S = compute_S_for_container([hex_vertices(inner[i][0], inner[i][1], np.radians(inner[i][2])) for i in range(N)], ocx, ocy, oangle)

    return inner, (ocx, ocy), S, np.degrees(oangle)
# EVOLVE-BLOCK-END
