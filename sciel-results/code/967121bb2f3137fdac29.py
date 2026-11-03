"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # Unit hexagon vertices (side 1), centered at origin
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees, distance 1 from center
    base_vertices = np.array([[np.cos(np.pi/3 * k), np.sin(np.pi/3 * k)] for k in range(6)])
    
    def rotate_points(pts, angle_rad):
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        return pts @ R.T
    
    def get_hex_vertices(cx, cy, angle_deg):
        """Get 6 vertices of a unit hexagon centered at (cx,cy) with rotation angle_deg."""
        a = np.radians(angle_deg)
        v = rotate_points(base_vertices, a)
        return v + np.array([cx, cy])
    
    def min_enclosing_hexagon(vertices, theta_deg):
        """
        Given a set of vertices (M x 2) and outer hexagon rotation theta_deg,
        find the minimal side length S and center C of a regular hexagon 
        with rotation theta_deg that contains all vertices.
        Returns (S, C) where C is (cx, cy).
        """
        theta = np.radians(theta_deg)
        # 6 outward normals at angles theta + 30 + k*60
        normals = np.array([[np.cos(theta + np.pi/6 + np.pi/3 * k), 
                            np.sin(theta + np.pi/6 + np.pi/3 * k)] for k in range(6)])
        
        # For a given center C, S = (2/sqrt(3)) * max_k max_v (v - C) dot n_k
        # Minimize over C
        
        centroid = np.mean(vertices, axis=0)
        
        def compute_S(C):
            extents = np.array([(vertices - C) @ n for n in normals])
            return (2.0 / np.sqrt(3.0)) * np.max(extents)
        
        best_S = compute_S(centroid)
        best_C = centroid.copy()
        
        # Coordinate descent on C
        step = 0.1
        for _ in range(100):
            improved = False
            for dx in [step, -step]:
                for dy in [step, -step]:
                    new_C = best_C + np.array([dx, dy])
                    new_S = compute_S(new_C)
                    if new_S < best_S - 1e-10:
                        best_S = new_S
                        best_C = new_C
                        improved = True
            if not improved:
                step *= 0.5
                if step < 0.001:
                    break
            if time.time() > deadline - 2:
                break
        
        return best_S, best_C
    
    # Triangular lattice basis
    sqrt3 = np.sqrt(3.0)
    e1 = np.array([sqrt3, 0.0])
    e2 = np.array([sqrt3/2.0, 1.5])
    
    def lattice_point(i, j):
        return i * e1 + j * e2
    
    # Generate candidate clusters of 12 lattice points
    candidate_sets = []
    
    # Shape A: 3x4
    pts_A = [lattice_point(i, j) for j in range(3) for i in range(4)]
    candidate_sets.append(np.array(pts_A))
    
    # Shape B: 4x3
    pts_B = [lattice_point(i, j) for j in range(4) for i in range(3)]
    candidate_sets.append(np.array(pts_B))
    
    # Shape C: hexagonal 1+6+5
    ring1 = [lattice_point(i, j) for i, j in [(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]]
    ring2 = [lattice_point(i, j) for i, j in [(2,0),(-2,0),(0,2),(0,-2),(2,-1),(-1,2)]]
    pts_C = [lattice_point(0,0)] + ring1 + ring2[:5]
    candidate_sets.append(np.array(pts_C))
    
    # Shape D: 2x6
    pts_D = [lattice_point(i, j) for j in range(2) for i in range(6)]
    candidate_sets.append(np.array(pts_D))
    
    # Shape E: 6x2
    pts_E = [lattice_point(i, j) for j in range(6) for i in range(2)]
    candidate_sets.append(np.array(pts_E))
    
    # Shape F: 3x4 shifted
    pts_F = [lattice_point(i, j) for j in range(3) for i in range(-1, 3)]
    candidate_sets.append(np.array(pts_F))
    
    # Shape G: 4x3 shifted
    pts_G = [lattice_point(i, j) for j in range(4) for i in range(-1, 2)]
    candidate_sets.append(np.array(pts_G))
    
    # Shape H: More symmetric hexagonal
    # Use points where hex distance <= 1 (7 pts) + 5 from hex distance 2
    hex_pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            k = -i - j
            if max(abs(i), abs(j), abs(k)) <= 1:
                hex_pts.append((i, j))
    # That's 7 points. Add 5 more from distance 2
    dist2 = [(i,j) for i in range(-2,3) for j in range(-2,3) 
             if max(abs(i), abs(j), abs(-i-j)) == 2]
    pts_H = [lattice_point(i,j) for i,j in hex_pts] + [lattice_point(i,j) for i,j in dist2[:5]]
    candidate_sets.append(np.array(pts_H))
    
    best_S = float('inf')
    best_inner = None
    best_C = None
    best_phi = 0.0
    best_theta = 0.0
    
    for cluster in candidate_sets:
        if time.time() > deadline - 10:
            break
        
        centroid = np.mean(cluster, axis=0)
        centered = cluster - centroid
        
        for phi_deg in [0, 15, 30, 45, 60, 75]:
            if time.time() > deadline - 5:
                break
            
            all_vertices = []
            for pt in centered:
                verts = get_hex_vertices(pt[0], pt[1], phi_deg)
                all_vertices.append(verts)
            all_vertices = np.vstack(all_vertices)
            
            for theta_deg in range(0, 60, 3):
                if time.time() > deadline - 2:
                    break
                
                S, C = min_enclosing_hexagon(all_vertices, theta_deg)
                
                if S < best_S:
                    best_S = S
                    best_inner = centered + C
                    best_C = C
                    best_phi = phi_deg
                    best_theta = theta_deg
    
    if best_inner is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Fine-tune theta around best_theta
    for theta_fine in np.arange(best_theta - 3, best_theta + 3, 0.5):
        if time.time() > deadline - 2:
            break
        # Recompute with fine theta
        centroid = np.mean(best_inner, axis=0)
        centered_fine = best_inner - centroid
        all_vertices = []
        for pt in centered_fine:
            verts = get_hex_vertices(pt[0], pt[1], best_phi)
            all_vertices.append(verts)
        all_vertices = np.vstack(all_vertices)
        
        S_fine, C_fine = min_enclosing_hexagon(all_vertices, theta_fine)
        if S_fine < best_S:
            best_S = S_fine
            best_inner = centered_fine + C_fine
            best_C = C_fine
            best_theta = theta_fine
    
    inner = [[float(p[0]), float(p[1]), float(best_phi)] for p in best_inner]
    return inner, (float(best_C[0]), float(best_C[1])), float(best_S), float(best_theta)
# EVOLVE-BLOCK-END
