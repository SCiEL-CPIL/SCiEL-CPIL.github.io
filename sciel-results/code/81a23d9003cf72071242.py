import numpy as np
from scipy.optimize import minimize

def construct():
    N = 12
    sqrt3 = np.sqrt(3)

    # Generate 12 centers from a hexagonal tiling (pointy-top)
    # 3 columns, 4 rows
    centers = []
    for i in range(-1, 2):
        for j in range(-1, 3):
            x = i * sqrt3
            y = 2 * j + (1 if i % 2 != 0 else 0)
            centers.append([x, y])
    centers = np.array(centers)
    centers -= np.mean(centers, axis=0)

    # Unit pointy-top hexagon vertices
    angles = np.deg2rad([90, 150, 210, 270, 330, 30])
    verts = np.column_stack([np.cos(angles), np.sin(angles)])

    # Precompute edge midpoints and normals for overlap checking
    # Each edge connects verts[k] to verts[(k+1)%6]
    edge_mids = np.zeros((6, 2))
    edge_normals = np.zeros((6, 2))
    for k in range(6):
        v1 = verts[k]
        v2 = verts[(k + 1) % 6]
        mid = (v1 + v2) / 2
        edge_mids[k] = mid
        # Normal pointing outward
        edge_vec = v2 - v1
        # For pointy-top, outward normal
        nx = edge_vec[1]
        ny = -edge_vec[0]
        norm = np.sqrt(nx**2 + ny**2)
        if norm > 1e-10:
            edge_normals[k] = [nx / norm, ny / norm]
        else:
            edge_normals[k] = [0, 0]

    def get_all_vertices(rot_deg, cx, cy):
        """Get all 72 vertices of the 12 hexagons."""
        rot = np.deg2rad(rot_deg)
        c, s = np.cos(rot), np.sin(rot)
        R = np.array([[c, -s], [s, c]])
        
        all_verts = []
        for center in centers:
            rc = R @ center
            rc[0] += cx
            rc[1] += cy
            rv = (R @ verts.T).T + rc
            all_verts.append(rv)
        return np.vstack(all_verts)

    def calc_L(vertices):
        """Calculate required outer hexagon side length (pointy-top, centered at origin)."""
        x = vertices[:, 0]
        y = vertices[:, 1]
        L1 = np.abs(y)
        L2 = (2 / sqrt3) * np.abs(x)
        L3 = (2 / sqrt3) * np.abs(y + x / sqrt3)
        L4 = (2 / sqrt3) * np.abs(y - x / sqrt3)
        return np.max(np.maximum(np.maximum(L1, L2), np.maximum(L3, L4)))

    def check_overlap(rot_deg, cx, cy, tolerance=1e-4):
        """Check if any two hexagons overlap. Returns minimum distance between any two hexagons."""
        rot = np.deg2rad(rot_deg)
        c, s = np.cos(rot), np.sin(rot)
        R = np.array([[c, -s], [s, c]])
        
        min_dist = float('inf')
        
        # For each pair of hexagons, check if they overlap
        for i in range(N):
            ci = R @ centers[i]
            ci[0] += cx
            ci[1] += cy
            vi = (R @ verts.T).T + ci
            
            for j in range(i + 1, N):
                cj = R @ centers[j]
                cj[0] += cx
                cj[1] += cy
                vj = (R @ verts.T).T + cj
                
                # Check if any vertex of i is inside j or vice versa
                # Use edge-based separation: for each edge of hex i, check if all vertices of j are on the same side
                # If not, they overlap
                overlap = False
                for k in range(6):
                    # Edge k of hex i
                    mid_i = vi[k]
                    normal_i = edge_normals[k]
                    
                    # Check all vertices of j against this edge
                    # A vertex is "inside" if it's on the opposite side of the edge from the outward normal
                    # Actually, for a convex polygon, if all vertices of j are on the inside side of edge k of i,
                    # then j is not separated by this edge
                    dots = (vj - mid_i) @ normal_i
                    # If all dots are <= 0, then all of j is on the inside of edge k of i
                    if np.all(dots <= tolerance):
                        # This edge doesn't separate them, but we need ALL edges to not separate
                        # Actually, if ANY edge separates them, they don't overlap
                        # So if NO edge separates them, they overlap
                        pass
                    else:
                        # This edge separates them
                        overlap = False
                        break
                else:
                    # No edge of i separates them, check edges of j
                    for k in range(6):
                        mid_j = vj[k]
                        normal_j = edge_normals[k]
                        dots = (vi - mid_j) @ normal_j
                        if np.all(dots <= tolerance):
                            pass
                        else:
                            overlap = False
                            break
                    else:
                        overlap = True
                
                if overlap:
                    # Calculate minimum distance between the two hexagons
                    # Use point-to-edge distance
                    for k in range(6):
                        mid_i = vi[k]
                        normal_i = edge_normals[k]
                        # Distance from each vertex of j to edge k of i
                        dots = (vj - mid_i) @ normal_i
                        # Only consider vertices on the inside (dots < 0)
                        inside = dots < -tolerance
                        if np.any(inside):
                            # Project onto edge line
                            edge_vec = vi[(k + 1) % 6] - vi[k]
                            t = np.dot(vj[inside] - vi[k], edge_vec) / np.dot(edge_vec, edge_vec)
                            t = np.clip(t, 0, 1)
                            proj = vi[k] + t[:, np.newaxis] * edge_vec
                            dists = np.linalg.norm(vj[inside] - proj, axis=1)
                            min_dist = min(min_dist, np.min(dists))
                    
                    for k in range(6):
                        mid_j = vj[k]
                        normal_j = edge_normals[k]
                        dots = (vi - mid_j) @ normal_j
                        inside = dots < -tolerance
                        if np.any(inside):
                            edge_vec = vj[(k + 1) % 6] - vj[k]
                            t = np.dot(vi[inside] - vj[k], edge_vec) / np.dot(edge_vec, edge_vec)
                            t = np.clip(t, 0, 1)
                            proj = vj[k] + t[:, np.newaxis] * edge_vec
                            dists = np.linalg.norm(vi[inside] - proj, axis=1)
                            min_dist = min(min_dist, np.min(dists))
                    
                    return min_dist
        
        return min_dist

    def objective(x):
        rot_deg, cx, cy = x
        vertices = get_all_vertices(rot_deg, cx, cy)
        L = calc_L(vertices)
        
        # Check overlap
        min_dist = check_overlap(rot_deg, cx, cy)
        if min_dist < 1e-6:
            return L + 1000.0 * (1e-6 - min_dist) ** 2
        return L

    # Optimize
    best_L = float('inf')
    best_params = [0, 0, 0]
    
    # Multiple starting points
    for rot_init in range(0, 180, 15):
        for cx_init in [-0.5, 0, 0.5]:
            for cy_init in [-0.5, 0, 0.5]:
                x0 = [float(rot_init), float(cx_init), float(cy_init)]
                try:
                    result = minimize(objective, x0, method='Nelder-Mead',
                                     options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10})
                    if result.fun < best_L:
                        best_L = result.fun
                        best_params = result.x
                except:
                    pass

    rot_deg, cx, cy = best_params

    # Final verification and refinement
    vertices = get_all_vertices(rot_deg, cx, cy)
    final_L = calc_L(vertices)

    # Construct output
    rot_rad = np.deg2rad(rot_deg)
    c, s = np.cos(rot_rad), np.sin(rot_rad)
    R = np.array([[c, -s], [s, c]])

    inner = []
    for center in centers:
        rc = R @ center
        rc[0] += cx
        rc[1] += cy
        inner.append([float(rc[0]), float(rc[1]), float(rot_deg)])

    return inner, (float(cx), float(cy)), float(final_L), 0.0
