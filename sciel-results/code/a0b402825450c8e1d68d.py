"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    # We need to pack N unit regular hexagons inside a regular hexagon of minimum side length.
    # The inner hexagons can translate and rotate.
    # The outer hexagon center, side length, and rotation are also variables to optimize.
    # However, for simplicity and due to symmetry, we can fix the outer hexagon center at (0,0)
    # and rotation at 0. The side length will be determined by the maximum distance of any
    # inner hexagon vertex from the center, projected onto the outer hexagon's support function.
    
    # Actually, the score is the outer side length. We want to minimize it.
    # Let's parameterize the positions and rotations of the N inner hexagons.
    # Each inner hexagon is defined by (cx, cy, angle).
    # We need to ensure non-overlap between all pairs of inner hexagons.
    # And all inner hexagons must be inside the outer hexagon.
    
    # For a regular hexagon with side length s and center (cx, cy) and rotation angle theta,
    # the vertices are at:
    # (cx + s * cos(theta + k*pi/3), cy + s * sin(theta + k*pi/3)) for k=0..5
    
    # The outer hexagon has side length S, center (0,0), rotation 0.
    # A point (x,y) is inside the outer hexagon if:
    # max(|x|, |y|/sqrt(3), |x/2 + y*sqrt(3)/2|) <= S
    # Wait, let's derive the support function properly.
    # For a regular hexagon with side length S centered at origin, flat-top orientation:
    # The vertices are at (S, 0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S, 0), (-S/2, -S*sqrt(3)/2), (S/2, -S*sqrt(3)/2)
    # The inequalities for being inside are:
    # |x| <= S
    # |y| <= S*sqrt(3)/2
    # |x/2 + y*sqrt(3)/2| <= S
    # Actually, the correct condition for a point (x,y) to be inside a regular hexagon of side S, center origin, flat-top:
    # The six sides are lines at distance S*sqrt(3)/2 from the center? No.
    # The apothem (distance from center to side) of a regular hexagon with side S is S*sqrt(3)/2.
    # So the hexagon is the intersection of six half-planes.
    # For flat-top hexagon, the normals to the sides are at angles 0, 60, 120, 180, 240, 300 degrees.
    # The distance from center to each side is apothem = S*sqrt(3)/2.
    # So the condition is: for each k=0..5, x*cos(k*pi/3) + y*sin(k*pi/3) <= S*sqrt(3)/2
    # Wait, that's for pointy-top? Let me think again.
    
    # For a regular hexagon with side length S, the circumradius is S, and the inradius (apothem) is S*sqrt(3)/2.
    # The support function h(u) = max_{v in hex} u·v.
    # For the hexagon to contain a point p, we need p·n_i <= apothem for all 6 normal directions n_i.
    # The 6 outward normals for a flat-top hexagon (vertices at 0, 60, 120, 180, 240, 300) are at 30, 90, 150, 210, 270, 330 degrees.
    # So n_k = (cos(30+k*60), sin(30+k*60)) for k=0..5.
    # Condition: x*cos(30+k*60) + y*sin(30+k*60) <= S*sqrt(3)/2 for all k.
    
    # To find the minimum S such that all inner hexagon vertices are inside, we compute for each vertex
    # the required S = max over all vertices and all k of (v_x*cos(30+k*60) + v_y*sin(30+k*60)) / (sqrt(3)/2)
    # And also we need the negative side, so actually:
    # S >= max over all vertices v and all k of |v·n_k| / (sqrt(3)/2)
    # Wait, no. The condition is v·n_k <= apothem for all k, where n_k are outward normals.
    # Since the hexagon is symmetric, we need v·n_k <= apothem AND (-v)·n_k <= apothem which is equivalent to |v·n_k| <= apothem.
    # So S >= max_{v, k} |v·n_k| / (sqrt(3)/2)
    
    # For non-overlap of two unit hexagons with centers c1, c2 and angles a1, a2:
    # This is complex. A simpler approach: use a distance field.
    # Two convex polygons don't overlap if and only if there exists a separating axis.
    # For two regular hexagons, we can check the 12 axes (6 from each hexagon's normals).
    # But this is expensive for optimization.
    
    # Alternative: use a simple penalty-based approach with a distance metric.
    # For two unit hexagons, the minimum distance between their boundaries when they are "touching" in a standard lattice arrangement is known.
    # In a hexagonal lattice of unit hexagons, the center-to-center distance is 2*sqrt(3) for adjacent hexagons (sharing an edge).
    # Wait, for regular hexagons of side 1, the distance between centers of two adjacent hexagons (sharing an edge) is 2*apothem = 2*sqrt(3)/2 = sqrt(3)? No.
    # Apothem = sqrt(3)/2. Distance between centers = 2 * apothem = sqrt(3).
    # But if they are rotated, the minimum center distance to avoid overlap depends on the relative rotation.
    
    # Let's use a simpler approach: for each pair, compute the minimum distance between the two hexagons.
    # If they overlap, penalize.
    
    # Given the complexity, let's try a different approach: 
    # Start with a hexagonal lattice arrangement and optimize the positions slightly.
    # Or, use a known optimal packing configuration.
    
    # For N=11, a common arrangement is a central hexagon surrounded by 6, and then 4 more.
    # But let's just use an optimization approach.
    
    # Variables: 3*N parameters (cx, cy, angle for each inner hexagon)
    # Objective: S = max over all vertices of all inner hexagons of |v·n_k| / (sqrt(3)/2)
    # Constraints: for each pair (i,j), the two hexagons don't overlap.
    
    # This is a non-convex problem. Let's use a penalty method.
    
    def hex_vertices(cx, cy, angle, s=1.0):
        """Return the 6 vertices of a regular hexagon."""
        vertices = []
        for k in range(6):
            theta = angle + k * np.pi / 3
            x = cx + s * np.cos(theta)
            y = cy + s * np.sin(theta)
            vertices.append((x, y))
        return np.array(vertices)
    
    def poly_overlap(v1, v2):
        """Check if two convex polygons (given as arrays of vertices) overlap using SAT."""
        # SAT: check if there's a separating axis
        # For each edge of each polygon, project all vertices onto the normal
        # If there's a gap, they don't overlap
        polys = [v1, v2]
        for poly in polys:
            n = len(poly)
            for i in range(n):
                p1 = poly[i]
                p2 = poly[(i+1) % n]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                # Project all vertices of both polygons onto this normal
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                min1, max1 = proj1.min(), proj1.max()
                min2, max2 = proj2.min(), proj2.max()
                if max1 < min2 or max2 < min1:
                    return False  # Separating axis found, no overlap
        return True  # No separating axis, they overlap
    
    def overlap_penalty(v1, v2):
        """Return a penalty if two hexagons overlap, 0 otherwise."""
        # Use a simpler metric: distance between centers
        c1 = v1.mean(axis=0)
        c2 = v2.mean(axis=0)
        dist = np.linalg.norm(c1 - c2)
        # Minimum distance for non-overlap depends on rotation, but let's use a conservative bound
        # For two unit hexagons, the maximum possible distance from center to any point is 1 (circumradius)
        # So if dist >= 2, they definitely don't overlap
        # If dist < 2, they might overlap
        # For a better estimate, use SAT
        if poly_overlap(v1, v2):
            # Return a penalty proportional to the overlap
            # Approximate overlap by how much the center distance is less than the minimum required
            # This is rough; let's just return a large penalty
            return 100.0 * (2.0 - dist) if dist < 2.0 else 0.0
        return 0.0
    
    def objective(params):
        """Compute the outer side length S and penalty for overlaps."""
        n = len(params) // 3
        S = 0.0
        vertices_list = []
        for i in range(n):
            cx = params[3*i]
            cy = params[3*i+1]
            angle = params[3*i+2]
            verts = hex_vertices(cx, cy, angle)
            vertices_list.append(verts)
            # Update S
            for v in verts:
                for k in range(6):
                    theta_n = np.pi/6 + k * np.pi/3
                    proj = v[0]*np.cos(theta_n) + v[1]*np.sin(theta_n)
                    S = max(S, abs(proj))
        S = S / (np.sqrt(3)/2)
        
        # Overlap penalty
        penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                penalty += overlap_penalty(vertices_list[i], vertices_list[j])
        
        return S + penalty
    
    # Initial guess: hexagonal lattice
    params = np.zeros(3*N)
    # Place hexagons in a hexagonal pattern
    # Center at (0,0)
    params[0:3] = [0, 0, 0]
    idx = 1
    # Ring 1: 6 hexagons at distance sqrt(3) from center
    for k in range(6):
        theta = k * np.pi/3
        cx = np.sqrt(3) * np.cos(theta)
        cy = np.sqrt(3) * np.sin(theta)
        params[3*idx] = cx
        params[3*idx+1] = cy
        params[3*idx+2] = 0
        idx += 1
    # Ring 2: need 4 more to reach 11
    # Place them at distance 2*sqrt(3) from center, at angles 0, 60, 120, 180
    for k in range(4):
        theta = k * np.pi/3
        cx = 2*np.sqrt(3) * np.cos(theta)
        cy = 2*np.sqrt(3) * np.sin(theta)
        params[3*idx] = cx
        params[3*idx+1] = cy
        params[3*idx+2] = 0
        idx += 1
    
    # Optimize
    result = minimize(objective, params, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
    
    # Extract optimal parameters
    opt_params = result.x
    inner = []
    for i in range(N):
        cx = opt_params[3*i]
        cy = opt_params[3*i+1]
        angle = np.degrees(opt_params[3*i+2])
        inner.append([cx, cy, angle])
    
    # Compute the final S
    S = 0.0
    for i in range(N):
        cx = opt_params[3*i]
        cy = opt_params[3*i+1]
        angle = opt_params[3*i+2]
        verts = hex_vertices(cx, cy, angle)
        for v in verts:
            for k in range(6):
                theta_n = np.pi/6 + k * np.pi/3
                proj = v[0]*np.cos(theta_n) + v[1]*np.sin(theta_n)
                S = max(S, abs(proj))
    S = S / (np.sqrt(3)/2)
    
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
