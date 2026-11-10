# EVOLVE-BLOCK-START
import numpy as np

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    n = 11
    # Normals for the hexagon edges (pointing outwards)
    # Vertices at 0, 60, ... deg. Edges perpendicular to 30, 90, ... deg.
    angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0
    normals = np.stack([np.cos(angles), np.sin(angles)], axis=1) # shape (6, 2)
    
    def get_required_side(centers):
        """
        Calculates the minimum side length S of the outer hexagon (angle 0) 
        required to contain all inner hexagons (angle 0) centered at 'centers'.
        """
        # Vertices of a unit hexagon (angle 0)
        v_angles = np.arange(6) * np.pi / 3.0
        vertices_rel = np.stack([np.cos(v_angles), np.sin(v_angles)], axis=1) # shape (6, 2)
        
        max_proj = 0.0
        # For each center, compute all 6 vertices
        # centers shape (11, 2), vertices_rel shape (6, 2)
        # We need to check projection of each vertex onto each of the 6 normals.
        # Due to symmetry, we can just check the max dot product with the normals.
        # Actually, since the container is centered at 0 and symmetric, 
        # we just need the max projection of any vertex in the direction of the normals.
        
        # Efficiently:
        # For each hexagon i, vertices are centers[i] + vertices_rel
        # Project onto normals: (centers[i] + v) . n
        # = centers[i] . n + v . n
        # Max over v for a fixed i and n is centers[i].n + max_v(v.n)
        # max_v(v.n) for unit hexagon is the apothem? No.
        # The max projection of a unit hexagon onto a normal n is the distance from center to the vertex in direction n?
        # No, the support function.
        # For a regular hexagon with side 1, the support function h(theta) is:
        # h(theta) = cos(theta - k*pi/3) * 1? No.
        # The vertices are at distance 1.
        # The max dot product with a unit vector n is 1 if n aligns with a vertex.
        # If n aligns with an edge normal (apothem direction), the max dot product is the apothem = sqrt(3)/2?
        # Wait. The vertex is at (1,0). Normal is (1,0). Dot is 1.
        # Normal is (cos 30, sin 30). Vertex at (1,0) dot n = cos 30 = 0.866.
        # Vertex at (0.5, 0.866) dot n = 0.5*0.866 + 0.866*0.5 = 0.866.
        # So max projection is always 1? No.
        # Let's check: h(theta) for regular polygon with circumradius R is R * cos(pi/m) / cos(theta - ...)?
        # For hexagon (m=6), R=1.
        # h(theta) = cos(theta) if theta is close to 0?
        # Actually, the max projection of a unit hexagon onto any direction is 1 (the circumradius).
        # Is that true? Yes, the farthest point from the origin in any direction is a vertex.
        # So, for a hexagon at c, the max projection onto normal n is c.n + 1.
        
        # Let's verify.
        # If n = (1,0), max proj is 1. c.n + 1.
        # If n = (cos 30, sin 30), max proj is 1?
        # Vertices: (1,0), (0.5, 0.866), (-0.5, 0.866)...
        # Dot with (0.866, 0.5):
        # (1,0): 0.866
        # (0.5, 0.866): 0.433 + 0.433 = 0.866
        # So yes, max projection is 1.
        
        # So, required S is such that:
        # max_i ( c_i . n + 1 ) <= S * sqrt(3)/2
        # S >= (max_i (c_i . n) + 1) * 2 / sqrt(3)
        
        # We need to satisfy this for all 6 normals.
        # So S = max_j [ (max_i (c_i . n_j) + 1) * 2 / sqrt(3) ]
        
        # Calculate c_i . n_j for all i, j
        # centers (11, 2), normals (6, 2)
        # projections (11, 6)
        projections = centers @ normals.T
        max_projections = np.max(projections, axis=0) # shape (6,)
        required_Ss = (max_projections + 1.0) * 2.0 / np.sqrt(3.0)
        return np.max(required_Ss)

    def check_overlap(centers):
        """
        Returns True if any pair of hexagons overlaps.
        Hexagons i and j overlap if max_k |(c_i - c_j) . n_k| <= sqrt(3).
        """
        # centers (11, 2)
        # Compute pairwise differences
        # diff[i, j] = centers[i] - centers[j]
        # We can use broadcasting
        # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :] # (11, 11, 2)
        # proj = diff @ normals.T # (11, 11, 6)
        # max_proj = np.max(np.abs(proj), axis=2) # (11, 11)
        # overlap if max_proj[i, j] <= sqrt(3) + eps for i != j
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        proj = diff @ normals.T
        max_abs_proj = np.max(np.abs(proj), axis=2)
        
        # Set diagonal to infinity to ignore self-overlap
        np.fill_diagonal(max_abs_proj, np.inf)
        
        # If any off-diagonal element is <= sqrt(3), they overlap
        # We want strict separation, so we check <= sqrt(3)
        # The prompt says "separated... by more than 1e-6".
        # So valid if min separation > 1e-6.
        # Overlap if separation <= 1e-6.
        # Separation is max_abs_proj - sqrt(3).
        # So overlap if max_abs_proj <= sqrt(3) + 1e-6.
        
        min_sep = np.min(max_abs_proj)
        return min_sep <= np.sqrt(3.0) + 1e-6

    def local_search(initial_centers, iterations=20000, step_size=0.5):
        current_centers = initial_centers.copy()
        current_S = get_required_side(current_centers)
        if check_overlap(current_centers):
            # If initial is invalid, try to fix or return invalid?
            # We should start with a valid one.
            pass
            
        best_centers = current_centers.copy()
        best_S = current_S
        
        for _ in range(iterations):
            idx = np.random.randint(n)
            # Random perturbation
            delta = np.random.normal(0, step_size, size=2)
            new_centers = current_centers.copy()
            new_centers[idx] += delta
            
            # Check overlap
            if check_overlap(new_centers):
                # Reject or try to push back?
                # For simplicity, reject.
                continue
            
            new_S = get_required_side(new_centers)
            if new_S < current_S:
                current_centers = new_centers
                current_S = new_S
                if current_S < best_S:
                    best_S = current_S
                    best_centers = current_centers.copy()
            
            # Adaptive step size
            # step_size = max(0.01, step_size * 0.99)
            
        return best_centers, best_S

    # Initial guess: 7-cluster + 4 in gaps
    # 7-cluster:
    c0 = np.array([0, 0])
    c1 = np.array([2, 0])
    c2 = np.array([1, np.sqrt(3)])
    c3 = np.array([-1, np.sqrt(3)])
    c4 = np.array([-2, 0])
    c5 = np.array([-1, -np.sqrt(3)])
    c6 = np.array([1, -np.sqrt(3)])
    
    # 4 more: place them at distance ~3.5 in the 4 "corner" gaps?
    # Gaps are at 30, 90, 150, 210, 270, 330.
    # Let's place them at 30, 150, 210, 330 (symmetric) or 0, 90, 180, 270?
    # If we place at 0, it overlaps with c1 (dist 2).
    # Place at distance 3.5 at angles 30, 150, 210, 330.
    r = 3.5
    c7 = np.array([r * np.cos(np.pi/6), r * np.sin(np.pi/6)])
    c8 = np.array([r * np.cos(5*np.pi/6), r * np.sin(5*np.pi/6)])
    c9 = np.array([r * np.cos(7*np.pi/6), r * np.sin(7*np.pi/6)])
    c10 = np.array([r * np.cos(11*np.pi/6), r * np.sin(11*np.pi/6)])
    
    initial_centers = np.array([c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10])
    
    # Check if initial is valid
    if not check_overlap(initial_centers):
        best_centers, best_S = local_search(initial_centers)
    else:
        # Try random restarts if initial is bad
        best_centers = None
        best_S = np.inf
        for _ in range(10):
            # Random valid-ish start
            centers = np.random.uniform(-4, 4, size=(n, 2))
            # Simple relaxation to remove overlaps
            for _ in range(1000):
                # Find overlapping pairs and push apart
                # This is complex, so we just rely on local search to fix it?
                # Local search rejects overlapping moves, so it might get stuck.
                # Better to start with a valid configuration.
                pass
            
            # Let's just use the 7-cluster and randomize the other 4
            base = np.array([c0, c1, c2, c3, c4, c5, c6])
            extra = np.random.uniform(-4, 4, size=(4, 2))
            centers = np.vstack([base, extra])
            
            # Try to fix overlaps by pushing
            for _ in range(500):
                # Check all pairs
                diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
                proj = diff @ normals.T
                max_abs_proj = np.max(np.abs(proj), axis=2)
                np.fill_diagonal(max_abs_proj, np.inf)
                
                # Find pairs with overlap
                overlapping = max_abs_proj <= np.sqrt(3.0)
                if not np.any(overlapping):
                    break
                
                # Pick a random overlapping pair and push apart
                idxs = np.argwhere(overlapping)
                i, j = idxs[np.random.randint(len(idxs))]
                if i == j: continue
                # Push i and j apart
                dir_vec = centers[i] - centers[j]
                if np.linalg.norm(dir_vec) < 1e-6:
                    dir_vec = np.array([1.0, 0.0])
                dir_vec = dir_vec / np.linalg.norm(dir_vec)
                centers[i] += 0.1 * dir_vec
                centers[j] -= 0.1 * dir_vec
            
            if not check_overlap(centers):
                c, s = local_search(centers, iterations=5000)
                if s < best_S:
                    best_S = s
                    best_centers = c

    # Construct return values
    # Inner hexagons: (x, y, 0)
    inner_hex_data = np.hstack([best_centers, np.zeros((n, 1))])
    # Outer hexagon: (0, 0, 0)
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = best_S
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
