import numpy as np
import time
from scipy.optimize import minimize
import itertools

def _hex_vertices(cx, cy, d, a):
    a_rad = np.deg2rad(a)
    angles = a_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _min_outer_side(positions, angle_deg):
    """
    Compute the minimum side length of a regular hexagon with the given angle
    that encloses all unit hexagons at the given positions.
    """
    unit_hex_verts = _hex_vertices(0, 0, 1, 0)
    
    # All inner hexagon vertices
    all_verts = positions[:, np.newaxis, :] + unit_hex_verts[np.newaxis, :, :]
    all_verts_flat = all_verts.reshape(-1, 2)
    
    a_rad = np.deg2rad(angle_deg)
    
    # Outward normals of the outer hexagon edges
    # Edge i connects vertex i and i+1.
    # Outward normal angle is a + 60i + 30 degrees.
    normals = []
    for i in range(6):
        n_angle = a_rad + i * np.pi/3 + np.pi/6
        n_unit = np.array([np.cos(n_angle), np.sin(n_angle)])
        normals.append(n_unit)
    normals = np.array(normals)
    
    # Project all vertices onto normals
    projections = all_verts_flat @ normals.T # (36, 6)
    max_projections = np.max(projections, axis=0) # (6,)
    
    # s * sqrt(3)/2 >= max_proj  =>  s >= 2 * max_proj / sqrt(3)
    min_s = np.max(max_projections) * 2 / np.sqrt(3)
    
    return min_s

def _check_validity(inner, outer, s):
    """
    Validates the solution according to the task contract.
    Returns True if valid, False otherwise.
    """
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(s):
        return False
    
    if s < np.sqrt(12) - 1e-9:
        return False
        
    outer_verts = _hex_vertices(outer[0], outer[1], s, outer[2])
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-9:
                return False
                
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hex_vertices(cx, cy, 1, ang)
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k+1)%6]
            if abs(np.linalg.norm(v2 - v1) - 1.0) > 1e-8:
                return False

    inner_verts_list = []
    for i in range(12):
        cx, cy, ang = inner[i]
        inner_verts_list.append(_hex_vertices(cx, cy, 1, ang))
        
    for i in range(12):
        for j in range(i+1, 12):
            va = inner_verts_list[i]
            vb = inner_verts_list[j]
            
            separated = False
            for k in range(6):
                v1 = va[k]
                v2 = va[(k+1)%6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-12: continue
                n = n / norm
                proj_a = np.dot(va, n)
                proj_b = np.dot(vb, n)
                if np.max(proj_a) < np.min(proj_b) - 1e-6 or np.max(proj_b) < np.min(proj_a) - 1e-6:
                    separated = True
                    break
            if separated: continue
            
            for k in range(6):
                v1 = vb[k]
                v2 = vb[(k+1)%6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-12: continue
                n = n / norm
                proj_a = np.dot(va, n)
                proj_b = np.dot(vb, n)
                if np.max(proj_a) < np.min(proj_b) - 1e-6 or np.max(proj_b) < np.min(proj_a) - 1e-6:
                    separated = True
                    break
            if not separated:
                return False
                
    for i in range(12):
        for v in inner_verts_list[i]:
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                e = p2 - p1
                cross_val = e[0] * (v[1] - p1[1]) - e[1] * (v[0] - p1[0])
                if cross_val < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
                
    return True

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580.0
    
    best_score = 0.0
    best_inner = None
    best_outer = None
    best_s = None
    
    # Define the hexagonal lattice vectors
    # h = sqrt(3)/2
    h = np.sqrt(3) / 2
    
    # Lattice basis for a hexagonal grid of unit hexagons
    # Center at (0,0)
    # Neighbors at distance 1.5 (edge touching) or sqrt(3) (vertex touching)?
    # For a "kissing" configuration (densest), neighbors share edges.
    # Distance between centers of edge-sharing unit hexagons is 1.5.
    # Let's define the 6 neighbors of the center in a hexagonal lattice.
    # Angles: 0, 60, 120, 180, 240, 300
    # But wait, the distance between centers of two unit hexagons sharing an edge is 1.5.
    # Let's verify.
    # Hexagon 1 center (0,0). Vertices at (1,0), (0.5, h), ...
    # Hexagon 2 center (1.5, 0). Vertices at (0.5, 0), (1, h), ...
    # Shared edge: (0.5, 0) to (0.5, h)? No.
    # Hex 1 right edge: (1,0) to (0.5, h). Midpoint (0.75, h/2).
    # Hex 2 left edge: (0.5, 0) to (1, h). Midpoint (0.75, h/2).
    # Yes, they share the edge. Distance is 1.5.
    
    # So the lattice vectors for the 6 neighbors are:
    # v0 = (1.5, 0)
    # v1 = (0.75, 1.5*h) = (0.75, sqrt(3)/2 * 1.5) = (0.75, 1.299)
    # ...
    
    # Actually, let's just use the standard hex grid coordinates.
    # Axial coordinates (q, r).
    # (0,0) -> (0,0)
    # (1,0) -> (1.5, 0)
    # (1,1) -> (0.75, 1.5*h)
    # (0,1) -> (-0.75, 1.5*h)
    # (-1,1) -> (-1.5, 0)
    # (-1,0) -> (-0.75, -1.5*h)
    # (0,-1) -> (0.75, -1.5*h)
    # (1,-1) -> (1.5, -1.5*h) ... wait, (1, -1) is not a neighbor of (0,0) in axial?
    # Axial neighbors of (0,0) are: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1).
    
    # Let's generate positions for the "12 hexagon" cluster.
    # The most compact shape for 12 hexagons is a hexagon of side 2 (which has 1+6+12=19? No).
    # Side 1: 1
    # Side 2: 1 + 6 = 7
    # Side 3: 1 + 6 + 12 = 19
    # So 12 is 7 + 5. We have a core of 7 (center + 6 neighbors) and 5 more on the outside.
    # The "defect" is 1 missing from the outer ring of the side-3 hexagon.
    # This creates a shape that is nearly a hexagon but with one corner cut off or flattened.
    
    # Let's enumerate the possible positions for the 5 extra hexagons.
    # The 6 neighbors of the center are at indices 0..5.
    # The next ring (distance 2 steps in lattice) has 12 positions.
    # We need to pick 5 of these 12 to minimize the bounding hexagon.
    
    # Lattice positions for the 2nd ring (neighbors of neighbors, not center or 1st ring)
    # Axial coords for 2nd ring:
    # (2,0), (2,-1), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-1,2), (0,2), (1,1), (2,1), (2,2)?
    # Let's just generate all lattice points within distance 2 from center and filter.
    
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = h * (r + q/2)
        return np.array([x, y])
        
    # Core 7 positions
    core_axial = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    core_pos = np.array([axial_to_xy(q, r) for q, r in core_axial])
    
    # Candidate positions for the 5 outer hexagons.
    # These must be adjacent to the core but not in the core.
    # They are at axial distance 2 from center.
    # List of all axial points at distance 2:
    # (2,0), (2,-1), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-1,2), (0,2), (1,1), (2,1), (2,2) -- wait, (2,2) is dist 2?
    # Distance in axial: (|q| + |r| + |q+r|) / 2
    # (2,0): (2+0+2)/2 = 2. OK.
    # (2,-1): (2+1+1)/2 = 2. OK.
    # (1,-2): (1+2+1)/2 = 2. OK.
    # (0,-2): (0+2+2)/2 = 2. OK.
    # (-1,-1): (1+1+2)/2 = 2. OK.
    # (-2,0): (2+0+2)/2 = 2. OK.
    # (-2,1): (2+1+1)/2 = 2. OK.
    # (-1,2): (1+2+1)/2 = 2. OK.
    # (0,2): (0+2+2)/2 = 2. OK.
    # (1,1): (1+1+2)/2 = 2. OK.
    # (2,1): (2+1+3)/2 = 3. No.
    # (1,2): (1+2+3)/2 = 3. No.
    # So there are 12 candidates in the 2nd ring.
    
    second_ring_axial = [
        (2, 0), (2, -1), (1, -2), (0, -2), (-1, -1), (-2, 0),
        (-2, 1), (-1, 2), (0, 2), (1, 1), (2, 1) # Wait, (2,1) is dist 3?
    ]
    # Let's re-verify (2,1). q=2, r=1. q+r=3. (2+1+3)/2 = 3. Correct, dist 3.
    # So the 12 candidates are:
    candidates_axial = [
        (2, 0), (2, -1), (1, -2), (0, -2), (-1, -1), (-2, 0),
        (-2, 1), (-1, 2), (0, 2), (1, 1), (1, 0) # (1,0) is core
    ]
    # I made a mistake. The 12 points at distance 2 are:
    # (2,0), (2,-1), (1,-2), (0,-2), (-1,-1), (-2,0)
    # (-2,1), (-1,2), (0,2), (1,1), (2,1) -- NO.
    # Let's just iterate q, r from -2 to 2 and check distance.
    
    cand_list = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            dist = (abs(q) + abs(r) + abs(q+r)) // 2
            if dist == 2:
                cand_list.append((q, r))
                
    # There should be 12.
    # (2,0), (2,-1), (1,-2), (0,-2), (-1,-1), (-2,0)
    # (-2,1), (-1,2), (0,2), (1,1), (2,1) is wrong.
    # (1,1) is dist 2. (2,0) is dist 2.
    # Let's trust the loop.
    
    # We need to choose 5 from these 12 candidates.
    # However, the chosen candidates must form a connected cluster with the core?
    # The problem doesn't require connectivity, just non-overlap.
    # But for minimal bounding box, they should be compact.
    # Also, the chosen candidates cannot overlap with each other.
    # In the hex lattice, two hexagons at distance 2 (axial) might be adjacent or not.
    # If they are adjacent, they share an edge, which is valid (non-overlapping).
    # If they are at distance 1 (axial), they would overlap? No, distance 1 is the core.
    # Any two distinct hexagons in the hex lattice are non-overlapping.
    # So we just need to pick any 5 from the 12 candidates.
    
    # But wait, if we pick two candidates that are "far apart", the bounding box will be larger.
    # We want to pick 5 that are "close together" to minimize the extent.
    # This is equivalent to finding the 5 points in the 2nd ring that have the smallest spread.
    
    # Let's generate all combinations of 5 from the 12 candidates.
    # C(12, 5) = 792 combinations. This is very small.
    
    best_candidate_set = None
    best_candidate_score = float('inf')
    
    # Pre-compute positions for candidates
    cand_pos = np.array([axial_to_xy(q, r) for q, r in cand_list])
    
    # We also need to consider that the "outer" hexagons might be rotated?
    # No, the task says "unit regular hexagons". It doesn't restrict their angle.
    # However, the current program fixes inner angles to 0.
    # The seed program also fixes inner angles to 0.
    # Can we improve by rotating inner hexagons?
    # If we rotate an inner hexagon, its vertices move.
    # The constraint is non-overlap.
    # In the hex lattice, the hexagons are packed edge-to-edge.
    # If we rotate one, it will likely overlap with its neighbors unless we move it.
    # Given the tight packing, rotating is risky and likely not optimal for the global minimum.
    # The "stiff" lattice arrangement is usually optimal for these small N.
    # So we stick to angle=0 for inner hexagons.
    
    # Optimization:
    # 1. Enumerate the 5 outer positions.
    # 2. For each combination, we have 12 positions.
    # 3. We can slightly perturb these positions and optimize the outer angle to minimize s.
    # 4. Since the lattice is rigid, we might not need to perturb positions much.
    #    But the outer hexagon angle can be varied to better fit the "irregular" shape of the 12-hex cluster.
    
    # Let's refine the best few combinations using local optimization.
    
    combinations = list(itertools.combinations(range(len(cand_list)), 5))
    
    # Sort combinations by a heuristic: sum of distances from centroid?
    # Or just iterate all 792, compute s for angle=0, and keep the top 10.
    
    top_combinations = []
    
    for comb in combinations:
        outer_indices = comb
        all_pos = np.vstack([core_pos, cand_pos[outer_indices]])
        
        # Compute s for angle 0
        s_val = _min_outer_side(all_pos, 0.0)
        top_combinations.append((s_val, comb))
        
    # Sort by s
    top_combinations.sort(key=lambda x: x[0])
    
    # Take top 10 combinations for detailed optimization
    for s_heuristic, comb in top_combinations[:10]:
        if time.time() > deadline:
            break
            
        outer_indices = comb
        initial_pos = np.vstack([core_pos, cand_pos[outer_indices]])
        
        # Optimize:
        # Variables: 24 positions + 1 outer angle.
        # But positions are constrained to be non-overlapping.
        # Since the initial positions are from a valid lattice, they are non-overlapping.
        # We can allow small movements.
        
        def objective(x):
            positions = x[:24].reshape(12, 2)
            angle_deg = x[24]
            
            s = _min_outer_side(positions, angle_deg)
            
            # Penalty for overlaps
            # We use the same penalty as before, but since we start from a valid lattice,
            # we can use a smaller penalty or just rely on the optimizer not moving too far.
            # However, Nelder-Mead might move far.
            # Let's use a strong penalty for d < 1.5.
            penalty = 0.0
            for i in range(12):
                for j in range(i+1, 12):
                    d = np.linalg.norm(positions[i] - positions[j])
                    if d < 1.5:
                        penalty += (1.5 - d)**2 * 100.0 # Strong penalty
                    elif d < 1.55:
                        penalty += (1.55 - d)**2 * 10.0
                        
            return s + penalty
            
        x0 = np.concatenate([initial_pos.flatten(), [0.0]])
        
        # Try a few different initial outer angles
        for init_angle in [0, 30, -30, 15, -15]:
            if time.time() > deadline:
                break
                
            x0_init = np.concatenate([initial_pos.flatten(), [init_angle]])
            
            try:
                res = minimize(
                    objective, 
                    x0_init, 
                    method='Nelder-Mead', 
                    options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True}
                )
                
                x_opt = res.x
                positions = x_opt[:24].reshape(12, 2)
                angle_deg = x_opt[24]
                
                s_candidate = _min_outer_side(positions, angle_deg)
                
                inner_candidate = np.zeros((12, 3))
                inner_candidate[:, :2] = positions
                inner_candidate[:, 2] = 0.0
                outer_candidate = np.array([0, 0, angle_deg])
                
                if _check_validity(inner_candidate, outer_candidate, s_candidate):
                    score_candidate = 1.0 / s_candidate
                    if score_candidate > best_score:
                        best_score = score_candidate
                        best_inner = inner_candidate
                        best_outer = outer_candidate
                        best_s = s_candidate
                        
            except Exception:
                continue

    # Fallback if optimization failed or didn't find anything better than a basic lattice
    if best_inner is None:
        # Use the best heuristic combination without optimization
        if top_combinations:
            s_heuristic, comb = top_combinations[0]
            outer_indices = comb
            positions = np.vstack([core_pos, cand_pos[outer_indices]])
            
            inner_candidate = np.zeros((12, 3))
            inner_candidate[:, :2] = positions
            inner_candidate[:, 2] = 0.0
            outer_candidate = np.array([0, 0, 0])
            
            s_candidate = _min_outer_side(positions, 0.0)
            
            if _check_validity(inner_candidate, outer_candidate, s_candidate):
                best_inner = inner_candidate
                best_outer = outer_candidate
                best_s = s_candidate
                best_score = 1.0 / s_candidate

    # Final fallback to a known valid configuration if all else fails
    if best_inner is None:
        inner_fallback = np.array([
            [0, 0, 0],
            [2.5, 0, 0],
            [-2.5, 0, 0],
            [1.25, 2.165, 0],
            [-1.25, 2.165, 0],
            [1.25, -2.165, 0],
            [-1.25, -2.165, 0],
            [3.75, 2.165, 0],
            [-3.75, 2.165, 0],
            [3.75, -2.165, 0],
            [-3.75, -2.165, 0],
            [0, 4.33, 0]
        ])
        outer_fallback = np.array([0, 0, 0])
        s_fallback = _min_outer_side(inner_fallback[:, :2], 0)
        
        if s_fallback < np.sqrt(12):
            s_fallback = np.sqrt(12)
            
        return inner_fallback, outer_fallback, s_fallback

    return best_inner, best_outer, best_s

# EVOLVE-BLOCK-END
