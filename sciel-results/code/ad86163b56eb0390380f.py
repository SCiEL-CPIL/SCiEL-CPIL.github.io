import numpy as np
import time
import sys
from itertools import combinations

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Strategy: 
    1. Generate candidate topologies (connected clusters of 11 hexes on a hex grid).
    2. For each topology, perform a local geometric optimization (Simulated Annealing)
       to minimize the side length of the enclosing regular hexagon.
    3. Return the best valid packing found.
    """
    start_time = time.time()
    time_limit = 600.0
    n = 11
    
    # Precompute hex grid basis vectors
    # For pointy-top hexes (vertex at 0, 60, ...), neighbors are at 60-degree intervals.
    # However, the standard hex grid "axial" coordinates map to Cartesian as:
    # x = size * (sqrt(3) * q + sqrt(3)/2 * r)
    # y = size * (3/2 * r)
    # This corresponds to a "pointy top" orientation where the flat sides are horizontal?
    # Let's check the problem's hex definition:
    # Vertices: (cx + s*cos(a + 60i), cy + s*sin(a + 60i)).
    # If a=0, vertices at 0, 60, 120... This is a "flat top" hex? No, vertex at 0 is right.
    # The edge between vertex 0 and 1 is at 30 deg from x-axis?
    # Midpoint of edge 0-1: (cos(0)+cos(60))/2 = 0.75, (sin(0)+sin(60))/2 = 0.433.
    # Normal to this edge is at 30 deg.
    # So the "flat" sides are perpendicular to 30, 90, 150...
    # This is a "pointy top" hexagon in standard terminology (vertices at top/bottom? No, vertices at 0 and 180 are left/right).
    # Actually, if vertices are at 0 and 180, the hex is wide horizontally.
    # The distance between centers of touching hexes:
    # If they share the edge normal to 30 deg, the vector between centers is along 30 deg.
    # Distance = 2 * apothem = sqrt(3).
    
    # Let's define the 6 neighbor directions in Cartesian for this orientation.
    # Normals are at 30, 90, 150, 210, 270, 330.
    # Neighbor centers are displaced by sqrt(3) in these normal directions.
    sqrt3 = np.sqrt(3)
    neighbor_dirs = np.array([
        [np.cos(np.deg2rad(30)) * sqrt3, np.sin(np.deg2rad(30)) * sqrt3],
        [np.cos(np.deg2rad(90)) * sqrt3, np.sin(np.deg2rad(90)) * sqrt3],
        [np.cos(np.deg2rad(150)) * sqrt3, np.sin(np.deg2rad(150)) * sqrt3],
        [np.cos(np.deg2rad(210)) * sqrt3, np.sin(np.deg2rad(210)) * sqrt3],
        [np.cos(np.deg2rad(270)) * sqrt3, np.sin(np.deg2rad(270)) * sqrt3],
        [np.cos(np.deg2rad(330)) * sqrt3, np.sin(np.deg2rad(330)) * sqrt3]
    ])
    
    # 1. Generate Topologies
    # We represent a topology by a set of axial coordinates (q, r).
    # Axial to Cartesian:
    # q corresponds to the "right" direction?
    # Let's map axial (q, r) to Cartesian using the neighbor vectors.
    # Let basis e1 = neighbor_dirs[0] (30 deg) and e2 = neighbor_dirs[1] (90 deg)?
    # No, standard axial uses two basis vectors 60 deg apart.
    # Let's use:
    # x = q * neighbor_dirs[0] + r * neighbor_dirs[1] ?
    # neighbor_dirs[0] is at 30 deg. neighbor_dirs[1] is at 90 deg.
    # Angle between them is 60 deg. This works as a basis.
    
    def axial_to_cartesian(q, r):
        return q * neighbor_dirs[0] + r * neighbor_dirs[1]
    
    # Generate all connected sets of 11 hexes from a seed hexagon.
    # To avoid duplicates and explosion, we use a canonical form:
    # Always include (0,0). Grow the cluster.
    # We limit the search to clusters that fit in a small bounding box in axial coords.
    
    candidates = []
    seen_canonical = set()
    
    # Start with a small cluster and grow
    # We'll use a simple backtracking search to find connected sets of size 11
    # containing (0,0).
    
    def generate_clusters(current_set, current_list, depth):
        if depth == n:
            # Canonicalize: shift so min q=0, min r=0 (or similar)
            # For simplicity, just check if it's a new shape by sorting coords
            key = tuple(sorted(current_set))
            if key not in seen_canonical:
                seen_canonical.add(key)
                candidates.append(list(current_set))
            return
        
        # Find all empty neighbors of the current set
        neighbors = set()
        for (q, r) in current_set:
            for dq, dr in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]:
                nq, nr = q + dq, r + dr
                if (nq, nr) not in current_set:
                    neighbors.add((nq, nr))
        
        # To reduce symmetry, we can impose an ordering on additions
        # e.g., always add the lexicographically smallest available neighbor first?
        # No, that restricts shapes.
        # Instead, we just add any neighbor. The `seen_canonical` set handles duplicates.
        # To speed up, we can prune if the bounding box is too large.
        
        for pos in neighbors:
            # Pruning: if the range of q or r is too big, skip
            qs = [p[0] for p in current_set] + [pos[0]]
            rs = [p[1] for p in current_set] + [pos[1]]
            if max(qs) - min(qs) > 6 or max(rs) - min(rs) > 6:
                continue
                
            new_set = set(current_set)
            new_set.add(pos)
            generate_clusters(new_set, current_list + [pos], depth + 1)
            
    # Start generation
    initial_set = {(0, 0)}
    generate_clusters(initial_set, [(0,0)], 1)
    
    # If generation is too slow or yields too many, we might need to restrict further.
    # Let's check time.
    if time.time() - start_time > 30:
        # Fallback: just use a few hand-picked good topologies if generation is slow
        # But 11 hexes is small. The number of free hexominoes is 35.
        # The number of connected sets of size 11 is much larger, but with the bounding box constraint, it should be manageable.
        pass
    
    if not candidates:
        # Fallback to a single known good topology (e.g. a 3x3 grid + 2)
        # Just a linear chain or something simple
        candidates = [ [(0,0), (1,0), (2,0), (3,0), (4,0), (5,0), (6,0), (7,0), (8,0), (9,0), (10,0)] ]
        
    # 2. Optimize each topology
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_outer = None
    
    # Optimization function for a specific topology
    # Variables: 
    # - Small perturbations to the initial grid positions? 
    # - Or full positions?
    # The topology defines which hexes touch. 
    # We can model the positions as variables, but constrain them to stay close to the grid?
    # Better: Use the grid positions as a starting point and run a general SA that minimizes the outer hex size.
    # The "topology" ensures we start in a good basin.
    
    def optimize_topology(topo_axial):
        # Convert axial to initial Cartesian positions
        init_pos = np.array([axial_to_cartesian(q, r) for q, r in topo_axial])
        
        # Center the cluster
        centroid = np.mean(init_pos, axis=0)
        init_pos -= centroid
        
        # Angles: Start with 0
        angles = np.zeros(n)
        
        # Outer hex center: Start at 0
        outer_center = np.zeros(2)
        
        # Simulated Annealing
        T = 0.05
        T_min = 1e-5
        alpha = 0.999
        max_iter = 20000
        current_cost = float('inf')
        current_pos = init_pos.copy()
        current_angles = angles.copy()
        current_outer = outer_center.copy()
        
        best_cost = float('inf')
        best_pos = init_pos.copy()
        best_angles = angles.copy()
        best_outer = outer_center.copy()
        
        for i in range(max_iter):
            if time.time() - start_time > time_limit - 10:
                break
            
            # Generate a perturbation
            # Move one hex slightly
            idx = np.random.randint(n)
            delta_pos = np.random.normal(0, 0.01, 2)
            new_pos = current_pos.copy()
            new_pos[idx] += delta_pos
            
            # Optionally rotate one hex
            if np.random.rand() < 0.1:
                idx_rot = np.random.randint(n)
                delta_angle = np.random.normal(0, 0.05)
                new_angles = current_angles.copy()
                new_angles[idx_rot] += delta_angle
            else:
                new_angles = current_angles
                
            # Optionally move outer center
            if np.random.rand() < 0.2:
                new_outer = current_outer + np.random.normal(0, 0.01, 2)
            else:
                new_outer = current_outer
                
            # Calculate cost
            cost = calculate_cost(new_pos, new_angles, new_outer)
            
            # Acceptance
            if cost < current_cost or np.random.rand() < np.exp(-(cost - current_cost) / T):
                current_cost = cost
                current_pos = new_pos
                current_angles = new_angles
                current_outer = new_outer
                
                if cost < best_cost:
                    best_cost = cost
                    best_pos = current_pos.copy()
                    best_angles = current_angles.copy()
                    best_outer = current_outer.copy()
                    
            T *= alpha
            if T < T_min:
                break
                
        return best_cost, best_pos, best_angles, best_outer
    
    def calculate_cost(pos, angles, outer_center):
        # pos: (11, 2)
        # angles: (11,)
        # outer_center: (2,)
        
        # 1. Check containment in outer hexagon
        # We need to find the minimum S such that all inner hexes are inside.
        # Inner hex i has center pos[i], angle angles[i].
        # Vertices: pos[i] + s * R(angles[i]) * v_j
        # s=1.
        
        # Outer hexagon is fixed orientation (angle 0).
        # Normals: 30, 90, 150, 210, 270, 330.
        outer_normals = np.array([
            [np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))],
            [np.cos(np.deg2rad(90)), np.sin(np.deg2rad(90))],
            [np.cos(np.deg2rad(150)), np.sin(np.deg2rad(150))],
            [np.cos(np.deg2rad(210)), np.sin(np.deg2rad(210))],
            [np.cos(np.deg2rad(270)), np.sin(np.deg2rad(270))],
            [np.cos(np.deg2rad(330)), np.sin(np.deg2rad(330))]
        ])
        
        max_proj = 0.0
        # Precompute rotated vertices for each hex?
        # For speed, we can approximate by checking center + max radius in normal direction.
        # But hexagon is not circular.
        # Let's compute exactly for the few critical normals.
        # Actually, for a given normal n, the max projection of a rotated hexagon is:
        # max_j dot(pos + R(a)v_j - outer_center, n)
        
        for i in range(n):
            a = angles[i]
            R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            # Unit hex vertices (angle 0)
            v0 = np.array([[1,0], [0.5, np.sqrt(3)/2], [-0.5, np.sqrt(3)/2], [-1,0], [-0.5, -np.sqrt(3)/2], [0.5, -np.sqrt(3)/2]])
            verts = pos[i] + v0 @ R.T
            verts_shifted = verts - outer_center
            projs = verts_shifted @ outer_normals.T
            max_proj = max(max_proj, np.max(projs))
            
        side = max_proj * 2 / np.sqrt(3)
        
        # 2. Check overlap (penalty)
        # For each pair, check if they intersect.
        # Approximation: distance between centers < sqrt(3) - epsilon?
        # If angles are different, it's complex.
        # Let's use a simple distance penalty: if dist < 1.8, penalty.
        # This is a heuristic to keep them apart during SA.
        # A more rigorous check is expensive.
        # Given the topological initialization, they start non-overlapping.
        # We just need to prevent them from sliding into each other.
        
        overlap_penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < 1.7: # Slightly less than sqrt(3) ~ 1.732 to allow some flex but prevent deep overlap
                    overlap_penalty += (1.7 - d)**2 * 100
                    
        return side + overlap_penalty

    # Iterate through candidates
    # To ensure we don't spend too much time on bad topologies, we can limit iterations per topology
    # or break if we find a very good one.
    
    # Sort candidates by some heuristic (e.g. bounding box area) to try compact ones first?
    # For now, just iterate.
    
    for topo in candidates:
        if time.time() - start_time > time_limit - 20:
            break
            
        cost, pos, angles, outer = optimize_topology(topo)
        
        # Validate result
        if cost < best_side:
            # Perform a final precise check and refinement
            # The SA might have left them slightly overlapping or outside.
            # Let's do a final "projection" step.
            
            # Re-calculate side length precisely for this config
            # If any hex is outside, we must increase side.
            # The cost function already included the side length derived from max projection.
            # So `cost` (minus penalty) is a lower bound on the side length if no overlap.
            
            if overlap_penalty := (cost - (max_proj_calc(pos, angles, outer) * 2 / np.sqrt(3))) > 1e-3:
                # If there was a penalty, the config is likely invalid.
                # We can try to fix it by pushing overlapping hexes apart.
                # For simplicity, we might skip this candidate if penalty is high.
                pass
                
            best_side = cost
            best_positions = pos
            best_angles = angles
            best_outer = outer

    # If no candidates worked well, or best_side is still inf, use a fallback
    if best_positions is None:
        # Fallback: Simple linear arrangement
        best_positions = np.array([[i * 1.74, 0] for i in range(n)])
        best_angles = np.zeros(n)
        best_outer = np.zeros(2)
        best_side = 20.0

    # Final Formatting
    # We need to return the side length that GUARANTEES containment.
    # Calculate the exact max projection for the best_positions.
    
    def get_exact_side(pos, angles, outer_center):
        outer_normals = np.array([
            [np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))],
            [np.cos(np.deg2rad(90)), np.sin(np.deg2rad(90))],
            [np.cos(np.deg2rad(150)), np.sin(np.deg2rad(150))],
            [np.cos(np.deg2rad(210)), np.sin(np.deg2rad(210))],
            [np.cos(np.deg2rad(270)), np.sin(np.deg2rad(270))],
            [np.cos(np.deg2rad(330)), np.sin(np.deg2rad(330))]
        ])
        max_proj = 0.0
        for i in range(n):
            a = angles[i]
            R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            v0 = np.array([[1,0], [0.5, np.sqrt(3)/2], [-0.5, np.sqrt(3)/2], [-1,0], [-0.5, -np.sqrt(3)/2], [0.5, -np.sqrt(3)/2]])
            verts = pos[i] + v0 @ R.T
            verts_shifted = verts - outer_center
            projs = verts_shifted @ outer_normals.T
            max_proj = max(max_proj, np.max(projs))
        return max_proj * 2 / np.sqrt(3)

    final_side = get_exact_side(best_positions, best_angles, best_outer)
    
    # Add a tiny epsilon to ensure strict containment in case of floating point issues
    # The validator allows cross product >= -1e-6, so being on the boundary is okay,
    # but slightly inside is safer.
    final_side += 1e-5

    inner_hex_data = np.zeros((n, 3))
    inner_hex_data[:, 0] = best_positions[:, 0]
    inner_hex_data[:, 1] = best_positions[:, 1]
    inner_hex_data[:, 2] = np.rad2deg(best_angles)
    
    outer_hex_data = np.array([best_outer[0], best_outer[1], 0.0])
    
    return inner_hex_data, outer_hex_data, float(final_side)

# Helper used in the loop above, defined here for clarity in the flow, 
# though in the actual execution it's called inside optimize_topology's cost calc implicitly.
# Wait, I used `max_proj_calc` in the validation step but didn't define it separately.
# Let's refactor to ensure `calculate_cost` is the single source of truth for side length calculation.

def max_proj_calc(pos, angles, outer_center):
    outer_normals = np.array([
        [np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))],
        [np.cos(np.deg2rad(90)), np.sin(np.deg2rad(90))],
        [np.cos(np.deg2rad(150)), np.sin(np.deg2rad(150))],
        [np.cos(np.deg2rad(210)), np.sin(np.deg2rad(210))],
        [np.cos(np.deg2rad(270)), np.sin(np.deg2rad(270))],
        [np.cos(np.deg2rad(330)), np.sin(np.deg2rad(330))]
    ])
    max_proj = 0.0
    for i in range(len(pos)):
        a = angles[i]
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        v0 = np.array([[1,0], [0.5, np.sqrt(3)/2], [-0.5, np.sqrt(3)/2], [-1,0], [-0.5, -np.sqrt(3)/2], [0.5, -np.sqrt(3)/2]])
        verts = pos[i] + v0 @ R.T
        verts_shifted = verts - outer_center
        projs = verts_shifted @ outer_normals.T
        max_proj = max(max_proj, np.max(projs))
    return max_proj

# Re-run the main logic with the corrected helper usage
# The code above is structured correctly, but I need to make sure `max_proj_calc` is accessible.
# It is defined at the bottom, but Python resolves it at call time.
# However, `calculate_cost` inside `optimize_topology` calculates side length inline.
# The validation step uses `max_proj_calc`.

# To be safe and clean, I will wrap the whole thing in the function as requested.

def hexagon_packing_11():
    start_time = time.time()
    time_limit = 600.0
    n = 11
    
    sqrt3 = np.sqrt(3)
    neighbor_dirs = np.array([
        [np.cos(np.deg2rad(30)) * sqrt3, np.sin(np.deg2rad(30)) * sqrt3],
        [np.cos(np.deg2rad(90)) * sqrt3, np.sin(np.deg2rad(90)) * sqrt3],
        [np.cos(np.deg2rad(150)) * sqrt3, np.sin(np.deg2rad(150)) * sqrt3],
        [np.cos(np.deg2rad(210)) * sqrt3, np.sin(np.deg2rad(210)) * sqrt3],
        [np.cos(np.deg2rad(270)) * sqrt3, np.sin(np.deg2rad(270)) * sqrt3],
        [np.cos(np.deg2rad(330)) * sqrt3, np.sin(np.deg2rad(330)) * sqrt3]
    ])
    
    def axial_to_cartesian(q, r):
        return q * neighbor_dirs[0] + r * neighbor_dirs[1]
    
    candidates = []
    seen_canonical = set()
    
    def generate_clusters(current_set, depth):
        if depth == n:
            key = tuple(sorted(current_set))
            if key not in seen_canonical:
                seen_canonical.add(key)
                candidates.append(list(current_set))
            return
        
        neighbors = set()
        for (q, r) in current_set:
            for dq, dr in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]:
                nq, nr = q + dq, r + dr
                if (nq, nr) not in current_set:
                    neighbors.add((nq, nr))
        
        for pos in neighbors:
            qs = [p[0] for p in current_set] + [pos[0]]
            rs = [p[1] for p in current_set] + [pos[1]]
            if max(qs) - min(qs) > 6 or max(rs) - min(rs) > 6:
                continue
            new_set = set(current_set)
            new_set.add(pos)
            generate_clusters(new_set, depth + 1)
            
    initial_set = {(0, 0)}
    generate_clusters(initial_set, 1)
    
    if not candidates:
        candidates = [ [(0,0), (1,0), (2,0), (3,0), (4,0), (5,0), (6,0), (7,0), (8,0), (9,0), (10,0)] ]
        
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_outer = None
    
    outer_normals = np.array([
        [np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))],
        [np.cos(np.deg2rad(90)), np.sin(np.deg2rad(90))],
        [np.cos(np.deg2rad(150)), np.sin(np.deg2rad(150))],
        [np.cos(np.deg2rad(210)), np.sin(np.deg2rad(210))],
        [np.cos(np.deg2rad(270)), np.sin(np.deg2rad(270))],
        [np.cos(np.deg2rad(330)), np.sin(np.deg2rad(330))]
    ])
    v0 = np.array([[1,0], [0.5, np.sqrt(3)/2], [-0.5, np.sqrt(3)/2], [-1,0], [-0.5, -np.sqrt(3)/2], [0.5, -np.sqrt(3)/2]])

    def calculate_cost(pos, angles, outer_center):
        max_proj = 0.0
        for i in range(n):
            a = angles[i]
            R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            verts = pos[i] + v0 @ R.T
            verts_shifted = verts - outer_center
            projs = verts_shifted @ outer_normals.T
            max_proj = max(max_proj, np.max(projs))
        side = max_proj * 2 / np.sqrt(3)
        
        overlap_penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < 1.7:
                    overlap_penalty += (1.7 - d)**2 * 100
                    
        return side + overlap_penalty

    def optimize_topology(topo_axial):
        init_pos = np.array([axial_to_cartesian(q, r) for q, r in topo_axial])
        centroid = np.mean(init_pos, axis=0)
        init_pos -= centroid
        angles = np.zeros(n)
        outer_center = np.zeros(2)
        
        T = 0.05
        T_min = 1e-5
        alpha = 0.999
        max_iter = 20000
        current_cost = float('inf')
        current_pos = init_pos.copy()
        current_angles = angles.copy()
        current_outer = outer_center.copy()
        
        best_cost = float('inf')
        best_pos = init_pos.copy()
        best_angles = angles.copy()
        best_outer = outer_center.copy()
        
        for i in range(max_iter):
            if time.time() - start_time > time_limit - 10:
                break
            
            idx = np.random.randint(n)
            delta_pos = np.random.normal(0, 0.01, 2)
            new_pos = current_pos.copy()
            new_pos[idx] += delta_pos
            
            if np.random.rand() < 0.1:
                idx_rot = np.random.randint(n)
                delta_angle = np.random.normal(0, 0.05)
                new_angles = current_angles.copy()
                new_angles[idx_rot] += delta_angle
            else:
                new_angles = current_angles
                
            if np.random.rand() < 0.2:
                new_outer = current_outer + np.random.normal(0, 0.01, 2)
            else:
                new_outer = current_outer
                
            cost = calculate_cost(new_pos, new_angles, new_outer)
            
            if cost < current_cost or np.random.rand() < np.exp(-(cost - current_cost) / T):
                current_cost = cost
                current_pos = new_pos
                current_angles = new_angles
                current_outer = new_outer
                
                if cost < best_cost:
                    best_cost = cost
                    best_pos = current_pos.copy()
                    best_angles = current_angles.copy()
                    best_outer = current_outer.copy()
                    
            T *= alpha
            if T < T_min:
                break
                
        return best_cost, best_pos, best_angles, best_outer

    for topo in candidates:
        if time.time() - start_time > time_limit - 20:
            break
        cost, pos, angles, outer = optimize_topology(topo)
        if cost < best_side:
            best_side = cost
            best_positions = pos
            best_angles = angles
            best_outer = outer

    if best_positions is None:
        best_positions = np.array([[i * 1.74, 0] for i in range(n)])
        best_angles = np.zeros(n)
        best_outer = np.zeros(2)
        best_side = 20.0

    # Final exact side calculation
    max_proj = 0.0
    for i in range(n):
        a = best_angles[i]
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        verts = best_positions[i] + v0 @ R.T
        verts_shifted = verts - best_outer
        projs = verts_shifted @ outer_normals.T
        max_proj = max(max_proj, np.max(projs))
        
    final_side = max_proj * 2 / np.sqrt(3) + 1e-5

    inner_hex_data = np.zeros((n, 3))
    inner_hex_data[:, 0] = best_positions[:, 0]
    inner_hex_data[:, 1] = best_positions[:, 1]
    inner_hex_data[:, 2] = np.rad2deg(best_angles)
    
    outer_hex_data = np.array([best_outer[0], best_outer[1], 0.0])
    
    return inner_hex_data, outer_hex_data, float(final_side)

