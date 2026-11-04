import numpy as np
import time
import math

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Constants
    sqrt3 = math.sqrt(3)
    
    # Helper: Get vertices of a unit hexagon at (x,y) with rotation theta
    def get_vertices(x, y, theta):
        # Vertices of unit hexagon centered at origin
        # Angles: 0, 60, 120, 180, 240, 300
        angles = np.array([0, 60, 120, 180, 240, 300]) * (np.pi / 180.0)
        # Rotate
        angles = angles + theta
        vx = np.cos(angles)
        vy = np.sin(angles)
        # Translate
        vx = vx + x
        vy = vy + y
        return np.column_stack((vx, vy))

    # Helper: Check overlap between two sets of vertices (convex polygons) using SAT
    # Returns max penetration depth (0 if disjoint)
    def check_overlap(v1, v2):
        # v1, v2 are (6, 2) arrays
        # Get edges of v1 and v2
        # For each edge, get normal
        # Project both polygons onto normal
        # Check for separation
        
        def get_normals(v):
            # v is (6,2)
            edges = np.roll(v, -1, axis=0) - v
            # Normal to (dx, dy) is (-dy, dx)
            normals = np.column_stack((-edges[:, 1], edges[:, 0]))
            # Normalize
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            norms[norms == 0] = 1
            return normals / norms
        
        n1 = get_normals(v1)
        n2 = get_normals(v2)
        
        max_pen = 0.0
        # Check axes from v1
        for n in n1:
            p1 = v1 @ n
            p2 = v2 @ n
            min1, max1 = np.min(p1), np.max(p1)
            min2, max2 = np.min(p2), np.max(p2)
            
            # If separated
            if max1 < min2 or max2 < min1:
                return 0.0 # Disjoint
            
            # Overlap
            overlap = min(max1, max2) - max(min1, min2)
            if overlap > max_pen:
                max_pen = overlap
                
        # Check axes from v2
        for n in n2:
            p1 = v1 @ n
            p2 = v2 @ n
            min1, max1 = np.min(p1), np.max(p1)
            min2, max2 = np.min(p2), np.max(p2)
            
            if max1 < min2 or max2 < min1:
                return 0.0 # Disjoint
            
            overlap = min(max1, max2) - max(min1, min2)
            if overlap > max_pen:
                max_pen = overlap
                
        return max_pen

    # Helper: Calculate required outer side length
    def calc_side(pos, rot, outer_rot, outer_center):
        # pos: (11, 2), rot: (11,)
        # outer_rot: scalar, outer_center: (2,)
        
        # Collect all vertices
        all_verts = []
        for i in range(N):
            v = get_vertices(pos[i, 0], pos[i, 1], rot[i])
            all_verts.append(v)
        
        all_verts = np.vstack(all_verts) # (66, 2)
        
        # Translate to outer center
        all_verts = all_verts - outer_center
        
        # Rotate by -outer_rot
        c = math.cos(-outer_rot)
        s = math.sin(-outer_rot)
        R = np.array([[c, -s], [s, c]])
        all_verts = all_verts @ R.T
        
        # Project onto 3 axes of outer hexagon
        # Outer hexagon sides are at angles 0, 60, 120 (normals)
        # Normals: (1, 0), (cos60, sin60), (cos120, sin120)
        # Actually, for a regular hexagon, the distance from center to side is R_in.
        # The condition is |v . n_k| <= R_in for k=0,1,2
        # n_0 = (1, 0)
        # n_1 = (cos(60), sin(60))
        # n_2 = (cos(120), sin(120))
        
        n0 = np.array([1.0, 0.0])
        n1 = np.array([0.5, sqrt3/2])
        n2 = np.array([-0.5, sqrt3/2])
        
        projs0 = np.abs(all_verts @ n0)
        projs1 = np.abs(all_verts @ n1)
        projs2 = np.abs(all_verts @ n2)
        
        max_proj = max(np.max(projs0), np.max(projs1), np.max(projs2))
        
        # R_in = max_proj
        # S = 2 * R_in / sqrt(3)
        return 2.0 * max_proj / sqrt3

    # Helper: Calculate total cost
    def calc_cost(pos, rot, outer_rot, outer_center):
        # 1. Calculate S
        S = calc_side(pos, rot, outer_rot, outer_center)
        
        # 2. Calculate overlap penalty
        # Pre-compute vertices for overlap check to avoid re-computation in SAT?
        # For N=11, 55 pairs. SAT is O(1) per pair. 55 SATs is fast.
        
        verts = []
        for i in range(N):
            v = get_vertices(pos[i, 0], pos[i, 1], rot[i])
            verts.append(v)
            
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                pen = check_overlap(verts[i], verts[j])
                if pen > 0:
                    penalty += pen * pen # Quadratic penalty to encourage separation
        
        return S + 10.0 * penalty # Weight for penalty

    # Initialize
    # Start with a hexagonal lattice pattern
    # Center at 0,0
    # 1 center, 6 ring 1, 4 more?
    # Let's just place them in a grid and let SA fix it
    pos = np.zeros((N, 2))
    rot = np.zeros(N)
    
    # Initial layout: Hexagonal close packing
    # 3-4-4 triangle? Or 1-6-4?
    # Let's try a 3x3 grid minus corners?
    # Or just random positions in a small box
    np.random.seed(42)
    pos = np.random.uniform(-5, 5, (N, 2))
    rot = np.random.uniform(0, 2*np.pi, N)
    outer_rot = 0.0
    outer_center = np.array([0.0, 0.0])
    
    best_cost = calc_cost(pos, rot, outer_rot, outer_center)
    best_state = (pos.copy(), rot.copy(), outer_rot, outer_center.copy())
    
    # Simulated Annealing
    T = 10.0
    T_min = 1e-6
    alpha = 0.995
    
    current_pos, current_rot, current_outer_rot, current_outer_center = best_state
    current_cost = best_cost
    
    step_size = 0.5
    
    while time.time() < deadline:
        # Pick a random variable to perturb
        var_type = np.random.randint(4) # 0: pos, 1: rot, 2: outer_rot, 3: outer_center
        
        new_pos = current_pos.copy()
        new_rot = current_rot.copy()
        new_outer_rot = current_outer_rot
        new_outer_center = current_outer_center.copy()
        
        if var_type == 0:
            i = np.random.randint(N)
            new_pos[i, 0] += np.random.normal(0, step_size)
            new_pos[i, 1] += np.random.normal(0, step_size)
        elif var_type == 1:
            i = np.random.randint(N)
            new_rot[i] += np.random.normal(0, step_size * 0.5)
        elif var_type == 2:
            new_outer_rot += np.random.normal(0, step_size * 0.1)
        else:
            new_outer_center[0] += np.random.normal(0, step_size * 0.5)
            new_outer_center[1] += np.random.normal(0, step_size * 0.5)
            
        new_cost = calc_cost(new_pos, new_rot, new_outer_rot, new_outer_center)
        
        delta = new_cost - current_cost
        if delta < 0 or np.random.rand() < math.exp(-delta / T):
            current_pos = new_pos
            current_rot = new_rot
            current_outer_rot = new_outer_rot
            current_outer_center = new_outer_center
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_state = (current_pos.copy(), current_rot.copy(), current_outer_rot, current_outer_center.copy())
                
        T *= alpha
        if T < T_min:
            T = T_min
            # Optional: restart with smaller step size if stuck?
            # For now, keep going with low T
            
    # Final refinement: Gradient descent? 
    # Or just return best SA result.
    # We need to ensure the result is valid (disjoint).
    # If best_state has overlaps, we might need to project them out or accept a slightly larger S.
    # The penalty ensures SA tries to avoid overlaps.
    # Let's verify the best state.
    
    p_best, r_best, o_rot_best, o_c_best = best_state
    S_best = calc_side(p_best, r_best, o_rot_best, o_c_best)
    
    # Check if disjoint
    verts = []
    for i in range(N):
        v = get_vertices(p_best[i, 0], p_best[i, 1], r_best[i])
        verts.append(v)
        
    is_disjoint = True
    for i in range(N):
        for j in range(i+1, N):
            if check_overlap(verts[i], verts[j]) > 1e-6:
                is_disjoint = False
                break
        if not is_disjoint:
            break
            
    if not is_disjoint:
        # Fallback: Use a known valid configuration or increase S
        # To be safe, we can just return the configuration and let the evaluator check.
        # But if it's invalid, score is 0.
        # We should try to "push" overlapping hexagons apart in a final cleanup.
        
        # Simple cleanup: Local search on positions only, minimizing overlap penalty, keeping S fixed?
        # Or just increase S until no overlaps? No, S is determined by the extent.
        # If they overlap, the "required S" is still the same, but the config is invalid.
        # We need to move them apart.
        
        # Let's run a quick local search to remove overlaps
        for _ in range(1000):
            moved = False
            for i in range(N):
                for j in range(i+1, N):
                    pen = check_overlap(verts[i], verts[j])
                    if pen > 1e-6:
                        # Push i and j apart along the separating axis? 
                        # SAT gives the axis of maximum penetration.
                        # We can just move them randomly in opposite directions.
                        dx = np.random.normal(0, 0.1)
                        dy = np.random.normal(0, 0.1)
                        p_best[i, 0] += dx
                        p_best[i, 1] += dy
                        p_best[j, 0] -= dx
                        p_best[j, 1] -= dy
                        # Update verts
                        verts[i] = get_vertices(p_best[i, 0], p_best[i, 1], r_best[i])
                        verts[j] = get_vertices(p_best[j, 0], p_best[j, 1], r_best[j])
                        moved = True
                        break
                if moved:
                    break
            if not moved:
                break
        
        # Recalculate S
        S_best = calc_side(p_best, r_best, o_rot_best, o_c_best)
        
    # Format output
    inner = []
    for i in range(N):
        inner.append([p_best[i, 0], p_best[i, 1], r_best[i] * 180.0 / np.pi])
        
    return inner, (o_c_best[0], o_c_best[1]), S_best, o_rot_best * 180.0 / np.pi

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time
    import math
    
    start_time = time.time()
    deadline = start_time + 110.0
    sqrt3 = math.sqrt(3)
    N = 11
    
    def get_vertices(x, y, theta):
        angles = np.array([0, 60, 120, 180, 240, 300]) * (np.pi / 180.0)
        angles = angles + theta
        vx = np.cos(angles) + x
        vy = np.sin(angles) + y
        return np.column_stack((vx, vy))
        
    def check_overlap(v1, v2):
        def get_normals(v):
            edges = np.roll(v, -1, axis=0) - v
            normals = np.column_stack((-edges[:, 1], edges[:, 0]))
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            norms[norms == 0] = 1
            return normals / norms
            
        n1 = get_normals(v1)
        n2 = get_normals(v2)
        max_pen = 0.0
        for n in n1:
            p1 = v1 @ n
            p2 = v2 @ n
            if np.max(p1) < np.min(p2) or np.max(p2) < np.min(p1):
                return 0.0
            overlap = min(np.max(p1), np.max(p2)) - max(np.min(p1), np.min(p2))
            if overlap > max_pen:
                max_pen = overlap
        for n in n2:
            p1 = v1 @ n
            p2 = v2 @ n
            if np.max(p1) < np.min(p2) or np.max(p2) < np.min(p1):
                return 0.0
            overlap = min(np.max(p1), np.max(p2)) - max(np.min(p1), np.min(p2))
            if overlap > max_pen:
                max_pen = overlap
        return max_pen
        
    def calc_side(pos, rot, outer_rot, outer_center):
        all_verts = []
        for i in range(N):
            v = get_vertices(pos[i, 0], pos[i, 1], rot[i])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        all_verts = all_verts - outer_center
        c = math.cos(-outer_rot)
        s = math.sin(-outer_rot)
        R = np.array([[c, -s], [s, c]])
        all_verts = all_verts @ R.T
        n0 = np.array([1.0, 0.0])
        n1 = np.array([0.5, sqrt3/2])
        n2 = np.array([-0.5, sqrt3/2])
        projs0 = np.abs(all_verts @ n0)
        projs1 = np.abs(all_verts @ n1)
        projs2 = np.abs(all_verts @ n2)
        max_proj = max(np.max(projs0), np.max(projs1), np.max(projs2))
        return 2.0 * max_proj / sqrt3
        
    def calc_cost(pos, rot, outer_rot, outer_center):
        S = calc_side(pos, rot, outer_rot, outer_center)
        verts = []
        for i in range(N):
            v = get_vertices(pos[i, 0], pos[i, 1], rot[i])
            verts.append(v)
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                pen = check_overlap(verts[i], verts[j])
                if pen > 0:
                    penalty += pen * pen
        return S + 10.0 * penalty

    np.random.seed(42)
    pos = np.random.uniform(-5, 5, (N, 2))
    rot = np.random.uniform(0, 2*np.pi, N)
    outer_rot = 0.0
    outer_center = np.array([0.0, 0.0])
    
    best_cost = calc_cost(pos, rot, outer_rot, outer_center)
    best_state = (pos.copy(), rot.copy(), outer_rot, outer_center.copy())
    
    T = 10.0
    T_min = 1e-6
    alpha = 0.995
    step_size = 0.5
    
    current_pos, current_rot, current_outer_rot, current_outer_center = best_state
    current_cost = best_cost
    
    while time.time() < deadline:
        var_type = np.random.randint(4)
        new_pos = current_pos.copy()
        new_rot = current_rot.copy()
        new_outer_rot = current_outer_rot
        new_outer_center = current_outer_center.copy()
        
        if var_type == 0:
            i = np.random.randint(N)
            new_pos[i, 0] += np.random.normal(0, step_size)
            new_pos[i, 1] += np.random.normal(0, step_size)
        elif var_type == 1:
            i = np.random.randint(N)
            new_rot[i] += np.random.normal(0, step_size * 0.5)
        elif var_type == 2:
            new_outer_rot += np.random.normal(0, step_size * 0.1)
        else:
            new_outer_center[0] += np.random.normal(0, step_size * 0.5)
            new_outer_center[1] += np.random.normal(0, step_size * 0.5)
            
        new_cost = calc_cost(new_pos, new_rot, new_outer_rot, new_outer_center)
        delta = new_cost - current_cost
        if delta < 0 or np.random.rand() < math.exp(-delta / T):
            current_pos = new_pos
            current_rot = new_rot
            current_outer_rot = new_outer_rot
            current_outer_center = new_outer_center
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_state = (current_pos.copy(), current_rot.copy(), current_outer_rot, current_outer_center.copy())
        T *= alpha
        if T < T_min:
            T = T_min

    p_best, r_best, o_rot_best, o_c_best = best_state
    S_best = calc_side(p_best, r_best, o_rot_best, o_c_best)
    
    verts = []
    for i in range(N):
        v = get_vertices(p_best[i, 0], p_best[i, 1], r_best[i])
        verts.append(v)
        
    is_disjoint = True
    for i in range(N):
        for j in range(i+1, N):
            if check_overlap(verts[i], verts[j]) > 1e-6:
                is_disjoint = False
                break
        if not is_disjoint:
            break
            
    if not is_disjoint:
        for _ in range(1000):
            moved = False
            for i in range(N):
                for j in range(i+1, N):
                    pen = check_overlap(verts[i], verts[j])
                    if pen > 1e-6:
                        dx = np.random.normal(0, 0.1)
                        dy = np.random.normal(0, 0.1)
                        p_best[i, 0] += dx
                        p_best[i, 1] += dy
                        p_best[j, 0] -= dx
                        p_best[j, 1] -= dy
                        verts[i] = get_vertices(p_best[i, 0], p_best[i, 1], r_best[i])
                        verts[j] = get_vertices(p_best[j, 0], p_best[j, 1], r_best[j])
                        moved = True
                        break
                if moved:
                    break
            if not moved:
                break
        S_best = calc_side(p_best, r_best, o_rot_best, o_c_best)
        
    inner = []
    for i in range(N):
        inner.append([p_best[i, 0], p_best[i, 1], r_best[i] * 180.0 / np.pi])
        
    return inner, (o_c_best[0], o_c_best[1]), S_best, o_rot_best * 180.0 / np.pi
# EVOLVE-BLOCK-END
