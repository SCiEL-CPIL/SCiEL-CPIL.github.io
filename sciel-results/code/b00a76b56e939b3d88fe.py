"""Packing 12 unit hexagons in minimal regular hexagon."""
import numpy as np
import math
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    sqrt3 = math.sqrt(3.0)
    normals = [(math.cos(math.radians(60*k+30)), math.sin(math.radians(60*k+30))) for k in range(6)]
    
    def get_hex_verts(cx, cy, theta):
        return [(cx + math.cos(math.radians(60*k) + theta), cy + math.sin(math.radians(60*k) + theta)) for k in range(6)]

    def point_in_poly(px, py, poly):
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(1, n + 1):
            p2x, p2y = poly[i % n]
            if (p1y > py) != (p2y > py):
                x_int = (px - p1x) * (p2y - p1y) / (p2y - p1y) + p1x
                if px < x_int:
                    inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def check_overlap(v1, v2):
        # Bounding box
        if max(v[0] for v in v1) < min(v[0] for v in v2) or max(v[0] for v in v2) < min(v[0] for v in v1): return 0
        if max(v[1] for v in v1) < min(v[1] for v in v2) or max(v[1] for v in v2) < min(v[1] for v in v1): return 0
        count = 0
        for v in v1:
            if point_in_poly(v[0], v[1], v2): count += 1
        for v in v2:
            if point_in_poly(v[0], v[1], v1): count += 1
        return count

    def calc_L(cx, cy, theta):
        verts = get_hex_verts(cx, cy, theta)
        max_proj = 0
        for nx, ny in normals:
            p = max(vx*nx + vy*ny for vx, vy in verts)
            if p > max_proj: max_proj = p
        return max_proj / (sqrt3/2.0)

    def is_valid(centers, angles):
        verts_list = [get_hex_verts(c[0], c[1], a) for c, a in zip(centers, angles)]
        for i in range(12):
            for j in range(i+1, 12):
                # Quick bounding box check
                if max(v[0] for v in verts_list[i]) < min(v[0] for v in verts_list[j]): continue
                if max(v[0] for v in verts_list[j]) < min(v[0] for v in verts_list[i]): continue
                if max(v[1] for v in verts_list[i]) < min(v[1] for v in verts_list[j]): continue
                if max(v[1] for v in verts_list[j]) < min(v[1] for v in verts_list[i]): continue
                
                # Precise overlap check
                if check_overlap(verts_list[i], verts_list[j]) > 0:
                    return False
        return True

    # Init: Hexagonal lattice subset (valid, touching)
    # Generate a larger hex grid and pick 12 closest to origin
    grid_centers = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            if abs(q + r) <= 3:
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                grid_centers.append((x, y, q*q + r*r + q*r))
    grid_centers.sort(key=lambda t: t[2])
    
    init_c = [c[:2] for c in grid_centers[:12]]
    init_a = [0.0] * 12
    
    # Verify initial validity (should be true)
    if not is_valid(init_c, init_a):
        # Fallback: Spread them out very far
        init_c = [(i * 10.0, 0.0) for i in range(12)]
        init_a = [0.0] * 12

    cur_L = max(calc_L(c[0], c[1], a) for c, a in zip(init_c, init_a))
    best_L = cur_L
    best_pos = [tuple(c) for c in init_c] + init_a[:]
    
    # Local Search
    it = 0
    max_iters = 5000
    while time.time() < deadline and it < max_iters:
        it += 1
        # Pick a random hexagon to move
        i = np.random.randint(12)
        cx, cy = init_c[i]
        a = init_a[i]
        
        # Random perturbation
        step = 0.1
        new_cx = cx + (np.random.rand() - 0.5) * step
        new_cy = cy + (np.random.rand() - 0.5) * step
        new_a = a + (np.random.rand() - 0.5) * 0.2
        
        # Try to move it
        old_c = init_c[i]
        old_a = init_a[i]
        init_c[i] = (new_cx, new_cy)
        init_a[i] = new_a
        
        # Check validity
        if is_valid(init_c, init_a):
            new_L = max(calc_L(c[0], c[1], ang) for c, ang in zip(init_c, init_a))
            if new_L < best_L:
                best_L = new_L
                best_pos = [tuple(c) for c in init_c] + init_a[:]
            # Accept move if it improves or is equal? 
            # To escape local minima, we might accept worse valid moves, 
            # but for safety and simplicity, let's stick to hill climbing 
            # or simple random walk with restarts if stuck.
            # Given the time limit, simple hill climbing on valid moves is robust.
            # If new_L <= cur_L: 
            #    cur_L = new_L
            # else:
            #    # Revert? Or keep? 
            #    # Keeping invalid moves is bad. Keeping worse valid moves can help escape.
            #    # Let's keep it if it's valid, even if L is worse, to allow exploration.
            #    pass 
        else:
            # Revert invalid move
            init_c[i] = old_c
            init_a[i] = old_a

    # Ensure best_pos is valid
    # best_pos structure: 12 centers + 12 angles
    bp_centers = [best_pos[i] for i in range(12)]
    bp_angles = [best_pos[12+i] for i in range(12)]
    
    if not is_valid(bp_centers, bp_angles):
         # This should not happen if we only update best_pos when valid, 
         # but just in case, use the last known valid state (init_c, init_a)
         # Note: init_c, init_a are the current state, which is valid at the end of loop
         bp_centers = [c for c in init_c]
         bp_angles = [a for a in init_a]
         best_L = max(calc_L(c[0], c[1], a) for c, a in zip(bp_centers, bp_angles))

    inner = [[bp_centers[i][0], bp_centers[i][1], math.degrees(bp_angles[i])] for i in range(12)]
    return inner, (0.0, 0.0), best_L + 1e-6, 0.0
# EVOLVE-BLOCK-END
