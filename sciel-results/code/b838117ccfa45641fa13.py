"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (evaluator limit 150s)
    
    SQRT3 = math.sqrt(3)
    
    # Helper: Get vertices of a unit hexagon centered at (cx, cy) with rotation angle
    def get_hex_vertices(cx, cy, angle_deg):
        verts = []
        for i in range(6):
            theta = math.radians(angle_deg + i * 60)
            x = cx + math.cos(theta)
            y = cy + math.sin(theta)
            verts.append((x, y))
        return verts

    # Helper: Check if two hexagons overlap using Separating Axis Theorem (SAT)
    def get_axes(poly):
        axes = []
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            norm = math.sqrt(dx*dx + dy*dy)
            if norm == 0: continue
            nx = -dy / norm
            ny = dx / norm
            axes.append((nx, ny))
        return axes

    def project(poly, axis):
        dots = [p[0]*axis[0] + p[1]*axis[1] for p in poly]
        return min(dots), max(dots)

    def is_disjoint(poly1, poly2, eps=1e-9):
        axes = get_axes(poly1) + get_axes(poly2)
        for axis in axes:
            min1, max1 = project(poly1, axis)
            min2, max2 = project(poly2, axis)
            if max1 < min2 + eps or max2 < min1 + eps:
                return True
        return False

    # Helper: Calculate minimum outer hexagon side length containing all given hexagons
    def calc_outer_side(hexagons, outer_angle_deg):
        base_angles = [30, 90, 150, 210, 270, 330]
        
        max_proj = 0.0
        for hex_verts in hexagons:
            for vx, vy in hex_verts:
                for a in base_angles:
                    rad = math.radians(a + outer_angle_deg)
                    nx = math.cos(rad)
                    ny = math.sin(rad)
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_proj:
                        max_proj = proj
        
        # Apothem of outer hex is S * SQRT3 / 2
        # Max projection must be <= Apothem
        # S >= 2 * max_proj / SQRT3
        return 2.0 * max_proj / SQRT3

    def axial_to_cartesian(q, r, spacing=1.0):
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    # Generate candidate positions on a dense triangular lattice
    # We will search in a box around the origin. 
    # The optimal side length for 11 hexagons is expected to be around 4-5.
    # Let's generate candidates in a range that covers a hexagon of side 6.
    
    # Lattice spacing: The distance between centers of touching hexagons is 2*apothem = sqrt(3).
    # But we can place them closer if rotated? No, minimum distance between centers of disjoint unit hexagons is 2*inradius = sqrt(3).
    # So a lattice with spacing sqrt(3) is natural.
    
    # Let's define a grid of candidate positions.
    # We'll use axial coordinates for the lattice.
    # The lattice points are (q, r) mapped to cartesian.
    # To allow for "non-lattice" optimal positions, we will also refine later.
    
    # Candidate angles: 0, 30, 60... but 0 and 30 are equivalent up to rotation of the whole set?
    # Actually, rotating a hex by 30 degrees makes it "pointy" vs "flat" relative to axes.
    # Let's try angles: 0, 30.
    
    candidate_angles = [0.0, 30.0]
    
    # Generate lattice points
    # Range of q, r such that the cartesian points are within radius ~4 from origin.
    # Max radius for side 5 hexagon is 5.
    # Let's use a generous range.
    lattice_points = []
    for q in range(-4, 5):
        for r in range(-4, 5):
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            # Check if within reasonable bounds
            if math.sqrt(x*x + y*y) < 5.0:
                lattice_points.append((x, y))
    
    # Sort lattice points by distance from origin to prioritize central placement
    lattice_points.sort(key=lambda p: p[0]*p[0] + p[1]*p[1])
    
    best_global_side = float('inf')
    best_global_config = None
    
    # Greedy placement strategy
    # We will try to place N hexagons.
    # We start with the center (0,0) for the first hexagon.
    # Then we pick the next best position from the remaining candidates that doesn't overlap.
    
    # To improve, we will run this greedy process multiple times with different random perturbations
    # or different orderings, and then refine the result using Nelder-Mead.
    
    def greedy_pack(num_hexagons, lattice_pts, angles, random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        placed = [] # List of (x, y, angle)
        placed_polys = [] # List of vertex lists
        
        # First hexagon at origin
        placed.append((0.0, 0.0, angles[0]))
        placed_polys.append(get_hex_vertices(0.0, 0.0, angles[0]))
        
        # Candidate pool: all lattice points and angles
        # We will iterate through candidates sorted by distance from origin
        # and pick the first one that fits.
        
        # To break ties and explore different configurations, we can shuffle the candidate list slightly
        # or use a random tie-breaker.
        
        candidates = []
        for p in lattice_pts:
            for a in angles:
                candidates.append((p[0], p[1], a))
        
        # Sort by distance to origin
        candidates.sort(key=lambda c: c[0]*c[0] + c[1]*c[1])
        
        # Add some randomness to the selection order to explore different local minima
        # We'll use a "jittered" sort or random sampling from the top K candidates
        
        for _ in range(num_hexagons - 1):
            best_candidate = None
            best_dist = float('inf')
            
            # Check a subset of candidates to save time? 
            # Or check all until we find a valid one?
            # Checking all is O(N * M * K) where M is candidates, K is placed.
            # M ~ 100, K ~ 10. 1000 checks per step. 10 steps. 10000 checks. Fast.
            
            # To diversify, we pick the first valid candidate from a randomly shuffled subset of the top 20 closest candidates
            # Or we can just pick the closest valid one, but that's deterministic.
            # Let's try: Pick from top 10 closest candidates that are valid, randomly.
            
            valid_candidates = []
            for c in candidates[:30]: # Look at top 30 closest
                x, y, a = c
                # Quick check: distance to all placed centers
                # If distance < 2*inradius = sqrt(3), they might overlap. 
                # But rotation matters.
                # Let's do full SAT check only if close.
                too_close = False
                for px, py, pa in placed:
                    dx = x - px
                    dy = y - py
                    if dx*dx + dy*dy < 3.0: # (sqrt(3))^2
                        too_close = True
                        break
                if too_close:
                    continue
                    
                poly = get_hex_vertices(x, y, a)
                overlap = False
                for pp in placed_polys:
                    if not is_disjoint(poly, pp):
                        overlap = True
                        break
                if not overlap:
                    valid_candidates.append(c)
                    
            if not valid_candidates:
                break # No more space?
            
            # Pick randomly from valid candidates to explore
            chosen = random.choice(valid_candidates)
            x, y, a = chosen
            placed.append((x, y, a))
            placed_polys.append(get_hex_vertices(x, y, a))
            
        return placed, placed_polys

    def refine_configuration(placed, placed_polys):
        """
        Use Nelder-Mead to optimize positions (x, y) of all hexagons to minimize outer side length.
        Angles are fixed from the greedy step.
        """
        if len(placed) < N:
            return None, None, float('inf')
            
        # Flatten positions: [x0, y0, x1, y1, ...]
        x0 = np.array([p[0] for p in placed])
        y0 = np.array([p[1] for p in placed])
        
        def objective(params):
            x = params[:N]
            y = params[N:2*N]
            
            # Check overlap
            polys = []
            for i in range(N):
                polys.append(get_hex_vertices(x[i], y[i], placed[i][2]))
            
            for i in range(N):
                for j in range(i+1, N):
                    if not is_disjoint(polys[i], polys[j]):
                        return 1e6 # Heavy penalty for overlap
            
            # Calculate outer side
            # Try a few outer angles
            min_side = float('inf')
            for oa in [0, 10, 20, 30, 40, 50]:
                side = calc_outer_side(polys, oa)
                if side < min_side:
                    min_side = side
            return min_side
            
        initial_params = np.concatenate([x0, y0])
        
        try:
            res = minimize(objective, initial_params, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False})
            
            if res.success or res.fun < 1e5:
                x_opt = res.x[:N]
                y_opt = res.x[N:2*N]
                
                # Reconstruct placed list
                new_placed = []
                new_polys = []
                for i in range(N):
                    new_placed.append((x_opt[i], y_opt[i], placed[i][2]))
                    new_polys.append(get_hex_vertices(x_opt[i], y_opt[i], placed[i][2]))
                
                # Final validation
                for i in range(N):
                    for j in range(i+1, N):
                        if not is_disjoint(new_polys[i], new_polys[j]):
                            return None, None, float('inf')
                            
                min_side = float('inf')
                best_oa = 0
                for oa in range(0, 60, 1):
                    side = calc_outer_side(new_polys, oa)
                    if side < min_side:
                        min_side = side
                        best_oa = oa
                        
                return new_placed, new_polys, min_side
            else:
                return None, None, float('inf')
        except Exception:
            return None, None, float('inf')

    # Main search loop
    # Run greedy packing multiple times with different seeds
    num_restarts = 50
    
    for i in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Different seeds for diversity
        seed = i * 12345 + 6789
        
        placed, placed_polys = greedy_pack(N, lattice_points, candidate_angles, random_seed=seed)
        
        if len(placed) < N:
            continue
            
        # Refine
        refined_placed, refined_polys, side = refine_configuration(placed, placed_polys)
        
        if refined_placed is not None and side < best_global_side:
            best_global_side = side
            best_global_config = refined_placed
            
    # If no valid configuration found, fallback to a known valid lattice
    if best_global_config is None:
        # Use the simple 1+6+4 lattice
        coords = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (2,1), (1,1)]
        placed = []
        polys = []
        for q, r in coords:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            placed.append((x, y, 0.0))
            polys.append(get_hex_vertices(x, y, 0.0))
            
        # Center it
        cx = sum(p[0] for p in placed) / N
        cy = sum(p[1] for p in placed) / N
        placed = [(p[0]-cx, p[1]-cy, p[2]) for p in placed]
        polys = [get_hex_vertices(p[0], p[1], p[2]) for p in placed]
        
        best_side = float('inf')
        best_oa = 0
        for oa in range(0, 60, 1):
            side = calc_outer_side(polys, oa)
            if side < best_side:
                best_side = side
                best_oa = oa
                
        best_global_side = best_side
        best_global_config = placed
        
    # Finalize
    # Center the best config
    cx = sum(p[0] for p in best_global_config) / N
    cy = sum(p[1] for p in best_global_config) / N
    final_placed = [(p[0]-cx, p[1]-cy, p[2]) for p in best_global_config]
    
    # Recalculate final side length with finer angle search
    final_polys = [get_hex_vertices(p[0], p[1], p[2]) for p in final_placed]
    
    best_side = float('inf')
    best_oa = 0
    for oa in range(0, 60, 1):
        side = calc_outer_side(final_polys, oa)
        if side < best_side:
            best_side = side
            best_oa = oa
            
    inner_list = []
    for p in final_placed:
        inner_list.append([p[0], p[1], p[2]])
        
    return (inner_list, (0.0, 0.0), best_side, best_oa)

# EVOLVE-BLOCK-END
