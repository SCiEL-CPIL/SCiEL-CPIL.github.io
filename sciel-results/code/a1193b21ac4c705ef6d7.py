"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard: 150s limit, keep 10s margin
    deadline = time.time() + 140.0
    
    # 1. Initial Configuration: Hexagonal Lattice
    # For 12 hexagons, a compact arrangement is a hexagon of side 2 (1+4+7=12)
    # Or a 3x4 rectangle. Let's use a 3x4 grid for simplicity and symmetry, 
    # or better, a hexagonal cluster to minimize bounding box.
    # Hexagonal cluster of radius 1 (center + 6 + 5 more? No, 1+6=7, 1+6+12=19).
    # 12 hexagons fit in a "hexagon" of side length 2 minus 5? 
    # Let's just use a 3x4 grid of flat-top hexagons.
    # Side length s=1.
    # Flat top hex: width = 2s, height = sqrt(3)s.
    # Horizontal spacing = 1.5s = 1.5
    # Vertical spacing = sqrt(3)s = 1.732
    # Offset for odd columns: sqrt(3)/2 s = 0.866
    
    # Let's try a more compact initial guess: a ring around a center?
    # 1 center, 6 ring, 5 more? 
    # Actually, a 3x4 grid is easy to define and likely valid.
    # Let's define positions for a 3x4 grid centered at (0,0).
    # Rows: 3, Cols: 4
    # x positions: -1.5, 0, 1.5, 3.0 -> centered: -2.25, -0.75, 0.75, 2.25
    # y positions: -sqrt(3), 0, sqrt(3) -> centered: -1.732, 0, 1.732
    # Wait, 3 rows * 4 cols = 12.
    
    centers = []
    angles = []
    
    # Grid 3x4
    # x: 4 cols, spacing 1.5
    xs = np.array([-2.25, -0.75, 0.75, 2.25])
    # y: 3 rows, spacing sqrt(3)
    ys = np.array([-math.sqrt(3), 0.0, math.sqrt(3)])
    
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            # Stagger rows for hexagonal packing if needed, but grid is fine for start
            # Standard grid:
            centers.append([x, y])
            angles.append(0.0) # Flat top
            
    # Flatten to array
    x0 = []
    for i in range(N):
        x0.append(centers[i][0])
        x0.append(centers[i][1])
        x0.append(angles[i])
    x0 = np.array(x0)
    
    # 2. Objective Function
    # Variables: x0[0::3] are cx, x0[1::3] are cy, x0[2::3] are theta
    # We need to calculate the minimal side length S of an outer regular hexagon
    # (centered at origin, rotatable) that contains all inner hexagons.
    # 
    # A regular hexagon of side S and angle phi is defined by 6 half-planes.
    # The distance from origin to each side (apothem) is A = S * sqrt(3) / 2.
    # The normal vectors to the sides are at angles phi + k*pi/3 for k=0..5.
    # For the inner hexagons to be inside, for every inner vertex v of every inner hexagon:
    # v . n_k <= A for all k=0..5.
    # So A >= max_k (max_i (v_ij . n_k)).
    # Since A is proportional to S, minimizing A minimizes S.
    # S = 2 * A / sqrt(3).
    
    def objective(params):
        # Unpack
        cxs = params[0::3]
        cys = params[1::3]
        thetas = params[2::3]
        
        # Calculate vertices for all inner hexagons
        # Inner hexagon: side 1, center (cx, cy), angle theta
        # Vertices are at distance 1 from center.
        # Vertex angles: theta + k*pi/3
        # We can compute the support function h(n) for each inner hexagon.
        # For a regular polygon, the support function in direction n is:
        # h(n) = |n . c| + h_shape(n - n . c / |n|) ? No.
        # h(n) = max (v . n) for v in polygon.
        # v = c + R * u_k.
        # max (c . n + R * u_k . n) = c . n + max (R * u_k . n).
        # For unit regular hexagon, max (u_k . n) is the support function of unit hexagon.
        # Let n be unit vector. The unit hexagon has apothem sqrt(3)/2.
        # The support function of a regular hexagon with apothem a is:
        # h(n) = a / cos(delta) where delta is angle between n and nearest normal.
        # Or simply, we can compute the max dot product with the 6 vertices.
        # Since N=12 and 6 vertices each, 72 dot products. Fast enough.
        
        # Precompute unit vertex directions for a hexagon at angle 0
        # But angles vary.
        
        max_proj = 0.0
        
        # We need to find the outer hexagon orientation phi that minimizes A.
        # A(phi) = max_{inner hex i} max_{vertex v of i} (v . n_k(phi))
        # where n_k(phi) are the 6 normals of the outer hexagon.
        # n_k(phi) = (cos(phi + k*pi/3), sin(phi + k*pi/3))
        
        # To be efficient, we can sample phi? No, phi is continuous.
        # However, the max of linear functions (in terms of cos/sin) is convex.
        # We can optimize over phi as well? 
        # The problem asks to return outer_center, side, angle.
        # We can fix outer center at (0,0) by symmetry/translation invariance?
        # Yes, we can translate all inner hexagons so their "center of mass" or bounding box center is at origin, or just optimize outer center too.
        # Let's optimize outer center (ox, oy) and angle (phi) as well to be safe, 
        # but usually centering them helps.
        # Actually, the objective function S depends on phi. 
        # If we don't optimize phi, we might miss the optimal rotation.
        # Let's add phi to the optimization variables? 
        # If we add phi, the function becomes non-smooth (max of sinusoids). 
        # L-BFGS-B might struggle with the kinks.
        # Alternative: For a fixed set of inner hexagons, the optimal outer hexagon 
        # orientation can be found by checking the critical angles where a vertex 
        # of an inner hexagon aligns with a normal of the outer hexagon.
        # But that's complex.
        
        # Simpler approach: Include phi in the optimization. 
        # The objective is smooth enough for L-BFGS-B if we use a penalty or just the max.
        # Actually, max of smooth functions is smooth almost everywhere.
        
        # Let's add ox, oy, phi to the variables.
        # Total variables: 3*N + 3 = 39.
        
        # Wait, I defined x0 with only inner vars. I should update x0 to include outer vars.
        # Let's restart the variable definition in my head.
        # I will modify the objective to take (inner_vars, ox, oy, phi).
        
        # For this specific objective function implementation, I will assume 
        # the outer hexagon is centered at (0,0) and we only optimize its angle phi?
        # No, the centers of inner hexagons can move. 
        # If we optimize inner centers, we should also optimize outer center to keep them centered.
        # Let's just fix outer center at (0,0) and optimize inner positions and outer angle.
        # This is a common simplification.
        
        # Recalculating max_proj for a given phi
        # n_k = [cos(phi + k*pi/3), sin(phi + k*pi/3)] for k in 0..5
        
        # To avoid non-differentiability issues with max, we can use a soft-max 
        # or just rely on L-BFGS-B handling the kinks (it often does okay for packing).
        # Or, we can observe that the optimal phi will be such that at least 2 vertices 
        # from different hexagons touch the boundary.
        
        # Let's try including phi in the optimization.
        # I need to change the structure of the code to include phi in params.
        
        # Since I'm writing the code now, I will structure it properly.
        
        pass # Placeholder, see below for full implementation

    # Re-defining the strategy to include outer angle in optimization
    # Variables: 
    # 0..11: cx
    # 12..23: cy
    # 24..35: theta (inner)
    # 36: ox (outer center x)
    # 37: oy (outer center y)
    # 38: phi (outer angle)
    
    # Initial guess
    x0_full = np.zeros(3*N + 3)
    x0_full[0::3] = [c[0] for c in centers]
    x0_full[1::3] = [c[1] for c in centers]
    x0_full[2::3] = [0.0] * N
    x0_full[36] = 0.0
    x0_full[37] = 0.0
    x0_full[38] = 0.0
    
    def objective_full(params):
        cxs = params[0::3]
        cys = params[1::3]
        thetas = params[2::3]
        ox = params[36]
        oy = params[37]
        phi = params[38]
        
        # Outer normals
        # n_k = (cos(phi + k*pi/3), sin(phi + k*pi/3))
        
        max_apothem = 0.0
        
        for i in range(N):
            cx = cxs[i]
            cy = cys[i]
            theta = thetas[i]
            
            # Vertices of inner hexagon i
            # v_j = (cx + cos(theta + j*pi/3), cy + sin(theta + j*pi/3))
            
            # Relative to outer center (ox, oy)
            # dx = cx - ox
            # dy = cy - oy
            # rel_v_j = (dx + cos(...), dy + sin(...))
            
            # We need max dot product with outer normals.
            # dot = (dx + cos(t+j)) * cos(phi+k) + (dy + sin(t+j)) * sin(phi+k)
            # dot = dx*cos(phi+k) + dy*sin(phi+k) + cos(t+j - (phi+k))
            
            # Let alpha = t - phi.
            # term_base = dx*cos(phi+k) + dy*sin(phi+k)
            # term_var = cos(alpha + j*pi/3 - k*pi/3)
            
            # For a fixed inner hexagon, we want max over j (0..5) and k (0..5).
            # Note that max over k of (A_k + B_{j-k}) is what we need?
            # No, the outer hexagon constraint is:
            # For ALL k, (v . n_k) <= A.
            # So A must be >= max_k (v . n_k).
            # So for each inner hexagon, we compute M_i = max_{j,k} (v_j . n_k).
            # Then A = max_i M_i.
            
            # Optimization: 
            # v_j . n_k = (c - o) . n_k + u_j . n_k
            # u_j . n_k = cos(theta + j*pi/3 - (phi + k*pi/3))
            # Let delta = theta - phi.
            # u_j . n_k = cos(delta + (j-k)*pi/3)
            
            # We can precompute the max of u_j . n_k for all j,k?
            # No, it depends on delta.
            # But the set of values { cos(delta + m*pi/3) for m in Z } is periodic.
            # The maximum of cos(x) over the 6 specific angles is:
            # It's the support function of the unit hexagon in direction -delta?
            # Yes. h_hex(direction) = max_j (u_j . n_dir).
            # Here n_dir is effectively the outer normal rotated by -phi?
            
            # Let's just compute it directly. N=12, 36 pairs per hex. 432 ops. Negligible.
            
            for j in range(6):
                vj_x = cx + math.cos(theta + j * math.pi / 3)
                vj_y = cy + math.sin(theta + j * math.pi / 3)
                
                # Project onto 6 outer normals
                for k in range(6):
                    nk_x = math.cos(phi + k * math.pi / 3)
                    nk_y = math.sin(phi + k * math.pi / 3)
                    
                    proj = (vj_x - ox) * nk_x + (vj_y - oy) * nk_y
                    if proj > max_apothem:
                        max_apothem = proj
                        
        # Side length S = 2 * A / sqrt(3)
        return 2.0 * max_apothem / math.sqrt(3)

    # Run optimization
    try:
        res = minimize(objective_full, x0_full, method='L-BFGS-B', 
                       options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if time.time() > deadline:
             return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

        best_params = res.x
        cxs = best_params[0::3]
        cys = best_params[1::3]
        thetas = best_params[2::3]
        ox = best_params[36]
        oy = best_params[37]
        phi = best_params[38]
        
        # Construct output
        inner = []
        for i in range(N):
            inner.append([cxs[i], cys[i], math.degrees(thetas[i])])
            
        outer_center = (ox, oy)
        # Calculate side length from the optimized params to ensure consistency
        # The objective returned S. Let's recalculate to be sure.
        # Or just use res.fun
        side = res.fun
        
        # Ensure side is positive and valid
        if side <= 0 or math.isnan(side):
             return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

        return inner, outer_center, side, math.degrees(phi)
        
    except Exception as e:
        # Fallback to seed
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

# EVOLVE-BLOCK-END
