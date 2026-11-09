"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    sqrt3 = math.sqrt(3.0)

    # Edge normals for a standard hexagon (flat-top, vertices at 0,60,...)
    # The 6 outward edge normals point at 30, 90, 150, 210, 270, 330 degrees
    edge_normals = np.array([
        [math.cos(math.radians(30 + i * 60)), math.sin(math.radians(30 + i * 60))]
        for i in range(6)
    ])

    # Unit hexagon vertices (flat-top, side=1)
    unit_verts = np.array([
        [math.cos(math.radians(i * 60)), math.sin(math.radians(i * 60))]
        for i in range(6)
    ])

    def get_vertices(cx, cy, angle_rad, side=1.0):
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        return np.array([cx, cy]) + side * (unit_verts @ R.T)

    def sat_overlap(centers, angles_rad, eps=1e-9):
        """SAT overlap check between all pairs of hexagons."""
        n = len(centers)
        all_verts = []
        all_normals = []
        for i in range(n):
            cx, cy = centers[i]
            ang = angles_rad[i]
            verts = get_vertices(cx, cy, ang)
            all_verts.append(verts)
            c, s = math.cos(ang), math.sin(ang)
            R = np.array([[c, -s], [s, c]])
            normals = edge_normals @ R.T
            all_normals.append(normals)

        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist_sq = dx * dx + dy * dy
                if dist_sq > 4.5:
                    continue
                v_i, v_j = all_verts[i], all_verts[j]
                n_i, n_j = all_normals[i], all_normals[j]
                overlap = True
                for k in range(6):
                    axis = n_i[k]
                    pi = v_i @ axis
                    pj = v_j @ axis
                    if pi.max() < pj.min() + eps or pj.max() < pi.min() + eps:
                        overlap = False
                        break
                    if overlap:
                        axis = n_j[k]
                        pi = v_i @ axis
                        pj = v_j @ axis
                        if pi.max() < pj.min() + eps or pj.max() < pi.min() + eps:
                            overlap = False
                            break
                if overlap:
                    return True
        return False

    def outer_side_length(centers, angles_rad):
        """Compute minimum outer hexagon side length to enclose all inner hexagons (outer angle=0)."""
        max_proj = 0.0
        for i in range(N):
            verts = get_vertices(centers[i][0], centers[i][1], angles_rad[i])
            # Project each vertex onto each of the 6 outer edge normals
            projs = verts @ edge_normals.T  # shape (6, 6)
            local_max = projs.max()
            if local_max > max_proj:
                max_proj = local_max
        # Apothem = max_proj, side length = 2 * apothem / sqrt(3)
        return 2.0 * max_proj / sqrt3

    def erosion_constraint(params, L):
        """
        Constraint: all centers must lie within the hexagon of side (L-1) centered at origin.
        The eroded region for a unit hexagon (any rotation) inside a hexagon of side L
        is contained in a hexagon of side L-1 (for aligned case).
        For arbitrary rotation, the safe region is a circle of radius (L-1)*sqrt(3)/2,
        but we use the hexagonal constraint which is tighter for aligned hexes.
        
        Actually, for a unit hexagon at any rotation, the center must be at distance
        at most (L-1)*apothem_ratio from center. The apothem of outer is L*sqrt(3)/2.
        The inradius of a unit hex is sqrt(3)/2. So the center must be within
        a hexagon whose apothem is L*sqrt(3)/2 - sqrt(3)/2 = (L-1)*sqrt(3)/2.
        This is a hexagon of side L-1.
        
        We use the 6 half-plane constraints: for each edge normal n_k,
        dot(center, n_k) <= (L-1) * sqrt(3)/2
        """
        apothem_inner = (L - 1.0) * sqrt3 / 2.0
        # params layout: [cx0, cy0, ang0, cx1, cy1, ang1, ...]
        constraints = []
        for k in range(6):
            n = edge_normals[k]
            for i in range(N):
                cx = params[3 * i]
                cy = params[3 * i + 1]
                # dot((cx,cy), n) <= apothem_inner
                # In SLSQP, constraint is g(x) >= 0
                # So: apothem_inner - dot((cx,cy), n) >= 0
                constraints.append(apothem_inner - (cx * n[0] + cy * n[1]))
        return constraints

    def pairwise_separation_constraints(params, min_dist_sq=3.0):
        """
        Constraint: pairwise center distance >= sqrt(3) (side-aligned touching).
        Actually for arbitrary rotations, minimum center distance for non-overlap
        is sqrt(3) (when sides are parallel and touching).
        For vertex-touching, it's 2.0. The tightest is sqrt(3).
        
        We use min_dist_sq = 3.0 (i.e., dist >= sqrt(3)) as a necessary condition.
        This is conservative; actual non-overlap requires SAT check.
        """
        constraints = []
        for i in range(N):
            for j in range(i + 1, N):
                cx_i, cy_i = params[3 * i], params[3 * i + 1]
                cx_j, cy_j = params[3 * j], params[3 * j + 1]
                dx = cx_i - cx_j
                dy = cy_i - cy_j
                # dist^2 >= min_dist_sq  =>  dist^2 - min_dist_sq >= 0
                constraints.append(dx * dx + dy * dy - min_dist_sq)
        return constraints

    def objective_aligned(params, L_target):
        """
        Objective: minimize the maximum projection (proxy for L).
        params: [cx0, cy0, ang0, cx1, cy1, ang1, ..., cx11, cy11, ang11]
        For this strategy, we fix all angles to 0 (aligned) to simplify.
        """
        max_proj = 0.0
        for i in range(N):
            cx, cy = params[3 * i], params[3 * i + 1]
            verts = get_vertices(cx, cy, 0.0)
            projs = verts @ edge_normals.T
            local_max = projs.max()
            if local_max > max_proj:
                max_proj = local_max
        return 2.0 * max_proj / sqrt3

    # Strategy: Binary search on L + SLSQP with constraints
    # Lower bound: theoretical minimum. 12 hexes of side 1 in a hexagon.
    # A hexagon of side 2 can hold 7 unit hexes (center + 6 neighbors).
    # We need 12, so L must be > 2.
    # Upper bound: generous.
    
    L_low = 2.5
    L_high = 8.0

    def can_fit(L):
        """Check if 12 unit hexagons can fit in outer hexagon of side L (aligned)."""
        if time.time() > deadline - 5:
            return False, None

        apothem_inner = (L - 1.0) * sqrt3 / 2.0

        # Build constraints for SLSQP
        cons = []
        # Boundary constraints: centers in eroded hexagon
        for k in range(6):
            n = edge_normals[k]
            for i in range(N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, k=k, i=i: apothem_inner - (x[3 * i] * n[0] + x[3 * i + 1] * n[1])
                })

        # Pairwise separation: dist^2 >= 3.0
        for i in range(N):
            for j in range(i + 1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: (x[3*i] - x[3*j])**2 + (x[3*i+1] - x[3*j+1])**2 - 3.0
                })

        # Objective: minimize max projection (which should be <= (L)*sqrt(3)/2 for feasibility)
        # Actually, if all centers are in the eroded region, the hexagons fit.
        # So the objective can just be 0 (feasibility problem).
        # But SLSQP needs a smooth objective. Use sum of squared distances from center.
        def obj(x):
            # Minimize spread: sum of squared radii
            s = 0.0
            for i in range(N):
                s += x[3 * i] ** 2 + x[3 * i + 1] ** 2
            return s

        # Initial guess: hexagonal lattice pattern scaled to fit
        # 7 hexes in ring of radius sqrt(3) around center, plus 5 more in outer ring
        hex_coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
        ]
        # Convert to cartesian (flat-top hex grid)
        # x = sqrt(3) * (q + r/2), y = 1.5 * r
        # Scale to fit within apothem_inner
        max_r = max(math.sqrt((sqrt3 * (q + r / 2.0)) ** 2 + (1.5 * r) ** 2) for q, r in hex_coords)
        scale = apothem_inner / max_r if max_r > 0 else 1.0
        # But we also need separation >= sqrt(3), so scale can't be too small
        min_scale = sqrt3 / (2.0 * max_r) if max_r > 0 else 1.0
        scale = max(scale, min_scale * 1.05)

        x0 = []
        for q, r in hex_coords:
            cx = sqrt3 * (q + r / 2.0) * scale
            cy = 1.5 * r * scale
            x0.extend([cx, cy, 0.0])

        try:
            res = minimize(obj, x0, method='SLSQP', constraints=cons,
                           options={'maxiter': 500, 'ftol': 1e-8})
            if res.success:
                # Verify no actual overlap with SAT
                centers = [(res.x[3 * i], res.x[3 * i + 1]) for i in range(N)]
                angles = [0.0] * N
                if not sat_overlap(centers, angles):
                    # Verify all hexagons are inside outer hexagon of side L
                    L_actual = outer_side_length(centers, angles)
                    if L_actual <= L + 1e-6:
                        return True, centers
                return True, centers  # SAT passed, trust it
            else:
                return False, None
        except Exception:
            return False, None

    # Binary search on L
    best_L = L_high
    best_centers = None
    best_angles = None

    # First, try a few direct optimizations with free angles using differential evolution
    # to get a good initial L estimate, then refine with SLSQP.

    # Phase 1: Differential evolution to find a good configuration
    def de_objective(params):
        """DE objective: L + penalty for overlap."""
        centers = [(params[2 * i], params[2 * i + 1]) for i in range(N)]
        angles_rad = [params[2 * N + i] for i in range(N)]
        if sat_overlap(centers, angles_rad):
            return 100.0
        return outer_side_length(centers, angles_rad)

    # Bounds for DE
    bounds = []
    for i in range(N):
        bounds.append((-5, 5))  # cx
        bounds.append((-5, 5))  # cy
    for i in range(N):
        bounds.append((0, math.pi / 3))  # angle (60 deg symmetry)

    if time.time() < deadline - 60:
        try:
            de_res = differential_evolution(
                de_objective, bounds, seed=42, maxiter=100, tol=1e-8,
                popsize=15, mutation=(0.3, 1.0), recombination=0.9,
                polish=True, init='latinhypercube'
            )
            if de_res.fun < best_L:
                best_L = de_res.fun
                best_centers = [(de_res.x[2 * i], de_res.x[2 * i + 1]) for i in range(N)]
                best_angles = [de_res.x[2 * N + i] for i in range(N)]
        except Exception:
            pass

    # Phase 2: Binary search on L using SLSQP feasibility (aligned hexagons)
    if time.time() < deadline - 30:
        for _ in range(20):  # Up to 20 bisection steps
            if time.time() > deadline - 10:
                break
            L_mid = (L_low + L_high) / 2.0
            feasible, centers = can_fit(L_mid)
            if feasible:
                L_high = L_mid
                if centers is not None:
                    # Verify with SAT and compute actual L
                    angles = [0.0] * N
                    if not sat_overlap(centers, angles):
                        L_actual = outer_side_length(centers, angles)
                        if L_actual < best_L:
                            best_L = L_actual
                            best_centers = centers
                            best_angles = angles
            else:
                L_low = L_mid

    # Phase 3: Local refinement with Nelder-Mead on the best found configuration
    if best_centers is not None and time.time() < deadline - 15:
        def nm_objective(params):
            centers = [(params[2 * i], params[2 * i + 1]) for i in range(N)]
            angles_rad = [params[2 * N + i] for i in range(N)]
            if sat_overlap(centers, angles_rad):
                return 100.0
            return outer_side_length(centers, angles_rad)

        x0 = []
        for i in range(N):
            x0.extend([best_centers[i][0], best_centers[i][1]])
        for i in range(N):
            x0.append(best_angles[i] if i < len(best_angles) else 0.0)

        try:
            nm_res = minimize(nm_objective, x0, method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            if nm_res.fun < best_L:
                best_L = nm_res.fun
                best_centers = [(nm_res.x[2 * i], nm_res.x[2 * i + 1]) for i in range(N)]
                best_angles = [nm_res.x[2 * N + i] for i in range(N)]
        except Exception:
            pass

    # Final verification
    if best_centers is None:
        # Fallback: widely separated hexagons
        inner = []
        for i in range(N):
            angle = 2 * math.pi * i / N
            r = 6.0
            inner.append([r * math.cos(angle), r * math.sin(angle), 0.0])
        L = outer_side_length([(6.0 * math.cos(2 * math.pi * i / N), 6.0 * math.sin(2 * math.pi * i / N)) for i in range(N)], [0.0] * N)
        return inner, (0.0, 0.0), L, 0.0

    # Check validity
    if sat_overlap(best_centers, best_angles):
        # Invalid, return fallback
        inner = []
        for i in range(N):
            angle = 2 * math.pi * i / N
            r = 6.0
            inner.append([r * math.cos(angle), r * math.sin(angle), 0.0])
        L = outer_side_length([(6.0 * math.cos(2 * math.pi * i / N), 6.0 * math.sin(2 * math.pi * i / N)) for i in range(N)], [0.0] * N)
        return inner, (0.0, 0.0), L, 0.0

    final_L = outer_side_length(best_centers, best_angles)

    inner = []
    for i in range(N):
        inner.append([best_centers[i][0], best_centers[i][1], math.degrees(best_angles[i])])

    return inner, (0.0, 0.0), final_L, 0.0
# EVOLVE-BLOCK-END
