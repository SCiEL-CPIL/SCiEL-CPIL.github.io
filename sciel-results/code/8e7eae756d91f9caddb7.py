"""Rewritten program: rigid honeycomb cluster with breathing scale + global rotation."""
import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s limit

    def hex_vertices(cx, cy, angle_deg, scale=1.0):
        """Get vertices of a regular hexagon with side length=scale, centered at (cx,cy), rotated by angle_deg."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + scale * math.cos(a)
            y = cy + scale * math.sin(a)
            verts.append((x, y))
        return np.array(verts)

    def hexagons_overlap(v1, v2):
        """SAT check for two convex hexagons (6 vertices each)."""
        axes = []
        for i in range(6):
            edge = v1[i] - v1[(i + 1) % 6]
            normal = np.array([-edge[1], edge[0]])
            axes.append(normal)
            edge2 = v2[i] - v2[(i + 1) % 6]
            normal2 = np.array([-edge2[1], edge2[0]])
            axes.append(normal2)
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            if proj1.max() < proj2.min() or proj2.max() < proj1.min():
                return False
        return True

    def check_all_overlaps(verts_list):
        for i in range(len(verts_list)):
            for j in range(i + 1, len(vars_list) if False else len(verts_list)):
                if hexagons_overlap(verts_list[i], verts_list[j]):
                    return True
        return False

    def compute_min_side(all_verts, outer_angle_deg):
        """
        Compute minimum side length of outer hexagon (centered at origin, rotated by outer_angle_deg)
        to contain all points in all_verts.
        The outer hexagon with side s has apothem = s*sqrt(3)/2.
        For a point at distance d from center in direction of normal k, we need d <= apothem.
        So s >= 2*d/sqrt(3) for each normal direction.
        """
        if len(all_verts) == 0:
            return 0.0
        theta = math.radians(outer_angle_deg)
        max_proj = 0.0
        for k in range(6):
            a = theta + k * math.pi / 3
            normal = np.array([math.cos(a), math.sin(a)])
            proj = np.dot(all_verts, normal)
            max_proj = max(max_proj, proj.max())
        # For a regular hexagon centered at origin, the constraint is:
        # For each of the 6 face normals, max projection <= apothem = s*sqrt(3)/2
        # So s >= 2 * max_proj / sqrt(3)
        return 2.0 * max_proj / math.sqrt(3.0)

    # Define honeycomb cluster: center + 6 ring1 + 5 selected from ring2
    # Using axial coordinates (q, r) -> cartesian for unit hexagons
    # Axial to cartesian: x = sqrt(3)*(q + r/2), y = 1.5*r
    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    # Build the 12-hexagon cluster in axial coordinates
    # Center: (0,0)
    # Ring 1: 6 neighbors
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    # Ring 2: all hexagons at distance 2 from center
    ring2_all = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            # Distance in hex grid: (|q| + |r| + |q+r|) / 2
            dist = (abs(q) + abs(r) + abs(q + r)) / 2
            if dist == 2:
                ring2_all.append((q, r))
    # ring2_all should have 12 elements; pick 5 for symmetry
    # Sort by angle for consistent selection
    def hex_angle(q, r):
        x, y = axial_to_cartesian(q, r)
        return math.atan2(y, x)
    ring2_all.sort(key=lambda p: hex_angle(p[0], p[1]))
    # Pick every other one starting from index 0 to get 5 (or 6, we need 5)
    # Actually we need 5 more to make 12 total (1 + 6 + 5 = 12)
    # For symmetry, pick 5 that are spread out
    # With 12 in ring2, pick indices 0, 2, 4, 6, 8 (every other, 5 of them)
    ring2_selected = [ring2_all[i] for i in range(0, 12, 2)][:5]
    # If we only got 5, good. If not, adjust.
    while len(ring2_selected) < 5:
        ring2_selected.append(ring2_all[len(ring2_selected)])

    cluster_axial = [(0, 0)] + ring1 + ring2_selected
    # Convert to cartesian (relative to center of mass)
    cluster_cart = []
    for q, r in cluster_axial:
        x, y = axial_to_cartesian(q, r)
        cluster_cart.append((x, y))
    # Center the cluster
    cx = sum(p[0] for p in cluster_cart) / len(cluster_cart)
    cy = sum(p[1] for p in cluster_cart) / len(cluster_cart)
    cluster_cart = [(x - cx, y - cy) for x, y in cluster_cart]
    cluster_cart = np.array(cluster_cart)

    def build_config(scale, rotation_deg, tx, ty, outer_angle_deg):
        """
        Build configuration:
        - scale: breathing factor for inter-center distances
        - rotation_deg: global rotation of the cluster
        - tx, ty: translation
        - outer_angle_deg: rotation of outer hexagon
        Returns: (positions, angles, outer_angle_deg)
        All inner hexagons have angle=0 (axis-aligned) for simplicity, 
        or we can rotate them too. Let's rotate them with the cluster.
        """
        theta = math.radians(rotation_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        positions = []
        angles = [rotation_deg] * N  # rotate each hexagon with the cluster
        for i in range(N):
            x, y = cluster_cart[i] * scale
            # Rotate
            rx = x * cos_t - y * sin_t
            ry = x * sin_t + y * cos_t
            positions.append((rx + tx, ry + ty))
        return positions, angles, outer_angle_deg

    def evaluate_config(scale, rotation_deg, tx, ty, outer_angle_deg):
        """
        Evaluate a configuration. Returns (side_length, is_valid).
        """
        positions, angles, oa = build_config(scale, rotation_deg, tx, ty, outer_angle_deg)
        verts_list = [hex_vertices(positions[i][0], positions[i][1], angles[i]) for i in range(N)]
        
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                if hexagons_overlap(verts_list[i], verts_list[j]):
                    return float('inf'), False
        
        # Compute outer side
        all_verts = np.vstack(verts_list)
        side = compute_min_side(all_verts, outer_angle_deg)
        return side, True

    # Optimization variables: [scale, rotation_deg, tx, ty, outer_angle_deg]
    # We'll use a penalty-based objective for scipy optimization.
    
    def objective(x):
        if time.time() > deadline - 2:
            return 1e10
        scale, rot_deg, tx, ty, oa_deg = x
        scale = max(0.5, min(2.0, scale))  # clamp scale
        rot_deg = rot_deg % 360
        oa_deg = oa_deg % 360
        
        positions, angles, oa = build_config(scale, rot_deg, tx, ty, oa_deg)
        verts_list = [hex_vertices(positions[i][0], positions[i][1], angles[i]) for i in range(N)]
        
        # Overlap penalty
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                if hexagons_overlap(verts_list[i], verts_list[j]):
                    d = np.array(positions[j]) - np.array(positions[i])
                    dist = np.linalg.norm(d)
                    overlap_pen += max(0.0, 2.0 - dist) ** 2
        
        if overlap_pen > 1e-4:
            return 100.0 + overlap_pen
        
        all_verts = np.vstack(verts_list)
        side = compute_min_side(all_verts, oa_deg)
        return side

    # Initial guesses
    # Natural scale for honeycomb: centers are ~2 apart for touching hexagons
    # In our axial->cartesian, adjacent centers are sqrt(3) apart in x and 1.5 in y
    # Distance between adjacent centers = 2 (for unit hexagons, center-to-center = 2*apothem = 2*sqrt(3)/2*... 
    # Actually for unit side hexagons, apothem = sqrt(3)/2, so center-to-center for touching = 2*apothem = sqrt(3)
    # Wait: two unit hexagons touching share an edge. The distance between centers = 2 * apothem = sqrt(3).
    # In our axial_to_cartesian, (0,0) to (1,0): x=sqrt(3), y=0, so distance = sqrt(3). Correct.
    # So at scale=1, they should be just touching (no overlap, no gap).
    # We want to try scales slightly less than 1 to see if we can compress.
    
    best_score = float('inf')
    best_params = None

    # Grid search over rotation and outer angle, then optimize scale, tx, ty
    rot_candidates = list(range(0, 180, 15))
    oa_candidates = list(range(0, 180, 15))
    scale_candidates = [0.95, 0.98, 1.0, 1.02, 1.05]

    for rot in rot_candidates:
        if time.time() > deadline - 10:
            break
        for oa in oa_candidates:
            if time.time() > deadline - 5:
                break
            for sc in scale_candidates:
                if time.time() > deadline - 3:
                    break
                
                # Try a few initial translations
                for tx in [-0.5, 0.0, 0.5]:
                    for ty in [-0.5, 0.0, 0.5]:
                        score, valid = evaluate_config(sc, float(rot), tx, ty, float(oa))
                        if valid and score < best_score:
                            best_score = score
                            best_params = (sc, float(rot), tx, ty, float(oa))

    # Now do local optimization around the best found
    if best_params is not None:
        x0 = np.array(best_params)
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass

        # Try a few more local optimizations with perturbed starts
        np.random.seed(42)
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            x_pert = x0 + np.random.uniform(-0.1, 0.1, size=5)
            try:
                result = minimize(objective, x_pert, method='Nelder-Mead',
                                  options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
                if result.fun < best_score:
                    best_score = result.fun
                    best_params = result.x
            except Exception:
                pass

    # If still inf, try a broader search
    if best_score == float('inf'):
        # Try differential evolution with wider bounds
        bounds = [(0.8, 1.2), (0, 180), (-2, 2), (-2, 2), (0, 180)]
        try:
            result = differential_evolution(objective, bounds, maxiter=50, popsize=15,
                                            tol=1e-4, seed=42, polish=False,
                                            update_interval=1)
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass

    # Fallback if still invalid
    if best_score == float('inf') or best_params is None:
        # Use scale=1.0, no rotation, centered
        best_params = (1.0, 0.0, 0.0, 0.0, 0.0)
        best_score, valid = evaluate_config(1.0, 0.0, 0.0, 0.0, 0.0)
        if not valid:
            # Spread out more
            best_params = (1.1, 0.0, 0.0, 0.0, 0.0)
            best_score, valid = evaluate_config(1.1, 0.0, 0.0, 0.0, 0.0)

    # Build final configuration
    scale, rot_deg, tx, ty, oa_deg = best_params
    positions, angles, outer_angle = build_config(scale, rot_deg, tx, ty, oa_deg)

    # Center the entire configuration so outer hexagon is at origin
    all_x = [p[0] for p in positions]
    all_y = [p[1] for p in positions]
    cx_final = sum(all_x) / N
    cy_final = sum(all_y) / N
    positions = [(x - cx_final, y - cy_final) for x, y in positions]

    # Recompute side length after centering
    verts_list = [hex_vertices(positions[i][0], positions[i][1], angles[i]) for i in range(N)]
    all_verts = np.vstack(verts_list)
    final_side = compute_min_side(all_verts, outer_angle)

    # Verify no overlaps
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            if hexagons_overlap(verts_list[i], verts_list[j]):
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback: spread out
        positions = []
        for i in range(N):
            r = i // 4
            c = i % 4
            x = c * 2.5 + (r % 2) * 1.25
            y = r * 2.2
            positions.append((x, y))
        cx_f = sum(p[0] for p in positions) / N
        cy_f = sum(p[1] for p in positions) / N
        positions = [(x - cx_f, y - cy_f) for x, y in positions]
        angles = [0.0] * N
        outer_angle = 0.0
        verts_list = [hex_vertices(positions[i][0], positions[i][1], angles[i]) for i in range(N)]
        all_verts = np.vstack(verts_list)
        final_side = compute_min_side(all_verts, outer_angle)

    inner = [[positions[i][0], positions[i][1], angles[i]] for i in range(N)]
    return inner, (0.0, 0.0), final_side, outer_angle
# EVOLVE-BLOCK-END
