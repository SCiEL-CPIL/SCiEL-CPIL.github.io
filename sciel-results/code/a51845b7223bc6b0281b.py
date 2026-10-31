import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0  # Safety margin
    
    def hex_vertices(cx, cy, S, angle_deg):
        """Return 6 vertices of regular hexagon (pointy-top by default)."""
        a = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = a + math.radians(60 * i)
            verts.append((cx + S * math.cos(theta), cy + S * math.sin(theta)))
        return verts

    def get_axes(cx, cy, angle_deg):
        """Return the 6 unit normal vectors of the hexagon edges."""
        a = math.radians(angle_deg)
        axes = []
        for i in range(6):
            # Normals are at 30, 90, 150, 210, 270, 330 degrees relative to rotation
            theta = a + math.radians(30 + 60 * i)
            axes.append((math.cos(theta), math.sin(theta)))
        return axes

    def project_points(points, axis):
        """Project points onto axis, return (min_proj, max_proj)."""
        proj = [p[0] * axis[0] + p[1] * axis[1] for p in points]
        return min(proj), max(proj)

    def hexagon_overlap_depth(verts_a, axes_a, verts_b, axes_b):
        """
        Calculate penetration depth between two hexagons.
        Returns 0 if disjoint, otherwise the minimum distance to separate them.
        """
        all_axes = axes_a + axes_b
        min_overlap = float('inf')
        for axis in all_axes:
            min_a, max_a = project_points(verts_a, axis)
            min_b, max_b = project_points(verts_b, axis)
            
            # Check for separation
            if max_a <= min_b or max_b <= min_a:
                return 0.0
            
            # Calculate overlap length on this axis
            overlap = min(max_a, max_b) - max(min_a, min_b)
            if overlap < min_overlap:
                min_overlap = overlap
                
        return min_overlap

    def compute_cost(params):
        """
        params: [x1, y1, a1, x2, y2, a2, ..., x11, y11, a11]
        Returns: S_required + Penalty
        """
        # Unpack
        centers = []
        angles = []
        for i in range(N):
            x = params[i * 3]
            y = params[i * 3 + 1]
            a = params[i * 3 + 2]
            centers.append((x, y))
            angles.append(a)
            
        # Precompute vertices and axes for all inner hexagons
        inner_verts = []
        inner_axes = []
        for i in range(N):
            v = hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            inner_verts.append(v)
            inner_axes.append(get_axes(centers[i][0], centers[i][1], angles[i]))
            
        # Calculate S_required for outer hexagon (centered at 0,0, rotation 0)
        # Outer hexagon normals (for S=1, rotation 0)
        outer_axes = [
            (math.cos(math.radians(30)), math.sin(math.radians(30))),
            (math.cos(math.radians(90)), math.sin(math.radians(90))),
            (math.cos(math.radians(150)), math.sin(math.radians(150))),
            (math.cos(math.radians(210)), math.sin(math.radians(210))),
            (math.cos(math.radians(270)), math.sin(math.radians(270))),
            (math.cos(math.radians(330)), math.sin(math.radians(330)))
        ]
        
        max_proj = 0.0
        for axis in outer_axes:
            for v in inner_verts:
                min_p, max_p = project_points(v, axis)
                if max_p > max_proj:
                    max_proj = max_p
                # Also need to check min_p because outer hex is centered at 0,0
                # The constraint is that all points must be within [-Apothem, Apothem]
                # So max(|min_p|, |max_p|) <= Apothem
                # Apothem = S * sqrt(3)/2
                # So S >= max(|min_p|, |max_p|) / (sqrt(3)/2)
                abs_max = max(abs(min_p), abs(max_p))
                if abs_max > max_proj:
                    max_proj = abs_max
                    
        S_required = max_proj / (math.sqrt(3) / 2)
        
        # Calculate Overlap Penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                depth = hexagon_overlap_depth(inner_verts[i], inner_axes[i], inner_verts[j], inner_axes[j])
                if depth > 1e-6:
                    penalty += depth * depth
                    # Early exit if penalty is huge
                    if penalty > 100.0:
                        return S_required + penalty
        
        return S_required + 10.0 * penalty

    def run_optimizer(initial_params):
        try:
            # Use Nelder-Mead as it doesn't require gradients and handles non-smooth max functions well
            res = minimize(compute_cost, initial_params, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8, 'disp': False})
            return res.x, res.fun
        except Exception as e:
            return initial_params, compute_cost(initial_params)

    best_params = None
    best_score = float('inf')
    
    # Generate initial configurations
    initial_configs = []
    
    # 1. Hexagonal Lattice (1 center, 6 ring 1, 4 ring 2)
    sp = math.sqrt(3)
    config1 = []
    config1.append([0.0, 0.0, 0.0])
    for i in range(6):
        theta = math.radians(60 * i)
        config1.append([sp * math.cos(theta), sp * math.sin(theta), 0.0])
    # 4 more in ring 2. Place them at angles 30, 150, 210, 330? 
    # Distance 2*sp
    for i in [1, 3, 4, 6]: # Indices 1, 3, 4, 6 correspond to angles 60, 180, 240, 360(0)? 
        # Let's just place them at specific angles to fill gaps
        pass
    # Better: Place 4 hexagons in the "corners" of the hexagon formed by ring 1.
    # Angles: 30, 150, 210, 330 degrees?
    # Let's try angles 0, 90, 180, 270 for the 4 extra?
    for i in [0, 1, 2, 3]:
        theta = math.radians(90 * i)
        config1.append([2 * sp * math.cos(theta), 2 * sp * math.sin(theta), 0.0])
    while len(config1) < N:
        config1.append([0.0, 0.0, 0.0]) # Placeholder, will be optimized
    initial_configs.append([item for sublist in config1[:N] for item in sublist])

    # 2. Triangular-ish
    config2 = []
    for i in range(N):
        row = i // 3
        col = i % 3
        x = col * 1.8 - 1.8
        y = row * 1.6
        config2.append([x, y, 0.0])
    initial_configs.append([item for sublist in config2[:N] for item in sublist])

    # 3. Random
    rng = random.Random(42)
    for _ in range(10):
        config3 = []
        for i in range(N):
            x = rng.uniform(-2, 2)
            y = rng.uniform(-2, 2)
            a = rng.uniform(0, 180)
            config3.append([x, y, a])
        initial_configs.append([item for sublist in config3 for item in sublist])

    # Optimize each config
    for i, init in enumerate(initial_configs):
        if time.time() > deadline:
            break
        
        # Normalize angles to 0-180 range to help optimizer?
        # Nelder-Mead can handle periodicity poorly, so we should probably fix angles or use a mapping.
        # However, for hexagons, 180 degrees is same as 0. 60 degrees is different.
        # Let's just let it optimize.
        
        params, score = run_optimizer(init)
        
        # Check validity of the result
        # Re-evaluate cost to ensure no exceptions
        try:
            final_cost = compute_cost(params)
            # Extract S_required from final_cost? 
            # The cost is S + 10*penalty. If penalty is 0, cost is S.
            # We need to check if penalty is actually 0.
            # Let's re-calculate penalty and S separately for the best params.
            
            # Re-calc S and Penalty
            centers = []
            angles = []
            for k in range(N):
                centers.append((params[k*3], params[k*3+1]))
                angles.append(params[k*3+2])
            
            inner_verts = []
            inner_axes = []
            for k in range(N):
                v = hex_vertices(centers[k][0], centers[k][1], 1.0, angles[k])
                inner_verts.append(v)
                inner_axes.append(get_axes(centers[k][0], centers[k][1], angles[k]))
            
            # Check overlaps
            valid = True
            for k in range(N):
                for m in range(k+1, N):
                    d = hexagon_overlap_depth(inner_verts[k], inner_axes[k], inner_verts[m], inner_axes[m])
                    if d > 1e-5:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Calculate S
                outer_axes = [
                    (math.cos(math.radians(30)), math.sin(math.radians(30))),
                    (math.cos(math.radians(90)), math.sin(math.radians(90))),
                    (math.cos(math.radians(150)), math.sin(math.radians(150))),
                    (math.cos(math.radians(210)), math.sin(math.radians(210))),
                    (math.cos(math.radians(270)), math.sin(math.radians(270))),
                    (math.cos(math.radians(330)), math.sin(math.radians(330)))
                ]
                max_proj = 0.0
                for axis in outer_axes:
                    for v in inner_verts:
                        min_p, max_p = project_points(v, axis)
                        abs_max = max(abs(min_p), abs(max_p))
                        if abs_max > max_proj:
                            max_proj = abs_max
                S = max_proj / (math.sqrt(3) / 2)
                
                if S < best_score:
                    best_score = S
                    best_params = params
        except Exception:
            pass

    # Construct return values from best_params
    if best_params is None:
        # Fallback
        return [[0.0, 0.0, 0.0]] * N, (0.0, 0.0), 10.0, 0.0

    inner = []
    for i in range(N):
        x = best_params[i * 3]
        y = best_params[i * 3 + 1]
        a = best_params[i * 3 + 2]
        inner.append([x, y, a])
    
    return inner, (0.0, 0.0), best_score, 0.0

# EVOLVE-BLOCK-END
