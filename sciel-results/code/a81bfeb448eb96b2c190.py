"""Optimized hexagon packing: minimize outer hexagon side for N unit hexagons.

Approach:
- Represent each inner unit hexagon by center (x,y) and rotation angle (deg).
- Objective = smallest side length S of a (rotated, translated) regular hexagon
  that contains all inner hexagons.
- Feasibility = all inner hexagons are pairwise disjoint (strict separation).
- Search = multi-start simulated annealing over (x,y,angle) per hexagon,
  with a penalty for overlaps and a penalty for exceeding a max side budget.
- Deadline guard: track wall-clock and return best feasible result found.
"""

import math
import time
import random
import numpy as np

N = 11
DEADLINE = 135.0  # safety margin below 150s
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def hex_vertices(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon (side length `side`) centered at (cx,cy),
    rotated by angle_deg degrees. Returns (6,2) array."""
    a = math.radians(angle_deg)
    # Regular hexagon vertices at angles 0,60,...,300 offset by a
    verts = np.zeros((6, 2))
    for i in range(6):
        theta = a + math.radians(60.0 * i)
        verts[i, 0] = cx + side * math.cos(theta)
        verts[i, 1] = cy + side * math.sin(theta)
    return verts

def polygon_overlap_area_sign(v1, v2):
    """Return a positive scalar if two convex polygons (6,2 arrays) overlap,
    else 0.0. Uses separating-axis theorem for convex polygons.
    Returns the minimum penetration depth (positive if overlapping)."""
    # SAT: check all edges of both polygons
    min_overlap = float('inf')
    # Edges of v1
    for i in range(6):
        j = (i + 1) % 6
        edge = v1[j] - v1[i]
        # Normal (perpendicular)
        nx, ny = -edge[1], edge[0]
        norm = math.sqrt(nx*nx + ny*ny)
        if norm < 1e-12:
            continue
        nx /= norm; ny /= norm
        # Project both polygons onto this normal
        d1_min = float('inf'); d1_max = -float('inf')
        for p in v1:
            d = p[0]*nx + p[1]*ny
            d1_min = min(d1_min, d); d1_max = max(d1_max, d)
        d2_min = float('inf'); d2_max = -float('inf')
        for p in v2:
            d = p[0]*nx + p[1]*ny
            d2_min = min(d2_min, d); d2_max = max(d2_max, d)
        # Separation?
        if d1_max <= d2_min or d2_max <= d1_min:
            return 0.0  # separating axis found
        overlap = min(d1_max, d2_max) - max(d1_min, d2_min)
        min_overlap = min(min_overlap, overlap)
    # Edges of v2
    for i in range(6):
        j = (i + 1) % 6
        edge = v2[j] - v2[i]
        nx, ny = -edge[1], edge[0]
        norm = math.sqrt(nx*nx + ny*ny)
        if norm < 1e-12:
            continue
        nx /= norm; ny /= norm
        d1_min = float('inf'); d1_max = -float('inf')
        for p in v1:
            d = p[0]*nx + p[1]*ny
            d1_min = min(d1_min, d); d1_max = max(d1_max, d)
        d2_min = float('inf'); d2_max = -float('inf')
        for p in v2:
            d = p[0]*nx + p[1]*ny
            d2_min = min(d2_min, d); d2_max = max(d2_max, d)
        if d1_max <= d2_min or d2_max <= d1_min:
            return 0.0
        overlap = min(d1_max, d2_max) - max(d1_min, d2_min)
        min_overlap = min(min_overlap, overlap)
    return min_overlap if min_overlap < float('inf') else 0.0

def max_overlap_depth(inner_states):
    """inner_states: list of (cx, cy, angle_deg). Returns max overlap depth
    among all pairs (0.0 if all disjoint)."""
    verts_list = [hex_vertices(cx, cy, 1.0, ang) for cx, cy, ang in inner_states]
    max_ov = 0.0
    for i in range(N):
        for j in range(i+1, N):
            ov = polygon_overlap_area_sign(verts_list[i], verts_list[j])
            if ov > max_ov:
                max_ov = ov
    return max_ov

def required_outer_side(inner_states, outer_angle_deg):
    """Given inner hexagon states and an outer hexagon rotation, compute the
    minimum side length S of a regular hexagon (centered at origin, rotated by
    outer_angle_deg) that contains all inner hexagons.
    
    For a regular hexagon with side S and rotation phi, the support function
    in direction theta is S * max over vertices of (v . u) where u is unit vector.
    Actually, for a regular hexagon centered at origin with 'pointy-top' or
    'flat-top' orientation, the distance from center to each edge is S * sqrt(3)/2
    (apothem). The hexagon contains a point p iff for all 6 edge normals n_k,
    p . n_k <= apothem = S * sqrt(3)/2.
    
    So S = max over all inner hexagon vertices v, and all 6 outer edge normals n_k:
        v . n_k / (sqrt(3)/2)
    
    But we also need to consider that the outer hexagon can be translated.
    To minimize S, we should center the outer hexagon at the centroid of the
    bounding region. A good approximation: center at the average of all inner
    hexagon centers, then compute required S.
    """
    # Compute all vertices of all inner hexagons
    all_verts = []
    for cx, cy, ang in inner_states:
        v = hex_vertices(cx, cy, 1.0, ang)
        all_verts.append(v)
    all_verts = np.vstack(all_verts)  # (6N, 2)
    
    # Outer hexagon edge normals for given rotation
    phi = math.radians(outer_angle_deg)
    # Edge normals at angles phi + 30 + 60*i for i=0..5 (for flat-top base)
    # Actually for a regular hexagon, the 6 outward normals are at angles
    # phi + 0, 60, 120, 180, 240, 300 (if phi is the angle of the first vertex)
    # Let's use: normals at phi + 30 + 60*i (apothem directions)
    # For a hexagon with vertices at angles phi, phi+60, ..., the edge normals
    # (perpendicular to edges, pointing outward) are at phi+30, phi+90, ...
    normals = np.zeros((6, 2))
    for i in range(6):
        theta = phi + math.radians(30 + 60 * i)
        normals[i, 0] = math.cos(theta)
        normals[i, 1] = math.sin(theta)
    
    # Project all vertices onto each normal
    # projections[k, m] = all_verts[:, k] . normals[m]
    projections = all_verts @ normals.T  # (6N, 6)
    
    # The outer hexagon centered at origin with side S has apothem = S*sqrt(3)/2
    # It contains all points iff for all normals n_k: max_v(v . n_k) <= apothem
    # So S = max_k (max_v(v . n_k)) / (sqrt(3)/2)
    # But we can translate the outer hexagon. Optimal translation centers the
    # bounding box in each normal direction.
    # For each normal k, the extent is [min_v(v.n_k), max_v(v.n_k)]
    # After translating by t, extent becomes [min - t.n_k, max - t.n_k]
    # We need both <= apothem and >= -apothem (for the opposite normal)
    # Actually normal k and normal k+3 are opposite.
    # For direction pair (k, k+3): we need
    #   max_v((v-t).n_k) <= apothem  =>  max_v(v.n_k) - t.n_k <= apothem
    #   max_v((v-t).n_{k+3}) <= apothem  =>  max_v(v.n_{k+3}) - t.n_{k+3} <= apothem
    # Since n_{k+3} = -n_k:
    #   max_v(v.n_k) - t.n_k <= apothem
    #   max_v(-v.n_k) + t.n_k <= apothem  =>  -min_v(v.n_k) + t.n_k <= apothem
    # So: t.n_k >= max_v(v.n_k) - apothem
    #     t.n_k <= apothem + min_v(v.n_k)
    # For feasibility: max_v(v.n_k) - apothem <= apothem + min_v(v.n_k)
    #   => max_v(v.n_k) - min_v(v.n_k) <= 2 * apothem = S * sqrt(3)
    # So S >= (max_v(v.n_k) - min_v(v.n_k)) / sqrt(3) for each k
    # Optimal t.n_k = (max_v(v.n_k) + min_v(v.n_k)) / 2
    
    S = 0.0
    for k in range(3):  # only need 3 pairs (k and k+3)
        proj = projections[:, k]
        extent = proj.max() - proj.min()
        S = max(S, extent / math.sqrt(3.0))
    return S

def objective(inner_states, outer_angle_deg):
    """Total cost = required_outer_side + penalty * max_overlap_depth."""
    S = required_outer_side(inner_states, outer_angle_deg)
    ov = max_overlap_depth(inner_states)
    penalty = 50.0
    return S + penalty * ov

def is_valid(inner_states, outer_angle_deg, max_side):
    """Check if packing is valid: no overlaps and fits in outer hex of side max_side."""
    ov = max_overlap_depth(inner_states)
    if ov > 1e-9:
        return False
    S = required_outer_side(inner_states, outer_angle_deg)
    return S <= max_side + 1e-9

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_result = None
    best_score = float('inf')
    
    def try_update(states, outer_angle):
        nonlocal best_result, best_score
        if _elapsed() > DEADLINE:
            return
        ov = max_overlap_depth(states)
        if ov > 1e-9:
            return
        S = required_outer_side(states, outer_angle)
        if S < best_score:
            best_score = S
            best_result = (list(states), outer_angle, S)
    
    # Initial configuration: place hexagons in a roughly hexagonal cluster
    def make_initial_config(seed):
        rng = random.Random(seed)
        # Start with a compact arrangement
        # Hexagonal close packing of 11 hexagons:
        # Center + ring of 6 + partial ring of 4
        # For unit hexagons with flat-top orientation, adjacent centers are
        # at distance sqrt(3) in 6 directions.
        states = []
        # Center
        states.append((0.0, 0.0, 0.0))
        # First ring: 6 hexagons
        d = math.sqrt(3.0)
        for i in range(6):
            theta = math.radians(60 * i)
            states.append((d * math.cos(theta), d * math.sin(theta), 0.0))
        # Second ring: 4 more hexagons (partial)
        d2 = 2.0 * d
        for i in range(4):
            theta = math.radians(60 * i + 30)
            states.append((d2 * math.cos(theta), d2 * math.sin(theta), 0.0))
        # Add small random perturbation
        for i in range(N):
            states[i] = (states[i][0] + rng.gauss(0, 0.1),
                         states[i][1] + rng.gauss(0, 0.1),
                         states[i][2])
        return states
    
    # Simulated annealing
    def simulated_annealing(states, outer_angle, n_steps, t_start, t_end, seed):
        rng = random.Random(seed)
        current_states = [list(s) for s in states]
        current_cost = objective(current_states, outer_angle)
        best_local = [list(s) for s in current_states]
        best_local_cost = current_cost
        
        for step in range(n_steps):
            if _elapsed() > DEADLINE:
                break
            # Temperature schedule
            t = t_start * (t_end / t_start) ** (step / max(n_steps - 1, 1))
            
            # Generate perturbation
            new_states = [list(s) for s in current_states]
            idx = rng.randint(0, N - 1)
            dx = rng.gauss(0, 0.5)
            dy = rng.gauss(0, 0.5)
            dangle = rng.gauss(0, 15.0)
            new_states[idx][0] += dx
            new_states[idx][1] += dy
            new_states[idx][2] = (new_states[idx][2] + dangle) % 360.0
            
            new_cost = objective(new_states, outer_angle)
            delta = new_cost - current_cost
            if delta < 0 or rng.random() < math.exp(-delta / max(t, 1e-10)):
                current_states = new_states
                current_cost = new_cost
                if current_cost < best_local_cost:
                    best_local_cost = current_cost
                    best_local = [list(s) for s in current_states]
        
        return best_local, best_local_cost
    
    # Multi-start
    n_starts = 20
    for start in range(n_starts):
        if _elapsed() > DEADLINE - 10:
            break
        states = make_initial_config(start * 1000 + 42)
        
        # Optimize outer angle first (coarse)
        best_outer_angle = 0.0
        best_outer_cost = float('inf')
        for oa in range(0, 60, 5):
            c = objective(states, float(oa))
            if c < best_outer_cost:
                best_outer_cost = c
                best_outer_angle = float(oa)
        
        # Fine-tune outer angle
        for oa in np.linspace(best_outer_angle - 5, best_outer_angle + 5, 11):
            c = objective(states, float(oa))
            if c < best_outer_cost:
                best_outer_cost = c
                best_outer_angle = float(oa)
        
        # Simulated annealing on positions and angles
        n_sa_steps = 5000
        t_start = 2.0
        t_end = 0.01
        states, cost = simulated_annealing(states, best_outer_angle, 
                                           n_sa_steps, t_start, t_end, 
                                           start * 7 + 13)
        
        # Check if valid and update best
        try_update(states, best_outer_angle)
        
        # If still not valid, try to fix overlaps with a penalty-based approach
        ov = max_overlap_depth(states)
        if ov > 1e-9:
            # Try a few more SA steps with higher temperature
            states2, cost2 = simulated_annealing(states, best_outer_angle,
                                                  3000, 5.0, 0.05,
                                                  start * 13 + 7)
            try_update(states2, best_outer_angle)
    
    # Final refinement: local search on best result
    if best_result is not None:
        states, outer_angle, S = best_result
        # Fine local search
        for _ in range(5):
            if _elapsed() > DEADLINE - 5:
                break
            improved = False
            for i in range(N):
                for dx in [-0.05, 0.05]:
                    for dy in [-0.05, 0.05]:
                        new_states = [list(s) for s in states]
                        new_states[i][0] += dx
                        new_states[i][1] += dy
                        ov = max_overlap_depth(new_states)
                        if ov <= 1e-9:
                            new_S = required_outer_side(new_states, outer_angle)
                            if new_S < S - 1e-6:
                                states = new_states
                                S = new_S
                                improved = True
            if not improved:
                break
        best_result = (states, outer_angle, S)
        best_score = S
    
    # If no valid result found, return a fallback (valid but not optimal)
    if best_result is None:
        # Fallback: spread hexagons far apart in a line with large outer
        states = []
        for i in range(N):
            states.append([float(i * math.sqrt(3)), 0.0, 0.0])
        outer_angle = 0.0
        S = required_outer_side(states, outer_angle)
        best_result = (states, outer_angle, S)
        best_score = S
    
    states, outer_angle, S = best_result
    # Convert to required format
    inner = [[s[0], s[1], s[2]] for s in states]
    # Center the outer hexagon: compute optimal translation
    all_verts = np.vstack([hex_vertices(s[0], s[1], 1.0, s[2]) for s in states])
    phi = math.radians(outer_angle)
    normals = np.zeros((6, 2))
    for i in range(6):
        theta = phi + math.radians(30 + 60 * i)
        normals[i, 0] = math.cos(theta)
        normals[i, 1] = math.sin(theta)
    projections = all_verts @ normals.T
    # Optimal center: for each normal pair, center = (max+min)/2
    cx = 0.0; cy = 0.0
    for k in range(3):
        proj = projections[:, k]
        center_proj = (proj.max() + proj.min()) / 2.0
        cx += normals[k, 0] * center_proj
        cy += normals[k, 1] * center_proj
    cx /= 3.0; cy /= 3.0
    
    return inner, (cx, cy), S, outer_angle

# Ensure construct is the entry point
if __name__ == "__main__":
    result = construct()
    print(f"Outer side: {result[2]:.4f}")
