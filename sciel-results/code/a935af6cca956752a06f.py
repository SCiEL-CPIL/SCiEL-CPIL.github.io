import math
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate hexagonal lattice points in axial coordinates (q, r)
    # We generate a sufficiently large hexagonal region and pick the N closest to the center.
    
    # Max radius needed: For N points, radius in lattice units is roughly sqrt(N/3).
    # Let's generate up to radius 10 to be safe for small N, or scale with N.
    max_radius = int(math.ceil(math.sqrt(N))) + 2
    
    points = []
    for q in range(-max_radius, max_radius + 1):
        for r in range(-max_radius, max_radius + 1):
            # Hex distance from origin is (|q| + |r| + |q+r|) / 2
            dist = (abs(q) + abs(r) + abs(q + r)) / 2.0
            if dist <= max_radius:
                points.append((dist, q, r))
    
    # Sort by distance from center to get the most compact cluster
    points.sort(key=lambda x: (x[0], x[1], x[2]))
    
    # Take the first N points
    selected = points[:N]
    
    # Convert axial to Cartesian coordinates
    # Standard flat-topped hexagon lattice:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    # This corresponds to unit hexagons of side 1.
    centers = []
    for _, q, r in selected:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
    
    # Calculate the centroid to center the cluster at (0,0)
    if N == 0:
        return [], (0.0, 0.0), 0.0, 0.0
        
    cx = sum(c[0] for c in centers) / N
    cy = sum(c[1] for c in centers) / N
    
    # Shift centers so centroid is at (0,0)
    shifted_centers = [(x - cx, y - cy) for x, y in centers]
    
    # Calculate the required outer hexagon side length
    # The outer hexagon is aligned with the inner ones (angle 0).
    # Normals to the sides of a regular hexagon (flat-topped, angle 0)
    # are at angles 0, 60, 120, 180, 240, 300 degrees.
    # n_k = (cos(theta), sin(theta))
    normals = []
    for k in range(6):
        theta = math.pi / 3 * k
        normals.append((math.cos(theta), math.sin(theta)))
    
    # For each normal, find the maximum projection of the shifted centers
    max_projections = []
    for n in normals:
        max_proj = max(c[0] * n[0] + c[1] * n[1] for c in shifted_centers)
        max_projections.append(max_proj)
    
    # The maximum projection of the *centers* is M_max.
    # The inner hexagons extend by their apothem (sqrt(3)/2) in the direction of the normal
    # because they are aligned with the outer hexagon.
    # So the required apothem of the outer hexagon is M_max + apothem_inner.
    M_max = max(max_projections)
    apothem_inner = math.sqrt(3) / 2.0
    required_apothem = M_max + apothem_inner
    
    # Side length S is related to apothem h by h = S * sqrt(3) / 2
    # So S = 2 * h / sqrt(3)
    S = 2.0 * required_apothem / math.sqrt(3)
    
    # Construct the inner list: (x, y, angle)
    # We return the coordinates relative to the outer center (which is 0,0)
    inner = []
    for x, y in shifted_centers:
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
