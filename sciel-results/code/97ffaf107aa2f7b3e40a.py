import math
import numpy as np
from itertools import combinations

N = 11

def construct():
    # Generate hex lattice points
    v1 = np.array([1.5, math.sqrt(3)/2])
    v2 = np.array([0, math.sqrt(3)])
    
    # Helper to get cartesian from axial
    def axial_to_cart(q, r):
        x = 1.5 * q
        y = math.sqrt(3) * (r + q / 2.0)
        return np.array([x, y])

    # Generate points by ring
    ring0 = [(0,0)]
    ring1 = []
    # Directions for ring 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Actually, let's just generate all points with hex_dist == 1
    for q in range(-1, 2):
        for r in range(-1, 2):
            if (abs(q) + abs(r) + abs(q+r)) // 2 == 1:
                ring1.append((q, r))
                
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if (abs(q) + abs(r) + abs(q+r)) // 2 == 2:
                ring2.append((q, r))

    # Base points: Ring 0 + Ring 1 (7 points)
    base_points = [axial_to_cart(q, r) for q, r in ring0 + ring1]
    
    # Candidate points: Ring 2 (12 points)
    candidate_points = [axial_to_cart(q, r) for q, r in ring2]
    
    # We need N - 7 = 4 points from candidates
    k = N - len(base_points)
    
    if k <= 0:
        selected = base_points[:N]
    else:
        best_S = float('inf')
        best_combo = None
        
        # Normals for flat-top hexagon
        normals = [
            np.array([0, 1]),
            np.array([math.sqrt(3)/2, 0.5]),
            np.array([math.sqrt(3)/2, -0.5])
        ]
        
        def calc_S(points):
            max_proj = 0.0
            for c in points:
                for n in normals:
                    proj = abs(np.dot(c, n))
                    if proj > max_proj:
                        max_proj = proj
            return 1.0 + (2.0 / math.sqrt(3)) * max_proj

        # Iterate combinations
        for combo in combinations(candidate_points, k):
            all_points = base_points + list(combo)
            S = calc_S(all_points)
            if S < best_S:
                best_S = S
                best_combo = combo
                
        selected = base_points + list(best_combo)
        
    # Final calculation of S for the selected set
    normals = [
        np.array([0, 1]),
        np.array([math.sqrt(3)/2, 0.5]),
        np.array([math.sqrt(3)/2, -0.5])
    ]
    max_proj = 0.0
    for c in selected:
        for n in normals:
            proj = abs(np.dot(c, n))
            if proj > max_proj:
                max_proj = proj
    S = 1.0 + (2.0 / math.sqrt(3)) * max_proj

    # Format output
    inner = []
    for c in selected:
        inner.append([c[0], c[1], 0.0])
        
    return inner, (0.0, 0.0), S, 0.0
