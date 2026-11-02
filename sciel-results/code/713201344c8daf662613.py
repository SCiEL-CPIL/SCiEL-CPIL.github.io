import numpy as np
import time
from scipy.optimize import linprog

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    # Precompute unit hexagon vertices (side 1, angle 0)
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    unit_verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    # Normals for the 6 faces of the hexagon (outward pointing)
    # Face 0: between v0 and v1. Midpoint (0.75, sqrt(3)/4). Normal direction (1, 1/sqrt(3))? 
    # Let's calculate exactly.
    # Edge 0: v1 - v0 = (-0.5, sqrt(3)/2). Normal (perp, outward) = (sqrt(3)/2, 0.5).
    # Wait, let's just use the standard 6 directions for a regular hexagon.
    # The support function of a unit hexagon in direction u is h(u).
    # For non-overlap: h_i(u) + h_j(u) <= ||p_i - p_j||_u? No.
    # Separating Axis Theorem: Two convex polygons P and Q are disjoint iff there exists a line L such that P and Q are on opposite sides.
    # For regular hexagons of same size, the separating axes are the 6 normals of the hexagon faces.
    # Let n_k be the unit normal to face k.
    # Projection of hexagon i onto n_k: [p_i . n_k - h(n_k), p_i . n_k + h(n_k)]
    # Where h(n_k) is the support function of the unit hexagon in direction n_k.
    # For a regular hexagon of side 1, the apothem (distance from center to side) is sqrt(3)/2.
    # The support function in the direction normal to a face is exactly the apothem.
    # So h(n_k) = sqrt(3)/2 for the 6 face normals.
    
    # Let's verify the normals.
    # Vertex 0: (1, 0). Vertex 1: (1/2, sqrt(3)/2).
    # Midpoint of edge 0-1: (3/4, sqrt(3)/4).
    # Vector from center to midpoint: (3/4, sqrt(3)/4).
    # Length: sqrt(9/16 + 3/16) = sqrt(12/16) = sqrt(3)/2. Correct.
    # Unit normal n0 = (3/4, sqrt(3)/4) / (sqrt(3)/2) = (3/4 * 2/sqrt(3), sqrt(3)/4 * 2/sqrt(3)) = (sqrt(3)/2, 1/2).
    
    normals = np.array([
        [np.sqrt(3)/2, 0.5],
        [0.0, 1.0],
        [-np.sqrt(3)/2, 0.5],
        [-np.sqrt(3)/2, -0.5],
        [0.0, -1.0],
        [np.sqrt(3)/2, -0.5]
    ])
    
    apothem = np.sqrt(3) / 2.0
    
    def solve_lp(init_positions):
        """
        Solve the LP for given initial positions to determine the sign pattern of separating axes.
        Variables:
        0..2*N-1: x_i, y_i for i=0..N-1
        2*N: S (outer side length)
        
        Objective: Minimize S
        c = [0, ..., 0, 1]
        
        Constraints:
        1. Containment:
           For each hexagon i and each normal direction k (0..5):
           (p_i . n_k) + apothem <= S * apothem
           Note: The outer hexagon is centered at (0,0) with side S.
           Its support function in direction n_k is S * apothem.
           The inner hexagon i has support function (p_i . n_k) + apothem.
           So: p_i . n_k + apothem - S * apothem <= 0.
           
        2. Non-overlap:
           For each pair (i, j) with i < j:
           We need to ensure they are disjoint.
           Using SAT, they are disjoint if there exists a normal n_k such that:
           (p_i . n_k) + apothem < (p_j . n_k) - apothem  (i is "below" j)
           OR
           (p_j . n_k) + apothem < (p_i . n_k) - apothem  (j is "below" i)
           
           This is an OR constraint. To linearize, we fix the "orientation" or "sign" for each pair.
           We determine the sign based on the initial positions:
           If (p_j - p_i) . n_k > 0 for some k, we guess that j is "above" i in that direction.
           Actually, we just pick the axis k that currently separates them (or is closest to separating) and fix the inequality direction.
           
           Let's refine: For each pair (i,j), look at the initial positions.
           Find the axis k that maximizes the separation: (p_j - p_i) . n_k.
           Let d = (p_j - p_i) . n_k.
           If d > 0, we constrain: (p_i . n_k) + apothem <= (p_j . n_k) - apothem
           => (p_j - p_i) . n_k >= 2 * apothem
           If d < 0, we constrain: (p_i - p_j) . n_k >= 2 * apothem
           => (p_i - p_j) . n_k >= 2 * apothem
           If d == 0, we can pick either, but usually one will dominate.
           
           Wait, is it sufficient to fix ONE axis per pair?
           Yes, if the hexagons are disjoint, there is at least one separating axis.
           If we fix the axis that *was* separating them in the initial config, and the optimization moves them slightly, it might still hold.
           However, if the optimal solution requires a *different* separating axis, this LP might fail or find a suboptimal solution where they touch on a different axis but violate the fixed one?
           No, if they are disjoint, ALL separating axes work. If we force them to be separated by axis k, they are disjoint.
           The risk is that the "natural" separating axis changes during optimization, causing the fixed constraint to be too restrictive or the wrong one to be chosen.
           However, for small perturbations, the separating axis is stable.
           
           Let's implement this.
        """
        n_vars = 2 * N + 1
        c = np.zeros(n_vars)
        c[-1] = 1.0  # Minimize S
        
        A_ub = []
        b_ub = []
        
        # 1. Containment Constraints
        # For each i, k:
        # p_i . n_k - S * apothem <= -apothem
        # x_i * n_k[0] + y_i * n_k[1] - S * apothem <= -apothem
        for i in range(N):
            for k in range(6):
                row = np.zeros(n_vars)
                row[2*i] = normals[k, 0]
                row[2*i+1] = normals[k, 1]
                row[-1] = -apothem
                A_ub.append(row)
                b_ub.append(-apothem)
        
        # 2. Non-overlap Constraints
        for i in range(N):
            for j in range(i + 1, N):
                # Determine sign based on initial positions
                diff = init_positions[j] - init_positions[i]
                # Project diff onto all normals
                projs = diff @ normals.T
                # Find axis with max projection (most separating)
                k_max = np.argmax(projs)
                # If projs[k_max] is close to 0, it's ambiguous. 
                # But for valid initial configs, they should be separated.
                # If projs[k_max] > 0, j is "forward" along n_k_max.
                # Constraint: (p_j - p_i) . n_k_max >= 2 * apothem
                # => -(p_j - p_i) . n_k_max <= -2 * apothem
                # => -x_j*n_x - y_j*n_y + x_i*n_x + y_i*n_y <= -2*apothem
                
                if projs[k_max] >= 0:
                    row = np.zeros(n_vars)
                    row[2*i] = normals[k_max, 0]
                    row[2*i+1] = normals[k_max, 1]
                    row[2*j] = -normals[k_max, 0]
                    row[2*j+1] = -normals[k_max, 1]
                    A_ub.append(row)
                    b_ub.append(-2 * apothem)
                else:
                    row = np.zeros(n_vars)
                    row[2*i] = -normals[k_max, 0]
                    row[2*i+1] = -normals[k_max, 1]
                    row[2*j] = normals[k_max, 0]
                    row[2*j+1] = normals[k_max, 1]
                    A_ub.append(row)
                    b_ub.append(-2 * apothem)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds:
        # Positions can be anywhere, but bounded by S.
        # S is positive.
        # We don't know S beforehand. 
        # We can set loose bounds. Max side length maybe 10?
        # x, y in [-10, 10], S in [0, 10]
        bounds = [(-10, 10)] * (2 * N) + [(0, 10)]
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                S = res.x[-1]
                positions = res.x[:2*N].reshape(N, 2)
                return positions, S
            else:
                return None, float('inf')
        except Exception:
            return None, float('inf')

    # Generate diverse initial configurations
    configs = []
    
    # 1. Hexagonal lattice
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    init1 = []
    for q, r in coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        init1.append([x, y])
    configs.append(np.array(init1))
    
    # 2. Circular
    init2 = np.zeros((N, 2))
    for i in range(N):
        angle = 2 * np.pi * i / N
        r = 2.5
        init2[i, 0] = r * np.cos(angle)
        init2[i, 1] = r * np.sin(angle)
    configs.append(init2)
    
    # 3. Random seeds
    for seed in [42, 123, 999, 555]:
        np.random.seed(seed)
        init_rand = np.random.uniform(-4, 4, (N, 2))
        # Scale to ensure initial validity roughly
        # Just add it, LP will handle if valid, else fail
        configs.append(init_rand)
        
    # 4. Grid-like
    init4 = np.array([
        [0, 0], [3, 0], [6, 0], [9, 0],
        [1.5, 2.6], [4.5, 2.6], [7.5, 2.6],
        [0, 5.2], [3, 5.2], [6, 5.2], [9, 5.2],
        [1.5, 7.8]
    ])
    configs.append(init4)

    best_positions = None
    best_side = float('inf')
    
    for init in configs:
        if time.time() > deadline:
            break
            
        # Check if initial config is roughly valid (not strictly, just to avoid obvious overlaps that break sign determination)
        # Actually, if they overlap, the "max projection" might be small or negative, leading to ambiguous signs.
        # We can try to solve anyway.
        
        try:
            pos, side = solve_lp(init)
            if pos is not None and side < best_side:
                best_side = side
                best_positions = pos
        except Exception:
            pass
            
    # Fallback if no LP solution found
    if best_positions is None:
        best_positions = configs[0]
        # Calculate side for fallback
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            # Simple hexagon vertices
            angles = np.radians(np.arange(0, 360, 60))
            verts[i] = np.column_stack([
                best_positions[i, 0] + np.cos(angles),
                best_positions[i, 1] + np.sin(angles)
            ])
        verts_flat = verts.reshape(-1, 2)
        x = verts_flat[:, 0]
        y = verts_flat[:, 1]
        sqrt3 = np.sqrt(3.0)
        L1 = x + y / sqrt3
        L2 = y * 2.0 / sqrt3
        L3 = -x + y / sqrt3
        L4 = -x - y / sqrt3
        L5 = -y * 2.0 / sqrt3
        L6 = x - y / sqrt3
        all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
        best_side = np.max(all_L)
        
    # Final validation and refinement
    # The LP solution is exact for the fixed topology.
    # We can try to "polish" by re-running LP with the solution as init (to catch sign flips?)
    # But usually the first LP is stable.
    
    # Calculate final exact side to be safe
    verts = np.zeros((N, 6, 2))
    for i in range(N):
        angles = np.radians(np.arange(0, 360, 60))
        verts[i] = np.column_stack([
            best_positions[i, 0] + np.cos(angles),
            best_positions[i, 1] + np.sin(angles)
        ])
    verts_flat = verts.reshape(-1, 2)
    x = verts_flat[:, 0]
    y = verts_flat[:, 1]
    sqrt3 = np.sqrt(3.0)
    L1 = x + y / sqrt3
    L2 = y * 2.0 / sqrt3
    L3 = -x + y / sqrt3
    L4 = -x - y / sqrt3
    L5 = -y * 2.0 / sqrt3
    L6 = x - y / sqrt3
    all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
    final_side = np.max(all_L)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), 0.0])
        
    return inner, (0.0, 0.0), float(final_side), 0.0

# EVOLVE-BLOCK-END
