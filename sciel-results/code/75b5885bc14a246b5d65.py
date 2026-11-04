import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    # Initial configuration: Hexagonal lattice
    axials = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,1), (0,2), (-1,2), (-2,1)
    ]
    
    sqrt3 = np.sqrt(3)
    
    # Try both orientations
    best_S = float('inf')
    best_centers = None
    best_angle = 0.0
    
    for orientation in [0, 1]: # 0: pointy-top, 1: flat-top
        centers = []
        for q, r in axials:
            if orientation == 0: # Pointy
                x = sqrt3 * (q + r/2.0)
                y = 1.5 * r
            else: # Flat
                x = 1.5 * q
                y = sqrt3 * (r + q/2.0)
            centers.append([x, y])
            
        centers = np.array(centers)
        centers -= np.mean(centers, axis=0)
        
        # Optimize positions
        def objective(params):
            c = params.reshape(-1, 2)
            # Calculate S
            if orientation == 0:
                # Pointy-top container
                # Vertices of pointy-top unit hex
                rel_verts = np.array([
                    [sqrt3/2, 0.5], [0, 1], [-sqrt3/2, 0.5],
                    [-sqrt3/2, -0.5], [0, -1], [sqrt3/2, -0.5]
                ])
                all_verts = np.vstack([c[i] + rel_verts for i in range(N)])
                abs_y = np.abs(all_verts[:, 1])
                e1 = np.abs(all_verts[:, 0]/sqrt3 + all_verts[:, 1])
                e2 = np.abs(all_verts[:, 0]/sqrt3 - all_verts[:, 1])
                S = np.max(np.vstack([abs_y, e1, e2]))
            else:
                # Flat-top container
                # Vertices of flat-top unit hex
                rel_verts = np.array([
                    [1, 0], [0.5, sqrt3/2], [-0.5, sqrt3/2],
                    [-1, 0], [-0.5, -sqrt3/2], [0.5, -sqrt3/2]
                ])
                all_verts = np.vstack([c[i] + rel_verts for i in range(N)])
                abs_x = np.abs(all_verts[:, 0])
                e1 = np.abs(sqrt3*all_verts[:, 0] + all_verts[:, 1])/sqrt3
                e2 = np.abs(sqrt3*all_verts[:, 0] - all_verts[:, 1])/sqrt3
                S = np.max(np.vstack([abs_x, e1, e2]))
            
            # Penalty for overlap (approx distance < sqrt(3))
            # We want to keep them disjoint.
            # Simple penalty: sum of (min_dist - dist)^2 for dist < min_dist
            # min_dist for aligned hexes is sqrt(3)
            penalty = 0
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(c[i] - c[j])
                    if d < sqrt3:
                        penalty += (sqrt3 - d)**2
            return S + 100 * penalty
        
        x0 = centers.flatten()
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100})
        
        if res.success:
            c_opt = res.x.reshape(-1, 2)
            # Recalculate true S without penalty
            if orientation == 0:
                rel_verts = np.array([
                    [sqrt3/2, 0.5], [0, 1], [-sqrt3/2, 0.5],
                    [-sqrt3/2, -0.5], [0, -1], [sqrt3/2, -0.5]
                ])
                all_verts = np.vstack([c_opt[i] + rel_verts for i in range(N)])
                S = np.max(np.vstack([np.abs(all_verts[:, 1]), 
                                      np.abs(all_verts[:, 0]/sqrt3 + all_verts[:, 1]),
                                      np.abs(all_verts[:, 0]/sqrt3 - all_verts[:, 1])]))
            else:
                rel_verts = np.array([
                    [1, 0], [0.5, sqrt3/2], [-0.5, sqrt3/2],
                    [-1, 0], [-0.5, -sqrt3/2], [0.5, -sqrt3/2]
                ])
                all_verts = np.vstack([c_opt[i] + rel_verts for i in range(N)])
                S = np.max(np.vstack([np.abs(all_verts[:, 0]),
                                      np.abs(sqrt3*all_verts[:, 0] + all_verts[:, 1])/sqrt3,
                                      np.abs(sqrt3*all_verts[:, 0] - all_verts[:, 1])/sqrt3]))
            
            if S < best_S:
                best_S = S
                best_centers = c_opt
                best_angle = 0.0 if orientation == 0 else 0.0 # Inner angle is 0 relative to their own frame, but we need to match container?
                # The task says inner rows (x, y, angle_degrees).
                # If orientation 0 (pointy), inner hexes are pointy. Angle 0.
                # If orientation 1 (flat), inner hexes are flat. Angle 0.
                # Container angle is 0.
                # So we just return the coordinates.

    # Fallback if optimization fails
    if best_centers is None:
        # Use the static pointy-top config
        centers = []
        for q, r in axials:
            x = sqrt3 * (q + r/2.0)
            y = 1.5 * r
            centers.append([x, y])
        centers = np.array(centers)
        centers -= np.mean(centers, axis=0)
        best_centers = centers
        # Calculate S for this
        rel_verts = np.array([
            [sqrt3/2, 0.5], [0, 1], [-sqrt3/2, 0.5],
            [-sqrt3/2, -0.5], [0, -1], [sqrt3/2, -0.5]
        ])
        all_verts = np.vstack([best_centers[i] + rel_verts for i in range(N)])
        best_S = np.max(np.vstack([np.abs(all_verts[:, 1]), 
                                  np.abs(all_verts[:, 0]/sqrt3 + all_verts[:, 1]),
                                  np.abs(all_verts[:, 0]/sqrt3 - all_verts[:, 1])]))

    inner = [[x, y, 0.0] for x, y in best_centers]
    return inner, (0.0, 0.0), best_S, 0.0
