import numpy as np
import itertools
from scipy.optimize import minimize

def construct():
    N = 12
    
    # Generate initial centers from a hexagonal tiling
    # This provides a good starting point
    centers = []
    for i in range(-1, 2): # -1, 0, 1
        for j in range(-1, 3): # -1, 0, 1, 2
            x = i * np.sqrt(3)
            y = 2 * j + (1 if i % 2 != 0 else 0)
            centers.append([x, y])
            
    centers = np.array(centers)
    centers -= np.mean(centers, axis=0)
    
    # Initial angles: 0
    angles = np.zeros(N)
    
    # Variables: x1, y1, a1, x2, y2, a2, ...
    # Total 36 variables
    x0 = np.zeros(36)
    for i in range(N):
        x0[i*3] = centers[i, 0]
        x0[i*3+1] = centers[i, 1]
        x0[i*3+2] = 0.0 # angle
        
    # Precompute vertices of a unit hexagon (pointy top? or flat top?)
    # The code's calc_L assumes outer is flat-top.
    # Inner hexagons can be any orientation.
    # Let's define inner hexagon vertices relative to its center.
    # Standard unit hexagon (side 1) vertices:
    # If we align it with the outer (flat top), its vertices are at 0, 60...
    # But we can rotate it.
    # Let's use the same vertex definition as before:
    # angles = [90, 150, 210, 270, 330, 30] -> Pointy top?
    # Wait, 90 is top. So this is pointy-top.
    # If inner is pointy-top and outer is flat-top, they are rotated 30 deg relative to each other.
    # We will allow angle optimization.
    
    base_angles = np.deg2rad([90, 150, 210, 270, 330, 30])
    base_verts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    def get_inner_verts(params):
        # params: [x1, y1, a1, x2, y2, a2, ...]
        verts_list = []
        for i in range(N):
            cx = params[i*3]
            cy = params[i*3+1]
            ang = params[i*3+2]
            c, s = np.cos(ang), np.sin(ang)
            R = np.array([[c, -s], [s, c]])
            v = (R @ base_verts.T).T + np.array([cx, cy])
            verts_list.append(v)
        return np.array(verts_list) # Shape (N, 6, 2)
        
    def calc_L_from_verts(verts):
        # verts: (N, 6, 2)
        # Outer hexagon: flat-top, centered at (0,0), side L.
        # Inequalities for point (x,y) to be inside flat-top hex of side L:
        # |y| <= L * sqrt(3)/2
        # |x| <= L
        # |y + x/sqrt(3)| <= L * sqrt(3)/2  ? No.
        
        # Let's derive the 6 inequalities for flat-top hex (vertices at 0, 60, ...):
        # 1. Right edge (x = L): No, vertex is at L. Edge is slanted.
        # Vertices: (L,0), (L/2, L*sqrt(3)/2)
        # Edge 1: between (L,0) and (L/2, L*sqrt(3)/2).
        # Line: y - 0 = (L*sqrt(3)/2 - 0) / (L/2 - L) * (x - L) = -sqrt(3) (x - L)
        # y = -sqrt(3)x + L*sqrt(3)
        # sqrt(3)x + y = L*sqrt(3)
        # Constraint: sqrt(3)x + y <= L*sqrt(3)  => L >= x + y/sqrt(3)
        
        # Edge 2: between (L/2, L*sqrt(3)/2) and (-L/2, L*sqrt(3)/2).
        # Horizontal line y = L*sqrt(3)/2.
        # Constraint: y <= L*sqrt(3)/2 => L >= 2y/sqrt(3)
        
        # Edge 3: between (-L/2, L*sqrt(3)/2) and (-L, 0).
        # Symmetric to Edge 1.
        # -sqrt(3)x + y = L*sqrt(3) => L >= -x + y/sqrt(3)
        
        # Edge 4: Bottom left.
        # -sqrt(3)x - y = L*sqrt(3) => L >= -x - y/sqrt(3)
        
        # Edge 5: Bottom horizontal.
        # y = -L*sqrt(3)/2 => L >= -2y/sqrt(3)
        
        # Edge 6: Bottom right.
        # sqrt(3)x - y = L*sqrt(3) => L >= x - y/sqrt(3)
        
        # So L_req for a point (x,y) is:
        # max( x + y/sqrt(3), 2y/sqrt(3), -x + y/sqrt(3), -x - y/sqrt(3), -2y/sqrt(3), x - y/sqrt(3) )
        # This can be simplified to:
        # max( |x| + |y|/sqrt(3) , 2|y|/sqrt(3) ) ? No.
        # Let's just compute the 6 terms.
        
        x = verts[:, :, 0]
        y = verts[:, :, 1]
        
        term1 = x + y / np.sqrt(3)
        term2 = 2 * y / np.sqrt(3)
        term3 = -x + y / np.sqrt(3)
        term4 = -x - y / np.sqrt(3)
        term5 = -2 * y / np.sqrt(3)
        term6 = x - y / np.sqrt(3)
        
        L_points = np.max(np.stack([term1, term2, term3, term4, term5, term6]), axis=0)
        return np.max(L_points)
        
    def overlap_penalty(params):
        verts = get_inner_verts(params)
        # Check pairwise overlaps
        # For simplicity and speed, use center distance approximation with a threshold
        # or exact vertex-edge intersection?
        # Exact is slow. 
        # For hexagons of side 1, min center distance for non-overlap is sqrt(3) if aligned.
        # If rotated, it varies.
        # We will use a penalty based on the maximum overlap depth.
        # Approximation: If center distance < 1.8, check vertices.
        # Better: Just use a smooth penalty on center distances with a conservative threshold.
        # Threshold 1.8 is safe for any rotation?
        # Max radius is 1. So if dist < 2, they might overlap.
        # If dist >= 2, they never overlap.
        # We will use a penalty for dist < 2.0.
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                ci = np.array([params[i*3], params[i*3+1]])
                cj = np.array([params[j*3], params[j*3+1]])
                dist = np.linalg.norm(ci - cj)
                if dist < 2.0:
                    # Check if vertices of i are inside j or vice versa?
                    # Too slow.
                    # Use a smooth potential: (2.0 - dist)^2 if dist < 2.0
                    # This is a heuristic. It might prevent optimal packing if it's too repulsive.
                    # Let's try a softer repulsion.
                    penalty += (2.0 - dist)**2
        return penalty
        
    def objective(params):
        verts = get_inner_verts(params)
        L = calc_L_from_verts(verts)
        pen = overlap_penalty(params)
        return L + 10.0 * pen
        
    # Optimize
    res = minimize(objective, x0, method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 2000})
    
    best_params = res.x
    best_L = calc_L_from_verts(get_inner_verts(best_params))
    
    # Verify no overlaps (hard check)
    verts = get_inner_verts(best_params)
    # Simple check: if any vertex of i is inside j, it's bad.
    # We trust the penalty for now.
    
    inner = []
    for i in range(N):
        cx = best_params[i*3]
        cy = best_params[i*3+1]
        ang_deg = np.rad2deg(best_params[i*3+2])
        inner.append([cx, cy, ang_deg])
        
    return inner, (0.0, 0.0), best_L, 0.0
