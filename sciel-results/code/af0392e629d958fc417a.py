import math
import time
import numpy as np
from scipy.optimize import minimize_scalar

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin

    # Honeycomb lattice basis
    # u = (1.5, sqrt(3)/2), v = (0, sqrt(3))
    # Points are i*u + j*v
    sqrt3 = math.sqrt(3)
    
    # Generate a pool of lattice points
    # We need enough points to pick 12.
    # Ring 0: 1
    # Ring 1: 6
    # Ring 2: 12
    # Total 19.
    # Let's generate points with index (i,j) such that max(|i|,|j|,|i+j|) <= 2?
    # Hex distance d = max(|i|, |j|, |i+j|) is not quite right for axial coords.
    # Axial coords (i,j). Distance = (|i| + |j| + |i+j|) / 2.
    
    lattice_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            # Calculate distance from origin
            dist = (abs(i) + abs(j) + abs(i+j)) / 2.0
            if dist <= 2.5: # Include ring 2 and some of ring 3 just in case
                x = i * 1.5
                y = i * sqrt3 / 2.0 + j * sqrt3
                lattice_points.append((x, y, i, j))
                
    # Sort by distance
    lattice_points.sort(key=lambda p: math.hypot(p[0], p[1]))
    
    # Candidate topologies
    # Topology 1: Center + 6 Neighbors + 5 from Ring 2
    # We need to identify which points are Ring 1 and Ring 2.
    ring0 = []
    ring1 = []
    ring2 = []
    for p in lattice_points:
        d = (abs(p[2]) + abs(p[3]) + abs(p[2]+p[3])) / 2.0
        if d == 0:
            ring0.append(p)
        elif d == 1:
            ring1.append(p)
        elif d == 2:
            ring2.append(p)
            
    # Base set: ring0 + ring1 (7 points)
    base = ring0 + ring1
    
    # We need 5 more from ring2.
    # To maintain symmetry, we can pick 5 out of 12.
    # Due to 6-fold symmetry, there are a few distinct ways to pick 5.
    # Let's generate all subsets of 5 from ring2? 12 choose 5 is 792. That's fine.
    # But many are equivalent by rotation.
    # Let's just try a few specific patterns.
    # Pattern A: 5 consecutive points in ring2.
    # Pattern B: 2 opposite pairs + 1 opposite? No, 5 is odd.
    # Let's just iterate over all combinations of 5 from ring2, but prune by symmetry?
    # Simpler: Just try all 792 combinations. It's fast.
    
    best_side = float('inf')
    best_config = None
    
    # Precompute normals for outer hexagon (aligned)
    normals = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(30, 360, 60)])
    
    def calc_side(points, angle_rad):
        # Rotate points
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        rot = np.array([[c, -s], [s, c]])
        pts = np.array([[p[0], p[1]] for p in points])
        pts_rot = pts @ rot.T
        
        # Center the points to minimize max projection?
        # The optimal translation is to center the bounding box.
        # For a regular hexagon, centering the "centroid" of the projections is good.
        # Let's just compute max projection and min projection for each normal.
        # The side length S is determined by the max extent.
        # Extent in direction n is (max_proj - min_proj).
        # Wait, the outer hexagon is centered at (0,0).
        # We can translate the inner cluster.
        # Let the cluster be translated by t.
        # The condition is: (p + t) . n <= S * sqrt(3)/2
        # And (p + t) . n >= -S * sqrt(3)/2
        # So 2 * S * sqrt(3)/2 >= max((p+t).n) - min((p+t).n)
        # S >= (max - min) / sqrt(3)
        # To minimize S, we want to minimize the range of projections for all 6 directions.
        # The optimal t for a given direction n centers the projection: t.n = -(max+min)/2.
        # But t must work for all directions.
        # This is a convex optimization problem (minimize S subject to linear constraints).
        # However, for small N, we can just assume the cluster is centered at the origin (t=0) if we pick symmetric points.
        # Or we can optimize t as well.
        
        # Let's just compute the range for t=0 first.
        projs = pts_rot @ normals.T
        max_projs = np.max(projs, axis=0)
        min_projs = np.min(projs, axis=0)
        extents = max_projs - min_projs
        # The side length S must satisfy S * sqrt(3) >= extent_k for all k.
        # Also, the inner hexagons have width sqrt(3).
        # So the total extent is extent_k + sqrt(3)?
        # No.
        # The projection of an inner hexagon at p is [ (p.n - sqrt(3)/2), (p.n + sqrt(3)/2) ].
        # The union of these intervals for all i is [ min_i(p_i.n - sqrt(3)/2), max_i(p_i.n + sqrt(3)/2) ].
        # Total width = max_i(p_i.n) - min_i(p_i.n) + sqrt(3).
        # So S * sqrt(3) >= max_i(p_i.n) - min_i(p_i.n) + sqrt(3).
        # S >= (max - min + sqrt(3)) / sqrt(3) = (max - min)/sqrt(3) + 1.
        
        max_extent = np.max(extents)
        return max_extent / sqrt3 + 1.0

    # Iterate over combinations of 5 from ring2
    import itertools
    
    # To save time, we can just try a few symmetric ones.
    # Let's try all 792.
    for combo in itertools.combinations(ring2, 5):
        points = base + list(combo)
        
        # Optimize rotation angle
        def obj(angle):
            return calc_side(points, angle)
            
        res = minimize_scalar(obj, bounds=(0, math.pi/3), method='bounded', options={'xatol': 1e-6})
        
        if res.fun < best_side:
            best_side = res.fun
            best_config = (points, res.x)
            
        if time.time() > deadline - 20:
            break
            
    # Refine the best configuration with local optimization of positions
    # We allow the points to move slightly to reduce the bounding box.
    # This is a continuous relaxation.
    
    if best_config is not None:
        points, angle = best_config
        pts = np.array([[p[0], p[1]] for p in points])
        
        # Define objective for positions
        def obj_pos(params):
            pos = params.reshape(N, 2)
            # Rotate
            c = math.cos(angle)
            s = math.sin(angle)
            rot = np.array([[c, -s], [s, c]])
            pos_rot = pos @ rot.T
            
            projs = pos_rot @ normals.T
            max_projs = np.max(projs, axis=0)
            min_projs = np.min(projs, axis=0)
            extents = max_projs - min_projs
            max_extent = np.max(extents)
            return max_extent / sqrt3 + 1.0
            
        x0 = pts.flatten()
        
        # Constraints: non-overlap
        # For aligned hexagons, distance between centers must be >= sqrt(3) in the lattice sense?
        # Actually, the forbidden zone is a hexagon of side 2.
        # We can use a penalty method.
        
        def penalty(params):
            pos = params.reshape(N, 2)
            pen = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    d = pos[i] - pos[j]
                    # Check if d is inside the Minkowski sum (hexagon of side 2)
                    # Project d onto normals
                    projs = np.abs(d @ normals.T)
                    # If all projections < sqrt(3), it's inside (overlapping)
                    # The distance to boundary is min(sqrt(3) - proj_k)
                    # We want to penalize if min > 0? No, if min < 0 (inside).
                    # Actually, if max(projs) < sqrt(3), it is inside.
                    # So penalty = sum( max(0, sqrt(3) - max(projs))^2 )?
                    # Better: if max(projs) < sqrt(3), then it's overlapping.
                    # The "depth" is sqrt(3) - max(projs).
                    if np.max(projs) < sqrt3:
                        pen += (sqrt3 - np.max(projs))**2
            return pen
            
        def obj_total(params):
            return obj_pos(params) + 100.0 * penalty(params)
            
        res2 = minimize(obj_total, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10})
        
        if res2.fun < best_side:
            # Check validity
            pos = res2.x.reshape(N, 2)
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    d = pos[i] - pos[j]
                    projs = np.abs(d @ normals.T)
                    if np.max(projs) < sqrt3 - 1e-4:
                        valid = False
                        break
                if not valid: break
            
            if valid:
                best_side = obj_pos(res2.x)
                best_config = (pos, angle)
    
    # Finalize
    if best_config is not None:
        points, angle = best_config
        pts = np.array([[p[0], p[1]] for p in points]) if not isinstance(points, np.ndarray) else points
        
        # Center the points
        center = np.mean(pts, axis=0)
        pts -= center
        
        # Calculate final side
        c = math.cos(angle)
        s = math.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        pts_rot = pts @ rot.T
        projs = pts_rot @ normals.T
        max_projs = np.max(projs, axis=0)
        min_projs = np.min(projs, axis=0)
        extents = max_projs - min_projs
        final_side = np.max(extents) / sqrt3 + 1.0
        
        # The outer hexagon is centered at (0,0)
        # Inner hexagons are at pts, angle 0 (since we rotated the points to align with outer)
        # Wait, if we rotated the points, the inner hexagons are effectively rotated?
        # No, we rotated the coordinates. The hexagons themselves are aligned with the axes.
        # So angle_degrees for inner hexagons is 0.
        
        inner = []
        for i in range(N):
            inner.append([pts[i][0], pts[i][1], 0.0])
            
        return inner, (0.0, 0.0), final_side, 0.0
    else:
        # Fallback
        return [[0,0,0] for _ in range(N)], (0,0), 10.0, 0.0

