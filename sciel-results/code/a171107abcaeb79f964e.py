"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def _covering_radius(points):
    """Calculate the covering radius of the points on the unit sphere."""
    if len(points) < 3:
        return np.inf
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.inf
        
    faces = hull.simplices
    if len(faces) == 0:
        return np.inf

    max_rad = 0.0
    for i in range(len(faces)):
        p1 = points[faces[i, 0]]
        p2 = points[faces[i, 1]]
        p3 = points[faces[i, 2]]
        
        # Compute circumcenter of the spherical triangle (p1, p2, p3)
        # v = (p2 x p3) + (p3 x p1) + (p1 x p2)
        v = np.cross(p2, p3) + np.cross(p3, p1) + np.cross(p1, p2)
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            # Degenerate, use centroid
            cc = p1 + p2 + p3
            norm = np.linalg.norm(cc)
            if norm < 1e-10:
                continue
            cc = cc / norm
        else:
            cc = v / norm
            
        # Angular distance from p1 to cc
        dot = np.dot(p1, cc)
        dot = np.clip(dot, -1.0, 1.0)
        rad = np.arccos(dot)
        if rad > max_rad:
            max_rad = rad
            
    return max_rad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s limit, 20s margin
    
    best_points = None
    best_rad = np.inf
    
    # Multiple restarts
    for restart in range(5):
        # Initialize
        if restart == 0:
            # Fibonacci spiral
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        else:
            # Random initialization
            points = np.random.randn(N, 3)
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
            
        curr_rad = _covering_radius(points)
        if curr_rad < best_rad:
            best_rad = curr_rad
            best_points = points.copy()
            
        alpha = 0.1
        min_alpha = 1e-4
        stuck_counter = 0
        
        while time.time() < deadline - 5.0:
            # Compute Hull and Forces
            try:
                hull = ConvexHull(points)
                faces = hull.simplices
            except Exception:
                # Degenerate hull, perturb and break
                points = points + 0.01 * np.random.randn(N, 3)
                points = points / np.linalg.norm(points, axis=1, keepdims=True)
                break
                
            # Compute Circumcenters
            ccs = np.zeros((len(faces), 3))
            for i in range(len(faces)):
                p1 = points[faces[i, 0]]
                p2 = points[faces[i, 1]]
                p3 = points[faces[i, 2]]
                v = np.cross(p2, p3) + np.cross(p3, p1) + np.cross(p1, p2)
                norm = np.linalg.norm(v)
                if norm > 1e-10:
                    ccs[i] = v / norm
                else:
                    cc = p1 + p2 + p3
                    norm = np.linalg.norm(cc)
                    if norm > 1e-10:
                        ccs[i] = cc / norm
                    else:
                        ccs[i] = p1
            
            # Find farthest circumcenter for each point
            farthest_cc = np.zeros_like(points)
            farthest_dist = np.zeros(N)
            
            for i in range(len(faces)):
                f = faces[i]
                cc = ccs[i]
                for idx in f:
                    p = points[idx]
                    dist = np.arccos(np.clip(np.dot(p, cc), -1.0, 1.0))
                    if dist > farthest_dist[idx]:
                        farthest_dist[idx] = dist
                        farthest_cc[idx] = cc
            
            # Calculate Repulsive Forces (optional, helps with stability)
            # Skip for now to keep it simple, Voronoi update should suffice
            
            # Update Points
            for i in range(N):
                if farthest_dist[i] > 1e-6:
                    # Move towards farthest CC
                    new_p = points[i] + alpha * (farthest_cc[i] - points[i])
                    new_p = new_p / np.linalg.norm(new_p)
                    points[i] = new_p
            
            # Normalize
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
            
            # Evaluate
            curr_rad = _covering_radius(points)
            
            if curr_rad < best_rad:
                best_rad = curr_rad
                best_points = points.copy()
                stuck_counter = 0
                alpha *= 0.99  # Slow decay
            else:
                stuck_counter += 1
                alpha *= 0.90  # Fast decay
                if stuck_counter > 50 and alpha < min_alpha:
                    # Perturb to escape local min
                    points = points + 0.05 * np.random.randn(N, 3)
                    points = points / np.linalg.norm(points, axis=1, keepdims=True)
                    alpha = 0.1
                    stuck_counter = 0
                    break # Break inner loop to try next restart or continue
    
    if best_points is None:
        # Fallback
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    return best_points
# EVOLVE-BLOCK-END
