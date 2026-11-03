import numpy as np
import time

N = 130

def _normalize(v):
    norms = np.linalg.norm(v, axis=1)
    norms[norms == 0] = 1.0
    return v / norms[:, np.newaxis]

def _covering_radius_and_worst_point(P):
    """
    Computes the covering radius and a point achieving it.
    """
    dots = np.dot(P, P.T)
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    np.fill_diagonal(dists, np.inf)
    
    max_dist = 0.0
    worst_point = None
    worst_indices = None
    
    # Find 5 nearest neighbors for each point
    nn_indices = np.argsort(dists, axis=1)[:, :5]
    
    for i in range(N):
        nns = nn_indices[i]
        # Check triangles formed by i and pairs of its 5 nearest neighbors
        # There are C(5,2) = 10 such triangles
        for a in range(len(nns)):
            for b in range(a + 1, len(nns)):
                j = nns[a]
                k = nns[b]
                
                p1 = P[i]
                p2 = P[j]
                p3 = P[k]
                
                # Circumcenter of spherical triangle p1, p2, p3
                v1 = p1 - p2
                v2 = p1 - p3
                c = np.cross(v1, v2)
                norm_c = np.linalg.norm(c)
                if norm_c < 1e-10:
                    continue
                
                c = c / norm_c
                
                # Ensure c is on the same side as the triangle centroid?
                # The circumcenter can be on the "other side" of the sphere.
                # We want the one that is "inside" the cap defined by the triangle.
                # A simple check: dot(c, p1) should be close to dot(c, p2) and dot(c, p3).
                # They are always equal by construction.
                # We want the circumcenter that is close to the points.
                # If the triangle is large, there are two circumcenters.
                # The one with smaller distance to p1 is the one we want.
                # d1 = arccos(c . p1)
                # d2 = arccos(-c . p1)
                # One is small, one is large.
                # We want the small one.
                # arccos is in [0, pi].
                # If c . p1 is positive, d1 < pi/2.
                # If c . p1 is negative, d1 > pi/2.
                # The other center is -c.
                # (-c) . p1 = - (c . p1).
                # If c . p1 > 0, then -c . p1 < 0, so distance to -c is > pi/2.
                # So we should check if c . p1 > 0?
                # Not necessarily. If the triangle covers more than a hemisphere, the "inner" circumcenter might have negative dot product?
                # For small caps (which we expect), the circumcenter is close to the points, so dot product is positive.
                
                # Let's just use c. If the distance is large, we might be using the wrong one.
                # But min_dist will be small if c is close to some point.
                
                dots_c = np.dot(P, c)
                dots_c = np.clip(dots_c, -1.0, 1.0)
                dists_c = np.arccos(dots_c)
                
                min_dist = np.min(dists_c)
                
                if min_dist > max_dist:
                    max_dist = min_dist
                    worst_point = c
                    # Find which points are at min_dist
                    # For the move, we need the indices of the points that define the cap.
                    # These are the points at distance min_dist from c.
                    worst_indices = np.argwhere(np.isclose(dists_c, min_dist, atol=1e-6)).flatten()
                    
    return max_dist, worst_point, worst_indices

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initial configuration
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_P = P.copy()
    best_R = _covering_radius_and_worst_point(P)[0]
    
    step_size = 0.1
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() < deadline:
        R, y, indices = _covering_radius_and_worst_point(P)
        
        if R < best_R:
            best_R = R
            best_P = P.copy()
            no_improve_count = 0
            step_size = 0.1 # Reset step size on improvement? Or keep decaying?
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturb
                noise = np.random.normal(0, 0.05, P.shape)
                P = P + noise
                P = _normalize(P)
                no_improve_count = 0
                step_size = 0.1
                
        if y is None:
            break
            
        # Move points towards y
        # Use a smaller step for points already close to y?
        # Simple uniform move
        for idx in indices:
            P[idx] = P[idx] + step_size * (y - P[idx])
            
        P = _normalize(P)
        
        # Decay step size
        step_size *= 0.99
        
    return best_P
