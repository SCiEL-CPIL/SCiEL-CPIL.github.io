import numpy as np
from scipy.spatial import Delaunay
import time

N = 50

def _covering_radius(P):
    """
    Compute the covering radius of the configuration P.
    P is an (N, 3) array of unit vectors.
    """
    if len(P) < 3:
        return np.pi / 2
        
    # Delaunay triangulation of points on the sphere
    # scipy.spatial.Delaunay works on 3D points.
    # The triangulation of points on a convex hull (sphere) gives the spherical Delaunay triangulation.
    try:
        tri = Delaunay(P)
    except Exception:
        return np.pi / 2 # Fallback for degenerate cases

    max_r = 0.0
    for simplex in tri.simplices:
        p1, p2, p3 = P[simplex[0]], P[simplex[1]], P[simplex[2]]
        
        # Circumcenter of spherical triangle p1, p2, p3
        # c is orthogonal to (p1-p2) and (p2-p3)
        v1 = p1 - p2
        v2 = p2 - p3
        c = np.cross(v1, v2)
        
        # Check if c is zero (degenerate triangle)
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-10:
            continue
            
        c = c / norm_c
        
        # The circumradius is the angular distance from c to any vertex
        # We need to ensure c is on the correct side (facing the triangle)
        # The dot product c.p1 should be positive for the "inner" circumcenter
        # If c.p1 < 0, the circumcenter is on the other side, and the radius is pi - arccos(abs(c.p1))
        # Actually, the covering radius is determined by the empty cap.
        # The circumcenter of the Delaunay triangle defines the vertex of the Voronoi cell.
        # The radius is arccos(c.p1) if c.p1 > 0, else pi - arccos(-c.p1)?
        # Let's just compute the angle.
        dot = np.clip(np.dot(c, p1), -1.0, 1.0)
        r = np.arccos(dot)
        
        # If the circumcenter is "outside" the triangle in a way that the cap is not empty?
        # For Delaunay triangulation on a sphere, the circumcircle (great circle) is empty.
        # The vertex of the Voronoi diagram is the circumcenter.
        # The distance from the vertex to the sites is the circumradius.
        # We want the maximum of these distances.
        
        if r > max_r:
            max_r = r
            
    return max_r

def _move_towards_circumcenter(P, idx, c, step):
    """
    Move points at indices idx towards circumcenter c by 'step'.
    """
    P_new = P.copy()
    for i in idx:
        # Geodesic interpolation
        # p_new = p + step * (c - p)
        # Then normalize
        p_new = P[i] + step * (c - P[i])
        norm = np.linalg.norm(p_new)
        if norm > 1e-10:
            P_new[i] = p_new / norm
    return P_new

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 10s margin
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    current_r = _covering_radius(P)
    best_P = P.copy()
    best_r = current_r
    
    # Simulated Annealing / Local Search
    T = 0.05 # Initial temperature
    step_size = 0.1
    
    for iter_idx in range(2000):
        if time.time() > deadline:
            break
            
        # Compute Delaunay
        try:
            tri = Delaunay(P)
        except Exception:
            break
            
        # Find the worst triangle
        max_r_local = 0.0
        worst_simplex = None
        worst_c = None
        
        for simplex in tri.simplices:
            p1, p2, p3 = P[simplex[0]], P[simplex[1]], P[simplex[2]]
            v1 = p1 - p2
            v2 = p2 - p3
            c = np.cross(v1, v2)
            norm_c = np.linalg.norm(c)
            if norm_c < 1e-10:
                continue
            c = c / norm_c
            dot = np.clip(np.dot(c, p1), -1.0, 1.0)
            r = np.arccos(dot)
            if r > max_r_local:
                max_r_local = r
                worst_simplex = simplex
                worst_c = c
                
        if worst_simplex is None:
            break
            
        # Move the vertices of the worst triangle towards the circumcenter
        P_new = _move_towards_circumcenter(P, worst_simplex, worst_c, step_size)
        
        new_r = _covering_radius(P_new)
        
        # Acceptance criterion: Accept if better, or with probability exp(-(new_r - current_r)/T)
        if new_r < current_r:
            P = P_new
            current_r = new_r
            if current_r < best_r:
                best_r = current_r
                best_P = P.copy()
        else:
            delta = new_r - current_r
            if np.random.rand() < np.exp(-delta / T):
                P = P_new
                current_r = new_r
                
        # Cooling
        T *= 0.99
        
        # If step size is too large, we might oscillate. 
        # If we are stuck, we could reduce step size, but simple cooling is okay.
        
    # Final polish: Gradient descent on the worst point?
    # The current method is a discrete jump. Let's try a few steps of gradient descent on the worst facet.
    
    # Re-evaluate best_P
    return best_P

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
import time

N = 50

def _covering_radius(P):
    """
    Compute the covering radius of the configuration P.
    P is an (N, 3) array of unit vectors.
    """
    if len(P) < 3:
        return np.pi / 2
        
    try:
        tri = Delaunay(P)
    except Exception:
        return np.pi / 2

    max_r = 0.0
    for simplex in tri.simplices:
        p1, p2, p3 = P[simplex[0]], P[simplex[1]], P[simplex[2]]
        
        v1 = p1 - p2
        v2 = p2 - p3
        c = np.cross(v1, v2)
        
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-10:
            continue
            
        c = c / norm_c
        
        dot = np.clip(np.dot(c, p1), -1.0, 1.0)
        r = np.arccos(dot)
        
        if r > max_r:
            max_r = r
            
    return max_r

def _move_towards_circumcenter(P, idx, c, step):
    P_new = P.copy()
    for i in idx:
        p_new = P[i] + step * (c - P[i])
        norm = np.linalg.norm(p_new)
        if norm > 1e-10:
            P_new[i] = p_new / norm
    return P_new

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    current_r = _covering_radius(P)
    best_P = P.copy()
    best_r = current_r
    
    T = 0.05
    step_size = 0.05 # Smaller step for stability
    
    for iter_idx in range(3000):
        if time.time() > deadline:
            break
            
        try:
            tri = Delaunay(P)
        except Exception:
            break
            
        max_r_local = 0.0
        worst_simplex = None
        worst_c = None
        
        for simplex in tri.simplices:
            p1, p2, p3 = P[simplex[0]], P[simplex[1]], P[simplex[2]]
            v1 = p1 - p2
            v2 = p2 - p3
            c = np.cross(v1, v2)
            norm_c = np.linalg.norm(c)
            if norm_c < 1e-10:
                continue
            c = c / norm_c
            dot = np.clip(np.dot(c, p1), -1.0, 1.0)
            r = np.arccos(dot)
            if r > max_r_local:
                max_r_local = r
                worst_simplex = simplex
                worst_c = c
                
        if worst_simplex is None:
            break
            
        P_new = _move_towards_circumcenter(P, worst_simplex, worst_c, step_size)
        new_r = _covering_radius(P_new)
        
        if new_r < current_r:
            P = P_new
            current_r = new_r
            if current_r < best_r:
                best_r = current_r
                best_P = P.copy()
        else:
            delta = new_r - current_r
            if np.random.rand() < np.exp(-delta / T):
                P = P_new
                current_r = new_r
                
        T *= 0.995
        if T < 0.001:
            T = 0.001

    return best_P
# EVOLVE-BLOCK-END
