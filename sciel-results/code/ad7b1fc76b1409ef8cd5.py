"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def _covering_radius_fast(points):
    """
    Compute the covering radius in degrees for points on the unit sphere.
    Uses vectorized circumradius computation for triangular facets.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        
        max_circumradius = 0.0
        
        # Handle triangular facets (most common in 3D convex hull)
        tri_facets = [f for f in facets if len(f) == 3]
        
        if len(tri_facets) > 0:
            tri_facets = np.array(tri_facets)
            a = points[tri_facets[:, 0]]
            b = points[tri_facets[:, 1]]
            c = points[tri_facets[:, 2]]
            
            # Compute cross products
            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            
            # Triple product: a . (b x c)
            triple = np.einsum('ij,ij->i', a, np.cross(b, c))
            denom = 4.0 * np.abs(triple)
            
            # Avoid division by zero
            valid = denom > 1e-15
            if np.any(valid):
                norms_ab = np.linalg.norm(ab[valid], axis=1)
                norms_bc = np.linalg.norm(bc[valid], axis=1)
                norms_ca = np.linalg.norm(ca[valid], axis=1)
                
                sin_R = (norms_ab * norms_bc * norms_ca) / denom[valid]
                sin_R = np.clip(sin_R, 0.0, 1.0)
                R = np.arcsin(sin_R)
                max_circumradius = max(max_circumradius, np.max(R))
        
        # Handle non-triangular facets (rare, but possible)
        non_tri_facets = [f for f in facets if len(f) != 3]
        for simplex in non_tri_facets:
            pts = points[simplex]
            n_pts = len(pts)
            if n_pts < 3:
                continue
            if n_pts == 4:
                a, b, c, d = pts[0], pts[1], pts[2], pts[3]
                v1 = a - b
                v2 = a - c
                v3 = a - d
                x = np.cross(v1, np.cross(v2, v3))
                nx = np.linalg.norm(x)
                if nx < 1e-15:
                    continue
                x = x / nx
                cos_R = np.clip(np.dot(x, a), -1.0, 1.0)
                R = np.arccos(cos_R)
                max_circumradius = max(max_circumradius, R)
            else:
                centroid = np.mean(pts, axis=0)
                nc = np.linalg.norm(centroid)
                if nc < 1e-15:
                    continue
                centroid = centroid / nc
                for p in pts:
                    cos_R = np.clip(np.dot(centroid, p), -1.0, 1.0)
                    R = np.arccos(cos_R)
                    max_circumradius = max(max_circumradius, R)
        
        return np.degrees(max_circumradius)
    except Exception:
        return np.inf

def _fibonacci_sphere(n):
    """Generate points on a sphere using the Fibonacci lattice."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _simulate_annealing(points, max_time, T_start=0.1, T_end=1e-5, alpha=0.999):
    """
    Perform Simulated Annealing to minimize the covering radius.
    """
    start_time = time.time()
    current_points = points.copy()
    current_radius = _covering_radius_fast(current_points)
    best_points = current_points.copy()
    best_radius = current_radius
    T = T_start
    
    while time.time() - start_time < max_time:
        if T < T_end:
            break
            
        # Propose a move: pick a random point and perturb it
        idx = np.random.randint(N)
        perturb = np.random.randn(3) * 0.05
        new_point = current_points[idx] + perturb
        new_point /= np.linalg.norm(new_point)
        
        # Apply move
        old_point = current_points[idx].copy()
        current_points[idx] = new_point
        
        new_radius = _covering_radius_fast(current_points)
        
        # Acceptance criterion
        delta = new_radius - current_radius
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_radius = new_radius
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = current_points.copy()
        else:
            # Revert
            current_points[idx] = old_point
            
        # Cool down
        T *= alpha
        
    return best_points, best_radius

def _local_hill_climb(points, max_time):
    """
    Fast local hill climbing to polish the result.
    """
    start_time = time.time()
    current_points = points.copy()
    current_radius = _covering_radius_fast(current_points)
    best_points = current_points.copy()
    best_radius = current_radius
    step_size = 0.01
    
    while time.time() - start_time < max_time:
        improved = False
        # Try moving each point
        for i in range(N):
            if time.time() - start_time > max_time:
                break
            # Try a few random directions
            for _ in range(3):
                perturb = np.random.randn(3) * step_size
                new_point = current_points[i] + perturb
                new_point /= np.linalg.norm(new_point)
                
                old_point = current_points[i].copy()
                current_points[i] = new_point
                
                new_radius = _covering_radius_fast(current_points)
                
                if new_radius < current_radius:
                    current_radius = new_radius
                    if new_radius < best_radius:
                        best_radius = new_radius
                        best_points = current_points.copy()
                    improved = True
                else:
                    current_points[i] = old_point
        
        if not improved:
            step_size *= 0.8
            if step_size < 1e-6:
                break
                
    return best_points, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Start with a high-quality Fibonacci lattice
    initial_points = _fibonacci_sphere(N)
    
    best_points = initial_points
    best_radius = _covering_radius_fast(best_points)
    
    # Phase 1: Simulated Annealing from the Fibonacci start
    # This explores the landscape broadly
    sa_points, sa_radius = _simulate_annealing(best_points, max_time=60.0, T_start=0.05, T_end=1e-5, alpha=0.9995)
    if sa_radius < best_radius:
        best_points = sa_points
        best_radius = sa_radius
        
    # Phase 2: Local Hill Climbing to polish
    if time.time() < deadline - 10.0:
        lc_points, lc_radius = _local_hill_climb(best_points, max_time=30.0)
        if lc_radius < best_radius:
            best_points = lc_points
            best_radius = lc_radius
            
    # Phase 3: Another round of SA from the best found, with lower T
    if time.time() < deadline - 5.0:
        sa2_points, sa2_radius = _simulate_annealing(best_points, max_time=20.0, T_start=0.01, T_end=1e-6, alpha=0.999)
        if sa2_radius < best_radius:
            best_points = sa2_points
            best_radius = sa2_radius
            
        # Final polish
        if time.time() < deadline - 2.0:
            lc2_points, lc2_radius = _local_hill_climb(best_points, max_time=5.0)
            if lc2_radius < best_radius:
                best_points = lc2_points
                best_radius = lc2_radius
                
    return best_points
# EVOLVE-BLOCK-END
