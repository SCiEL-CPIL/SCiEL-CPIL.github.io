# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time

def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points, metric='euclidean')
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or np.isnan(d_min) or np.isnan(d_max):
        return 0.0
    return (d_min / d_max) ** 2

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Strategy: Symmetry-reduced optimization.
    We assume the optimal configuration for 14 points in 3D respects T_d symmetry (tetrahedral).
    The 14 points can be decomposed into 5 orbits under T_d:
    1. 4 vertices of a regular tetrahedron (Orbit A)
    2. 4 face centers of the tetrahedron (Orbit B)
    3. 6 edge midpoints of the tetrahedron (Orbit C)
    
    Total: 4 + 4 + 6 = 14 points.
    
    We parameterize each orbit by a radial scaling factor relative to the ideal tetrahedron geometry.
    Let the ideal tetrahedron have vertices at distance R from the origin.
    Orbit A: 4 points at distance s_A * R
    Orbit B: 4 points at distance s_B * R (face centers are at distance R/sqrt(3) in ideal tet)
    Orbit C: 6 points at distance s_C * R (edge midpoints are at distance R/sqrt(2) in ideal tet)
    
    Actually, to allow for full flexibility, we define the base geometry and then scale each orbit radially.
    The base geometry is the regular tetrahedron inscribed in a unit sphere.
    Vertices: (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1) normalized.
    Face centers: Negatives of vertices (since centroid is origin).
    Edge midpoints: Midpoints of edges.
    
    We optimize 3 scaling factors: s_A, s_B, s_C.
    Wait, the prompt suggests "5 symmetry orbits". Let's re-evaluate.
    T_d has 24 elements.
    Orbits:
    - 4 vertices
    - 4 face centers (which are equivalent to vertices of the dual tetrahedron)
    - 6 edge midpoints
    
    Are there other orbits?
    If we allow the points to not be strictly on the "ideal" radial lines, we might break symmetry.
    However, the directive says "optimize only the single radial scaling factor of each of the 5 symmetry orbits".
    This implies there are 5 orbits. My count is 3.
    Perhaps the "5 orbits" refers to a different decomposition or a less symmetric group?
    Or maybe the 6 edge midpoints split into two orbits of 3? No, T_d acts transitively on edges.
    
    Let's look at the inspiration text again: "4 vertices + 10 edge midpoints".
    A tetrahedron has 6 edges. 10 midpoints is impossible for a standard tetrahedron.
    Maybe it's a different polytope?
    "subdivided into 14 points using a high-symmetry combinatorial template (e.g., 4 vertices + 10 edge midpoints)"
    This example is geometrically incoherent for a single tetrahedron.
    
    Let's stick to the T_d symmetry with 3 orbits (4+4+6=14). This reduces the space from 42D to 3D.
    If 3D is too restrictive, we can add a global rotation, but T_d is rigid.
    
    Actually, let's consider the group D_4h or similar?
    Let's try a different high-symmetry set:
    Cube (8 vertices) + Octahedron (6 vertices) = 14 points.
    This is a very common configuration (Cubic/Octahedral symmetry).
    Orbits under O_h (48 elements):
    1. 8 Cube vertices: (±1, ±1, ±1)
    2. 6 Octahedron vertices: (±1, 0, 0), (0, ±1, 0), (0, 0, ±1)
    
    Total: 8 + 6 = 14 points.
    This fits the count perfectly.
    Symmetry group: O_h.
    Orbits:
    1. Cube vertices (8 points)
    2. Octahedron vertices (6 points)
    
    This gives 2 radial scaling factors.
    Let's optimize s_cube and s_octa.
    
    Let's implement this "Cube + Octahedron" strategy.
    Base points:
    Cube: (±1, ±1, ±1)
    Octa: (±1, 0, 0), (0, ±1, 0), (0, 0, ±1)
    
    Scaled points:
    P_cube = s_c * (±1, ±1, ±1)
    P_octa = s_o * (±1, 0, 0), ...
    
    We will optimize s_c and s_o.
    This is a 2D optimization problem.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin

    # Define the 14 base points for Cube + Octahedron
    # Cube vertices
    cube_verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
    ])
    
    # Octahedron vertices
    octa_verts = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ])
    
    def get_points(s_c, s_o):
        pts_c = s_c * cube_verts
        pts_o = s_o * octa_verts
        return np.vstack([pts_c, pts_o])

    def objective(params):
        s_c, s_o = params
        pts = get_points(s_c, s_o)
        return -_score(pts)

    # Initial guess: s_c = 1, s_o = 1
    # Let's run a grid search or Nelder-Mead on the 2D space
    # Since the space is small and smooth, we can be thorough.
    
    best_score = -np.inf
    best_params = None
    
    # Grid search to find a good starting point
    # s_c and s_o are likely close to each other or related by the geometry.
    # Let's search s_c in [0.5, 2.0] and s_o in [0.5, 2.0]
    
    # To save time, we'll use scipy.optimize.minimize with multiple starts.
    from scipy.optimize import minimize
    
    starts = [
        [1.0, 1.0],
        [1.0, 0.8],
        [1.0, 1.2],
        [0.8, 1.0],
        [1.2, 1.0],
        [1.5, 1.0],
        [1.0, 1.5],
        [0.5, 0.5],
        [2.0, 2.0]
    ]
    
    for start in starts:
        if time.time() > deadline:
            break
            
        res = minimize(objective, start, method='Nelder-Mead', 
                       options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 1000})
        
        if res.success:
            s_c, s_o = res.x
            pts = get_points(s_c, s_o)
            sc = _score(pts)
            if sc > best_score:
                best_score = sc
                best_params = (s_c, s_o)
                best_points = pts

    if best_points is None:
        # Fallback to uniform
        best_points = get_points(1.0, 1.0)
        best_score = _score(best_points)

    # Refine with a finer local search around the best found
    if time.time() < deadline - 10:
        s_c, s_o = best_params
        
        # Simple coordinate descent polish
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for step_size in [0.01, 0.001, 0.0001]:
                for delta in [-step_size, step_size]:
                    new_s_c = s_c + delta
                    new_s_o = s_o
                    if new_s_c > 0:
                        pts = get_points(new_s_c, new_s_o)
                        sc = _score(pts)
                        if sc > best_score:
                            best_score = sc
                            best_params = (new_s_c, new_s_o)
                            best_points = pts
                            s_c = new_s_c
                            improved = True
                            
                    new_s_c = s_c
                    new_s_o = s_o + delta
                    if new_s_o > 0:
                        pts = get_points(new_s_c, new_s_o)
                        sc = _score(pts)
                        if sc > best_score:
                            best_score = sc
                            best_params = (new_s_c, new_s_o)
                            best_points = pts
                            s_o = new_s_o
                            improved = True

    return best_points
# EVOLVE-BLOCK-END
