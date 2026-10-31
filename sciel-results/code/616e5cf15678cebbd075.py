import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import itertools

N = 11

# Precompute all triangle index triples
TRI_IDX = list(itertools.combinations(range(N), 3))
TRI_IDX = np.array(TRI_IDX)  # (165, 3)

def _compute_areas_from_distances(D):
    """Compute all triangle areas from pairwise distance matrix D using Cayley-Menger."""
    # For each triangle (i,j,k), area = 0.25 * sqrt(CM_det)
    # CM det for triangle: |0  d_ik^2  d_ij^2  0|
    #                       |d_ik^2  0  d_jk^2  0|
    #                       |d_ij^2  d_jk^2  0  0|
    #                       |0       0       0  1|
    # = d_ik^2 * d_jk^2 + d_ik^2 * d_ij^2 + d_jk^2 * d_ij^2 - 0.5*(d_ik^4 + d_jk^4 + d_ij^4)
    # Actually: 16*area^2 = 2*(a^2*b^2 + b^2*c^2 + c^2*a^2) - (a^4 + b^4 + c^4)
    
    i_idx = TRI_IDX[:, 0]
    j_idx = TRI_IDX[:, 1]
    k_idx = TRI_IDX[:, 2]
    
    a2 = D[i_idx, j_idx] ** 2
    b2 = D[j_idx, k_idx] ** 2
    c2 = D[i_idx, k_idx] ** 2
    
    # 16 * area^2
    sixteen_area_sq = 2.0 * (a2 * b2 + b2 * c2 + c2 * a2) - (a2**2 + b2**2 + c2**2)
    sixteen_area_sq = np.maximum(sixteen_area_sq, 0.0)
    
    areas = 0.25 * np.sqrt(sixteen_area_sq)
    return areas

def _min_area_from_distances(D):
    """Return minimum triangle area from pairwise distance matrix."""
    areas = _compute_areas_from_distances(D)
    return np.min(areas)

def _soft_min_from_distances(D, beta=50.0):
    """Soft minimum of triangle areas for smooth optimization."""
    areas = _compute_areas_from_distances(D)
    # Soft min: -log(sum(exp(-beta*a))) / beta
    a_max = np.max(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - a_max))))
    soft_min = a_max - log_sum / beta
    return soft_min

def _cartesian_to_distance_matrix(pts):
    """Compute pairwise distance matrix from Cartesian coordinates."""
    n = len(pts)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            D[i, j] = d
            D[j, i] = d
    return D

def _distance_matrix_to_cartesian(D, ref_triangle=None):
    """
    Embed distance matrix into 2D using the reference triangle.
    We place point 0 at origin, point 1 on x-axis, and use the reference triangle's
    third vertex to determine orientation.
    
    For points inside the unit equilateral triangle, we use the constraint that
    all points must be inside the triangle.
    """
    n = len(D)
    pts = np.zeros((n, 2))
    
    # Place point 0 at (0,0)
    pts[0] = [0.0, 0.0]
    
    # Place point 1 on x-axis
    pts[1] = [D[0, 1], 0.0]
    
    # Place remaining points using trilateration
    for i in range(2, n):
        # Trilateration: find point given distances to pts[0] and pts[1]
        d0 = D[0, i]
        d1 = D[1, i]
        
        # Distance between reference points
        d_ref = D[0, 1]
        
        if d_ref < 1e-12:
            # Degenerate case
            pts[i] = [d0, 0.0]
            continue
        
        # Standard trilateration
        a = d1
        b = d0
        c = d_ref
        
        # x coordinate relative to ref[0]
        x = (b**2 - a**2 + c**2) / (2.0 * c)
        
        # y coordinate
        y_sq = b**2 - x**2
        if y_sq < 0:
            y_sq = 0.0
        y = np.sqrt(y_sq)
        
        # Choose the side that keeps points inside the triangle (positive y)
        pts[i] = [x, y]
    
    return pts

def _project_to_triangle(pts):
    """Project points to be inside or on the unit equilateral triangle."""
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])
    
    n = len(pts)
    result = pts.copy()
    
    for i in range(n):
        p = result[i]
        
        # Check if point is inside the triangle using barycentric coordinates
        # p = u*A + v*B + w*C where u+v+w=1, u,v,w >= 0
        
        # Solve for barycentric coordinates
        # p = v*(B-A) + w*(C-A)
        # [px]   [1    0.5  ] [v]
        # [py] = [0    sqrt(3)/2] [w]
        
        px, py = p
        H = np.sqrt(3)/2
        
        # From the linear system:
        # v + 0.5*w = px
        # H*w = py
        # => w = py / H
        # => v = px - 0.5 * py / H
        
        w = py / H
        v = px - 0.5 * w
        u = 1.0 - v - w
        
        # If all barycentric coords are non-negative, point is inside
        if u >= -1e-10 and v >= -1e-10 and w >= -1e-10:
            # Clamp small negative values to zero
            u = max(u, 0.0)
            v = max(v, 0.0)
            w = max(w, 0.0)
            # Renormalize
            s = u + v + w
            if s > 1e-12:
                u /= s
                v /= s
                w /= s
            result[i] = u * A + v * B + w * C
        else:
            # Project to nearest point on triangle boundary
            # This is a simplified projection; for better results, use proper projection
            # For now, clamp to the triangle by adjusting barycentric coords
            if u < 0:
                u = 0
            if v < 0:
                v = 0
            if w < 0:
                w = 0
            s = u + v + w
            if s > 1e-12:
                u /= s
                v /= s
                w /= s
            result[i] = u * A + v * B + w * C
    
    return result

def _objective_from_distances(D_flat):
    """
    Objective: minimize negative soft-min area from distance matrix.
    D_flat is a flattened upper triangle of the distance matrix.
    """
    n = N
    D = np.zeros((n, n))
    
    # Reconstruct upper triangle
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            D[i, j] = D_flat[idx]
            D[j, i] = D_flat[idx]
            idx += 1
    
    # Penalty for invalid distances (negative or too large)
    penalty = 0.0
    upper = D[np.triu_indices(n, k=1)]
    if np.any(upper < 0):
        penalty += 1000.0 * np.sum(upper[upper < 0]**2)
    # Maximum distance in unit triangle is 1.0
    if np.any(upper > 1.0 + 1e-8):
        penalty += 1000.0 * np.sum((upper[upper > 1.0 + 1e-8] - 1.0)**2)
    
    # Soft minimum of triangle areas
    soft_min = _soft_min_from_distances(D, beta=50.0)
    
    return -soft_min + penalty

def _generate_random_distance_matrix(rng):
    """Generate a random valid distance matrix."""
    n = N
    D = np.zeros((n, n))
    
    # Start with a random configuration inside the triangle
    # and compute its distance matrix
    bary = rng.dirichlet(np.ones(3), size=n)
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])
    
    pts = np.array([bary[i, 0]*A + bary[i, 1]*B + bary[i, 2]*C for i in range(n)])
    
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            D[i, j] = d
            D[j, i] = d
    
    # Flatten upper triangle
    flat = []
    for i in range(n):
        for j in range(i+1, n):
            flat.append(D[i, j])
    
    return np.array(flat)

def _distance_matrix_to_valid_config(D_flat):
    """
    Convert a distance matrix to a valid point configuration inside the triangle.
    This involves embedding the metric space into 2D and projecting to the triangle.
    """
    n = N
    D = np.zeros((n, n))
    
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            D[i, j] = max(D_flat[idx], 0.0)
            D[j, i] = D[i, j]
            idx += 1
    
    # Embed into 2D using classical multidimensional scaling (CMDS)
    # Step 1: Double centering
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D**2) @ J
    
    # Step 2: Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(B)
    
    # Take the top 2 eigenvalues
    # Sort in descending order
    idx_sort = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx_sort]
    eigvecs = eigvecs[:, idx_sort]
    
    # Use top 2 dimensions
    if eigvals[0] > 0:
        pts = eigvecs[:, :2] * np.sqrt(np.maximum(eigvals[:2], 0))
    else:
        # Fallback: random configuration
        rng = np.random.RandomState(42)
        bary = rng.dirichlet(np.ones(3), size=n)
        A = np.array([0.0, 0.0])
        B = np.array([1.0, 0.0])
        C = np.array([0.5, np.sqrt(3)/2])
        pts = np.array([bary[i, 0]*A + bary[i, 1]*B + bary[i, 2]*C for i in range(n)])
    
    # Project to unit equilateral triangle
    pts = _project_to_triangle(pts)
    
    # Convert to barycentric coordinates for consistency
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])
    H = np.sqrt(3)/2
    
    bary = np.zeros((n, 2))
    for i in range(n):
        px, py = pts[i]
        w = py / H
        v = px - 0.5 * w
        bary[i] = [v, w]
    
    return bary

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_score = -float('inf')
    best_points = None
    
    # Phase 1: Optimize in distance space
    n_distances = N * (N - 1) // 2
    bounds = [(0, 1.0)] * n_distances
    
    rng = np.random.RandomState(42)
    
    # Generate initial distance matrices
    inits_dist = []
    for _ in range(20):
        init_d = _generate_random_distance_matrix(rng)
        inits_dist.append(init_d)
    
    # Add a structured initialization based on hexagonal lattice
    # Hexagonal lattice points in the triangle
    hex_bary = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
        [1/3, 1/3], [2/3, 0.0], [0.0, 2/3],
        [2/3, 1/3], [1/3, 2/3]
    ])
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])
    hex_pts = np.array([hex_bary[i, 0]*A + hex_bary[i, 1]*B + (1-hex_bary[i, 0]-hex_bary[i, 1])*C for i in range(N)])
    hex_D = _cartesian_to_distance_matrix(hex_pts)
    hex_flat = []
    for i in range(N):
        for j in range(i+1, N):
            hex_flat.append(hex_D[i, j])
    inits_dist.append(np.array(hex_flat))
    
    # Local optimization in distance space
    for init_d in inits_dist:
        if time.time() > deadline - 15:
            break
        
        try:
            result = minimize(
                _objective_from_distances,
                init_d,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            D_opt = result.x
            bary_opt = _distance_matrix_to_valid_config(D_opt)
            
            # Compute actual score from the Cartesian configuration
            A = np.array([0.0, 0.0])
            B = np.array([1.0, 0.0])
            C = np.array([0.5, np.sqrt(3)/2])
            pts_opt = np.array([bary_opt[i, 0]*A + bary_opt[i, 1]*B + (1-bary_opt[i, 0]-bary_opt[i, 1])*C for i in range(N)])
            
            D_actual = _cartesian_to_distance_matrix(pts_opt)
            score = _min_area_from_distances(D_actual)
            
            if score > best_score:
                best_score = score
                best_points = bary_opt.copy()
        except Exception:
            pass
    
    # Phase 2: Differential Evolution in Cartesian space (fallback and complement)
    if time.time() < deadline - 30:
        # Use Cartesian coordinates directly for DE
        def _cartesian_objective(params):
            pts = params.reshape(N, 2)
            # Project to triangle
            pts = _project_to_triangle(pts)
            D = _cartesian_to_distance_matrix(pts)
            return -_min_area_from_distances(D)
        
        cart_bounds = [(-0.5, 1.5)] * (2 * N)
        
        try:
            de_result = differential_evolution(
                _cartesian_objective,
                cart_bounds,
                maxiter=100,
                popsize=20,
                tol=1e-8,
                seed=42,
                init='latinhypercube',
                polish=False,
                updating='immediate'
            )
            
            pts_de = de_result.x.reshape(N, 2)
            pts_de = _project_to_triangle(pts_de)
            
            # Convert to barycentric
            H = np.sqrt(3)/2
            bary_de = np.zeros((N, 2))
            for i in range(N):
                px, py = pts_de[i]
                w = py / H
                v = px - 0.5 * w
                bary_de[i] = [v, w]
            
            D_de = _cartesian_to_distance_matrix(pts_de)
            score_de = _min_area_from_distances(D_de)
            
            if score_de > best_score:
                best_score = score_de
                best_points = bary_de.copy()
        except Exception:
            pass
    
    # Phase 3: Local refinement in Cartesian space
    if best_points is not None and time.time() < deadline - 10:
        A = np.array([0.0, 0.0])
        B = np.array([1.0, 0.0])
        C = np.array([0.5, np.sqrt(3)/2])
        H = np.sqrt(3)/2
        
        current_pts = np.array([best_points[i, 0]*A + best_points[i, 1]*B + (1-best_points[i, 0]-best_points[i, 1])*C for i in range(N)])
        
        # Nelder-Mead refinement
        try:
            nm_result = minimize(
                _cartesian_objective,
                current_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            pts_nm = nm_result.x.reshape(N, 2)
            pts_nm = _project_to_triangle(pts_nm)
            
            D_nm = _cartesian_to_distance_matrix(pts_nm)
            score_nm = _min_area_from_distances(D_nm)
            
            if score_nm > best_score:
                best_score = score_nm
                # Convert back to barycentric
                bary_nm = np.zeros((N, 2))
                for i in range(N):
                    px, py = pts_nm[i]
                    w = py / H
                    v = px - 0.5 * w
                    bary_nm[i] = [v, w]
                best_points = bary_nm.copy()
        except Exception:
            pass
    
    # Fallback if no valid points found
    if best_points is None:
        best_points = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
            [1/3, 1/3], [2/3, 0.0], [0.0, 2/3],
            [2/3, 1/3], [1/3, 2/3]
        ])
    
    # Convert barycentric to Cartesian for final output
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])
    
    pts = []
    for i in range(N):
        u, v = best_points[i]
        w = 1 - u - v
        p = u * A + v * B + w * C
        pts.append(p)
    
    return np.array(pts)
