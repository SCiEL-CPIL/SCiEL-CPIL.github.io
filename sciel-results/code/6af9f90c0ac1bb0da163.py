import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import itertools
import time

N_POINTS = 13
TRIANGLE_IDX = np.array(list(itertools.combinations(range(N_POINTS), 3)))

def sdp_initialization(time_budget=120.0):
    """
    Use an SDP-inspired approach: maximize minimum squared triangle area
    via a Gram matrix formulation, then extract points via Cholesky.
    
    The idea: for points p_0, ..., p_{n-1}, define differences d_i = p_i - p_0.
    The Gram matrix G = D D^T where D is (n-1) x 2.
    Triangle areas can be expressed in terms of G entries.
    We want to maximize the minimum triangle area subject to G being PSD and rank(G) <= 2.
    
    This is a relaxation; we solve it iteratively with projected gradient.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Work with centered points, p_0 at origin
    # d_i = p_i - p_0 for i = 1, ..., 12
    # G = D D^T, 12x12 PSD, rank <= 2
    # Triangle (i,j,k) area = 0.5 * |d_i x d_j| for appropriate indices
    # Actually, triangle (a,b,c) with a=0: area = 0.5 * |d_b x d_c|
    # Triangle (a,b,c) with a>0: area = 0.5 * |(d_b - d_a) x (d_c - d_a)|
    
    # Let's use a different parameterization:
    # Place points as p_i = (x_i, y_i) directly, but use the Gram structure
    # to guide initialization.
    
    # Alternative: use the fact that for optimal Heilbronn configurations,
    # points tend to be near a regular polygon or grid-like structure.
    # Use SDP to find a good Gram matrix, then extract 2D coordinates.
    
    # Initialize G as a 12x12 PSD matrix (rank 2)
    # Start from a regular 12-gon (excluding center)
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    D_init = np.column_stack([np.cos(angles), np.sin(angles)])
    G = D_init @ D_init.T
    
    # Gradient ascent on G to maximize min triangle area
    # Project back to PSD rank-2 after each step
    
    # Compute all triangle areas from G
    # For triangle (0, i, j): area^2 = 0.25 * (G[i,i]*G[j,j] - G[i,j]^2)
    # For triangle (i, j, k) with i,j,k > 0:
    #   v1 = d_j - d_i, v2 = d_k - d_i
    #   area^2 = 0.25 * |v1 x v2|^2
    #   |v1 x v2|^2 = |d_j - d_i|^2 * |d_k - d_i|^2 - ((d_j-d_i)·(d_k-d_i))^2
    #   = (G[j,j]-2G[i,j]+G[i,i])*(G[k,k]-2G[i,k]+G[i,i]) - (G[j,k]-G[j,i]-G[i,k]+G[i,i])^2
    
    def compute_areas_from_G(G):
        """Compute all 286 triangle areas from Gram matrix G (12x12)."""
        areas = np.zeros(286)
        idx = 0
        for a, b, c in TRIANGLE_IDX:
            if a == 0:
                # Triangle (0, b, c): use d_b and d_c
                b_idx = b - 1
                c_idx = c - 1
                det_sq = G[b_idx, b_idx] * G[c_idx, c_idx] - G[b_idx, c_idx] ** 2
                det_sq = max(det_sq, 0)
                areas[idx] = 0.5 * np.sqrt(det_sq)
            elif b == 0:
                # Triangle (a, 0, c): use d_a and d_c
                a_idx = a - 1
                c_idx = c - 1
                det_sq = G[a_idx, a_idx] * G[c_idx, c_idx] - G[a_idx, c_idx] ** 2
                det_sq = max(det_sq, 0)
                areas[idx] = 0.5 * np.sqrt(det_sq)
            elif c == 0:
                # Triangle (a, b, 0): use d_a and d_b
                a_idx = a - 1
                b_idx = b - 1
                det_sq = G[a_idx, a_idx] * G[b_idx, b_idx] - G[a_idx, b_idx] ** 2
                det_sq = max(det_sq, 0)
                areas[idx] = 0.5 * np.sqrt(det_sq)
            else:
                # Triangle (a, b, c) all > 0
                a_idx = a - 1
                b_idx = b - 1
                c_idx = c - 1
                # v1 = d_b - d_a, v2 = d_c - d_a
                v1_sq = G[b_idx, b_idx] - 2*G[a_idx, b_idx] + G[a_idx, a_idx]
                v2_sq = G[c_idx, c_idx] - 2*G[a_idx, c_idx] + G[a_idx, a_idx]
                dot = G[b_idx, c_idx] - G[b_idx, a_idx] - G[a_idx, c_idx] + G[a_idx, a_idx]
                det_sq = v1_sq * v2_sq - dot ** 2
                det_sq = max(det_sq, 0)
                areas[idx] = 0.5 * np.sqrt(det_sq)
            idx += 1
        return areas
    
    # Project G to be PSD with rank <= 2
    def project_psd_rank2(G):
        """Project G to nearest PSD matrix with rank <= 2."""
        try:
            eigvals, eigvecs = np.linalg.eigh(G)
            # Keep top 2 eigenvalues
            eigvals[2:] = 0
            G_proj = eigvecs @ np.diag(eigvals) @ eigvecs.T
            # Symmetrize
            G_proj = (G_proj + G_proj.T) / 2
            return G_proj
        except:
            return G
    
    # Iterative optimization
    lr = 0.01
    best_G = G.copy()
    best_min_area = np.min(compute_areas_from_G(G))
    
    for iteration in range(2000):
        if time.time() > deadline - 5:
            break
            
        areas = compute_areas_from_G(G)
        min_area = np.min(areas)
        
        if min_area > best_min_area:
            best_min_area = min_area
            best_G = G.copy()
        
        # Gradient of min area w.r.t. G
        # Use soft-min for smoothness
        T = 0.01
        x = -areas / T
        max_x = np.max(x)
        weights = np.exp(x - max_x)
        weights /= np.sum(weights)
        
        # Gradient of each area w.r.t. G
        # This is complex; use numerical gradient for the soft-min
        # Or use subgradient of min
        
        # For now, use a simpler approach: perturb G in direction that increases min area
        # Numerical gradient
        grad = np.zeros_like(G)
        eps = 1e-4
        for i in range(12):
            for j in range(i, 12):
                G_plus = G.copy()
                G_plus[i, j] += eps
                G_plus[j, i] += eps
                G_plus = project_psd_rank2(G_plus)
                areas_plus = compute_areas_from_G(G_plus)
                soft_min_plus = -T * np.log(np.sum(np.exp(-areas_plus / T)))
                
                G_minus = G.copy()
                G_minus[i, j] -= eps
                G_minus[j, i] -= eps
                G_minus = project_psd_rank2(G_minus)
                areas_minus = compute_areas_from_G(G_minus)
                soft_min_minus = -T * np.log(np.sum(np.exp(-areas_minus / T)))
                
                grad[i, j] = (soft_min_plus - soft_min_minus) / (2 * eps)
                grad[j, i] = grad[i, j]
        
        # Update G
        G = G + lr * grad
        G = project_psd_rank2(G)
        
        # Normalize: scale G so that trace(G) = 12 (unit variance)
        trace_G = np.trace(G)
        if trace_G > 1e-10:
            G = G * (12.0 / trace_G)
    
    # Extract points from best_G
    try:
        eigvals, eigvecs = np.linalg.eigh(best_G)
        # Keep top 2 components
        D = eigvecs[:, :2] * np.sqrt(np.maximum(eigvals[:2], 0))
    except:
        D = D_init
    
    # Construct full point set: p_0 = (0,0), p_i = D[i-1] for i=1..12
    points = np.vstack([np.array([0.0, 0.0]), D])
    
    # Center the points
    points = points - np.mean(points, axis=0)
    
    # Normalize
    hull = ConvexHull(points)
    area = hull.volume
    if area > 1e-10:
        points = points * (1.0 / np.sqrt(area))
    
    return points

def compute_areas(points):
    """
    Compute areas of all 286 triangles.
    Returns an array of shape (286,).
    """
    p1 = points[TRIANGLE_IDX[:, 0]]
    p2 = points[TRIANGLE_IDX[:, 1]]
    p3 = points[TRIANGLE_IDX[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    return areas

def compute_hull_area(points):
    """
    Compute the area of the convex hull.
    Returns 0.0 if hull calculation fails.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def compute_score(points):
    """
    Compute the Heilbronn score: A_min / A_hull.
    """
    if len(points) < 3:
        return 0.0
    
    A_hull = compute_hull_area(points)
    if A_hull <= 1e-10:
        return 0.0
        
    areas = compute_areas(points)
    A_min = np.min(areas)
    
    return A_min / A_hull

def soft_min_objective(flat_points, T=0.1):
    """
    Objective function for optimization: negative soft-min of triangle areas,
    normalized by hull area.
    
    We want to MAXIMIZE A_min / A_hull.
    So we MINIMIZE - (A_min / A_hull).
    
    Using soft-min: S = -1/T * log(sum(exp(-A_i/T))) approximates -min(A_i).
    To handle the ratio, we compute the soft_min_area and divide by hull_area.
    If hull_area is 0, return a large penalty.
    """
    points = flat_points.reshape(N_POINTS, 2)
    
    # Basic validity check
    if not np.all(np.isfinite(points)):
        return 1e10
        
    A_hull = compute_hull_area(points)
    if A_hull < 1e-8:
        return 1e10

    areas = compute_areas(points)
    
    # Avoid log(0)
    areas = np.maximum(areas, 1e-10)
    
    # Soft min calculation
    # log(sum(exp(-A/T))) = -T * logsumexp(-A/T)
    # We want -soft_min = - ( -T * logsumexp(-A/T) ) ? No.
    # Soft min is defined as -T * log(sum(exp(-A_i/T))).
    # This is a lower bound on the min. As T->0, it approaches min(A_i).
    # We want to maximize the min area. So we minimize -SoftMinArea.
    
    # Numerical stability for logsumexp
    # max_val = np.max(areas) # not needed for -A/T if T is small and A positive
    # Let's use scipy.special.logsumexp if available, else manual
    # Manual:
    # log_sum_exp(x) = max(x) + log(sum(exp(x - max(x))))
    x = -areas / T
    max_x = np.max(x)
    log_sum_exp = max_x + np.log(np.sum(np.exp(x - max_x)))
    soft_min_area = -T * log_sum_exp
    
    score = soft_min_area / A_hull
    
    # We want to maximize score, so minimize -score
    return -score

def polish_local(points, time_limit=30.0):
    """
    Local optimization using Nelder-Mead on the hard-min objective.
    """
    start_time = time.time()
    
    # Initial score
    current_score = compute_score(points)
    best_score = current_score
    best_points = points.copy()
    
    # Objective to minimize: -score
    def obj(flat_pts):
        if time.time() > start_time + time_limit:
            return 1e10
        pts = flat_pts.reshape(N_POINTS, 2)
        s = compute_score(pts)
        if s < 1e-10:
            return 1e10
        return -s
    
    # Run Nelder-Mead
    # We might need to run multiple times with different initial perturbations if it gets stuck
    for i in range(3):
        if time.time() > start_time + time_limit:
            break
            
        # Slight perturbation for diversity in restarts
        if i > 0:
            rng = np.random.default_rng(i)
            perturbed = points + rng.normal(0, 0.01, size=points.shape)
        else:
            perturbed = points
            
        try:
            res = minimize(obj, perturbed.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
            
            if res.success and res.fun < -best_score:
                best_score = -res.fun
                best_points = res.x.reshape(N_POINTS, 2)
        except Exception:
            pass
            
    return best_points, best_score

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area ratio.
    Uses SDP-inspired initialization, Differential Evolution, and local polish.
    """
    start_time = time.time()
    # Safety margin: return 10s before the 600s limit
    deadline = start_time + 590.0
    
    best_points_overall = None
    best_score_overall = -1.0
    
    # Strategy 0: SDP-inspired initialization
    if time.time() < deadline - 180:
        print("Running SDP-inspired initialization...")
        try:
            sdp_pts = sdp_initialization(time_budget=120.0)
            sdp_score = compute_score(sdp_pts)
            if sdp_score > best_score_overall:
                best_score_overall = sdp_score
                best_points_overall = sdp_pts.copy()
                print(f"SDP init score: {sdp_score}")
        except Exception as e:
            print(f"SDP init failed: {e}")
    
    # Strategy 1: Global Search with Differential Evolution
    # Boundaries: Points roughly in [-1, 1] to keep hull area manageable
    # We can center the points later
    bounds = [(-1.0, 1.0)] * (N_POINTS * 2)
    
    # DE parameters
    # popsize=10 means 10*26 = 260 evaluations per generation
    # maxiter=100 means 26000 evaluations.
    # Each evaluation is fast (~1ms), so 26s.
    # We have ~590s. We can afford much more.
    # Let's try popsize=20, maxiter=200. 20*26*200 = 104,000 evals. ~100s.
    # We can run multiple DE instances with different T (temperature) for soft-min.
    
    # We will run DE with a fixed T, then polish.
    # A good T is small enough to approximate min, but large enough for smoothness.
    # T=0.1 is a reasonable start.
    
    # To improve robustness, we can run DE for a few different T values or just one good one.
    # Let's stick to one strong DE run with adaptive T if possible, but DE doesn't support that directly.
    # We can run DE with T=0.05.
    
    T_de = 0.05
    print(f"Starting DE with T={T_de}")
    
    def de_objective(flat_pts):
        if time.time() > deadline - 60: # Stop DE if close to deadline
            return 1e10
        return soft_min_objective(flat_pts, T=T_de)
    
    try:
        # seed=42 for reproducibility
        res_de = differential_evolution(
            de_objective, 
            bounds, 
            seed=42, 
            maxiter=150, 
            popsize=15, 
            tol=1e-6, 
            mutation=(0.4, 1.0), 
            recombination=0.9, 
            polish=False, # We do our own polish
            init='sobol' # Better initial distribution
        )
        
        if res_de.success or res_de.fun < 1e9:
            de_points = res_de.x.reshape(N_POINTS, 2)
            de_score = compute_score(de_points)
            
            if de_score > best_score_overall:
                best_score_overall = de_score
                best_points_overall = de_points.copy()
                print(f"DE found score: {de_score}")
                
    except Exception as e:
        print(f"DE failed: {e}")
    
    # Strategy 2: Local Polish on the best result
    if best_points_overall is not None:
        remaining_time = deadline - time.time()
        if remaining_time > 10:
            print("Polishing best result...")
            polished_pts, polished_score = polish_local(best_points_overall, time_limit=min(remaining_time - 5, 60))
            if polished_score > best_score_overall:
                best_score_overall = polished_score
                best_points_overall = polished_pts
                print(f"Polished score: {polished_score}")
    
    # Strategy 2b: Additional local polish with more iterations
    if best_points_overall is not None:
        remaining_time = deadline - time.time()
        if remaining_time > 20:
            print("Additional fine polishing...")
            # Run more Nelder-Mead iterations with tighter tolerances
            def obj_fine(flat_pts):
                pts = flat_pts.reshape(N_POINTS, 2)
                s = compute_score(pts)
                if s < 1e-10:
                    return 1e10
                return -s
            
            try:
                res = minimize(obj_fine, best_points_overall.flatten(), method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
                if res.success:
                    fine_pts = res.x.reshape(N_POINTS, 2)
                    fine_score = compute_score(fine_pts)
                    if fine_score > best_score_overall:
                        best_score_overall = fine_score
                        best_points_overall = fine_pts
                        print(f"Fine polished score: {fine_score}")
            except Exception as e:
                print(f"Fine polish failed: {e}")
    
    # Strategy 3: If score is low, try multiple seed configurations
    if best_score_overall < 0.08: # Threshold for "bad" result
        print("Trying additional seed configurations...")
        
        seeds = []
        
        # Regular 13-gon
        angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
        seeds.append(np.column_stack([np.cos(angles), np.sin(angles)]))
        
        # 12-gon + center
        angles12 = np.linspace(0, 2*np.pi, 12, endpoint=False)
        pts12 = np.column_stack([np.cos(angles12), np.sin(angles12)])
        seeds.append(np.vstack([np.array([0.0, 0.0]), pts12]))
        
        # 4x4 grid with one removed
        grid = np.array([[i, j] for i in range(4) for j in range(4)], dtype=float)
        grid = grid[:13]
        grid = (grid - 1.5) / 3.0
        seeds.append(grid)
        
        # 3x5 grid
        grid35 = np.array([[i, j] for i in range(3) for j in range(5)], dtype=float)
        grid35 = grid35[:13]
        grid35 = (grid35 - np.array([1.0, 2.0])) / 4.0
        seeds.append(grid35)
        
        # Hexagonal packing
        hex_pts = []
        for i in range(3):
            for j in range(5):
                x = j + 0.5 * (i % 2)
                y = i * np.sqrt(3) / 2
                hex_pts.append([x, y])
        hex_pts = np.array(hex_pts[:13])
        hex_pts = (hex_pts - np.mean(hex_pts, axis=0)) / np.std(hex_pts)
        seeds.append(hex_pts)
        
        for seed_pts in seeds:
            remaining_time = deadline - time.time()
            if remaining_time < 15:
                break
            try:
                polished, score = polish_local(seed_pts, time_limit=min(remaining_time - 5, 20))
                if score > best_score_overall:
                    best_score_overall = score
                    best_points_overall = polished
                    print(f"Seed polished score: {score}")
            except Exception:
                pass

    # Final safety check
    if best_points_overall is None:
        # Return a known valid configuration
        angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
        best_points_overall = np.column_stack([np.cos(angles), np.sin(angles)])
        
    # Ensure finite
    if not np.all(np.isfinite(best_points_overall)):
        angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
        best_points_overall = np.column_stack([np.cos(angles), np.sin(angles)])
        
    # Ensure convex hull exists
    try:
        hull = ConvexHull(best_points_overall)
        if hull.volume <= 1e-10:
            angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
            best_points_overall = np.column_stack([np.cos(angles), np.sin(angles)])
    except:
        angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
        best_points_overall = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points_overall

if __name__ == "__main__":
    # Quick test
    import time
    start = time.time()
    pts = heilbronn_convex13()
    score = compute_score(pts)
    print(f"Time taken: {time.time() - start:.2f}s")
    print(f"Final Score: {score}")
    print(f"Points:\n{pts}")
