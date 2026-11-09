import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = 8
    
    def compute_score_from_points(points):
        """
        Compute the score Delta/n^n for a set of points.
        Delta is the product of distances for all ordered pairs (i, j) with i != j.
        The set is scaled to have diameter 2.
        """
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        mask = ~np.eye(n, dtype=bool)
        max_dist = np.max(dists[mask])
        
        if max_dist < 1e-10:
            return 0.0
            
        # Scale to diameter 2
        scale = 2.0 / max_dist
        dists_scaled = dists * scale
        
        # Product of distances for ordered pairs (i != j)
        # There are n*(n-1) such pairs.
        # log(Delta) = sum_{i!=j} log(d_ij)
        # We exclude i=j (dist 0)
        
        # Avoid log(0) just in case, though distinctness is enforced by edge lengths > 0
        dists_nonzero = dists_scaled[mask]
        
        log_prod = np.sum(np.log(dists_nonzero))
        
        # Score is Delta / n^n
        # log(Score) = log(Delta) - n * log(n)
        log_score = log_prod - n * np.log(n)
        
        return np.exp(log_score)

    def get_points_from_edges(edge_lengths):
        """
        Reconstruct points from edge lengths of a convex polygon.
        We fix the first point at the origin and the second on the x-axis.
        The remaining points are determined by the turning angles.
        For a convex polygon, the exterior angles sum to 2*pi.
        Given edge lengths L_i, the turning angles theta_i are determined by the geometry.
        Actually, for a general convex polygon, the shape is not uniquely determined by edge lengths alone (flexible polygon).
        However, we can assume the "maximally spread" configuration or use a specific parametrization.
        
        A more robust approach:
        Parametrize by 8 angles (turning angles) and 8 edge lengths.
        Constraint: sum(turning angles) = 2*pi.
        Closure: sum(L_i * exp(i * cumulative_angle)) = 0.
        
        This is complex. Let's use a simpler parametrization that guarantees convexity:
        Use polar coordinates relative to the centroid? No, that doesn't guarantee convexity easily.
        
        Alternative: Use the fact that for fixed edge lengths, the product of chord distances is maximized when the polygon is "as round as possible" or specific symmetric configurations.
        
        Actually, the directive suggests: "Parametrize the 8 points as vertices of a convex polygon defined by 8 positive edge lengths... optimizing these lengths".
        This implies that for a given set of edge lengths, there is a unique (or canonical) convex polygon?
        No, a convex polygon is flexible if n > 3.
        However, if we assume the polygon is cyclic (inscribed in a circle), then the edge lengths determine the polygon uniquely (up to congruence).
        The regular octagon is cyclic. It is highly likely the optimum is cyclic due to symmetry and the nature of the distance product (which favors uniform distribution).
        
        Let's assume the optimal configuration is a cyclic polygon.
        If the polygon is cyclic with radius R, and edge lengths L_i, then the central angle alpha_i satisfies L_i = 2 * R * sin(alpha_i / 2).
        Sum(alpha_i) = 2 * pi.
        So, given L_i, we have:
        sum(2 * arcsin(L_i / (2R))) = 2*pi  =>  sum(arcsin(L_i / (2R))) = pi.
        
        We can solve for R given L_i.
        Then the vertices are at angles cumulative sum of alpha_i.
        
        Let's define the objective in terms of L_i (normalized so sum(L_i) = 1 to fix scale, or just optimize L_i and let R adjust).
        Since the score is scale-invariant, we can fix the perimeter P = 1.
        So we optimize over L_i > 0, sum(L_i) = 1.
        
        Steps:
        1. Given L (shape 8,), sum(L)=1.
        2. Find R such that sum(arcsin(L_i / (2R))) = pi.
           Note: L_i < 2R is required. Since max L_i <= 1 and R will be roughly 1/(2*pi) * 8 = 4/pi ~ 1.27 for regular, L_i ~ 1/8 = 0.125. 2R ~ 2.5. So L_i/2R is small.
           f(R) = sum(arcsin(L_i / (2R))) - pi.
           f(R) is decreasing in R.
           As R -> infinity, arcsin(x) ~ x, sum(L_i/2R) -> 0. f(R) -> -pi.
           As R -> max(L_i)/2, one term goes to arcsin(0.5)=pi/6? No, arcsin(1/2)=pi/6.
           Wait, if L_i = 2R, arcsin(1) = pi/2.
           If one L_i dominates, R must be large enough.
           Actually, for a convex polygon, R must be at least max(L_i)/2.
           Let's use bisection to find R.
           
        3. Compute angles alpha_i = 2 * arcsin(L_i / (2R)).
        4. Compute cumulative angles to get vertex positions on the circle of radius R.
        5. Compute score.
        
        This reduces the problem to 7 dimensions (L_i with sum=1).
        """
        # L is a vector of 8 positive numbers summing to 1
        L = np.array(edge_lengths)
        
        # Find R
        # g(R) = sum(arcsin(L_i / (2R))) - pi
        # We need g(R) = 0.
        # Lower bound for R: max(L_i) / 2. If L_i = 2R, arcsin(1)=pi/2.
        # If we have one very large L, say 0.9, others small.
        # R must be > 0.9/2 = 0.45.
        # Let's set low = max(L) / 2 * 1.001
        # High: For regular octagon, R ~ 1.3.
        # If L is very uneven, R might be larger.
        # Let's start high = 10.0
        
        low = np.max(L) / 2.0 * 1.0001
        high = 100.0
        
        # Check if solution exists.
        # g(low) should be > 0?
        # If R = max(L)/2, the largest term is arcsin(1) = pi/2.
        # Other terms are arcsin(L_j / L_max) < pi/2.
        # Sum could be > pi or < pi.
        # Actually, for a valid convex polygon, the sum of angles must be 2pi.
        # If the polygon is degenerate (flat), R -> infinity? No, flat means angles sum to pi?
        # Let's just bisect.
        
        def g(R):
            if R <= 0: return np.inf
            args = L / (2.0 * R)
            # Clip args to [-1, 1] to avoid domain errors in arcsin
            args = np.clip(args, -1.0, 1.0)
            return np.sum(np.arcsin(args)) - np.pi
        
        if g(low) < 0:
            # No solution? This happens if the polygon cannot be closed with these edges in a convex way with this parametrization?
            # Actually, if g(low) < 0, it means sum(arcsin) < pi even at the smallest possible R.
            # This implies the "circumradius" would have to be smaller than max(L)/2, which is impossible.
            # In this case, the configuration is invalid or we should return a low score.
            # However, for convex polygons, there is always a circumcircle if it's cyclic.
            # Wait, not all convex polygons are cyclic. But we are restricting to cyclic ones.
            # If no cyclic polygon exists for these edge lengths, we penalize.
            return -np.inf, None
        
        # Bisect
        R = (low + high) / 2.0
        for _ in range(50):
            mid = (low + high) / 2.0
            val = g(mid)
            if val > 0:
                low = mid
            else:
                high = mid
            R = (low + high) / 2.0
            
        if g(R) < -1e-5:
             return -np.inf, None

        # Compute angles
        args = L / (2.0 * R)
        args = np.clip(args, -1.0, 1.0)
        alpha = 2.0 * np.arcsin(args)
        
        # Cumulative angles
        cum_alpha = np.cumsum(alpha)
        # The last one should be 2*pi
        # Vertices on circle
        # Start at angle 0
        angles = np.concatenate(([0], cum_alpha[:-1]))
        
        points = np.column_stack([R * np.cos(angles), R * np.sin(angles)])
        
        score = compute_score_from_points(points)
        return score, points

    def objective(log_L):
        """
        Optimize log(L_i) to ensure positivity.
        L_i = exp(log_L_i) / sum(exp(log_L_j))
        """
        # Softmax to get normalized L
        L = np.exp(log_L - np.max(log_L))
        L_sum = np.sum(L)
        L = L / L_sum
        
        score, _ = get_points_from_edges(L)
        if not np.isfinite(score):
            return 1e10
        return -score

    # Initial guess: Regular octagon
    # Edge length for regular octagon with perimeter 1 is 1/8.
    log_L_init = np.full(8, np.log(1.0/8.0))
    
    best_score = -1e10
    best_points = None
    
    # Regular octagon score is 1.0
    angles = 2 * np.pi * np.arange(n) / n
    p_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    best_score = compute_score_from_points(p_oct)
    best_points = p_oct
    
    # Optimization
    # We have 8 parameters, but they are scale invariant (sum=1 constraint handled by softmax).
    # So effectively 7 dimensions.
    
    try:
        # Use Nelder-Mead on the log-space
        res = minimize(objective, log_L_init, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
        
        if res.fun < -best_score:
            # Reconstruct best points
            L = np.exp(res.x - np.max(res.x))
            L = L / np.sum(L)
            _, pts = get_points_from_edges(L)
            if pts is not None:
                score = compute_score_from_points(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts
                    
    except Exception as e:
        pass

    # If time remains, try a second restart with perturbed initial guess
    if time.time() < deadline - 20:
        try:
            # Perturb regular octagon
            np.random.seed(42)
            log_L_pert = log_L_init + np.random.normal(0, 0.1, 8)
            res2 = minimize(objective, log_L_pert, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            
            if res2.fun < -best_score:
                L = np.exp(res2.x - np.max(res2.x))
                L = L / np.sum(L)
                _, pts = get_points_from_edges(L)
                if pts is not None:
                    score = compute_score_from_points(pts)
                    if score > best_score:
                        best_score = score
                        best_points = pts
        except Exception:
            pass

    # Final check and return
    if best_points is None:
        return [[float(x), float(y)] for x, y in p_oct]
        
    return [[float(x), float(y)] for x, y in best_points]
