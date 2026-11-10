import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    TIME_LIMIT = 600.0
    SAFETY_MARGIN = 30.0
    N = 16
    N_PAIRS = N * (N - 1) // 2

    def get_ratio(pts):
        """Calculate d_min / d_max. Higher is better."""
        d = pdist(pts)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max <= 1e-12:
            return 0.0
        return d_min / d_max

    def get_cost_fast(pts):
        """
        Calculate (d_max / d_min)^2. Lower is better.
        Used for optimization objectives.
        """
        # Using pdist is often faster than manual broadcasting for N=16 due to C implementation
        d = pdist(pts)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_min <= 1e-12 or d_max <= 1e-12:
            return 1e10
        return (d_max / d_min) ** 2

    def generate_initial_candidates():
        """Generate diverse initial configurations."""
        candidates = []
        
        # 1. Circle
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        circle = np.column_stack([np.cos(angles), np.sin(angles)])
        candidates.append(circle)
        
        # 2. Two concentric circles (4 inner, 12 outer)
        inner_angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
        outer_angles = np.linspace(0, 2*np.pi, 12, endpoint=False) + np.pi/12
        r_inner = 0.5
        r_outer = 1.0
        inner_pts = np.column_stack([r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)])
        outer_pts = np.column_stack([r_outer * np.cos(outer_angles), r_outer * np.sin(outer_angles)])
        two_circles = np.vstack([inner_pts, outer_pts])
        candidates.append(two_circles)
        
        # 3. Three concentric circles (2, 6, 8)
        inner_angles = np.linspace(0, 2*np.pi, 2, endpoint=False)
        mid_angles = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
        outer_angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        r1, r2, r3 = 0.3, 0.7, 1.0
        pts1 = np.column_stack([r1 * np.cos(inner_angles), r1 * np.sin(inner_angles)])
        pts2 = np.column_stack([r2 * np.cos(mid_angles), r2 * np.sin(mid_angles)])
        pts3 = np.column_stack([r3 * np.cos(outer_angles), r3 * np.sin(outer_angles)])
        three_circles = np.vstack([pts1, pts2, pts3])
        candidates.append(three_circles)
        
        # 4. Hexagonal lattice subset (approx)
        # Simple hex packing for 16 points is tricky, let's use a distorted grid or random perturbed circle
        # Let's stick to a robust random perturbation of the circle for variety
        np.random.seed(42)
        circle_perturbed = circle + np.random.randn(16, 2) * 0.1
        candidates.append(circle_perturbed)
        
        # 5. Square grid 4x4
        grid = []
        for i in range(4):
            for j in range(4):
                grid.append([i - 1.5, j - 1.5])
        grid = np.array(grid)
        candidates.append(grid)
        
        # 6. Random
        np.random.seed(123)
        random_pts = np.random.randn(16, 2)
        candidates.append(random_pts)
        
        # Normalize all candidates to unit max radius to ensure consistent scale
        for i, c in enumerate(candidates):
            c = c - np.mean(c, axis=0)
            max_r = np.max(np.linalg.norm(c, axis=1))
            if max_r > 0:
                c = c / max_r
            candidates[i] = c
            
        return candidates

    def simulated_annealing(initial_pts, time_budget):
        """
        Fast local search using Simulated Annealing.
        """
        pts = initial_pts.copy()
        current_cost = get_cost_fast(pts)
        best_pts_sa = pts.copy()
        best_cost_sa = current_cost
        
        # Adaptive temperature
        T = 0.5
        cooling_rate = 0.9999
        min_T = 1e-8
        no_improve_count = 0
        max_no_improve = 20000
        
        t_start = time.time()
        
        while time.time() - t_start < time_budget:
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            idx = np.random.randint(0, N)
            noise = np.random.randn(2) * T
            old_pos = pts[idx].copy()
            pts[idx] += noise
            
            new_cost = get_cost_fast(pts)
            
            if new_cost < current_cost:
                current_cost = new_cost
                no_improve_count = 0
                
                if current_cost < best_cost_sa:
                    best_cost_sa = current_cost
                    best_pts_sa = pts.copy()
            else:
                delta = new_cost - current_cost
                if T > min_T:
                    if delta > T * 50: 
                        accept = False
                    else:
                        prob = np.exp(-delta / T)
                        accept = np.random.rand() < prob
                else:
                    accept = False
                
                if accept:
                    current_cost = new_cost
                else:
                    pts[idx] = old_pos
                    no_improve_count += 1
                    
            T *= cooling_rate
            
            if no_improve_count > max_no_improve:
                pts = best_pts_sa.copy()
                current_cost = best_cost_sa
                T = 0.1
                no_improve_count = 0
                max_no_improve = 10000

        return best_pts_sa, best_cost_sa

    def powell_polish(pts, time_budget):
        """
        Refine using Powell's derivative-free method.
        Objective: Minimize (d_max / d_min)^2
        """
        def objective(x):
            p = x.reshape(N, 2)
            return get_cost_fast(p)
            
        x0 = pts.flatten()
        
        # Powell is derivative-free and robust to non-smoothness
        try:
            result = minimize(objective, x0, method='Powell',
                              options={'maxiter': 1000, 'maxfev': 5000})
            if result.fun < 1e10:
                return result.x.reshape(N, 2)
        except Exception:
            pass
        
        return pts

    # Main Loop
    best_pts = None
    best_cost = np.inf

    candidates = generate_initial_candidates()
    
    # Evaluate initial candidates
    for c in candidates:
        cost = get_cost_fast(c)
        if cost < best_cost:
            best_cost = cost
            best_pts = c.copy()

    if best_pts is None:
        best_pts = np.random.randn(16, 2)
        best_cost = get_cost_fast(best_pts)

    # Phase 1: Simulated Annealing from best candidate
    time_elapsed = time.time() - start_time
    sa_budget = min(300.0, TIME_LIMIT - time_elapsed - SAFETY_MARGIN)
    if sa_budget > 10:
        sa_pts, sa_cost = simulated_annealing(best_pts, sa_budget)
        if sa_cost < best_cost:
            best_pts = sa_pts
            best_cost = sa_cost

    # Phase 2: Powell Polish
    time_elapsed = time.time() - start_time
    powell_budget = min(150.0, TIME_LIMIT - time_elapsed - SAFETY_MARGIN)
    if powell_budget > 5:
        polished_pts = powell_polish(best_pts, powell_budget)
        polished_cost = get_cost_fast(polished_pts)
        if polished_cost < best_cost:
            best_pts = polished_pts
            best_cost = polished_cost

    # Phase 3: Try SA + Powell from second best candidate if time permits
    time_elapsed = time.time() - start_time
    if TIME_LIMIT - time_elapsed - SAFETY_MARGIN > 100:
        # Find second best candidate
        sorted_candidates = sorted(candidates, key=lambda c: get_cost_fast(c))
        if len(sorted_candidates) > 1:
            c2 = sorted_candidates[1]
            remaining = TIME_LIMIT - time.time() - SAFETY_MARGIN
            if remaining > 50:
                sa2_budget = min(50.0, remaining / 2)
                sa2_pts, sa2_cost = simulated_annealing(c2, sa2_budget)
                
                if sa2_cost < best_cost:
                    best_pts = sa2_pts
                    best_cost = sa2_cost
                
                remaining = TIME_LIMIT - time.time() - SAFETY_MARGIN
                if remaining > 20:
                    pol2_pts = powell_polish(best_pts, remaining)
                    pol2_cost = get_cost_fast(pol2_pts)
                    if pol2_cost < best_cost:
                        best_pts = pol2_pts
                        best_cost = pol2_cost

    # Ensure finite and correct shape
    final_pts = best_pts
    if final_pts is None or not np.all(np.isfinite(final_pts)) or final_pts.shape != (16, 2):
        final_pts = np.random.randn(16, 2)

    return final_pts
