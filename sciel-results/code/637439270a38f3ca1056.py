import numpy as np
import time
from scipy.spatial.distance import pdist

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    TIME_LIMIT = 600.0
    SAFETY_MARGIN = 15.0
    N = 16

    def normalize(points):
        points = points - np.mean(points, axis=0)
        norm = np.linalg.norm(points)
        if norm > 1e-10:
            points = points / norm
        return points

    def score(points):
        p = normalize(points)
        dists = pdist(p)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10 or d_min < 1e-10:
            return -np.inf
        return (d_min / d_max) ** 2

    def run_sa(initial_points, time_limit):
        np.random.seed(12345)
        current_points = normalize(initial_points.copy())
        current_score = score(current_points)
        
        best_points = current_points.copy()
        best_score = current_score
        
        T = 0.1
        T_min = 1e-4
        cooling_rate = 0.99995
        
        start_time = time.time()
        
        while True:
            if time.time() - start_time > time_limit:
                break
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
                
            candidate_points = current_points.copy()
            idx = np.random.randint(N)
            step = 0.1 * (T / 0.1 + 0.1)
            noise = np.random.randn(2) * step
            candidate_points[idx] += noise
            
            candidate_points = normalize(candidate_points)
            candidate_score = score(candidate_points)
            
            delta = candidate_score - current_score
            if delta > 0 or np.random.rand() < np.exp(delta / T):
                current_points = candidate_points
                current_score = candidate_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = candidate_points.copy()
                    
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
        return best_points, best_score

    def coordinate_descent(pts, max_iter=5000, time_budget=30.0):
        pts = pts.copy()
        t_start = time.time()
        step = 0.05
        
        for it in range(max_iter):
            if time.time() - t_start > time_budget:
                break
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            improved = False
            for i in range(N):
                best_pos = pts[i].copy()
                best_r = score(pts)
                
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        old = pts[i].copy()
                        pts[i] += np.array([dx, dy])
                        r = score(pts)
                        if r > best_r + 1e-10:
                            best_r = r
                            best_pos = pts[i].copy()
                            improved = True
                        pts[i] = old
                
                pts[i] = best_pos
            
            if not improved:
                step *= 0.9
                if step < 1e-6:
                    break
        
        return pts, score(pts)

    def generate_candidates():
        candidates = []
        
        # 1. 4x4 Grid
        x = np.arange(4) - 1.5
        y = np.arange(4) - 1.5
        xx, yy = np.meshgrid(x, y)
        grid_pts = np.column_stack((xx.ravel(), yy.ravel()))
        candidates.append(grid_pts)
        
        # 2. Regular 16-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        circle_pts = np.column_stack((np.cos(angles), np.sin(angles)))
        candidates.append(circle_pts)
        
        # 3. Hexagonal-like packing
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.5, np.sqrt(3)/2])
        hex_points = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                p = i * v1 + j * v2
                hex_points.append(p)
        hex_points = np.array(hex_points)
        hex_points = hex_points - np.mean(hex_points, axis=0)
        dists_to_center = np.linalg.norm(hex_points, axis=1)
        idx = np.argsort(dists_to_center)[:16]
        candidates.append(hex_points[idx])
        
        # 4. Multi-ring (1 center, 5 inner, 10 outer)
        c1 = np.array([0, 0])
        a2 = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
        r2 = 1.0
        inner = np.column_stack((r2 * np.cos(a2), r2 * np.sin(a2)))
        a3 = np.linspace(0, 2*np.pi, 10, endpoint=False)
        r3 = 2.0
        outer = np.column_stack((r3 * np.cos(a3), r3 * np.sin(a3)))
        ring_pts = np.vstack([c1, inner, outer])
        candidates.append(ring_pts)
        
        # 5. Random candidates
        np.random.seed(42)
        for _ in range(5):
            rand_pts = np.random.uniform(-2, 2, (N, 2))
            candidates.append(rand_pts)
            
        # Normalize all candidates
        for i, c in enumerate(candidates):
            candidates[i] = normalize(c)
            
        return candidates

    best_pts = None
    best_ratio = -np.inf

    candidates = generate_candidates()
    
    # Evaluate candidates to find the best starting point
    for c in candidates:
        if len(c) != 16:
            continue
        ratio = score(c)
        if ratio > best_ratio:
            best_ratio = ratio
            best_pts = c.copy()

    # Run Simulated Annealing from the best candidate
    # We allocate the majority of the time to SA, as it is a global search method
    # that can escape local minima better than simple coordinate descent.
    
    # Time budget allocation:
    # SA: ~80% of time
    # CD: ~20% of time for final polish
    
    time_elapsed = time.time() - start_time
    remaining_time = TIME_LIMIT - time_elapsed - SAFETY_MARGIN
    
    sa_budget = remaining_time * 0.8
    cd_budget = remaining_time * 0.2
    
    if sa_budget > 10:
        sa_pts, sa_ratio = run_sa(best_pts, sa_budget)
        if sa_ratio > best_ratio:
            best_pts = sa_pts
            best_ratio = sa_ratio
            
    # Final polish with coordinate descent
    time_elapsed = time.time() - start_time
    if TIME_LIMIT - time_elapsed - SAFETY_MARGIN > 5:
        cd_pts, cd_ratio = coordinate_descent(best_pts, max_iter=20000, 
                                             time_budget=min(30.0, TIME_LIMIT - time_elapsed - SAFETY_MARGIN))
        if cd_ratio > best_ratio:
            best_pts = cd_pts
            best_ratio = cd_ratio

    # Ensure finite and correct shape
    final_pts = best_pts
    if final_pts is None or not np.all(np.isfinite(final_pts)) or final_pts.shape != (16, 2):
        # Fallback to grid if something went wrong
        x = np.arange(4) - 1.5
        y = np.arange(4) - 1.5
        xx, yy = np.meshgrid(x, y)
        final_pts = np.column_stack((xx.ravel(), yy.ravel()))

    return final_pts
