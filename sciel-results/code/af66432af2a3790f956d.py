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

    def get_ratio(pts):
        """Calculate d_min / d_max. Higher is better."""
        d = pdist(pts)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max <= 1e-12:
            return 0.0
        return d_min / d_max

    def get_ratio_sq(pts):
        """Calculate (d_min / d_max)^2. Higher is better. This is the actual metric R."""
        d = pdist(pts)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max <= 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def ring_init(n_inner, n_outer, r_inner, r_outer):
        """Points on two concentric rings."""
        pts = []
        if n_inner > 0:
            for i in range(n_inner):
                angle = 2 * np.pi * i / n_inner
                pts.append([r_inner * np.cos(angle), r_inner * np.sin(angle)])
        if n_outer > 0:
            for i in range(n_outer):
                angle = 2 * np.pi * i / n_outer + np.pi / n_outer
                pts.append([r_outer * np.cos(angle), r_outer * np.sin(angle)])
        return np.array(pts)

    def grid_init():
        """4x4 square grid."""
        pts = []
        for i in range(4):
            for j in range(4):
                pts.append([i, j])
        pts = np.array(pts, dtype=float)
        pts = pts - np.mean(pts, axis=0)
        return pts

    def hex_init():
        """Hexagonal arrangement."""
        pts = []
        a = 1.0
        b = np.sqrt(3) / 2
        pts.append([0, 0])
        for i in range(6):
            angle = 2 * np.pi * i / 6
            pts.append([a * np.cos(angle), a * np.sin(angle)])
        ring2_pts = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i * a + j * 0.5 * a
                y = j * b
                r = np.sqrt(x*x + y*y)
                if 1.5 * a - 0.1 < r < 2.5 * a + 0.1:
                    ring2_pts.append([x, y])
        ring2_pts = np.array(ring2_pts)
        if len(ring2_pts) >= 9:
            r_target = 2.0 * a
            dists_to_target = np.abs(np.linalg.norm(ring2_pts, axis=1) - r_target)
            idx = np.argsort(dists_to_target)[:9]
            ring2_pts = ring2_pts[idx]
        pts = np.array(pts)
        pts = np.vstack([pts, ring2_pts])
        pts = pts - np.mean(pts, axis=0)
        return pts

    def circle_init(n=16):
        """Points on a circle."""
        pts = []
        for i in range(n):
            angle = 2 * np.pi * i / n
            pts.append([np.cos(angle), np.sin(angle)])
        return np.array(pts)

    def square_ring_init():
        """Points on boundary of a square with some interior."""
        pts = []
        for i in range(4):
            pts.append([-1 + 2*i/3, 1.0])
        for i in range(1, 4):
            pts.append([1.0, 1 - 2*i/3])
        for i in range(1, 4):
            pts.append([-1 + 2*i/3, -1.0])
        for i in range(1, 3):
            pts.append([-1.0, 1 - 2*i/3])
        pts.append([0, 0])
        pts.append([0.5, 0.5])
        pts.append([-0.5, 0.5])
        pts.append([0.5, -0.5])
        pts = np.array(pts[:16], dtype=float)
        pts = pts - np.mean(pts, axis=0)
        return pts

    def generate_candidates():
        candidates = []
        candidates.append(ring_init(1, 15, 0, 1.0))
        candidates.append(ring_init(4, 12, 0.5, 1.0))
        candidates.append(ring_init(7, 9, 0.4, 1.0))
        candidates.append(ring_init(9, 7, 0.4, 1.0))
        candidates.append(ring_init(1, 3, 0, 0.5))
        for i, c in enumerate(candidates):
            c = c - np.mean(c, axis=0)
            max_r = np.max(np.linalg.norm(c, axis=1))
            if max_r > 0:
                c = c / max_r
            candidates[i] = c
        candidates.append(grid_init())
        candidates.append(hex_init())
        candidates.append(circle_init(16))
        candidates.append(square_ring_init())
        return candidates

    best_pts = None
    best_ratio = 0.0

    for c in generate_candidates():
        if len(c) != 16:
            continue
        ratio = get_ratio(c)
        if ratio > best_ratio:
            best_ratio = ratio
            best_pts = c.copy()

    # Simulated Annealing optimization
    def simulated_annealing(pts, time_budget=300.0):
        """Improve points using simulated annealing with vectorized cost evaluation."""
        pts = pts.copy()
        t_start = time.time()
        
        # Initial temperature
        T = 0.1
        T_min = 1e-8
        cooling_rate = 0.9995
        
        # Precompute pair indices for vectorized distance computation
        # This avoids repeated pdist calls by working with squared distances
        n = N
        # Create index pairs for all pairs i < j
        i_idx, j_idx = np.triu_indices(n, k=1)
        
        current_ratio = get_ratio(pts)
        best_ratio = current_ratio
        best_pts = pts.copy()
        
        # Track the minimum and maximum distance pairs for efficient updates
        # But for simplicity, we'll recompute when needed
        
        it = 0
        while time.time() - t_start < time_budget and T > T_min:
            it += 1
            
            # Check time limit
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            # Randomly select a point to move
            idx = np.random.randint(n)
            
            # Generate a random perturbation
            scale = 0.05 * T / 0.1  # Scale perturbation with temperature
            delta = np.random.randn(2) * scale
            
            # Save old position
            old_pos = pts[idx].copy()
            
            # Apply perturbation
            pts[idx] += delta
            
            # Compute new ratio
            new_ratio = get_ratio(pts)
            
            # Acceptance criterion
            if new_ratio > current_ratio:
                # Always accept improvements
                current_ratio = new_ratio
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_pts = pts.copy()
            else:
                # Accept worse solutions with probability exp((new - current) / T)
                # Since we're maximizing, delta = new - current < 0
                delta_ratio = new_ratio - current_ratio
                # For small T, this probability is very small
                accept_prob = np.exp(delta_ratio / (T * 0.01))  # Scale by 0.01 to make it more likely to accept
                if np.random.rand() < accept_prob:
                    current_ratio = new_ratio
                else:
                    # Reject: revert
                    pts[idx] = old_pos
            
            # Cool down
            T *= cooling_rate
            
            # Adaptive cooling: if stuck, cool slower
            if it % 1000 == 0 and current_ratio == best_ratio:
                T = max(T, T_min * 10)
        
        return best_pts, best_ratio

    # Run simulated annealing from best candidate
    time_elapsed = time.time() - start_time
    sa_budget = min(400.0, TIME_LIMIT - time_elapsed - SAFETY_MARGIN)
    if sa_budget > 10:
        sa_pts, sa_ratio = simulated_annealing(best_pts, time_budget=sa_budget)
        if sa_ratio > best_ratio:
            best_pts = sa_pts
            best_ratio = sa_ratio

    # Final polish with coordinate descent on the SA result
    def coordinate_descent(pts, max_iter=5000, time_budget=30.0):
        """Improve points using coordinate descent."""
        pts = pts.copy()
        t_start = time.time()
        step = 0.02
        
        for it in range(max_iter):
            if time.time() - t_start > time_budget:
                break
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            improved = False
            for i in range(N):
                best_pos = pts[i].copy()
                best_r = get_ratio(pts)
                
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        old = pts[i].copy()
                        pts[i] += np.array([dx, dy])
                        r = get_ratio(pts)
                        if r > best_r + 1e-10:
                            best_r = r
                            best_pos = pts[i].copy()
                            improved = True
                        pts[i] = old
                
                pts[i] = best_pos
            
            if not improved:
                step *= 0.85
                if step < 1e-7:
                    break
        
        return pts, get_ratio(pts)

    time_elapsed = time.time() - start_time
    if TIME_LIMIT - time_elapsed - SAFETY_MARGIN > 15:
        cd_pts, cd_ratio = coordinate_descent(best_pts, max_iter=10000,
                                               time_budget=min(60.0, TIME_LIMIT - time_elapsed - SAFETY_MARGIN))
        if cd_ratio > best_ratio:
            best_pts = cd_pts
            best_ratio = cd_ratio

    # Ensure finite and correct shape
    final_pts = best_pts
    if final_pts is None or not np.all(np.isfinite(final_pts)) or final_pts.shape != (16, 2):
        final_pts = grid_init()

    return final_pts
