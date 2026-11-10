"""Improved packing of 300 equal circles in a unit circle using local optimization."""
import numpy as np
import time
import math

N = 300

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    def compute_score(points):
        """Compute the minimum radius for the entire configuration (O(N^2))."""
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        radii = np.linalg.norm(points, axis=1)
        min_boundary = np.min(1.0 - radii)
        
        return min(min_dist / 2.0, min_boundary)

    def get_limiting_pairs(points, score):
        """Find which pairs/boundary constraints are limiting the score."""
        # Find all pairs with distance close to 2*score
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Pairs that are limiting (distance <= 2*score + tolerance)
        tol = 1e-9
        limiting_mask = dists <= 2 * score + tol
        limiting_pairs = []
        for i in range(N):
            for j in range(i+1, N):
                if limiting_mask[i, j]:
                    limiting_pairs.append((i, j))
        
        # Boundary limiting points
        radii = np.linalg.norm(points, axis=1)
        boundary_limiting = np.where(1.0 - radii <= score + tol)[0].tolist()
        
        return limiting_pairs, boundary_limiting

    def local_optimize(points, max_iter=300, verbose=False):
        """Local optimization focusing on limiting constraints."""
        points = points.copy()
        current_score = compute_score(points)
        best_points = points.copy()
        best_score = current_score
        
        lr = 0.005
        no_improve_count = 0
        
        for it in range(max_iter):
            if time.time() > deadline - 3.0:
                break
            
            # Get current limiting constraints
            limiting_pairs, boundary_limiting = get_limiting_pairs(points, current_score)
            
            # Identify which points are involved in limiting constraints
            involved_points = set()
            for i, j in limiting_pairs:
                involved_points.add(i)
                involved_points.add(j)
            for i in boundary_limiting:
                involved_points.add(i)
            
            # Try to improve by moving involved points
            improved = False
            indices = list(involved_points)
            np.random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline - 3.0:
                    break
                
                # Try multiple random directions
                for attempt in range(5):
                    angle = np.random.uniform(0, 2 * np.pi)
                    dx = lr * math.cos(angle)
                    dy = lr * math.sin(angle)
                    
                    old_pos = points[i].copy()
                    new_pos = old_pos + np.array([dx, dy])
                    
                    # Clamp to boundary
                    r_new = np.linalg.norm(new_pos)
                    if r_new > 0.999:
                        new_pos *= 0.999 / r_new
                    
                    points[i] = new_pos
                    
                    # Compute new score
                    new_score = compute_score(points)
                    
                    if new_score > current_score + 1e-10:
                        current_score = new_score
                        improved = True
                        break
                    else:
                        points[i] = old_pos
                
                if improved:
                    break
            
            # Periodically recompute and track best
            if it % 20 == 0 or not improved:
                current_score = compute_score(points)
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    lr *= 0.9
                    if no_improve_count > 8 or lr < 1e-6:
                        break
            else:
                no_improve_count = 0
        
        return best_points, best_score

    def initial_hex(n):
        """Generate initial hexagonal lattice points with multiple spacing attempts."""
        s_est = math.sqrt(2 * math.pi * 0.95**2 / (n * math.sqrt(3)))
        
        best_points = None
        best_score = -1
        
        for trial in range(30):
            spacing = s_est * (0.95 + trial * 0.01)
            points = []
            r_max = int(1.0 / (spacing * math.sqrt(3) / 2)) + 3
            for j in range(-r_max, r_max + 1):
                for i in range(-r_max, r_max + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * math.sqrt(3) / 2
                    if x * x + y * y <= (1.0 - spacing / 2) ** 2:
                        points.append((x, y))
            
            if len(points) >= n:
                points_arr = np.array(points)
                dists = np.hypot(points_arr[:, 0], points_arr[:, 1])
                idx = np.argsort(dists)[:n]
                candidate = points_arr[idx]
                
                score = compute_score(candidate)
                if score > best_score:
                    best_score = score
                    best_points = candidate
            
            if time.time() > deadline - 10:
                break
        
        if best_points is None:
            best_points = np.random.randn(n, 2) * 0.8
            max_r = np.max(np.hypot(best_points[:, 0], best_points[:, 1]))
            if max_r > 0.9:
                best_points *= 0.9 / max_r
        
        return best_points

    # Generate initial configuration
    np.random.seed(123)
    points = initial_hex(N)
    
    # Local optimization
    points, score = local_optimize(points, max_iter=200)
    
    # If time remains, try random initialization
    if time.time() < deadline - 40:
        np.random.seed(456)
        points2 = np.random.randn(N, 2) * 0.6
        max_r = np.max(np.hypot(points2[:, 0], points2[:, 1]))
        if max_r > 0.9:
            points2 *= 0.9 / max_r
        
        points2, score2 = local_optimize(points2, max_iter=150)
        if score2 > score:
            points = points2
            score = score2
    
    # Final polish if time allows
    if time.time() < deadline - 15:
        points, score_final = local_optimize(points, max_iter=50)
        if score_final > score:
            score = score_final
    
    return points
# EVOLVE-BLOCK-END

