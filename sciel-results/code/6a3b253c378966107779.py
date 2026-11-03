import numpy as np
import time
from scipy.spatial import Delaunay

N = 50
DEADLINE = 145.0  # Safety margin before 150s limit

def _true_score(centers):
    """Compute the true packing radius for a given configuration."""
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > 0.5):
        return 0.0
    boundary = (0.5 - np.abs(centers)).min()
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = (diffs ** 2).sum(axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_dist = np.sqrt(dists_sq.min())
    return min(boundary, min_dist / 2.0)

def _init_grid():
    """Grid initialization trying multiple configurations."""
    best_pts = None
    best_r = -1

    for rows, cols in [(7, 8), (8, 7), (5, 10), (10, 5), (6, 9), (9, 6), (7, 7)]:
        total = rows * cols
        if total < N:
            continue

        xs = (np.arange(cols) + 0.5) / cols - 0.5
        ys = (np.arange(rows) + 0.5) / rows - 0.5
        grid = np.array([(x, y) for y in ys for x in xs])

        if total == N:
            pts = grid
        else:
            dists_to_center = np.sqrt((grid ** 2).sum(axis=1))
            indices = np.argsort(dists_to_center)[:N]
            pts = grid[indices]

        r = _true_score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    return best_pts if best_pts is not None else (np.random.rand(N, 2) - 0.5) * 0.9

def _init_hexagonal():
    """Improved hexagonal initialization with proper spacing."""
    best_pts = None
    best_r = -1

    for s in [0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]:
        pts = []
        row_height = s * np.sqrt(3) / 2
        for i in range(-5, 6):
            for j in range(-5, 6):
                x = j * s + (i % 2) * s / 2
                y = i * row_height
                if abs(x) <= 0.48 and abs(y) <= 0.48:
                    pts.append([x, y])

        pts = np.array(pts)
        if len(pts) < N:
            continue

        # Select N points from the hex grid
        # Simple selection: sort by distance to center, take N
        dists = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dists)[:N]
        pts_sel = pts[idx]
        
        r = _true_score(pts_sel)
        if r > best_r:
            best_r = r
            best_pts = pts_sel

    if best_pts is not None:
        return best_pts
    return _init_grid()

def _delaunay_refine(centers, max_iter=200, time_limit=5.0):
    """
    Refine a configuration by identifying the closest pair (via Delaunay edges)
    and pushing them apart to maximize the minimum distance.
    This targets the specific bottleneck of the packing.
    """
    centers = centers.copy()
    best_score = _true_score(centers)
    start_time = time.time()
    
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Compute Delaunay triangulation
        try:
            tri = Delaunay(centers)
        except Exception:
            break
            
        # Find edges
        edges = tri.simplices
        # For each edge, compute distance
        min_dist = np.inf
        min_i = 0
        min_j = 0
        
        # We only need to check edges, not all pairs
        for simplex in edges:
            for k in range(3):
                i = simplex[k]
                j = simplex[(k+1)%3]
                d = np.linalg.norm(centers[i] - centers[j])
                if d < min_dist:
                    min_dist = d
                    min_i = i
                    min_j = j
                    
        # Check boundary constraint too
        boundary_dists = (0.5 - np.abs(centers)).min(axis=1)
        min_boundary = boundary_dists.min()
        
        # The true bottleneck is min(min_pair_dist/2, min_boundary)
        # If boundary is the bottleneck, we might want to move the point closest to boundary inward?
        # But usually, for dense packings, pair distance is the bottleneck.
        # Let's focus on the closest pair for now, as it's the most common constraint.
        
        # Push i and j apart
        diff = centers[min_i] - centers[min_j]
        norm = np.linalg.norm(diff)
        if norm < 1e-9:
            # Random nudge
            diff = np.random.randn(2)
            norm = np.linalg.norm(diff)
            
        # Move them apart by a small step
        # We want to increase the distance.
        # Step size proportional to current distance or fixed?
        # Let's try a fixed small step relative to the scale
        step = 0.001 
        direction = diff / norm
        
        centers[min_i] += step * direction
        centers[min_j] -= step * direction
        
        # Clip to square
        centers = np.clip(centers, -0.5, 0.5)
        
        new_score = _true_score(centers)
        if new_score > best_score:
            best_score = new_score
        else:
            # If score didn't improve, maybe this move was bad? 
            # But we are doing a greedy ascent on the minimum distance.
            # If it didn't improve, we might be stuck or the step was too big/small.
            # Let's just continue, hoping to find a better configuration.
            pass
            
    return centers, best_score

def _simulated_annealing(centers, time_budget=20.0):
    """
    Multi-start Simulated Annealing with local moves.
    Moves:
    1. Random small displacement of one point.
    2. Delaunay-based push of closest pair.
    """
    start_time = time.time()
    current_centers = centers.copy()
    current_score = _true_score(current_centers)
    
    best_centers = current_centers.copy()
    best_score = current_score
    
    # Temperature schedule
    T_start = 0.01
    T_end = 0.0001
    alpha = 0.999 # Cooling factor per step
    
    T = T_start
    
    while time.time() - start_time < time_budget:
        # Choose a move
        if np.random.random() < 0.5:
            # Random move: displace a random point
            i = np.random.randint(N)
            displacement = np.random.randn(2) * 0.005
            new_centers = current_centers.copy()
            new_centers[i] += displacement
            new_centers = np.clip(new_centers, -0.5, 0.5)
        else:
            # Delaunay move: push closest pair
            try:
                tri = Delaunay(current_centers)
                edges = tri.simplices
                min_dist = np.inf
                min_i = 0
                min_j = 0
                for simplex in edges:
                    for k in range(3):
                        i = simplex[k]
                        j = simplex[(k+1)%3]
                        d = np.linalg.norm(current_centers[i] - current_centers[j])
                        if d < min_dist:
                            min_dist = d
                            min_i = i
                            min_j = j
                            
                diff = current_centers[min_i] - current_centers[min_j]
                norm = np.linalg.norm(diff)
                if norm < 1e-9:
                    diff = np.random.randn(2)
                    norm = np.linalg.norm(diff)
                direction = diff / norm
                
                new_centers = current_centers.copy()
                step = 0.002
                new_centers[min_i] += step * direction
                new_centers[min_j] -= step * direction
                new_centers = np.clip(new_centers, -0.5, 0.5)
            except Exception:
                new_centers = current_centers.copy()
                
        new_score = _true_score(new_centers)
        
        # Acceptance criterion
        if new_score > current_score:
            current_centers = new_centers
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()
        else:
            # Metropolis acceptance
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
                if np.random.random() < prob:
                    current_centers = new_centers
                    current_score = new_score
        
        T *= alpha
        if T < T_end:
            T = T_end
            
    return best_centers, best_score

def construct():
    """Main entry point combining multiple optimization strategies."""
    start_time = time.time()
    np.random.seed(42)

    # 1. Generate initial configurations
    inits = []
    init_hex = _init_hexagonal()
    inits.append(init_hex)
    init_grid = _init_grid()
    inits.append(init_grid)
    
    # Add a jittered grid
    init_jitter = init_grid + np.random.randn(N, 2) * 0.01
    init_jitter = np.clip(init_jitter, -0.5, 0.5)
    inits.append(init_jitter)
    
    best_centers = None
    best_score = -1

    # 2. For each init, do Delaunay Refine + Simulated Annealing
    for init in inits:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 30:
            break

        # Initial score
        score_init = _true_score(init)
        if score_init > best_score:
            best_score = score_init
            best_centers = init.copy()

        # Delaunay Refine (fast local improvement)
        t_refine = 5.0
        if time.time() - start_time + t_refine > DEADLINE - 10:
            t_refine = max(0, DEADLINE - 10 - (time.time() - start_time))
            
        if t_refine > 1:
            refined, score_refined = _delaunay_refine(init, time_limit=t_refine)
            if score_refined > best_score:
                best_score = score_refined
                best_centers = refined.copy()
            else:
                # Use refined config for SA even if score didn't improve immediately?
                # Usually refine improves it.
                pass

        # Simulated Annealing (global search within local basin)
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining > 15 and best_centers is not None:
            t_sa = min(remaining * 0.5, 20.0)
            centers_sa, score_sa = _simulated_annealing(best_centers, time_budget=t_sa)
            if score_sa > best_score:
                best_score = score_sa
                best_centers = centers_sa.copy()
        
        # Another round of Delaunay Refine on the SA result
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining > 5 and best_centers is not None:
            t_refine2 = min(remaining * 0.5, 5.0)
            if t_refine2 > 1:
                refined2, score_refined2 = _delaunay_refine(best_centers, time_limit=t_refine2)
                if score_refined2 > best_score:
                    best_score = score_refined2
                    best_centers = refined2.copy()

    # 3. Final Polish with a very short SA run on the best result
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 5 and best_centers is not None:
        t_final_sa = min(remaining * 0.8, 10.0)
        centers_final, score_final = _simulated_annealing(best_centers, time_budget=t_final_sa)
        if score_final > best_score:
            best_score = score_final
            best_centers = centers_final.copy()
            
        # Final Delaunay Refine
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining > 2 and best_centers is not None:
            t_final_refine = min(remaining * 0.5, 2.0)
            if t_final_refine > 0.5:
                refined_final, score_final_refine = _delaunay_refine(best_centers, time_limit=t_final_refine)
                if score_final_refine > best_score:
                    best_score = score_final_refine
                    best_centers = refined_final.copy()

    # Ensure validity
    if best_centers is None:
        best_centers = _init_grid()
    best_centers = np.clip(best_centers, -0.5, 0.5)

    return best_centers

if __name__ == "__main__":
    # Quick test
    t0 = time.time()
    c = construct()
    print(f"Time: {time.time()-t0:.2f}s")
    print(f"Score: {_true_score(c)}")
    print(f"Shape: {c.shape}")
