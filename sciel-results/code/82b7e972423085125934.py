# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Pack 32 circles with disjoint interiors inside [0,1]^2, maximizing sum of radii.
    Uses a 'Grow and Relax' physics-inspired heuristic.
    """
    N = 32
    
    # 1. Initialization: Grid placement
    # Use a square grid for initial positions
    side = int(np.ceil(np.sqrt(N)))
    # We need exactly N circles. If side*side > N, we just take the first N.
    # A better init might be to distribute them evenly.
    xs = np.linspace(0.5/side, 1 - 0.5/side, side)
    ys = np.linspace(0.5/side, 1 - 0.5/side, side)
    
    circles = np.zeros((N, 3))
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= N:
                break
            circles[count, 0] = xs[i]
            circles[count, 1] = ys[j]
            # Initial radius: half the grid spacing
            circles[count, 2] = 0.5 / side * 0.9 
            count += 1
            
    # Ensure we have N circles
    if count < N:
        # Fill remaining with small circles in center (should not happen for N=32, side=6 -> 36 cells)
        for k in range(count, N):
            circles[k] = [0.5, 0.5, 0.01]

    best_circles = circles.copy()
    best_score = np.sum(best_circles[:, 2])
    
    # 2. Relaxation Function
    # Attempts to resolve overlaps and wall collisions.
    # Returns True if a valid configuration is found, False otherwise.
    def relax(circs, max_iters=50):
        n = len(circs)
        for _ in range(max_iters):
            moved = False
            # Resolve pairwise overlaps
            for i in range(n):
                for j in range(i + 1, n):
                    xi, yi, ri = circs[i]
                    xj, yj, rj = circs[j]
                    dx = xj - xi
                    dy = yj - yi
                    dist_sq = dx*dx + dy*dy
                    min_dist = ri + rj
                    if dist_sq < min_dist * min_dist:
                        dist = np.sqrt(dist_sq)
                        if dist < 1e-9:
                            # Random nudge if centers coincide
                            dx = np.random.uniform(-0.01, 0.01)
                            dy = np.random.uniform(-0.01, 0.01)
                            dist = np.sqrt(dx*dx + dy*dy)
                        # Push apart
                        overlap = min_dist - dist
                        # Move each by half the overlap
                        push = overlap * 0.5
                        # Normalize direction
                        nx = dx / dist
                        ny = dy / dist
                        circs[i, 0] -= nx * push
                        circs[i, 1] -= ny * push
                        circs[j, 0] += nx * push
                        circs[j, 1] += ny * push
                        moved = True
            
            # Resolve wall collisions
            for i in range(n):
                x, y, r = circs[i]
                # Left
                if x - r < 0:
                    circs[i, 0] = r
                    moved = True
                # Right
                if x + r > 1:
                    circs[i, 0] = 1 - r
                    moved = True
                # Bottom
                if y - r < 0:
                    circs[i, 1] = r
                    moved = True
                # Top
                if y + r > 1:
                    circs[i, 1] = 1 - r
                    moved = True
            
            if not moved:
                return True
        
        # Check if still valid after max iters
        return _is_valid(circs)

    def _is_valid(circs, eps=1e-7):
        n = len(circs)
        for i in range(n):
            x, y, r = circs[i]
            if x - r < -eps or x + r > 1 + eps or y - r < -eps or y + r > 1 + eps:
                return False
            if r < -eps:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt((circs[i,0]-circs[j,0])**2 + (circs[i,1]-circs[j,1])**2)
                if dist < circs[i,2] + circs[j,2] - eps:
                    return False
        return True

    # 3. Grow and Relax Loop
    # Start with a small increment and decrease it
    delta_r = 0.005
    iterations = 2000
    
    for it in range(iterations):
        # Try to grow all circles
        new_circles = circles.copy()
        new_circles[:, 2] += delta_r
        
        # Relax to fix overlaps
        if relax(new_circles, max_iters=20):
            # If valid, accept
            new_score = np.sum(new_circles[:, 2])
            if new_score > best_score:
                best_circles = new_circles.copy()
                best_score = new_score
            circles = new_circles
        else:
            # If not valid, revert and try smaller increment
            pass
            
        # Cool down the increment
        delta_r *= 0.995
        if delta_r < 1e-6:
            break
            
    # 4. Final local optimization: Try to jiggle positions to squeeze in more radius
    # This is a simple local search on positions with fixed radii (from best)
    # Or better: alternate between growing radii and relaxing.
    # Let's do a second pass with finer granularity on the best solution found.
    
    current_circles = best_circles.copy()
    delta_r_fine = 0.0005
    for it in range(1000):
        new_circles = current_circles.copy()
        new_circles[:, 2] += delta_r_fine
        if relax(new_circles, max_iters=50):
            new_score = np.sum(new_circles[:, 2])
            if new_score > best_score:
                best_circles = new_circles.copy()
                best_score = new_score
            current_circles = new_circles
        else:
            break
        delta_r_fine *= 0.99

    return best_circles
# EVOLVE-BLOCK-END
