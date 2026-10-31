import numpy as np
import time

N = 21

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill
    
    # Start with a grid layout in a rectangle
    # Try different aspect ratios and grid configurations
    
    def evaluate(circles, w):
        """Check validity and compute score. circles: (N,3) array of (x,y,r)"""
        h = 2.0 - w
        if w <= 0 or h <= 0:
            return -1.0
        for i in range(N):
            x, y, r = circles[i]
            if r <= 0:
                return -1.0
            if x - r < -1e-9 or x + r > w + 1e-9:
                return -1.0
            if y - r < -1e-9 or y + r > h + 1e-9:
                return -1.0
        for i in range(N):
            for j in range(i+1, N):
                dx = circles[i,0] - circles[j,0]
                dy = circles[i,1] - circles[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < circles[i,2] + circles[j,2] - 1e-9:
                    return -1.0
        return np.sum(circles[:, 2])
    
    def initial_grid(rows, cols, w):
        """Create initial grid of circles in rectangle of width w, height 2-w"""
        h = 2.0 - w
        n = rows * cols
        if n < N:
            # Pad with extra circles
            circles = np.zeros((N, 3))
        else:
            circles = np.zeros((N, 3))
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= N:
                    break
                x = (j + 0.5) * w / cols
                y = (i + 0.5) * h / rows
                r = min(w / cols, h / rows) * 0.49
                circles[idx] = [x, y, r]
                idx += 1
        # Fill remaining circles with small radius at random positions
        while idx < N:
            circles[idx] = [np.random.uniform(0.1, w-0.1), np.random.uniform(0.1, h-0.1), 0.05]
            idx += 1
        return circles
    
    def optimize(circles, w, max_iter=5000):
        """Iterative local optimization"""
        h = 2.0 - w
        circles = circles.copy()
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            lr = 0.01  # learning rate for position adjustments
            
            # Compute forces on each circle
            forces_x = np.zeros(N)
            forces_y = np.zeros(N)
            
            # Wall constraints
            for i in range(N):
                x, y, r = circles[i]
                # Left wall
                if x - r < 0:
                    forces_x[i] += (x - r) * 10
                # Right wall
                if x + r > w:
                    forces_x[i] -= (x + r - w) * 10
                # Bottom wall
                if y - r < 0:
                    forces_y[i] += (y - r) * 10
                # Top wall
                if y + r > h:
                    forces_y[i] -= (y + r - h) * 10
            
            # Circle-circle repulsion
            for i in range(N):
                for j in range(i+1, N):
                    dx = circles[i,0] - circles[j,0]
                    dy = circles[i,1] - circles[j,1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = circles[i,2] + circles[j,2]
                    if dist < min_dist and dist > 1e-12:
                        overlap = min_dist - dist
                        fx = (dx / dist) * overlap
                        fy = (dy / dist) * overlap
                        forces_x[i] += fx
                        forces_y[i] += fy
                        forces_x[j] -= fx
                        forces_y[j] -= fy
                    elif dist <= 1e-12:
                        # Same position, push apart randomly
                        angle = np.random.uniform(0, 2*np.pi)
                        forces_x[i] += np.cos(angle)
                        forces_y[i] += np.sin(angle)
                        forces_x[j] -= np.cos(angle)
                        forces_y[j] -= np.sin(angle)
            
            # Apply forces
            for i in range(N):
                circles[i,0] += forces_x[i] * lr
                circles[i,1] += forces_y[i] * lr
            
            # Try to grow radii slightly
            grow_factor = 1.0 + 0.001
            for i in range(N):
                new_r = circles[i,2] * grow_factor
                # Check if this circle can grow
                valid = True
                x, y = circles[i,0], circles[i,1]
                if x - new_r < 0 or x + new_r > w or y - new_r < 0 or y + new_r > h:
                    valid = False
                else:
                    for j in range(N):
                        if i == j:
                            continue
                        dx = x - circles[j,0]
                        dy = y - circles[j,1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < new_r + circles[j,2] - 1e-12:
                            valid = False
                            break
                if valid:
                    circles[i,2] = new_r
                    improved = True
            
            if not improved and it > 100:
                # Try to adjust positions more aggressively
                for i in range(N):
                    circles[i,0] += np.random.uniform(-0.02, 0.02)
                    circles[i,1] += np.random.uniform(-0.02, 0.02)
        
        return circles
    
    # Try multiple configurations
    best_score = 0
    best_circles = None
    best_w = 1.0
    
    configs = [
        (5, 5, 1.0),
        (6, 4, 1.2),
        (4, 6, 0.8),
        (7, 3, 1.3),
        (3, 7, 0.7),
        (5, 4, 1.1),
        (4, 5, 0.9),
    ]
    
    for rows, cols, w_init in configs:
        if time.time() > deadline:
            break
        circles = initial_grid(rows, cols, w_init)
        circles = optimize(circles, w_init, max_iter=3000)
        
        # Optimize width
        for _ in range(20):
            if time.time() > deadline:
                break
            w_new = w_init + np.random.uniform(-0.1, 0.1)
            w_new = np.clip(w_new, 0.5, 1.5)
            circles2 = circles.copy()
            # Scale positions to new width
            h_old = 2.0 - w_init
            h_new = 2.0 - w_new
            circles2[:, 0] *= w_new / w_init
            circles2[:, 1] *= h_new / h_old
            # Scale radii
            scale = min(w_new / w_init, h_new / h_old)
            circles2[:, 2] *= scale
            circles2 = optimize(circles2, w_new, max_iter=1000)
            
            score = evaluate(circles2, w_new)
            if score > best_score:
                best_score = score
                best_circles = circles2.copy()
                best_w = w_new
        
        score = evaluate(circles, w_init)
        if score > best_score:
            best_score = score
            best_circles = circles.copy()
            best_w = w_init
    
    # Final polish
    if best_circles is not None and time.time() < deadline - 5:
        best_circles = optimize(best_circles, best_w, max_iter=5000)
    
    if best_circles is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_circles
