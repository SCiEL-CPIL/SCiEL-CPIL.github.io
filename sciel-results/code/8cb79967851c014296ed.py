import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_score = -np.inf
    best_circles = None

    def validate(circles, w, h, tol=1e-7):
        if w <= 0 or h <= 0:
            return False
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        if np.any(r < -tol):
            return False
            
        # Boundary check
        if np.any(x < r - tol) or np.any(x > w - r + tol) or np.any(y < r - tol) or np.any(y > h - r + tol):
            return False
            
        # Overlap check
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if dist < r[i] + r[j] - tol:
                    return False
        return True

    def repair(x, y, r, w, max_iters=50):
        """
        Repairs the configuration by pushing circles apart and clamping to bounds.
        Returns x, y, r.
        """
        h = 2.0 - w
        
        # Clamp to bounds first
        for i in range(N):
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - r[i])
            
        # Resolve overlaps
        for _ in range(max_iters):
            moved = False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r[i] + r[j]
                    
                    if dist < min_dist:
                        if dist < 1e-9:
                            angle = np.random.uniform(0, 2*np.pi)
                            dx = np.cos(angle)
                            dy = np.sin(angle)
                            dist = 1.0
                        else:
                            dx /= dist
                            dy /= dist
                        
                        overlap = min_dist - dist
                        push_x = dx * overlap * 0.5
                        push_y = dy * overlap * 0.5
                        
                        x[i] -= push_x
                        y[i] -= push_y
                        x[j] += push_x
                        y[j] += push_y
                        moved = True
            
            # Re-clamp to bounds after pushing
            for i in range(N):
                x[i] = np.clip(x[i], r[i], w - r[i])
                y[i] = np.clip(y[i], r[i], h - r[i])
                
            if not moved:
                break
                
        return x, y, r

    def local_search(w_init, max_time=100):
        """
        Runs local search for a fixed aspect ratio w_init.
        """
        h_init = 2.0 - w_init
        
        # Initialize with a grid
        cols = 5
        rows = 5
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count < N:
                    x[count] = (i + 0.5) * w_init / cols
                    y[count] = (j + 0.5) * h_init / rows
                    cell_w = w_init / cols
                    cell_h = h_init / rows
                    r[count] = 0.4 * min(cell_w, cell_h)
                    count += 1
        
        # Initial repair
        x, y, r = repair(x.copy(), y.copy(), r.copy(), w_init)
        current_score = np.sum(r)
        
        step_size = 0.01
        move_step = 0.05
        last_improvement_time = start_time
        
        while time.time() < deadline - 5:
            if time.time() - last_improvement_time > 10:
                step_size *= 0.8
                move_step *= 1.2
                if step_size < 0.001:
                    break
                
            improved = False
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                
                x_old = x.copy()
                y_old = y.copy()
                r_old = r.copy()
                
                action = np.random.choice(['radius', 'move'])
                
                if action == 'radius':
                    i = np.random.randint(N)
                    r[i] += step_size
                    if r[i] <= 0:
                        continue
                else:
                    i = np.random.randint(N)
                    x[i] += np.random.uniform(-move_step, move_step)
                    y[i] += np.random.uniform(-move_step, move_step)
                    
                x_new, y_new, r_new = repair(x.copy(), y.copy(), r.copy(), w_init)
                new_score = np.sum(r_new)
                
                if new_score > current_score + 1e-9:
                    x, y, r = x_new, y_new, r_new
                    current_score = new_score
                    last_improvement_time = time.time()
                    improved = True
                
            if not improved and time.time() - last_improvement_time > 5:
                for _ in range(3):
                    i = np.random.randint(N)
                    x[i] = np.random.uniform(r[i], w_init - r[i])
                    y[i] = np.random.uniform(r[i], h_init - r[i])
                x, y, r = repair(x.copy(), y.copy(), r.copy(), w_init)
                current_score = np.sum(r)
                last_improvement_time = time.time()

        return np.column_stack((x, y, r)), current_score

    # Try several aspect ratios
    aspect_ratios = [0.6, 0.8, 1.0, 1.2, 1.4]
    
    for w in aspect_ratios:
        if time.time() > deadline - 10:
            break
            
        try:
            circles, score = local_search(w)
            
            # Final validation
            if validate(circles, w, 2.0 - w):
                if score > best_score:
                    best_score = score
                    best_circles = circles
        except Exception:
            continue

    if best_circles is None:
        # Fallback: Simple grid in a square
        w = 1.0
        h = 1.0
        cells = []
        for i in range(5):
            for j in range(5):
                if len(cells) >= N:
                    break
                x = (i + 0.5) * 0.2
                y = (j + 0.5) * 0.2
                r = 0.08
                cells.append([x, y, r])
        best_circles = np.array(cells)
        
    return best_circles
