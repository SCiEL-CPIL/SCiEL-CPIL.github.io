"""Combined program: Greedy placement + force-directed relaxation for circle packing."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    best_result = None
    best_score = -np.inf

    def create_grid_init(w):
        h = 2.0 - w
        aspect = w / h
        if aspect >= 1.0:
            n_cols = int(np.round(np.sqrt(N * aspect)))
            n_rows = int(np.ceil(N / n_cols))
        else:
            n_rows = int(np.round(np.sqrt(N / aspect)))
            n_cols = int(np.ceil(N / n_rows))
        
        while n_cols * n_rows < N:
            if aspect >= 1.0:
                n_cols += 1
            else:
                n_rows += 1
            
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        cell_w = w / n_cols
        cell_h = h / n_rows
        max_r = min(cell_w, cell_h) * 0.45
        
        idx = 0
        for i in range(n_rows):
            for j in range(n_cols):
                if idx >= N:
                    break
                x[idx] = (j + 0.5) * cell_w
                y[idx] = (i + 0.5) * cell_h
                r[idx] = max_r
                idx += 1
        return x, y, r

    def validate(w, h, x, y, r, tol=1e-9):
        if w < 0.1 or h < 0.1:
            return False
        if np.any(r < 0):
            return False
        if np.any(x - r < -tol) or np.any(w - x - r < -tol):
            return False
        if np.any(y - r < -tol) or np.any(h - y - r < -tol):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                if dist_sq < (r[i] + r[j] - tol) ** 2:
                    return False
        return True

    def update_best(w, h, x, y, r):
        nonlocal best_result, best_score
        if validate(w, h, x, y, r):
            x = np.clip(x, 0, w)
            y = np.clip(y, 0, h)
            r = np.clip(r, 0, None)
            
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([x, y, r])
                return True
        return False

    def force_relax(w, h, x, y, r, max_iter=300, step_size=0.05, radius_growth=0.01):
        """
        Force-directed relaxation to remove overlaps and grow radii.
        Returns updated (x, y, r).
        """
        x = x.copy()
        y = y.copy()
        r = r.copy()
        
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
                
            fx.fill(0.0)
            fy.fill(0.0)
            
            # Circle-Circle Repulsion
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist_sq = dx * dx + dy * dy
                    dist = np.sqrt(dist_sq)
                    min_dist = r[i] + r[j]
                    
                    if dist < min_dist and dist > 1e-6:
                        overlap = min_dist - dist
                        force = overlap * 10.0
                        fx[i] += force * (dx / dist)
                        fy[i] += force * (dy / dist)
                        fx[j] -= force * (dx / dist)
                        fy[j] -= force * (dy / dist)
                    elif dist <= 1e-6:
                        fx[i] += np.random.uniform(-1, 1)
                        fy[i] += np.random.uniform(-1, 1)
                        fx[j] -= np.random.uniform(-1, 1)
                        fy[j] -= np.random.uniform(-1, 1)

            # Wall Repulsion
            for i in range(N):
                if x[i] < r[i]:
                    fx[i] += (r[i] - x[i]) * 10.0
                if w - x[i] < r[i]:
                    fx[i] -= (r[i] - (w - x[i])) * 10.0
                if y[i] < r[i]:
                    fy[i] += (r[i] - y[i]) * 10.0
                if h - y[i] < r[i]:
                    fy[i] -= (r[i] - (h - y[i])) * 10.0

            # Update positions
            x += fx * step_size * 0.1
            y += fy * step_size * 0.1
            x = np.clip(x, 0, w)
            y = np.clip(y, 0, h)
            
            # Try to increase radii
            new_r = r * (1.0 + radius_growth)
            if validate(w, h, x, y, new_r, tol=1e-8):
                r = new_r
            else:
                # If not valid, keep old radii
                pass

        return x, y, r

    def greedy_place(w, h, existing_x, existing_y, existing_r, max_attempts=200):
        """
        Place a new circle at maximal radius in free space.
        Returns (x, y, r) or None.
        """
        best_r = 0.0
        best_pos = None
        
        for _ in range(max_attempts):
            px = np.random.uniform(0, w)
            py = np.random.uniform(0, h)
            
            r_candidate = min(px, w - px, py, h - py)
            for i in range(len(existing_x)):
                dx = px - existing_x[i]
                dy = py - existing_y[i]
                dist = np.sqrt(dx*dx + dy*dy)
                r_candidate = min(r_candidate, dist - existing_r[i])
            
            if r_candidate > best_r:
                best_r = r_candidate
                best_pos = (px, py)
        
        if best_pos is None or best_r <= 1e-6:
            return None
        return best_pos[0], best_pos[1], best_r

    # Main search loop
    np.random.seed(42)
    
    # Try different aspect ratios
    aspect_ratios = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    
    # Shuffle to randomize order
    np.random.shuffle(aspect_ratios)
    
    for aspect in aspect_ratios:
        if time.time() > deadline - 20.0:
            break
            
        # Calculate w, h from aspect ratio
        # w/h = aspect, w + h = 2 => w = 2*aspect/(1+aspect)
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 - w
        
        # Strategy 1: Grid init + relaxation
        x, y, r = create_grid_init(w)
        r = r * 0.5  # Start small
        x, y, r = force_relax(w, h, x, y, r, max_iter=200)
        update_best(w, h, x, y, r)
        
        # Strategy 2: Greedy placement
        if time.time() > deadline - 10.0:
            break
            
        gx = np.array([])
        gy = np.array([])
        gr = np.array([])
        
        for i in range(N):
            if time.time() > deadline - 5.0:
                break
            res = greedy_place(w, h, gx, gy, gr, max_attempts=100)
            if res is None:
                break
            gx = np.append(gx, res[0])
            gy = np.append(gy, res[1])
            gr = np.append(gr, res[2])
        
        if len(gx) == N:
            # Relax the greedy packing
            gx, gy, gr = force_relax(w, h, gx, gy, gr, max_iter=100)
            update_best(w, h, gx, gy, gr)

    # Final Polish: Try to squeeze in a bit more by reducing tolerance in validation
    # and doing a few more iterations with smaller steps on the best result found
    if best_result is not None:
        x = best_result[:, 0]
        y = best_result[:, 1]
        r = best_result[:, 2]
        
        # Infer w, h from bounding box? No, we need to know w, h.
        # Since we don't store w, h in best_result, we need to track them.
        # Let's modify update_best to store w, h.
        
        # Actually, let's just re-run a light relaxation on the best config found.
        # But we don't know which w, h it came from.
        # Let's assume the best aspect ratio was one of the tried ones.
        # A safer approach: store the best w, h.

    # Re-implementing with w, h tracking for final polish
    best_w = 1.0
    best_h = 1.0

    def update_best_tracked(w, h, x, y, r):
        nonlocal best_result, best_score, best_w, best_h
        if validate(w, h, x, y, r):
            x = np.clip(x, 0, w)
            y = np.clip(y, 0, h)
            r = np.clip(r, 0, None)
            
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([x, y, r])
                best_w = w
                best_h = h
                return True
        return False

    # Re-run main loop with tracked update
    best_result = None
    best_score = -np.inf
    
    for aspect in aspect_ratios:
        if time.time() > deadline - 20.0:
            break
            
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 - w
        
        # Strategy 1: Grid init + relaxation
        x, y, r = create_grid_init(w)
        r = r * 0.5
        x, y, r = force_relax(w, h, x, y, r, max_iter=200)
        update_best_tracked(w, h, x, y, r)
        
        # Strategy 2: Greedy placement
        if time.time() > deadline - 10.0:
            break
            
        gx = np.array([])
        gy = np.array([])
        gr = np.array([])
        
        for i in range(N):
            if time.time() > deadline - 5.0:
                break
            res = greedy_place(w, h, gx, gy, gr, max_attempts=100)
            if res is None:
                break
            gx = np.append(gx, res[0])
            gy = np.append(gy, res[1])
            gr = np.append(gr, res[2])
        
        if len(gx) == N:
            gx, gy, gr = force_relax(w, h, gx, gy, gr, max_iter=100)
            update_best_tracked(w, h, gx, gy, gr)

    # Final Polish on the best tracked result
    if best_result is not None and time.time() < deadline - 2.0:
        x = best_result[:, 0]
        y = best_result[:, 1]
        r = best_result[:, 2]
        w = best_w
        h = best_h
        
        x, y, r = force_relax(w, h, x, y, r, max_iter=100, step_size=0.01, radius_growth=0.005)
        update_best_tracked(w, h, x, y, r)

    # Fallback
    if best_result is None:
        x, y, r = create_grid_init(1.0)
        best_result = np.column_stack([x, y, r])

    return best_result
# EVOLVE-BLOCK-END
