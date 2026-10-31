N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    best_result = None
    best_score = -np.inf
    best_w = 1.0
    best_h = 1.0

    def validate(w, h, x, y, r, tol=1e-9):
        if w < 0.05 or h < 0.05:
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

    def compute_max_radii(w, h, x, y):
        """Compute the maximum feasible radius for each circle given centers."""
        r = np.zeros(N)
        for i in range(N):
            # Distance to walls
            r[i] = min(x[i], w - x[i], y[i], h - y[i])
            if r[i] <= 0:
                return r
            # Distance to other circles
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r[i] = min(r[i], d / 2.0)
            r[i] = max(0.0, r[i])
        return r

    def local_refine(w, h, x, y, max_iters=100, step_size=0.01):
        """
        Coordinate descent refinement: move centers to increase sum of max radii.
        This is the refinement machinery from the DE approach.
        """
        x = x.copy().astype(float)
        y = y.copy().astype(float)
        
        def get_score(cx, cy):
            radii = compute_max_radii(w, h, cx, cy)
            return np.sum(radii), radii
        
        current_score, current_radii = get_score(x, y)
        best_x = x.copy()
        best_y = y.copy()
        best_score_local = current_score
        best_radii = current_radii.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            # Adaptive step size
            for i in range(N):
                # Try perturbations in x
                for dx in [-step_size, step_size]:
                    nx = x[i] + dx
                    if 0.01 < nx < w - 0.01:
                        new_x = x.copy()
                        new_x[i] = nx
                        new_score, _ = get_score(new_x, y)
                        if new_score > current_score + 1e-12:
                            x[i] = nx
                            current_score = new_score
                            improved = True
                            if current_score > best_score_local:
                                best_score_local = current_score
                                best_x = x.copy()
                                best_y = y.copy()
                                best_radii = compute_max_radii(w, h, x, y)
                
                # Try perturbations in y
                for dy in [-step_size, step_size]:
                    ny = y[i] + dy
                    if 0.01 < ny < h - 0.01:
                        new_y = y.copy()
                        new_y[i] = ny
                        new_score, _ = get_score(x, new_y)
                        if new_score > current_score + 1e-12:
                            y[i] = ny
                            current_score = new_score
                            improved = True
                            if current_score > best_score_local:
                                best_score_local = current_score
                                best_x = x.copy()
                                best_y = y.copy()
                                best_radii = compute_max_radii(w, h, x, y)
            
            if not improved:
                step_size *= 0.7
                if step_size < 1e-6:
                    break
            else:
                # Keep step size or slightly increase if improving fast
                step_size = min(step_size * 1.05, 0.05)
        
        return best_x, best_y, best_radii

    def spectral_init(w, h):
        """
        Spectral graph embedding initialization.
        Uses Laplacian eigenvalue decomposition to find low-energy configurations.
        """
        # Build a random weighted adjacency matrix
        A = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                # Random weights with some structure
                if random.random() < 0.3:
                    A[i, j] = A[j, i] = random.uniform(0.5, 1.0)
        
        # Compute Laplacian L = D - A
        D = np.diag(A.sum(axis=1))
        L = D - A
        
        # Get eigenvalues and eigenvectors
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
            # Use the second and third eigenvectors (Fiedler vectors) for 2D embedding
            # Skip the first (trivial) eigenvector
            if eigvecs.shape[1] >= 3:
                v1 = eigvecs[:, 1]
                v2 = eigvecs[:, 2]
            else:
                v1 = np.random.randn(N)
                v2 = np.random.randn(N)
            
            # Normalize and scale to fit in the box
            v1 = (v1 - v1.min()) / (v1.max() - v1.min() + 1e-10)
            v2 = (v2 - v2.min()) / (v2.max() - v2.min() + 1e-10)
            
            x = v1 * (w - 0.2) + 0.1
            y = v2 * (h - 0.2) + 0.1
            
            # Add small random perturbation to break symmetry
            x += np.random.uniform(-0.02, 0.02, N)
            y += np.random.uniform(-0.02, 0.02, N)
            
            # Clip to valid range
            x = np.clip(x, 0.05, w - 0.05)
            y = np.clip(y, 0.05, h - 0.05)
            
            # Compute initial radii
            r = compute_max_radii(w, h, x, y)
            
            return x, y, r
            
        except Exception:
            # Fallback to random
            x = np.random.uniform(0.1, w - 0.1, N)
            y = np.random.uniform(0.1, h - 0.1, N)
            r = compute_max_radii(w, h, x, y)
            return x, y, r

    def hexagonal_init(w, h):
        """Initialize circles in a hexagonal pattern."""
        x = np.zeros(N)
        y = np.zeros(N)
        
        spacing = 0.15
        rows = 0
        count = 0
        while count < N:
            cols = int(w / spacing) + 1
            if rows % 2 == 1:
                cols = int(w / spacing)
            count += cols
            rows += 1
        
        idx = 0
        for row in range(rows):
            if idx >= N:
                break
            n_in_row = int(w / spacing) + 1 if row % 2 == 0 else int(w / spacing)
            offset = spacing / 2.0 if row % 2 == 1 else 0.0
            for col in range(n_in_row):
                if idx >= N:
                    break
                cx = offset + (col + 0.5) * spacing
                cy = (row + 0.5) * spacing * 0.866
                if cx > w or cy > h:
                    continue
                x[idx] = cx
                y[idx] = cy
                idx += 1
        
        while idx < N:
            x[idx] = w / 2.0 + random.uniform(-0.2, 0.2)
            y[idx] = h / 2.0 + random.uniform(-0.2, 0.2)
            idx += 1
        
        r = compute_max_radii(w, h, x, y)
        return x, y, r

    def grid_init(w, h):
        """Initialize circles in a grid pattern."""
        x = np.zeros(N)
        y = np.zeros(N)
        
        n_cols = int(np.round(np.sqrt(N * w / h)))
        n_rows = int(np.ceil(N / n_cols))
        while n_cols * n_rows < N:
            n_cols += 1
        
        cell_w = w / n_cols
        cell_h = h / n_rows
        
        idx = 0
        for i in range(n_rows):
            for j in range(n_cols):
                if idx >= N:
                    break
                x[idx] = (j + 0.5) * cell_w
                y[idx] = (i + 0.5) * cell_h
                idx += 1
        
        r = compute_max_radii(w, h, x, y)
        return x, y, r

    def random_init(w, h):
        """Random initialization."""
        x = np.random.uniform(0.1, w - 0.1, N)
        y = np.random.uniform(0.1, h - 0.1, N)
        r = compute_max_radii(w, h, x, y)
        return x, y, r

    def force_relax(w, h, x, y, r, max_iters=2000, dt=0.01, damping=0.9):
        """
        Force-directed relaxation for fine-tuning.
        """
        x = x.copy().astype(float)
        y = y.copy().astype(float)
        r = r.copy().astype(float)
        
        best_local_x = x.copy()
        best_local_y = y.copy()
        best_local_r = r.copy()
        best_local_score = -np.inf
        
        if validate(w, h, x, y, r):
            best_local_score = np.sum(r)
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = r[i] + r[j]
                    if dist < 1e-10:
                        dx = random.uniform(-0.01, 0.01)
                        dy = random.uniform(-0.01, 0.01)
                        dist = np.sqrt(dx * dx + dy * dy)
                    
                    if dist < min_dist:
                        overlap = min_dist - dist
                        force = 10.0 * overlap
                        fx[i] += force * dx / dist
                        fy[i] += force * dy / dist
                        fx[j] -= force * dx / dist
                        fy[j] -= force * dy / dist
            
            for i in range(N):
                if x[i] < r[i]:
                    fx[i] += 10.0 * (r[i] - x[i])
                if w - x[i] < r[i]:
                    fx[i] -= 10.0 * (r[i] - (w - x[i]))
                if y[i] < r[i]:
                    fy[i] += 10.0 * (r[i] - y[i])
                if h - y[i] < r[i]:
                    fy[i] -= 10.0 * (r[i] - (h - y[i]))
            
            cx, cy = w / 2.0, h / 2.0
            for i in range(N):
                fx[i] += 0.01 * (cx - x[i])
                fy[i] += 0.01 * (cy - y[i])
            
            x += dt * fx * damping
            y += dt * fy * damping
            
            for i in range(N):
                r[i] = max(0.01, r[i])
                x[i] = np.clip(x[i], r[i], w - r[i])
                y[i] = np.clip(y[i], r[i], h - r[i])
            
            if validate(w, h, x, y, r):
                score = np.sum(r)
                if score > best_local_score:
                    best_local_score = score
                    best_local_x = x.copy()
                    best_local_y = y.copy()
                    best_local_r = r.copy()
        
        return best_local_x, best_local_y, best_local_r

    # Main search loop
    np.random.seed(42)
    random.seed(42)
    
    # Try different aspect ratios
    aspect_ratios = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    
    # Initialize with multiple strategies including spectral
    init_strategies = ['spectral', 'hex', 'grid', 'random']
    
    for aspect in aspect_ratios:
        if time.time() > deadline - 15.0:
            break
        
        # Calculate w, h from aspect ratio
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 - w
        
        for strategy in init_strategies:
            if time.time() > deadline - 10.0:
                break
            
            if strategy == 'spectral':
                x0, y0, r0 = spectral_init(w, h)
            elif strategy == 'hex':
                x0, y0, r0 = hexagonal_init(w, h)
            elif strategy == 'grid':
                x0, y0, r0 = grid_init(w, h)
            elif strategy == 'random':
                x0, y0, r0 = random_init(w, h)
            else:
                continue
            
            # First, use local refinement (coordinate descent)
            x_ref, y_ref, r_ref = local_refine(w, h, x0, y0, max_iters=80, step_size=0.015)
            update_best(w, h, x_ref, y_ref, r_ref)
            
            # Then, force relaxation for fine-tuning
            if time.time() < deadline - 8.0:
                x_force, y_force, r_force = force_relax(w, h, x_ref, y_ref, r_ref, max_iters=1500)
                update_best(w, h, x_force, y_force, r_force)

    # Final polish on best result
    if best_result is not None and time.time() < deadline - 3.0:
        x = best_result[:, 0]
        y = best_result[:, 1]
        r = best_result[:, 2]
        w = best_w
        h = best_h
        
        # Final local refinement
        x_polished, y_polished, r_polished = local_refine(w, h, x, y, max_iters=50, step_size=0.005)
        update_best(w, h, x_polished, y_polished, r_polished)

    # Fallback
    if best_result is None:
        w, h = 1.0, 1.0
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        n_cols = 5
        for i in range(N):
            row = i // 5
            col = i % 5
            x[i] = (col + 0.5) * 0.2
            y[i] = (row + 0.5) * 0.2
            r[i] = 0.09
        best_result = np.column_stack([x, y, r])

    return best_result
# EVOLVE-BLOCK-END
