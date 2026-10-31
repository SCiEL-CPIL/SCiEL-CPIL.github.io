import numpy as np
import time

N = 21

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 110.0  # 10s margin
    
    # Initial state: Grid in 1x1 square
    side = 5
    w0 = 1.0
    h0 = 1.0
    xs = np.array([(i + 0.5) / side for i in range(side)] * side)
    ys = np.array([(j + 0.5) / side for j in range(side) for _ in range(side)])
    # Reshape to N
    xs = xs[:N]
    ys = ys[:N]
    
    # Initial radii
    r0 = 0.5 / side
    R = np.full(N, r0)
    
    # State: X (N,2), R (N,), w (scalar)
    X = np.column_stack((xs, ys))
    w = 1.0
    
    # Objective function
    def evaluate(X, R, w):
        h = 2.0 - w
        # Boundary violations
        # x in [r, w-r], y in [r, h-r]
        vx = np.maximum(0, R - X[:, 0]) + np.maximum(0, X[:, 0] - (w - R))
        vy = np.maximum(0, R - X[:, 1]) + np.maximum(0, X[:, 1] - (h - R))
        boundary_pen = np.sum(vx**2) + np.sum(vy**2)
        
        # Overlap violations
        # Distance matrix
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        # Min distance required
        min_dist = R[:, np.newaxis] + R[np.newaxis, :]
        # Overlap amount
        overlap = np.maximum(0, min_dist - dist)
        # Zero diagonal
        np.fill_diagonal(overlap, 0)
        overlap_pen = np.sum(overlap**2)
        
        # Sum of radii
        sum_r = np.sum(R)
        
        # Total energy
        # We want to minimize -sum_r + penalties
        # Penalties should be large
        alpha = 100.0
        energy = -sum_r + alpha * (boundary_pen + overlap_pen)
        
        return energy, boundary_pen, overlap_pen, sum_r

    # Simulated Annealing
    T = 0.1
    T_min = 0.001
    cool_rate = 0.999
    
    best_energy, best_boundary, best_overlap, best_sum_r = evaluate(X, R, w)
    best_X = X.copy()
    best_R = R.copy()
    best_w = w
    
    # Track best valid solution
    best_valid_sum_r = -np.inf
    best_valid_X = None
    best_valid_R = None
    best_valid_w = None
    
    if best_boundary < 1e-6 and best_overlap < 1e-6:
        best_valid_sum_r = best_sum_r
        best_valid_X = X.copy()
        best_valid_R = R.copy()
        best_valid_w = w

    steps = 0
    while time.time() < deadline:
        # Perturb
        # Randomly choose what to perturb: one circle's x, y, r, or w
        choice = np.random.rand()
        if choice < 0.3:
            # Perturb w
            dw = np.random.normal(0, 0.01)
            new_w = np.clip(w + dw, 0.5, 1.5)
            new_X = X
            new_R = R
        elif choice < 0.8:
            # Perturb one circle's position and radius
            i = np.random.randint(N)
            new_X = X.copy()
            new_R = R.copy()
            dx = np.random.normal(0, 0.05)
            dy = np.random.normal(0, 0.05)
            dr = np.random.normal(0, 0.02)
            new_X[i, 0] += dx
            new_X[i, 1] += dy
            new_R[i] = np.clip(new_R[i] + dr, 0.01, 1.0)
            new_w = w
        else:
            # Perturb all positions slightly (jitter)
            new_X = X + np.random.normal(0, 0.01, X.shape)
            new_R = R
            new_w = w

        new_energy, new_boundary, new_overlap, new_sum_r = evaluate(new_X, new_R, new_w)
        delta_e = new_energy - best_energy
        
        if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
            best_energy = new_energy
            best_boundary = new_boundary
            best_overlap = new_overlap
            best_sum_r = new_sum_r
            X = new_X
            R = new_R
            w = new_w
            
            # Check validity
            if best_boundary < 1e-6 and best_overlap < 1e-6:
                if best_sum_r > best_valid_sum_r:
                    best_valid_sum_r = best_sum_r
                    best_valid_X = X.copy()
                    best_valid_R = R.copy()
                    best_valid_w = w
        
        T *= cool_rate
        if T < T_min:
            T = T_min
        steps += 1

    # If we found a valid solution, return it
    if best_valid_X is not None:
        # Final check and slight adjustment if needed
        # The evaluator requires disjoint interiors and bounding rect perimeter <= 4
        # Our constraints ensure w+h=2, so perimeter=4.
        # We need to ensure strict disjointness and interior containment.
        # The penalties were squared, so < 1e-6 is very small.
        # To be safe, we can slightly shrink radii if any violation is detected in a final check.
        
        # Final verification
        h = 2.0 - best_valid_w
        vx = np.maximum(0, best_valid_R - best_valid_X[:, 0]) + np.maximum(0, best_valid_X[:, 0] - (best_valid_w - best_valid_R))
        vy = np.maximum(0, best_valid_R - best_valid_X[:, 1]) + np.maximum(0, best_valid_X[:, 1] - (h - best_valid_R))
        if np.any(vx > 1e-5) or np.any(vy > 1e-5):
            # Shrink radii to fit boundaries
            scale = 1.0
            for _ in range(10):
                best_valid_R *= 0.99
                vx = np.maximum(0, best_valid_R - best_valid_X[:, 0]) + np.maximum(0, best_valid_X[:, 0] - (best_valid_w - best_valid_R))
                vy = np.maximum(0, best_valid_R - best_valid_X[:, 1]) + np.maximum(0, best_valid_X[:, 1] - (h - best_valid_R))
                if not (np.any(vx > 1e-5) or np.any(vy > 1e-5)):
                    break
            
        diff = best_valid_X[:, np.newaxis, :] - best_valid_X[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        min_dist = best_valid_R[:, np.newaxis] + best_valid_R[np.newaxis, :]
        overlap = np.maximum(0, min_dist - dist)
        np.fill_diagonal(overlap, 0)
        
        if np.any(overlap > 1e-5):
            # Shrink radii to resolve overlaps
            for _ in range(10):
                best_valid_R *= 0.99
                diff = best_valid_X[:, np.newaxis, :] - best_valid_X[np.newaxis, :]
                dist = np.linalg.norm(diff, axis=2)
                min_dist = best_valid_R[:, np.newaxis] + best_valid_R[np.newaxis, :]
                overlap = np.maximum(0, min_dist - dist)
                np.fill_diagonal(overlap, 0)
                if not np.any(overlap > 1e-5):
                    break

        result = np.column_stack((best_valid_X, best_valid_R))
        return result

    # Fallback to seed if no valid solution found
    side = 5
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
