import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 26
DEADLINE = 110.0  # Safety margin for 120s limit

def construct():
    start_time = time.time()
    
    def get_state(x, y, r):
        return np.concatenate([x, y, r])

    def unpack_state(z):
        x = z[0:N]
        y = z[N:2*N]
        r = z[2*N:3*N]
        return x, y, r

    def sum_radii(x, y, r):
        return np.sum(r)

    def enforce_constraints(x, y, r, max_iter=50):
        """
        Directly enforce non-overlap and boundary constraints by adjusting positions and radii.
        This is a projection-based approach: after each gradient step, we project back to feasibility.
        """
        x = x.copy()
        y = y.copy()
        r = r.copy()
        
        for _ in range(max_iter):
            changed = False
            
            # Enforce boundary constraints
            for i in range(N):
                # Circle must be inside [0,1]^2
                if x[i] < r[i]:
                    x[i] = r[i]
                    changed = True
                if x[i] > 1 - r[i]:
                    x[i] = 1 - r[i]
                    changed = True
                if y[i] < r[i]:
                    y[i] = r[i]
                    changed = True
                if y[i] > 1 - r[i]:
                    y[i] = 1 - r[i]
                    changed = True
            
            # Enforce non-overlap by moving circles apart or shrinking radii
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r[i] + r[j]
                    
                    if dist < min_dist:
                        # Overlap detected
                        if dist < 1e-12:
                            # Circles are at same point, push apart randomly
                            angle = np.random.uniform(0, 2*np.pi)
                            dx = np.cos(angle)
                            dy = np.sin(angle)
                            dist = 1e-12
                        else:
                            # Normalize direction
                            dx /= dist
                            dy /= dist
                        
                        overlap = min_dist - dist
                        # Move each circle half the overlap distance apart
                        move = overlap / 2.0 + 1e-8
                        x[i] += move * dx
                        y[i] += move * dy
                        x[j] -= move * dx
                        y[j] -= move * dy
                        changed = True
                        
                        # Re-enforce boundaries for these two circles
                        for idx in [i, j]:
                            if x[idx] < r[idx]:
                                x[idx] = r[idx]
                            if x[idx] > 1 - r[idx]:
                                x[idx] = 1 - r[idx]
                            if y[idx] < r[idx]:
                                y[idx] = r[idx]
                            if y[idx] > 1 - r[idx]:
                                y[idx] = 1 - r[idx]
                        
                        # If still overlapping after moving, shrink radii
                        dx2 = x[i] - x[j]
                        dy2 = y[i] - y[j]
                        dist2 = np.sqrt(dx2*dx2 + dy2*dy2)
                        if dist2 < r[i] + r[j]:
                            # Shrink both radii
                            max_r_i = min(x[i], 1-x[i], y[i], 1-y[i])
                            max_r_j = min(x[j], 1-x[j], y[j], 1-y[j])
                            new_r_i = min(r[i], max_r_i * 0.99)
                            new_r_j = min(r[j], max_r_j * 0.99)
                            # Scale down so they just touch
                            if new_r_i + new_r_j > dist2:
                                scale = dist2 / (new_r_i + new_r_j) * 0.999
                                new_r_i *= scale
                                new_r_j *= scale
                            r[i] = max(0.001, new_r_i)
                            r[j] = max(0.001, new_r_j)
                            changed = True
            
            if not changed:
                break
        
        return x, y, r

    def compute_forces(x, y, r):
        """
        Compute force-directed forces:
        - Repulsion between overlapping circles
        - Attraction to grow radii (handled separately)
        - Boundary repulsion
        """
        fx = np.zeros(N)
        fy = np.zeros(N)
        fr = np.zeros(N)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                
                if dist < 1e-12:
                    angle = np.random.uniform(0, 2*np.pi)
                    dx = np.cos(angle)
                    dy = np.sin(angle)
                    dist = 1e-12
                else:
                    dx /= dist
                    dy /= dist
                
                if dist < min_dist:
                    # Strong repulsion when overlapping
                    overlap = min_dist - dist
                    force = 10.0 * overlap + 1.0
                    fx[i] += force * dx
                    fy[i] += force * dy
                    fx[j] -= force * dx
                    fy[j] -= force * dy
                    # Reduce radii when overlapping
                    fr[i] -= 0.5 * overlap
                    fr[j] -= 0.5 * overlap
                elif dist < min_dist * 1.5:
                    # Mild repulsion when close
                    force = 0.5 * (min_dist * 1.5 - dist)
                    fx[i] += force * dx
                    fy[i] += force * dy
                    fx[j] -= force * dx
                    fy[j] -= force * dy
        
        # Boundary forces
        for i in range(N):
            margin = 0.05
            if x[i] < r[i] + margin:
                fx[i] += 5.0 * (r[i] + margin - x[i])
            if x[i] > 1 - r[i] - margin:
                fx[i] -= 5.0 * (x[i] - (1 - r[i] - margin))
            if y[i] < r[i] + margin:
                fy[i] += 5.0 * (r[i] + margin - y[i])
            if y[i] > 1 - r[i] - margin:
                fy[i] -= 5.0 * (y[i] - (1 - r[i] - margin))
        
        # Growth force: try to increase radii
        fr += 0.3
        
        return fx, fy, fr

    def force_directed_optimize(x0, y0, r0, max_iter=2000, lr=0.01, cooling=0.999):
        """
        Force-directed optimization with constraint enforcement.
        """
        x = x0.copy()
        y = y0.copy()
        r = r0.copy()
        
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        best_score = sum_radii(x, y, r)
        
        current_lr = lr
        rng = np.random.default_rng(42)
        
        for it in range(max_iter):
            # Compute forces
            fx, fy, fr = compute_forces(x, y, r)
            
            # Propose move
            noise_scale = 0.01 * (1 - it / max_iter)
            x_prop = x + current_lr * fx + rng.normal(0, noise_scale, N)
            y_prop = y + current_lr * fy + rng.normal(0, noise_scale, N)
            r_prop = r + current_lr * fr * 0.1 + rng.normal(0, noise_scale * 0.1, N)
            
            # Clip to valid ranges
            x_prop = np.clip(x_prop, 0.001, 0.999)
            y_prop = np.clip(y_prop, 0.001, 0.999)
            r_prop = np.clip(r_prop, 0.001, 0.49)
            
            # Enforce constraints
            x_new, y_new, r_new = enforce_constraints(x_prop, y_prop, r_prop, max_iter=10)
            
            # Check if this is an improvement
            score = sum_radii(x_new, y_new, r_new)
            if score > best_score:
                best_score = score
                best_x, best_y, best_r = x_new.copy(), y_new.copy(), r_new.copy()
            
            # Accept if not worse (or with some probability)
            if score >= sum_radii(x, y, r) or np.random.random() < 0.1:
                x, y, r = x_new, y_new, r_new
            
            # Cool down
            current_lr *= cooling
            
            # Time check
            if time.time() - start_time > DEADLINE - 5:
                break
        
        return best_x, best_y, best_r

    def objective(z, penalty_weight=1000.0):
        x, y, r = unpack_state(z)
        
        obj = -np.sum(r)
        
        # Boundary violations
        viol = np.maximum(0, r - x) + np.maximum(0, r - y) + np.maximum(0, r - (1-x)) + np.maximum(0, r - (1-y))
        boundary_pen = np.sum(viol**2)
        
        # Overlap violations
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap_pen += (min_dist - dist)**2
        
        return obj + penalty_weight * (boundary_pen + overlap_pen)

    def gradient(z, penalty_weight=1000.0):
        x, y, r = unpack_state(z)
        grad = np.zeros(3*N)
        
        grad[2*N:3*N] = -1.0
        
        # Boundary gradients
        for i in range(N):
            if r[i] > x[i]:
                grad[i] -= 2 * penalty_weight * (r[i] - x[i])
                grad[2*N+i] += 2 * penalty_weight * (r[i] - x[i])
            if r[i] > y[i]:
                grad[N+i] -= 2 * penalty_weight * (r[i] - y[i])
                grad[2*N+i] += 2 * penalty_weight * (r[i] - y[i])
            if r[i] > 1 - x[i]:
                grad[i] += 2 * penalty_weight * (r[i] - (1-x[i]))
                grad[2*N+i] -= 2 * penalty_weight * (r[i] - (1-x[i]))
            if r[i] > 1 - y[i]:
                grad[N+i] += 2 * penalty_weight * (r[i] - (1-y[i]))
                grad[2*N+i] -= 2 * penalty_weight * (r[i] - (1-y[i]))
        
        # Overlap gradients
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                min_dist = r[i] + r[j]
                
                if dist < min_dist:
                    diff = min_dist - dist
                    factor = 2 * penalty_weight * diff
                    
                    grad[i] += factor * (-dx / dist)
                    grad[N+i] += factor * (-dy / dist)
                    grad[j] += factor * (dx / dist)
                    grad[N+j] += factor * (dy / dist)
                    grad[2*N+i] += factor
                    grad[2*N+j] += factor
        
        return grad

    def local_optimize(x0, y0, r0, max_iter=300, penalty_weight=1000.0):
        """Run local optimization from initial guess."""
        z0 = get_state(x0, y0, r0)
        
        bounds = []
        for i in range(N):
            bounds.append((0.001, 0.999))
            bounds.append((0.001, 0.999))
            bounds.append((0.001, 0.49))
            
        try:
            res = minimize(
                objective, 
                z0, 
                method='L-BFGS-B', 
                jac=gradient, 
                bounds=bounds,
                args=(penalty_weight,),
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            x_opt, y_opt, r_opt = unpack_state(res.x)
            r_opt = np.maximum(r_opt, 0.001)
            
            # Enforce constraints
            x_opt, y_opt, r_opt = enforce_constraints(x_opt, y_opt, r_opt, max_iter=30)
            
            return x_opt, y_opt, r_opt
        except Exception:
            return None

    def hexagonal_init(n_circles, perturb=0.0, seed=42):
        """Create hexagonal packing initialization."""
        rng = np.random.default_rng(seed)
        x = np.zeros(n_circles)
        y = np.zeros(n_circles)
        r = np.zeros(n_circles)
        
        # Try different grid configurations
        configs = [
            (5, 6),  # rows, cols
            (6, 5),
            (4, 7),
            (7, 4),
            (5, 5),  # with some extra
        ]
        
        best_config = None
        best_score = -1
        
        for rows, cols in configs:
            total = rows * cols
            if total < n_circles:
                continue
            
            x_test = np.zeros(n_circles)
            y_test = np.zeros(n_circles)
            r_test = np.zeros(n_circles)
            
            idx = 0
            for row in range(rows):
                for col in range(cols):
                    if idx >= n_circles:
                        break
                    x_offset = 0.5 if row % 2 == 1 else 0.0
                    x_test[idx] = (col + 0.5 + x_offset) / (cols + 0.5)
                    y_test[idx] = (row + 0.5) / (rows + 0.5)
                    
                    spacing_x = 1.0 / (cols + 0.5)
                    spacing_y = 1.0 / (rows + 0.5)
                    r_test[idx] = 0.48 * min(spacing_x, spacing_y)
                    
                    idx += 1
            
            # Enforce constraints
            x_test, y_test, r_test = enforce_constraints(x_test, y_test, r_test, max_iter=20)
            score = sum_radii(x_test, y_test, r_test)
            
            if score > best_score:
                best_score = score
                best_config = (x_test.copy(), y_test.copy(), r_test.copy())
        
        if best_config is None:
            # Fallback: simple grid
            side = int(np.ceil(np.sqrt(n_circles)))
            for i in range(n_circles):
                row = i // side
                col = i % side
                x[i] = (col + 0.5) / side
                y[i] = (row + 0.5) / side
                r[i] = 0.45 / side
            x, y, r = enforce_constraints(x, y, r, max_iter=20)
        else:
            x, y, r = best_config
        
        if perturb > 0:
            x += rng.uniform(-perturb, perturb, n_circles)
            y += rng.uniform(-perturb, perturb, n_circles)
            x = np.clip(x, 0.001, 0.999)
            y = np.clip(y, 0.001, 0.999)
            x, y, r = enforce_constraints(x, y, r, max_iter=20)
        
        return x, y, r

    def random_init(rng, n_circles=N):
        """Random initialization with constraint enforcement."""
        x = rng.uniform(0.1, 0.9, n_circles)
        y = rng.uniform(0.1, 0.9, n_circles)
        r = rng.uniform(0.02, 0.08, n_circles)
        x, y, r = enforce_constraints(x, y, r, max_iter=20)
        return x, y, r

    best_score = 0.0
    best_state = None
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Multi-start with hexagonal and random initializations
    # Try several hexagonal starts with different perturbations
    for k in range(6):
        if time.time() - start_time > DEADLINE - 30:
            break
        x0, y0, r0 = hexagonal_init(N, perturb=0.01 * k, seed=42+k)
        result = local_optimize(x0, y0, r0, max_iter=200, penalty_weight=1000.0)
        if result is not None:
            x_opt, y_opt, r_opt = result
            score = sum_radii(x_opt, y_opt, r_opt)
            if score > best_score:
                best_score = score
                best_state = (x_opt.copy(), y_opt.copy(), r_opt.copy())
    
    # Random starts
    for k in range(8):
        if time.time() - start_time > DEADLINE - 30:
            break
        x0, y0, r0 = random_init(rng)
        result = local_optimize(x0, y0, r0, max_iter=150, penalty_weight=1000.0)
        if result is not None:
            x_opt, y_opt, r_opt = result
            score = sum_radii(x_opt, y_opt, r_opt)
            if score > best_score:
                best_score = score
                best_state = (x_opt.copy(), y_opt.copy(), r_opt.copy())
    
    # Phase 2: Force-directed refinement from best state
    if best_state is not None:
        x_best, y_best, r_best = best_state
        fd_result = force_directed_optimize(x_best, y_best, r_best, max_iter=1500, lr=0.02, cooling=0.9995)
        if fd_result is not None:
            x_opt, y_opt, r_opt = fd_result
            score = sum_radii(x_opt, y_opt, r_opt)
            if score > best_score:
                best_score = score
                best_state = (x_opt.copy(), y_opt.copy(), r_opt.copy())
    
    # Phase 3: Final local optimization polish
    if best_state is not None:
        x_best, y_best, r_best = best_state
        for pw in [500.0, 2000.0]:
            if time.time() - start_time > DEADLINE - 5:
                break
            result = local_optimize(x_best, y_best, r_best, max_iter=200, penalty_weight=pw)
            if result is not None:
                x_opt, y_opt, r_opt = result
                score = sum_radii(x_opt, y_opt, r_opt)
                if score > best_score:
                    best_score = score
                    best_state = (x_opt.copy(), y_opt.copy(), r_opt.copy())
    
    # Final fallback
    if best_state is None:
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row = i // side
            col = i % side
            x[i] = (col + 0.5) / side
            y[i] = (row + 0.5) / side
            r[i] = 0.45 / side
        x, y, r = enforce_constraints(x, y, r, max_iter=30)
        best_state = (x, y, r)
    
    x_final, y_final, r_final = best_state
    
    # Final constraint enforcement
    x_final, y_final, r_final = enforce_constraints(x_final, y_final, r_final, max_iter=50)
    
    result = np.column_stack([x_final, y_final, r_final])
    return result

if __name__ == "__main__":
    res = construct()
    print(np.sum(res[:, 2]))
