"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _penalty_and_sum(circles):
    """Compute penalty (overlap + boundary violations) and sum of radii.
    circles: (N, 3) array of (x, y, r)."""
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    n = len(circles)
    
    # Boundary violations
    boundary_pen = 0.0
    for i in range(n):
        boundary_pen += max(0, r[i] - x[i])
        boundary_pen += max(0, r[i] - (1.0 - x[i]))
        boundary_pen += max(0, r[i] - y[i])
        boundary_pen += max(0, r[i] - (1.0 - y[i]))
    
    # Overlap violations
    overlap_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                overlap_pen += (min_dist - dist) ** 2
    
    total_penalty = boundary_pen ** 2 + overlap_pen
    radius_sum = np.sum(r)
    return total_penalty, radius_sum

def _is_valid(circles, tol=1e-9):
    """Check if circles are valid (no overlaps, inside unit square)."""
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    n = len(circles)
    
    for i in range(n):
        if x[i] < -tol or x[i] > 1 + tol or y[i] < -tol or y[i] > 1 + tol:
            return False
        if r[i] < -tol:
            return False
        if x[i] < r[i] - tol or x[i] > 1 - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > 1 - r[i] + tol:
            return False
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _construct():
    """Main optimization: use simulated annealing with gradient-based refinement."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    random.seed(42)
    np.random.seed(42)
    
    # Initialize with a reasonable starting configuration
    # Use a grid-based initialization with some perturbation
    side = 5
    init_circles = np.zeros((N, 3))
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            base_r = 0.5 / side
            # Add some randomness to break symmetry
            init_circles[idx, 0] = cx + random.uniform(-0.02, 0.02)
            init_circles[idx, 1] = cy + random.uniform(-0.02, 0.02)
            init_circles[idx, 2] = base_r * random.uniform(0.8, 1.2)
            idx += 1
    
    # Ensure initial circles are roughly valid
    for i in range(N):
        init_circles[i, 0] = np.clip(init_circles[i, 0], init_circles[i, 2], 1 - init_circles[i, 2])
        init_circles[i, 1] = np.clip(init_circles[i, 1], init_circles[i, 2], 1 - init_circles[i, 2])
    
    best_circles = init_circles.copy()
    best_penalty, best_sum = _penalty_and_sum(best_circles)
    best_score = best_sum - 100.0 * best_penalty  # Combined objective
    
    current = init_circles.copy()
    current_penalty, current_sum = _penalty_and_sum(current)
    current_score = current_sum - 100.0 * current_penalty
    
    # Simulated annealing parameters
    T_start = 0.1
    T_end = 0.001
    cooling_rate = 0.995
    T = T_start
    
    n_iterations = 0
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        # Check for timeout
        if time.time() > deadline - 5.0:
            break
        
        # Generate a perturbation
        n_perturb = random.randint(1, 3)
        candidate = current.copy()
        for _ in range(n_perturb):
            i = random.randint(0, N - 1)
            # Perturb position
            if random.random() < 0.7:
                candidate[i, 0] += random.gauss(0, 0.05)
                candidate[i, 1] += random.gauss(0, 0.05)
            # Perturb radius
            if random.random() < 0.3:
                candidate[i, 2] *= random.uniform(0.9, 1.1)
        
        # Clamp to valid range
        for i in range(N):
            candidate[i, 2] = np.clip(candidate[i, 2], 0.01, 0.5)
            candidate[i, 0] = np.clip(candidate[i, 0], 0, 1)
            candidate[i, 1] = np.clip(candidate[i, 1], 0, 1)
        
        cand_penalty, cand_sum = _penalty_and_sum(candidate)
        cand_score = cand_sum - 100.0 * cand_penalty
        
        delta_score = cand_score - current_score
        
        # Accept if better, or with probability exp(delta/T)
        if delta_score > 0 or random.random() < np.exp(delta_score / max(T, 1e-10)):
            current = candidate
            current_penalty, current_sum = cand_penalty, cand_sum
            current_score = cand_score
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_circles = current.copy()
            best_penalty, best_sum = current_penalty, current_sum
            no_improve_count = 0
        
        # Cool down
        T *= cooling_rate
        if T < T_end:
            T = T_end
        
        n_iterations += 1
        
        # If stuck for too long, restart from best with larger perturbation
        if no_improve_count > max_no_improve:
            current = best_circles.copy()
            for i in range(N):
                current[i, 0] += random.gauss(0, 0.03)
                current[i, 1] += random.gauss(0, 0.03)
                current[i, 2] *= random.uniform(0.95, 1.05)
            for i in range(N):
                current[i, 2] = np.clip(current[i, 2], 0.01, 0.5)
                current[i, 0] = np.clip(current[i, 0], 0, 1)
                current[i, 1] = np.clip(current[i, 1], 0, 1)
            current_penalty, current_sum = _penalty_and_sum(current)
            current_score = current_sum - 100.0 * current_penalty
            no_improve_count = 0
            T = max(T, 0.05)  # Reheat
    
    # Final refinement: try to fix any small violations by shrinking circles
    # and then use a local search to maximize radii
    final = best_circles.copy()
    
    # Try to improve by adjusting radii and positions with a simple local search
    for iteration in range(5000):
        if time.time() > deadline - 2.0:
            break
        
        improved = False
        for i in range(N):
            # Try to increase radius slightly
            old_r = final[i, 2]
            new_r = min(old_r * 1.001, 0.5)
            if new_r > old_r + 1e-12:
                test = final.copy()
                test[i, 2] = new_r
                # Check if still valid
                valid = True
                x, y, r = test[:, 0], test[:, 1], test[:, 2]
                # Boundary check
                if x[i] < r[i] or x[i] > 1 - r[i] or y[i] < r[i] or y[i] > 1 - r[i]:
                    valid = False
                # Overlap check
                if valid:
                    for j in range(N):
                        if j == i:
                            continue
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist = np.sqrt(dx * dx + dy * dy)
                        if dist < r[i] + r[j] - 1e-9:
                            valid = False
                            break
                if valid:
                    final = test
                    improved = True
        
        if not improved:
            # Try position adjustments
            for i in range(N):
                for dx in [-0.005, 0, 0.005]:
                    for dy in [-0.005, 0, 0.005]:
                        if dx == 0 and dy == 0:
                            continue
                        test = final.copy()
                        test[i, 0] += dx
                        test[i, 1] += dy
                        if test[i, 0] < 0 or test[i, 0] > 1 or test[i, 1] < 0 or test[i, 1] > 1:
                            continue
                        # Quick validity check
                        x, y, r = test[:, 0], test[:, 1], test[:, 2]
                        valid = True
                        if x[i] < r[i] or x[i] > 1 - r[i] or y[i] < r[i] or y[i] > 1 - r[i]:
                            continue
                        for j in range(N):
                            if j == i:
                                continue
                            ddx = x[i] - x[j]
                            ddy = y[i] - y[j]
                            dist = np.sqrt(ddx * ddx + ddy * ddy)
                            if dist < r[i] + r[j] - 1e-9:
                                valid = False
                                break
                        if valid:
                            # Check if this allows larger radius
                            final = test
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
        
        if not improved:
            break
    
    # Final validation
    if _is_valid(final):
        return final
    
    # If final isn't valid, return best that is valid
    if _is_valid(best_circles):
        return best_circles
    
    # Fallback: return a valid grid configuration
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            r = 0.5 / side - 1e-3
            cells.append(((i + 0.5) / side, (j + 0.5) / side, r))
    return np.array(cells[:N])

def construct():
    """Construct 26 disjoint circles in the unit square maximizing sum of radii."""
    return _construct()
# EVOLVE-BLOCK-END
