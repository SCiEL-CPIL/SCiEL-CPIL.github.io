"""Circle packing in unit square n=32: maximize sum of radii via simulated annealing."""
import numpy as np
import time
import math

N = 32
DEADLINE_SECONDS = 115.0  # safety margin under 120s limit


def _clamp_circles(circles):
    """Ensure circles are inside [0,1]^2 by clamping centers and reducing radii if needed."""
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    # Clamp radii to be non-negative
    r = np.maximum(r, 1e-9)
    # Clamp centers so that circles stay inside [0,1]^2
    x = np.clip(x, r, 1.0 - r)
    y = np.clip(y, r, 1.0 - r)
    return np.column_stack([x, y, r])


def _objective(circles):
    """Return sum of radii minus a penalty for overlaps and boundary violations."""
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Penalty for circles going outside [0,1]^2
    boundary_penalty = 0.0
    for i in range(N):
        if x[i] < r[i]:
            boundary_penalty += (r[i] - x[i]) ** 2
        if x[i] > 1.0 - r[i]:
            boundary_penalty += (x[i] - (1.0 - r[i])) ** 2
        if y[i] < r[i]:
            boundary_penalty += (r[i] - y[i]) ** 2
        if y[i] > 1.0 - r[i]:
            boundary_penalty += (y[i] - (1.0 - r[i])) ** 2
    
    # Penalty for overlapping circles
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = math.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                overlap_penalty += (min_dist - dist) ** 2
    
    return np.sum(r) - 10.0 * (boundary_penalty + overlap_penalty)


def _is_valid(circles, tol=1e-6):
    """Check if circles are disjoint and inside [0,1]^2."""
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    for i in range(N):
        if x[i] < -tol or x[i] > 1.0 + tol or y[i] < -tol or y[i] > 1.0 + tol:
            return False
        if r[i] < -tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _make_hexagonal_init():
    """Create a hexagonal packing initialization for 32 circles."""
    # Use a hexagonal grid pattern
    # For 32 circles, use roughly 6x6 grid with hex offset
    cols = 6
    rows = 6
    dx = 1.0 / cols
    dy = dx * math.sqrt(3) / 2
    r_base = dx * 0.45
    
    circles = []
    for j in range(rows):
        for i in range(cols):
            if len(circles) >= N:
                break
            x = (i + 0.5) * dx + (0.5 * dx if j % 2 else 0)
            y = (j + 0.5) * dy
            # Scale to fit in [0,1]
            if y > 1.0:
                break
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
            circles.append([x, y, r_base])
    
    # If we don't have enough, add random ones
    while len(circles) < N:
        circles.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9), 0.05])
    
    return np.array(circles[:N])


def _make_random_init():
    """Create a random initialization."""
    x = np.random.uniform(0.1, 0.9, N)
    y = np.random.uniform(0.1, 0.9, N)
    r = np.random.uniform(0.03, 0.08, N)
    return np.column_stack([x, y, r])


def _make_grid_init():
    """Create a grid initialization."""
    side = int(math.ceil(math.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 0.001
            cells.append([x, y, r])
    return np.array(cells[:N])


def _simulated_annealing(init_circles, t_start=1.0, t_end=0.001, n_iter=50000, deadline=None):
    """Run simulated annealing to optimize circle packing."""
    circles = init_circles.copy()
    best_circles = circles.copy()
    best_score = _objective(circles)
    
    current_score = best_score
    n_accepted = 0
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Generate a perturbation
        new_circles = circles.copy()
        
        # Randomly select some circles to perturb
        n_perturb = np.random.randint(1, 4)
        idx = np.random.choice(N, n_perturb, replace=False)
        
        for i in idx:
            # Perturb position
            new_circles[i, 0] += np.random.normal(0, 0.02)
            new_circles[i, 1] += np.random.normal(0, 0.02)
            # Perturb radius
            new_circles[i, 2] += np.random.normal(0, 0.005)
        
        # Clamp to valid range
        new_circles = _clamp_circles(new_circles)
        
        new_score = _objective(new_circles)
        delta = new_score - current_score
        
        # Temperature schedule
        progress = it / n_iter
        temp = t_start * (t_end / t_start) ** progress
        
        if delta > 0 or np.random.random() < math.exp(delta / max(temp, 1e-10)):
            circles = new_circles
            current_score = new_score
            n_accepted += 1
            if current_score > best_score:
                best_score = current_score
                best_circles = circles.copy()
    
    return best_circles, best_score


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    np.random.seed(42)
    
    # Generate multiple initializations
    inits = []
    inits.append(_make_hexagonal_init())
    inits.append(_make_grid_init())
    for _ in range(8):
        inits.append(_make_random_init())
    
    best_circles = None
    best_score = -np.inf
    
    # Run simulated annealing on each initialization
    for init in inits:
        if time.time() > deadline - 5:
            break
        
        # Run SA with decreasing iterations as time runs out
        remaining = deadline - time.time()
        if remaining < 2:
            break
        
        n_iter = int(min(50000, remaining * 5000))
        if n_iter < 1000:
            break
        
        circles, score = _simulated_annealing(init, t_start=0.5, t_end=0.001, n_iter=n_iter, deadline=deadline - 2)
        
        if score > best_score:
            best_score = score
            best_circles = circles
    
    # Final polish: local search on the best result
    if best_circles is not None and time.time() < deadline - 5:
        remaining = deadline - time.time()
        n_polish = int(min(20000, remaining * 3000))
        if n_polish > 500:
            polished, polished_score = _simulated_annealing(
                best_circles, t_start=0.1, t_end=0.0001, n_iter=n_polish, deadline=deadline - 1
            )
            if polished_score > best_score:
                best_circles = polished
                best_score = polished_score
    
    # Ensure validity
    if best_circles is not None:
        best_circles = _clamp_circles(best_circles)
        # Verify and fix if needed
        if not _is_valid(best_circles, tol=1e-5):
            # Fallback to grid
            best_circles = _make_grid_init()
    
    return best_circles
