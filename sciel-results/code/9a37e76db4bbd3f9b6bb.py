"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
def _is_valid(x, w, h, tol=1e-6):
    """Check if solution is valid: all circles inside rectangle and non-overlapping."""
    for i in range(N):
        cx, cy, r = x[i]
        if cx - r < -tol or cx + r > w + tol:
            return False
        if cy - r < -tol or cy + r > h + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i, 0] - x[j, 0]
            dy = x[i, 1] - x[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < x[i, 2] + x[j, 2] - tol:
                return False
    return True

def _build_contact_graph(x, w, h):
    """Build a simple contact graph based on proximity."""
    n = len(x)
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i, 0] - x[j, 0]
            dy = x[i, 1] - x[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            # Consider "close" if within sum of radii + small margin
            if dist < x[i, 2] + x[j, 2] + 0.05:
                adj[i].append(j)
                adj[j].append(i)
    return adj

def _topological_init(w, h, seed=42):
    """Generate a structured initialization using grid + jitter."""
    rng = np.random.RandomState(seed)
    n = N
    # Use a grid layout with some jitter
    cols = int(np.ceil(np.sqrt(n * w / h))) if h > 0 else 5
    rows = int(np.ceil(n / cols))
    cell_w = w / cols
    cell_h = h / rows
    
    x = np.zeros((n, 3))
    for i in range(n):
        row = i // cols
        col = i % cols
        cx = (col + 0.5) * cell_w + rng.uniform(-0.1 * cell_w, 0.1 * cell_w)
        cy = (row + 0.5) * cell_h + rng.uniform(-0.1 * cell_h, 0.1 * cell_h)
        # Initial radius: fraction of the smaller cell dimension
        r = 0.3 * min(cell_w, cell_h)
        x[i] = [cx, cy, r]
    
    # Clip to bounds
    x[:, 0] = np.clip(x[:, 0], 0.05, w - 0.05)
    x[:, 1] = np.clip(x[:, 1], 0.05, h - 0.05)
    x[:, 2] = np.clip(x[:, 2], 0.02, min(w, h) / 4)
    
    return x

def _random_init(w, h, seed=42):
    """Generate a random initialization."""
    rng = np.random.RandomState(seed)
    x = np.zeros((N, 3))
    x[:, 0] = rng.uniform(0.1, w - 0.1, N)
    x[:, 1] = rng.uniform(0.1, h - 0.1, N)
    x[:, 2] = rng.uniform(0.03, 0.12, N)
    return x

def _unified_objective(params):
    """
    Unified objective: params = [w, cx_0, cy_0, r_0, ..., cx_N, cy_N, r_N]
    h = 2 - w
    Minimize -sum(r) with penalty for constraint violations.
    """
    w = params[0]
    h = 2.0 - w
    if w <= 0.1 or h <= 0.1:
        return 1e10
    
    x = params[1:].reshape(N, 3)
    cx, cy, r = x[:, 0], x[:, 1], x[:, 2]
    
    # Boundary penalties
    pen = 0.0
    pen += np.sum(np.clip(r - cx, 0, None)**2)
    pen += np.sum(np.clip(cx + r - w, 0, None)**2)
    pen += np.sum(np.clip(r - cy, 0, None)**2)
    pen += np.sum(np.clip(cy + r - h, 0, None)**2)
    
    # Overlap penalties
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_d = r[i] + r[j]
            viol = np.clip(min_d - dist, 0, None)
            pen += viol**2
    
    return -np.sum(r) + 50.0 * pen

def _unified_bounds():
    """Bounds for unified parameters: w in [0.3, 1.7], then per-circle bounds."""
    bounds = [(0.3, 1.7)]  # w
    # We'll use loose bounds for cx, cy, r; the penalty handles feasibility
    for i in range(N):
        bounds.append((0.0, 2.0))  # cx
        bounds.append((0.0, 2.0))  # cy
        bounds.append((0.01, 0.5))  # r
    return bounds

def _polish_result(x, w, h):
    """Attempt to polish a near-feasible solution by adjusting radii up."""
    if not _is_valid(x, w, h, tol=1e-5):
        return x
    
    # Try to increase radii slightly while maintaining feasibility
    for _ in range(50):
        improved = False
        for i in range(N):
            # Try increasing radius of circle i
            old_r = x[i, 2]
            new_r = old_r * 1.02
            if new_r >= min(w, h) / 2:
                continue
            x[i, 2] = new_r
            if _is_valid(x, w, h, tol=1e-6):
                improved = True
            else:
                x[i, 2] = old_r
        if not improved:
            break
    
    return x

def construct():
    start_time = time.time()
    timeout = 145  # Safety margin for 150s limit
    
    best_score = 0.0
    best_solution = None
    best_w = 1.0
    best_h = 1.0
    
    random.seed(42)
    np.random.seed(42)
    
    # Phase 1: Multi-start with unified SLSQP optimization
    num_starts = 30
    bounds = _unified_bounds()
    
    for start_idx in range(num_starts):
        if time.time() - start_time > timeout:
            break
        
        # Choose initialization strategy
        if start_idx % 3 == 0:
            # Topological/grid-based
            w_init = random.uniform(0.7, 1.3)
            h_init = 2.0 - w_init
            x_init = _topological_init(w_init, h_init, seed=start_idx * 100 + 1)
        elif start_idx % 3 == 1:
            # Random
            w_init = random.uniform(0.5, 1.5)
            h_init = 2.0 - w_init
            x_init = _random_init(w_init, h_init, seed=start_idx * 100 + 2)
        else:
            # Square-ish with small jitter
            w_init = 1.0 + random.uniform(-0.1, 0.1)
            h_init = 2.0 - w_init
            x_init = _topological_init(w_init, h_init, seed=start_idx * 100 + 3)
        
        # Build initial parameter vector: [w, cx_0, cy_0, r_0, ...]
        params0 = np.concatenate(([w_init], x_init.flatten()))
        
        # Optimize with SLSQP
        try:
            res = minimize(
                _unified_objective,
                params0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-6}
            )
            
            # Extract solution
            w_opt = res.x[0]
            h_opt = 2.0 - w_opt
            x_opt = res.x[1:].reshape(N, 3)
            
            # Clip to valid range
            x_opt[:, 0] = np.clip(x_opt[:, 0], 0, w_opt)
            x_opt[:, 1] = np.clip(x_opt[:, 1], 0, h_opt)
            x_opt[:, 2] = np.clip(x_opt[:, 2], 0.01, min(w_opt, h_opt) / 2)
            
            # Check validity
            if _is_valid(x_opt, w_opt, h_opt):
                score = np.sum(x_opt[:, 2])
                if score > best_score:
                    best_score = score
                    best_solution = x_opt.copy()
                    best_w = w_opt
                    best_h = h_opt
                    
        except Exception:
            pass
    
    # Phase 2: Polish the best solution
    if best_solution is not None:
        best_solution = _polish_result(best_solution, best_w, best_h)
        if _is_valid(best_solution, best_w, best_h):
            best_score = np.sum(best_solution[:, 2])
    
    # Fallback if no valid solution found
    if best_solution is None:
        w = 1.0
        h = 1.0
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            cx = (col + 0.5) / side
            cy = (row + 0.5) / side
            r = 0.5 / side - 0.005
            cells.append([cx, cy, r])
        best_solution = np.array(cells)
        best_w = w
        best_h = h
    
    return best_solution
# EVOLVE-BLOCK-END
