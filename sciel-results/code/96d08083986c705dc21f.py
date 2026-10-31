<<<<<<< SEARCH
"""Circle packing in perimeter-4 rectangle using grid initialization with force-directed relaxation and SLSQP optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    best_value = 0.0
    best_result = None
    
    # Try multiple aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0, 0.4, 0.9, 1.1, 0.75, 1.33]
    
    for aspect_idx, aspect in enumerate(aspect_ratios):
        if time.time() > deadline:
            break
        
        # For perimeter 4: w + h = 2, so h = 2 - w
        # aspect = w/h => w = aspect * h => aspect*h + h = 2 => h = 2/(aspect+1)
        h_rect = 2.0 / (aspect + 1.0)
        w_rect = aspect * h_rect
        
        # Try multiple grid configurations for this aspect
        for cols in range(3, 8):
            if time.time() > deadline:
                break
            
            rows = int(np.ceil(N / cols))
            actual_n = min(N, cols * rows)
            
            # Generate grid initialization
            W_cell = w_rect / cols
            H_cell = h_rect / rows
            cx = np.array([(i + 0.5) * W_cell for i in range(cols) for _ in range(rows)][:N])
            cy = np.array([(j + 0.5) * H_cell for _ in range(cols) for j in range(rows)][:N])
            r = np.full(N, min(W_cell, H_cell) / 2 * 0.85)
            
            # Force-directed relaxation to fix initial overlaps
            cx, cy = _force_directed_relax(cx, cy, r, w_rect, h_rect, max_iter=300)
            
            # Optimize using SLSQP
            result = _slsqp_optimize(cx, cy, r, w_rect, h_rect, 
                                     deadline - (time.time() - start_time))
            if result is not None:
                value = np.sum(result[:, 2])
                if value > best_value:
                    best_value = value
                    best_result = result
            
            if time.time() > deadline:
                break
    
    if best_result is None:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result


def _force_directed_relax(cx, cy, r, W, H, max_iter=500, dt=0.002):
    """Simple force-directed relaxation to fix overlaps and wall violations."""
    n = len(cx)
    cx = cx.copy()
    cy = cy.copy()
    for _ in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)
        # Wall repulsion
        for i in range(n):
            if cx[i] - r[i] < 0:
                fx[i] += (cx[i] - r[i]) * 10
            if cy[i] - r[i] < 0:
                fy[i] += (cy[i] - r[i]) * 10
            if cx[i] + r[i] > W:
                fx[i] -= (cx[i] + r[i] - W) * 10
            if cy[i] + r[i] > H:
                fy[i] -= (cy[i] + r[i] - H) * 10
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-12:
                    overlap = min_d - d
                    fx[i] += (dx / d) * overlap * 5
                    fy[i] += (dy / d) * overlap * 5
                    fx[j] -= (dx / d) * overlap * 5
                    fy[j] -= (dy / d) * overlap * 5
        cx += dt * fx
        cy += dt * fy
        # Clamp to rectangle
        cx = np.clip(cx, r, W - r)
        cy = np.clip(cy, r, H - r)
    return cx, cy


def _slsqp_optimize(cx, cy, r, W, H, time_budget):
    """Optimize circle positions and radii using SLSQP."""
    n = N
    
    # Parameter vector: [x1, y1, r1, x2, y2, r2, ..., xn, yn, rn]
    def objective(x):
        cx_opt = x[:n]
        cy_opt = x[n:2*n]
        r_opt = x[2*n:3*n]
        
        # Penalty for violations
        pen = 0.0
        for i in range(n):
            if cx_opt[i] - r_opt[i] < 0:
                pen += (cx_opt[i] - r_opt[i]) ** 2
            if cy_opt[i] - r_opt[i] < 0:
                pen += (cy_opt[i] - r_opt[i]) ** 2
            if cx_opt[i] + r_opt[i] > W:
                pen += (cx_opt[i] + r_opt[i] - W) ** 2
            if cy_opt[i] + r_opt[i] > H:
                pen += (cy_opt[i] + r_opt[i] - H) ** 2
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx_opt[i] - cx_opt[j]
                dy = cy_opt[i] - cy_opt[j]
                d = np.sqrt(dx * dx + dy * dy)
                if d < r_opt[i] + r_opt[j]:
                    pen += (r_opt[i] + r_opt[j] - d) ** 2
        return -np.sum(r_opt) + 10 * pen
    
    x0 = np.concatenate([cx, cy, r])
    bounds = []
    for i in range(n):
        bounds.append((0, W))
    for i in range(n):
        bounds.append((0, H))
    for i in range(n):
        bounds.append((1e-4, min(W, H) / 2))
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       options={'maxiter': 200, 'ftol': 1e-8})
        
        if res.success or res.fun < objective(x0):
            cx_opt = res.x[:n]
            cy_opt = res.x[n:2*n]
            r_opt = res.x[2*n:3*n]
            
            # Validate solution
            if _is_valid(cx_opt, cy_opt, r_opt, W, H):
                return np.column_stack([cx_opt, cy_opt, r_opt])
    except Exception:
        pass
    
    return None


def _is_valid(cx, cy, r, W, H):
    """Check if circles fit in rectangle [0,W]x[0,H] with disjoint interiors."""
    n = len(cx)
    # Wall constraints
    if np.any(cx - r < -1e-9) or np.any(cy - r < -1e-9):
        return False
    if np.any(cx + r > W + 1e-9) or np.any(cy + r > H + 1e-9):
        return False
    # Pairwise non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - 1e-9:
                return False
    return True
=======
"""Circle packing in perimeter-4 rectangle using joint aspect-ratio and position optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    best_value = 0.0
    best_result = None
    
    # Try a few aspect ratios with joint optimization
    aspect_ratios = [1.0, 0.75, 1.33, 0.6, 1.67, 0.5, 2.0, 0.8, 1.25]
    
    for aspect in aspect_ratios:
        if time.time() > deadline - 5.0:
            break
        
        h_rect = 2.0 / (aspect + 1.0)
        w_rect = aspect * h_rect
        
        # Use hexagonal packing initialization
        cx, cy, r = _hex_init(N, w_rect, h_rect)
        
        # Quick force-directed relaxation
        cx, cy = _force_directed_relax(cx, cy, r, w_rect, h_rect, max_iter=200)
        
        # Joint optimization: optimize positions, radii, AND aspect ratio
        result = _joint_optimize(cx, cy, r, aspect, deadline - (time.time() - start_time))
        if result is not None:
            value = np.sum(result[:, 2])
            if value > best_value:
                best_value = value
                best_result = result
        
        # Also try fixed-aspect SLSQP as fallback for this aspect
        if time.time() < deadline - 10.0:
            result2 = _slsqp_fixed_aspect(cx, cy, r, w_rect, h_rect)
            if result2 is not None:
                value2 = np.sum(result2[:, 2])
                if value2 > best_value:
                    best_value = value2
                    best_result = result2
    
    if best_result is None:
        # Fallback: uniform grid in square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result


def _hex_init(n, W, H):
    """Generate hexagonal packing initialization."""
    # Estimate radius from area
    avg_r = np.sqrt(n * np.pi / 12 / (W * H) / np.sqrt(3))
    avg_r = min(avg_r, min(W, H) / 2 * 0.9)
    
    # Hexagonal spacing
    dx = avg_r * 2 * 0.95  # slight compression
    dy = dx * np.sqrt(3) / 2
    
    cx_list = []
    cy_list = []
    r_list = []
    
    row = 0
    while len(cx_list) < n:
        x_offset = (row % 2) * dx / 2
        col = 0
        while len(cx_list) < n:
            x = x_offset + col * dx + avg_r
            y = row * dy + avg_r
            if x + avg_r <= W and y + avg_r <= H:
                cx_list.append(x)
                cy_list.append(y)
                r_list.append(avg_r * 0.9)
            col += 1
        row += 1
    
    cx = np.array(cx_list[:n])
    cy = np.array(cy_list[:n])
    r = np.array(r_list[:n])
    
    # If not enough circles fit, use grid fallback
    if len(cx) < n:
        cols = int(np.ceil(np.sqrt(n * W / H)))
        rows = int(np.ceil(n / cols))
        W_cell = W / cols
        H_cell = H / rows
        cx = np.array([(i + 0.5) * W_cell for i in range(cols) for _ in range(rows)][:n])
        cy = np.array([(j + 0.5) * H_cell for _ in range(cols) for j in range(rows)][:n])
        r = np.full(n, min(W_cell, H_cell) / 2 * 0.85)
    
    return cx, cy, r


def _force_directed_relax(cx, cy, r, W, H, max_iter=300, dt=0.003):
    """Force-directed relaxation to fix overlaps and wall violations."""
    n = len(cx)
    cx = cx.copy()
    cy = cy.copy()
    for _ in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)
        # Wall repulsion
        for i in range(n):
            if cx[i] - r[i] < 0:
                fx[i] += (cx[i] - r[i]) * 10
            if cy[i] - r[i] < 0:
                fy[i] += (cy[i] - r[i]) * 10
            if cx[i] + r[i] > W:
                fx[i] -= (cx[i] + r[i] - W) * 10
            if cy[i] + r[i] > H:
                fy[i] -= (cy[i] + r[i] - H) * 10
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-12:
                    overlap = min_d - d
                    fx[i] += (dx / d) * overlap * 5
                    fy[i] += (dy / d) * overlap * 5
                    fx[j] -= (dx / d) * overlap * 5
                    fy[j] -= (dy / d) * overlap * 5
        cx += dt * fx
        cy += dt * fy
        cx = np.clip(cx, r, W - r)
        cy = np.clip(cy, r, H - r)
    return cx, cy


def _joint_optimize(cx, cy, r, aspect_init, time_budget):
    """Optimize positions, radii, and aspect ratio jointly."""
    n = N
    # Parameter vector: [x1..xn, y1..yn, r1..rn, aspect]
    # Total: 3n + 1 dimensions
    
    def objective(x):
        cx_opt = x[:n]
        cy_opt = x[n:2*n]
        r_opt = x[2*n:3*n]
        aspect = x[3*n]
        
        h_rect = 2.0 / (aspect + 1.0)
        w_rect = aspect * h_rect
        
        # Penalty for violations
        pen = 0.0
        for i in range(n):
            if cx_opt[i] - r_opt[i] < 0:
                pen += (cx_opt[i] - r_opt[i]) ** 2
            if cy_opt[i] - r_opt[i] < 0:
                pen += (cy_opt[i] - r_opt[i]) ** 2
            if cx_opt[i] + r_opt[i] > w_rect:
                pen += (cx_opt[i] + r_opt[i] - w_rect) ** 2
            if cy_opt[i] + r_opt[i] > h_rect:
                pen += (cy_opt[i] + r_opt[i] - h_rect) ** 2
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx_opt[i] - cx_opt[j]
                dy = cy_opt[i] - cy_opt[j]
                d = np.sqrt(dx * dx + dy * dy)
                if d < r_opt[i] + r_opt[j]:
                    pen += (r_opt[i] + r_opt[j] - d) ** 2
        return -np.sum(r_opt) + 50 * pen
    
    h_init = 2.0 / (aspect_init + 1.0)
    w_init = aspect_init * h_init
    
    x0 = np.concatenate([cx, cy, r, [aspect_init]])
    bounds = []
    for i in range(n):
        bounds.append((0, 2.0))  # x can be up to W which is at most 2
    for i in range(n):
        bounds.append((0, 2.0))  # y can be up to H which is at most 2
    for i in range(n):
        bounds.append((1e-4, 1.0))  # radius at most 1
    bounds.append((0.3, 3.0))  # aspect ratio bounds
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       options={'maxiter': 150, 'ftol': 1e-7, 'iprint': 0})
        
        if res.success or res.fun < objective(x0):
            cx_opt = res.x[:n]
            cy_opt = res.x[n:2*n]
            r_opt = res.x[2*n:3*n]
            aspect_opt = res.x[3*n]
            h_opt = 2.0 / (aspect_opt + 1.0)
            w_opt = aspect_opt * h_opt
            
            if _is_valid(cx_opt, cy_opt, r_opt, w_opt, h_opt):
                return np.column_stack([cx_opt, cy_opt, r_opt])
    except Exception:
        pass
    
    return None


def _slsqp_fixed_aspect(cx, cy, r, W, H):
    """Optimize circle positions and radii with fixed rectangle using SLSQP."""
    n = N
    
    def objective(x):
        cx_opt = x[:n]
        cy_opt = x[n:2*n]
        r_opt = x[2*n:3*n]
        
        pen = 0.0
        for i in range(n):
            if cx_opt[i] - r_opt[i] < 0:
                pen += (cx_opt[i] - r_opt[i]) ** 2
            if cy_opt[i] - r_opt[i] < 0:
                pen += (cy_opt[i] - r_opt[i]) ** 2
            if cx_opt[i] + r_opt[i] > W:
                pen += (cx_opt[i] + r_opt[i] - W) ** 2
            if cy_opt[i] + r_opt[i] > H:
                pen += (cy_opt[i] + r_opt[i] - H) ** 2
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx_opt[i] - cx_opt[j]
                dy = cy_opt[i] - cy_opt[j]
                d = np.sqrt(dx * dx + dy * dy)
                if d < r_opt[i] + r_opt[j]:
                    pen += (r_opt[i] + r_opt[j] - d) ** 2
        return -np.sum(r_opt) + 50 * pen
    
    x0 = np.concatenate([cx, cy, r])
    bounds = []
    for i in range(n):
        bounds.append((0, W))
    for i in range(n):
        bounds.append((0, H))
    for i in range(n):
        bounds.append((1e-4, min(W, H) / 2))
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       options={'maxiter': 100, 'ftol': 1e-7, 'iprint': 0})
        
        if res.success or res.fun < objective(x0):
            cx_opt = res.x[:n]
            cy_opt = res.x[n:2*n]
            r_opt = res.x[2*n:3*n]
            
            if _is_valid(cx_opt, cy_opt, r_opt, W, H):
                return np.column_stack([cx_opt, cy_opt, r_opt])
    except Exception:
        pass
    
    return None


def _is_valid(cx, cy, r, W, H):
    """Check if circles fit in rectangle [0,W]x[0,H] with disjoint interiors."""
    n = len(cx)
    if np.any(cx - r < -1e-9) or np.any(cy - r < -1e-9):
        return False
    if np.any(cx + r > W + 1e-9) or np.any(cy + r > H + 1e-9):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - 1e-9:
                return False
    return True
>>>>>>> REPLACE
