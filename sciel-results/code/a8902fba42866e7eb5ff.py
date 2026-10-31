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
=======
"""Circle packing in perimeter-4 rectangle using differentiable physics simulation with gradient descent."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_value = 0.0
    best_result = None
    
    # Try multiple aspect ratios with physics-based optimization
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0, 0.9, 1.1, 0.75, 1.33]
    
    for aspect_idx, aspect in enumerate(aspect_ratios):
        if time.time() > deadline:
            break
        
        # For perimeter 4: w + h = 2
        h_rect = 2.0 / (aspect + 1.0)
        w_rect = aspect * h_rect
        
        # Try multiple grid configurations
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
            r = np.full(N, min(W_cell, H_cell) / 2 * 0.9)
            
            # Differentiable physics simulation with gradient descent
            result = _physics_optimize(cx, cy, r, w_rect, h_rect, 
                                       time_budget=deadline - (time.time() - start_time))
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
>>>>>>> REPLACE
<<<<<<< SEARCH
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
                pen += (cx_opt[i] + r_opt[i] - H) ** 2
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
=======
def _physics_optimize(cx, cy, r, W, H, time_budget):
    """Differentiable physics simulation with gradient descent on potential energy."""
    n = N
    cx = cx.copy()
    cy = cy.copy()
    r = r.copy()
    
    # Physics parameters
    k_repel = 50.0      # Repulsion strength
    k_wall = 100.0      # Wall constraint strength
    k_gravity = 0.1     # Small attractive force to encourage growth
    dt = 0.001          # Time step
    damping = 0.95      # Velocity damping
    
    # Initialize velocities
    vx = np.zeros(n)
    vy = np.zeros(n)
    vr = np.zeros(n)
    
    max_iter = int(min(5000, time_budget * 20))  # Cap iterations based on time budget
    best_local_value = 0.0
    best_local_state = None
    
    for iter_num in range(max_iter):
        if time.time() > time_budget + (time.time() - time_budget):  # Check against absolute deadline
            break
        
        # Compute forces
        fx = np.zeros(n)
        fy = np.zeros(n)
        fr = np.zeros(n)
        
        # Wall constraints (soft penalties)
        for i in range(n):
            # Left wall
            if cx[i] - r[i] < 0:
                penetration = -(cx[i] - r[i])
                fx[i] += k_wall * penetration
                fr[i] -= k_wall * 0.5 * penetration
            # Right wall
            if cx[i] + r[i] > W:
                penetration = cx[i] + r[i] - W
                fx[i] -= k_wall * penetration
                fr[i] -= k_wall * 0.5 * penetration
            # Bottom wall
            if cy[i] - r[i] < 0:
                penetration = -(cy[i] - r[i])
                fy[i] += k_wall * penetration
                fr[i] -= k_wall * 0.5 * penetration
            # Top wall
            if cy[i] + r[i] > H:
                penetration = cy[i] + r[i] - H
                fy[i] -= k_wall * penetration
                fr[i] -= k_wall * 0.5 * penetration
        
        # Pairwise repulsion (elastic soft spheres)
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d_sq = dx * dx + dy * dy
                d = np.sqrt(d_sq + 1e-12)
                min_d = r[i] + r[j]
                
                if d < min_d:
                    overlap = min_d - d
                    # Elastic repulsion force (Hooke's law style)
                    force_mag = k_repel * overlap
                    fx[i] += (dx / d) * force_mag
                    fy[i] += (dy / d) * force_mag
                    fx[j] -= (dx / d) * force_mag
                    fy[j] -= (dy / d) * force_mag
                    # Radii should decrease when overlapping
                    fr[i] -= k_repel * 0.1 * overlap
                    fr[j] -= k_repel * 0.1 * overlap
        
        # Gravity-like force to encourage radius growth (only when not overlapping)
        for i in range(n):
            # Check if circle i is not significantly overlapping
            is_overlapping = False
            for j in range(n):
                if i == j:
                    continue
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx * dx + dy * dy + 1e-12)
                if d < r[i] + r[j]:
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                fr[i] += k_gravity
            else:
                fr[i] -= k_gravity * 2
        
        # Update velocities with damping
        vx = damping * vx + dt * fx
        vy = damping * vy + dt * fy
        vr = damping * vr + dt * fr
        
        # Update positions
        cx += dt * vx
        cy += dt * vy
        r += dt * vr
        
        # Enforce hard constraints (clamp)
        cx = np.clip(cx, r, W - r)
        cy = np.clip(cy, r, H - r)
        r = np.clip(r, 1e-4, min(W, H) / 2)
        
        # Track best valid state
        if _is_valid(cx, cy, r, W, H):
            current_value = np.sum(r)
            if current_value > best_local_value:
                best_local_value = current_value
                best_local_state = np.column_stack([cx, cy, r])
        
        # Early termination if converged
        if np.max(np.abs(vx)) < 1e-8 and np.max(np.abs(vy)) < 1e-8 and np.max(np.abs(vr)) < 1e-8:
            break
    
    return best_local_state
>>>>>>> REPLACE
<<<<<<< SEARCH
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


def _polish_with_slsqp(cx, cy, r, W, H, time_budget):
    """Final polish using SLSQP on the physics-optimized result."""
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
                       options={'maxiter': 100, 'ftol': 1e-10})
        
        if res.success or res.fun < objective(x0):
            cx_opt = res.x[:n]
            cy_opt = res.x[n:2*n]
            r_opt = res.x[2*n:3*n]
            
            if _is_valid(cx_opt, cy_opt, r_opt, W, H):
                return np.column_stack([cx_opt, cy_opt, r_opt])
    except Exception:
        pass
    
    return None
>>>>>>> REPLACE

Now I need to update the `construct` function to use the polish step:

