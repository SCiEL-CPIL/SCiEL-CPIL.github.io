"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def construct():
    """
    Pack 32 circles in the unit square maximizing sum of radii.
    
    Uses a differentiable optimization with a soft contact graph derived
    from nearest-neighbor distances. The objective maximizes sum of radii
    with smooth penalties for overlap and boundary violations, where
    contact weights focus optimization on near-contact pairs.
    """
    n = N
    best_result = None
    best_score = -np.inf
    
    def softplus(x, beta=100.0):
        """Numerically stable softplus for smooth max(0, x)."""
        return np.log1p(np.exp(-beta * x)) / beta
    
    def objective(x, n_circles, sigma=0.03, penalty_w=500.0, boundary_w=200.0):
        """
        Differentiable objective: -sum(r) + penalties for violations.
        
        x: shape (3*n,) -> [x1,y1,r1, x2,y2,r2, ...]
        """
        pos = x.reshape(n_circles, 3)[:, :2]
        r = x.reshape(n_circles, 3)[:, 2]
        
        # Pairwise distances
        d = cdist(pos, pos)
        np.fill_diagonal(d, np.inf)
        
        # Sum of radii for each pair
        rsum = r[:, None] + r[None, :]
        
        # Overlap violations: max(0, rsum - d)
        overlap = np.maximum(0, rsum - d)
        
        # Soft contact weights: emphasize near-contact pairs
        # w_ij = exp(-d_ij^2 / (2*sigma^2)), but only where overlap is possible
        w = np.exp(-d**2 / (2 * sigma**2))
        np.fill_diagonal(w, 0)
        
        # Weighted overlap penalty (smooth)
        overlap_penalty = penalty_w * np.sum(w * overlap**2)
        
        # Boundary violations
        # Circle i must satisfy: r_i <= x_i <= 1-r_i, r_i <= y_i <= 1-r_i
        bviol_x = np.maximum(0, r - pos[:, 0]) + np.maximum(0, r - (1 - pos[:, 0]))
        bviol_y = np.maximum(0, r - pos[:, 1]) + np.maximum(0, r - (1 - pos[:, 1]))
        boundary_penalty = boundary_w * (np.sum(bviol_x**2) + np.sum(bviol_y**2))
        
        return -np.sum(r) + overlap_penalty + boundary_penalty
    
    def gradient(x, n_circles, sigma=0.03, penalty_w=500.0, boundary_w=200.0):
        """Analytical gradient for L-BFGS-B efficiency."""
        pos = x.reshape(n_circles, 3)[:, :2]
        r = x.reshape(n_circles, 3)[:, 2]
        
        d = cdist(pos, pos)
        np.fill_diagonal(d, 1.0)  # avoid div by zero, will be masked
        rsum = r[:, None] + r[None, :]
        
        # Overlap
        overlap = np.maximum(0, rsum - d)
        w = np.exp(-d**2 / (2 * sigma**2))
        np.fill_diagonal(w, 0)
        
        # Gradient of overlap penalty w.r.t. positions
        # d(overlap^2)/d(pos_i) involves chain rule through distance
        grad = np.zeros_like(x)
        
        # Position gradients from overlap
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                if overlap[i, j] <= 0:
                    continue
                # grad w.r.t. pos_i
                diff = pos[i] - pos[j]
                d_ij = d[i, j]
                if d_ij < 1e-10:
                    continue
                # d(overlap)/d(pos_i) = (pos_i - pos_j) / d_ij
                grad_pos_i = (diff / d_ij)
                # d(overlap^2 * w)/d(pos_i) = 2*overlap*w*grad_pos_i
                # + overlap^2 * dw/d(pos_i) -- ignore weight gradient for simplicity
                g_i = 2 * penalty_w * overlap[i, j] * w[i, j] * grad_pos_i
                grad[3*i] += g_i[0]
                grad[3*i+1] += g_i[1]
                grad[3*j] -= g_i[0]
                grad[3*j+1] -= g_i[1]
                # Radius gradients
                grad[3*i+2] += 2 * penalty_w * overlap[i, j] * w[i, j]
                grad[3*j+2] += 2 * penalty_w * overlap[i, j] * w[i, j]
        
        # Boundary gradients
        for i in range(n_circles):
            # x-boundary
            if r[i] > pos[i, 0]:
                grad[3*i] += -2 * boundary_w * (r[i] - pos[i, 0])
                grad[3*i+2] += -2 * boundary_w * (r[i] - pos[i, 0])
            if r[i] > (1 - pos[i, 0]):
                grad[3*i] += 2 * boundary_w * (r[i] - (1 - pos[i, 0]))
                grad[3*i+2] += -2 * boundary_w * (r[i] - (1 - pos[i, 0]))
            # y-boundary
            if r[i] > pos[i, 1]:
                grad[3*i+1] += -2 * boundary_w * (r[i] - pos[i, 1])
                grad[3*i+2] += -2 * boundary_w * (r[i] - pos[i, 1])
            if r[i] > (1 - pos[i, 1]):
                grad[3*i+1] += 2 * boundary_w * (r[i] - (1 - pos[i, 1]))
                grad[3*i+2] += -2 * boundary_w * (r[i] - (1 - pos[i, 1]))
        
        # Objective gradient: -sum(r)
        for i in range(n_circles):
            grad[3*i+2] -= 1.0
        
        return grad
    
    def make_bounds(n_circles):
        bounds = []
        for i in range(n_circles):
            bounds.append((0.0, 1.0))  # x
            bounds.append((0.0, 1.0))  # y
            bounds.append((0.001, 0.45))  # radius
        return bounds
    
    def validate(pos, rad, tol=1e-7):
        """Check if configuration is valid."""
        for i in range(n):
            if pos[i, 0] < rad[i] - tol or pos[i, 0] > 1 - rad[i] + tol:
                return False
            if pos[i, 1] < rad[i] - tol or pos[i, 1] > 1 - rad[i] + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2)
                if d < rad[i] + rad[j] - tol:
                    return False
        return True
    
    def solve_from_init(pos, rad, maxiter=800, sigma=0.03):
        """Run optimization from initial configuration."""
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3*i] = pos[i, 0]
            x0[3*i+1] = pos[i, 1]
            x0[3*i+2] = rad[i]
        
        bounds = make_bounds(n)
        
        result = minimize(
            objective,
            x0,
            jac=gradient,
            args=(n, sigma),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
        )
        return result
    
    # Generate diverse initializations
    inits = []
    
    # 1. Uniform grid (6x6 = 36, take 32)
    for rows, cols in [(6, 6), (5, 7), (7, 5), (8, 4), (4, 8)]:
        pos = []
        rad = []
        for i in range(rows):
            for j in range(cols):
                if len(pos) >= n:
                    break
                pos.append([(j + 0.5) / cols, (i + 0.5) / rows])
                rad.append(0.45 / max(rows, cols))
        if len(pos) == n:
            inits.append((np.array(pos), np.array(rad)))
    
    # 2. Hexagonal packing
    r_hex = 0.07
    dx = 2 * r_hex
    dy = r_hex * np.sqrt(3)
    pos_hex = []
    rad_hex = []
    for row in range(8):
        for col in range(8):
            if len(pos_hex) >= n:
                break
            x = dx * (col + 0.5 * (row % 2))
            y = dy * row
            # Scale and center
            x = 0.5 + (x - 3.5 * dx) * 0.12
            y = 0.5 + (y - 3.5 * dy) * 0.12
            if 0.08 < x < 0.92 and 0.08 < y < 0.92:
                pos_hex.append([x, y])
                rad_hex.append(r_hex * 0.85)
    while len(pos_hex) < n:
        pos_hex.append([0.5 + np.random.uniform(-0.3, 0.3), 
                        0.5 + np.random.uniform(-0.3, 0.3)])
        rad_hex.append(0.05)
    inits.append((np.array(pos_hex[:n]), np.array(rad_hex[:n])))
    
    # 3. Random initializations with various seeds
    rng = np.random.RandomState(42)
    for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]:
        rng = np.random.RandomState(seed)
        pos_r = rng.uniform(0.15, 0.85, size=(n, 2))
        rad_r = rng.uniform(0.04, 0.08, size=n)
        inits.append((pos_r, rad_r))
    
    # 4. Corner-clustered init (4 corners + center)
    pos_c = []
    rad_c = []
    corners = [[0.2, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.8], [0.5, 0.5]]
    for cx, cy in corners:
        for dx, dy in [(-0.15, -0.15), (0, -0.15), (0.15, -0.15), 
                       (-0.15, 0), (0, 0), (0.15, 0),
                       (-0.15, 0.15), (0, 0.15), (0.15, 0.15)]:
            if len(pos_c) >= n:
                break
            x, y = cx + dx, cy + dy
            if 0.1 < x < 0.9 and 0.1 < y < 0.9:
                pos_c.append([x, y])
                rad_c.append(0.06)
    while len(pos_c) < n:
        pos_c.append([0.5 + np.random.uniform(-0.2, 0.2), 
                      0.5 + np.random.uniform(-0.2, 0.2)])
        rad_c.append(0.05)
    inits.append((np.array(pos_c[:n]), np.array(rad_c[:n])))
    
    # Solve from each initialization
    for idx, (pos, rad) in enumerate(inits):
        try:
            # Try with different sigma values
            for sigma in [0.02, 0.03, 0.05]:
                result = solve_from_init(pos, rad, maxiter=600, sigma=sigma)
                
                if result.fun < best_score or (best_result is None and result.success):
                    x_opt = result.x
                    pos_opt = x_opt[0::3]
                    rad_opt = x_opt[2::3]
                    
                    if validate(pos_opt, rad_opt):
                        score = np.sum(rad_opt)
                        if score > best_score:
                            best_score = score
                            best_result = np.column_stack([pos_opt, rad_opt])
        except Exception as e:
            continue
    
    # Refine best result with tighter parameters if found
    if best_result is not None:
        try:
            pos_best = best_result[:, :2]
            rad_best = best_result[:, 2]
            for sigma in [0.01, 0.015, 0.02]:
                result = solve_from_init(pos_best, rad_best, maxiter=400, sigma=sigma)
                x_opt = result.x
                pos_opt = x_opt[0::3]
                rad_opt = x_opt[2::3]
                if validate(pos_opt, rad_opt):
                    score = np.sum(rad_opt)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack([pos_opt, rad_opt])
        except Exception:
            pass
    
    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(n)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= n:
                    break
                cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.45 / side])
        return np.array(cells[:n])
    
    return best_result
# EVOLVE-BLOCK-END
