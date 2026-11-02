"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time

DEADLINE = 115.0  # safety margin before 120s kill

def _init_positions(n, aspect):
    """Initialize circle centers in a grid within a unit-height rectangle of width=aspect."""
    cols = int(np.ceil(np.sqrt(n * aspect)))
    rows = int(np.ceil(n / cols))
    xs = np.linspace(0.05, aspect - 0.05, cols)
    ys = np.linspace(0.05, 1.0 - 0.05, rows)
    positions = []
    for i in range(n):
        c = i % cols
        r = i // cols
        positions.append([xs[c], ys[r]])
    return np.array(positions[:n])

def _soft_min_penalty(dists, radii, margin=1e-4):
    """Soft penalty for pairwise circle overlaps: sum of soft-min of (d_ij - r_i - r_j).
    Returns (penalty, gradient contribution placeholder)."""
    n = len(radii)
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            d = dists[i, j]
            gap = d - radii[i] - radii[j]
            if gap < 0:
                penalty += gap * gap
    return penalty

def _boundary_penalty(positions, radii, aspect):
    """Penalty for circles exceeding rectangle [0,aspect]x[0,1]."""
    pen = 0.0
    for i in range(len(radii)):
        x, y = positions[i]
        r = radii[i]
        # left
        if x - r < 0: pen += (x - r)**2
        # right
        if x + r > aspect: pen += (x + r - aspect)**2
        # bottom
        if y - r < 0: pen += (y - r)**2
        # top
        if y + r > 1: pen += (y + r - 1)**2
    return pen

def construct():
    t_start = time.time()
    
    # Initial guess: aspect ratio around 1.0 (square-ish)
    # We optimize: positions (2N), radii (N), aspect (1)
    # Total variables: 3N + 1 = 64
    
    best_result = None
    best_obj = np.inf
    
    # Try several initial aspect ratios
    for aspect0 in [1.0, 0.8, 1.25, 0.6, 1.5]:
        if time.time() - t_start > DEADLINE:
            break
        
        positions = _init_positions(N, aspect0)
        # Initial radii: equal, small enough to fit
        # Perimeter = 2*(w+h) = 4 => w+h=2, with h=1, w=aspect0
        # Max radius for grid: roughly 0.5/sqrt(N)
        r0 = 0.5 / np.sqrt(N) * 0.9
        radii = np.full(N, r0)
        
        x0 = np.concatenate([positions.flatten(), radii, [aspect0]])
        
        def objective(x):
            if time.time() - t_start > DEADLINE:
                return 1e10
            pos = x[:2*N].reshape(N, 2)
            rad = x[2*N:3*N]
            asp = x[3*N]
            
            # Ensure positive radii and aspect
            rad = np.maximum(rad, 1e-6)
            asp = max(asp, 0.1)
            
            # Pairwise distances
            d = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    d[i, j] = np.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2)
            
            # Overlap penalty (soft)
            overlap_pen = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    gap = d[i, j] - rad[i] - rad[j]
                    if gap < 0:
                        overlap_pen += gap * gap
            
            # Boundary penalty
            bpen = _boundary_penalty(pos, rad, asp)
            
            # Objective: maximize sum of radii, penalize violations
            # Use a large penalty weight
            W = 100.0
            val = -np.sum(rad) + W * (overlap_pen + bpen)
            return val
        
        def jac(x):
            # Numerical gradient (simplified; use finite differences)
            eps = 1e-5
            grad = np.zeros_like(x)
            f0 = objective(x)
            for k in range(len(x)):
                xp = x.copy()
                xp[k] += eps
                grad[k] = (objective(xp) - f0) / eps
            return grad
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', jac=jac,
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8,
                                   'maxfun': 5000, 'disp': False})
            
            if time.time() - t_start > DEADLINE:
                if best_result is None:
                    pos = x0[:2*N].reshape(N, 2)
                    rad = x0[2*N:3*N]
                    best_result = np.column_stack([pos, rad])
                break
            
            x_opt = res.x
            pos = x_opt[:2*N].reshape(N, 2)
            rad = np.maximum(x_opt[2*N:3*N], 1e-6)
            asp = max(x_opt[3*N], 0.1)
            
            # Compute the actual objective value (sum of radii) for comparison
            # Need to verify validity: check for overlaps and boundary violations
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2)
                    if d < rad[i] + rad[j] - 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                for i in range(N):
                    if pos[i,0] - rad[i] < -1e-6 or pos[i,0] + rad[i] > asp + 1e-6:
                        valid = False
                        break
                    if pos[i,1] - rad[i] < -1e-6 or pos[i,1] + rad[i] > 1 + 1e-6:
                        valid = False
                        break
            
            # Perimeter check: 2*(asp + 1) <= 4 => asp + 1 <= 2 => asp <= 1
            # Wait: perimeter = 2*(w+h) = 2*(asp*1 + 1) = 2*(asp+1)
            # We need 2*(asp+1) <= 4 => asp+1 <= 2 => asp <= 1
            # But we allowed asp up to 1.5. Let me reconsider.
            # Actually, the rectangle has height h and width w, perimeter 2(w+h)=4 => w+h=2.
            # We normalized h=1, so w=asp, and w+h = asp+1 = 2 => asp=1.
            # So aspect is actually fixed at 1 if h=1! 
            # I need to re-parameterize. Let me use w and h directly with w+h=2.
            
            # Actually, let me just scale the result to fit perimeter 4.
            # Current rectangle: width=asp, height=1. Perimeter=2*(asp+1).
            # We need perimeter <= 4, so scale factor s = 4 / (2*(asp+1)) = 2/(asp+1)
            # But if asp+1 > 2, we need to shrink.
            
            scale = 2.0 / (asp + 1.0)
            pos_scaled = pos * scale
            rad_scaled = rad * scale
            asp_scaled = asp * scale
            h_scaled = 1.0 * scale
            
            # Verify perimeter: 2*(asp_scaled + h_scaled) = 2*(asp+1)*scale = 2*(asp+1)*2/(asp+1) = 4. Good.
            
            # Check validity in scaled coordinates
            valid_scaled = True
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((pos_scaled[i,0]-pos_scaled[j,0])**2 + (pos_scaled[i,1]-pos_scaled[j,1])**2)
                    if d < rad_scaled[i] + rad_scaled[j] - 1e-6:
                        valid_scaled = False
                        break
                if not valid_scaled:
                    break
            if valid_scaled:
                for i in range(N):
                    if pos_scaled[i,0] - rad_scaled[i] < -1e-6 or pos_scaled[i,0] + rad_scaled[i] > asp_scaled + 1e-6:
                        valid_scaled = False
                        break
                    if pos_scaled[i,1] - rad_scaled[i] < -1e-6 or pos_scaled[i,1] + rad_scaled[i] > h_scaled + 1e-6:
                        valid_scaled = False
                        break
            
            if valid_scaled:
                obj_val = np.sum(rad_scaled)
                if obj_val > best_obj:
                    best_obj = obj_val
                    best_result = np.column_stack([pos_scaled, rad_scaled])
                    
        except Exception:
            continue
    
    if best_result is None:
        # Fallback to seed solution
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
