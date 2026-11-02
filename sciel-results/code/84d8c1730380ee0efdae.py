"""Circle packing in perimeter-4 rectangle: differentiable rigid-body + LP radius assignment."""
import time
import numpy as np
from scipy.optimize import linprog

N = 21

def _validate(centers, radii, L, W):
    """Check validity: disjoint interiors, inside rectangle, perimeter <= 4."""
    if len(centers) != N:
        return False
    if L + W > 2.0 + 1e-9:
        return False
    # Inside rectangle
    if np.any(centers[:, 0] - radii < -1e-9) or np.any(centers[:, 0] + radii > L + 1e-9):
        return False
    if np.any(centers[:, 1] - radii < -1e-9) or np.any(centers[:, 1] + radii > W + 1e-9):
        return False
    # Disjoint interiors
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < radii[i] + radii[j] - 1e-7:
                return False
    return True

def _solve_radii_lp(centers, L, W, max_iter=200):
    """
    Solve LP: maximize sum(r_i) subject to:
      r_i + r_j <= dist(i,j)  for all pairs
      r_i <= min(centers[i][0], L - centers[i][0], centers[i][1], W - centers[i][1])
      r_i >= 0
    """
    n = N
    # Variables: r_0, ..., r_{n-1}
    # Objective: maximize sum(r) -> minimize -sum(r)
    c = -np.ones(n)
    
    # Inequality constraints: A_ub @ r <= b_ub
    # Pair constraints: r_i + r_j <= d_ij
    rows = []
    b = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d <= 1e-12:
                continue
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            rows.append(row)
            b.append(d)
    
    # Boundary constraints: r_i <= min(cxi, L-cxi, cyi, W-cyi)
    for i in range(n):
        max_r = min(centers[i][0], L - centers[i][0], centers[i][1], W - centers[i][1])
        if max_r < 0:
            max_r = 0.0
        row = np.zeros(n)
        row[i] = 1.0
        rows.append(row)
        b.append(max_r)
    
    A_ub = np.array(rows) if rows else None
    b_ub = np.array(b) if b else None
    
    bounds = [(0, None)] * n
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            radii = np.maximum(res.x, 0)
            return radii
    except Exception:
        pass
    
    # Fallback: greedy radius assignment
    radii = np.zeros(n)
    for i in range(n):
        max_r = min(centers[i][0], L - centers[i][0], centers[i][1], W - centers[i][1])
        max_r = max(max_r, 0)
        for j in range(n):
            if i != j:
                d = np.linalg.norm(centers[i] - centers[j])
                max_r = min(max_r, d - radii[j])
        radii[i] = max(max_r, 0)
    return radii

def _compute_score(centers, L, W):
    """Compute the sum of radii for given centers and rectangle."""
    radii = _solve_radii_lp(centers, L, W)
    return np.sum(radii)

def _gradient_descent(centers, L, W, lr=0.01, steps=500, beta=50):
    """
    Gradient descent on a smooth objective: maximize sum of radii via LP,
    but use a differentiable proxy: maximize sum of min-pairwise-distances / boundary-distances.
    Actually, we optimize centers to maximize the LP objective.
    Since LP is piecewise linear, we use subgradient or smooth approximation.
    
    Here we use a smooth proxy: for each circle, its "effective radius" is bounded by
    min over all other circles of (d_ij - r_j) and boundary distances.
    We approximate by maximizing sum of min(distances) with a soft-min.
    """
    centers = centers.copy()
    n = N
    
    for step in range(steps):
        # Compute current radii via LP
        radii = _solve_radii_lp(centers, L, W)
        
        # Compute gradient of sum(radii) w.r.t. centers using subgradient of LP
        # Alternative: use a smooth surrogate
        # For each circle i, r_i is limited by:
        #   - boundary: min(cx_i, L-cx_i, cy_i, W-cy_i)
        #   - other circles: min_j (d_ij - r_j)
        # We compute a smooth approximation of the gradient
        
        grad = np.zeros_like(centers)
        
        for i in range(n):
            # Boundary contribution
            # r_i <= min(cx_i, L-cx_i, cy_i, W-cy_i)
            # Gradient of min is subgradient; use the active constraint
            bnd = [centers[i][0], L - centers[i][0], centers[i][1], W - centers[i][1]]
            min_bnd = min(bnd)
            
            # Circle-circle contribution
            min_cc = np.inf
            for j in range(n):
                if i == j:
                    continue
                d = np.linalg.norm(centers[i] - centers[j])
                limit = d - radii[j]
                if limit < min_cc:
                    min_cc = limit
            
            # The effective radius is min(min_bnd, min_cc)
            # We want to maximize this, so gradient points away from constraints
            
            # Subgradient from boundary
            if min_bnd <= min_cc:
                # Active boundary constraint
                if centers[i][0] == min_bnd:
                    grad[i][0] += 1.0
                elif L - centers[i][0] == min_bnd:
                    grad[i][0] -= 1.0
                elif centers[i][1] == min_bnd:
                    grad[i][1] += 1.0
                elif W - centers[i][1] == min_bnd:
                    grad[i][1] -= 1.0
            
            # Subgradient from circle-circle
            if min_cc < min_bnd:
                for j in range(n):
                    if i == j:
                        continue
                    d = np.linalg.norm(centers[i] - centers[j])
                    if d < 1e-12:
                        continue
                    limit = d - radii[j]
                    if limit < min_cc + 1e-6:
                        # Active constraint: move away from j
                        diff = centers[i] - centers[j]
                        grad[i] += diff / d
        
        # Update centers
        centers += lr * grad
        
        # Project back into rectangle (with small margin)
        margin = 1e-4
        centers[:, 0] = np.clip(centers[:, 0], margin, L - margin)
        centers[:, 1] = np.clip(centers[:, 1], margin, W - margin)
    
    return centers

def _rectangular_grid_init(L, W):
    """Initialize with a grid pattern."""
    side = int(np.ceil(np.sqrt(N)))
    centers = []
    for i in range(side):
        for j in range(side):
            if len(centers) >= N:
                break
            x = (i + 0.5) * L / side
            y = (j + 0.5) * W / side
            centers.append([x, y])
    return np.array(centers[:N])

def _random_init(L, W, n_trials=50):
    """Random initialization with rejection sampling for non-overlapping."""
    best_centers = None
    best_score = -1
    
    for _ in range(n_trials):
        centers = np.random.rand(N, 2) * np.array([L, W])
        score = _compute_score(centers, L, W)
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
    
    return best_centers

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    best_score = 0.0
    best_centers = None
    best_L = 1.0
    best_W = 1.0
    
    # Try several aspect ratios
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.667, 1.333, 0.75, 1.25, 1.1, 0.9]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        # Perimeter = 2(L+W) = 4, so L+W = 2
        # L/W = ar -> L = ar*W, so ar*W + W = 2 -> W = 2/(1+ar), L = 2*ar/(1+ar)
        W = 2.0 / (1.0 + ar)
        L = 2.0 * ar / (1.0 + ar)
        
        # Initialize with grid
        centers = _rectangular_grid_init(L, W)
        
        # Run gradient descent
        centers = _gradient_descent(centers, L, W, lr=0.02, steps=300)
        
        # Compute score
        score = _compute_score(centers, L, W)
        radii = _solve_radii_lp(centers, L, W)
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
            best_L = L
            best_W = W
    
    # Try some random initializations on the best aspect ratio
    if time.time() < deadline - 30:
        for _ in range(3):
            if time.time() > deadline:
                break
            centers = _random_init(best_L, best_W, n_trials=20)
            centers = _gradient_descent(centers, best_L, best_W, lr=0.02, steps=200)
            score = _compute_score(centers, best_L, best_W)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
    
    # Final polish: more gradient steps
    if time.time() < deadline - 10 and best_centers is not None:
        best_centers = _gradient_descent(best_centers, best_L, best_W, lr=0.005, steps=200)
        best_score = _compute_score(best_centers, best_L, best_W)
    
    # Get final radii
    radii = _solve_radii_lp(best_centers, best_L, best_W)
    
    # Build result
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = [best_centers[i][0], best_centers[i][1], radii[i]]
    
    # Validate
    if not _validate(best_centers, radii, best_L, best_W):
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        L = W = 1.0
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3])
        result = np.array(cells[:N])
    
    return result
