"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle.

Strategy: Optimize over pairwise distances using Cayley-Menger determinants,
then embed the optimal metric into the fixed triangle via least-squares.
"""
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Pre-compute distance index mapping: for each triangle (i,j,k), get indices into dist array
# dist array is indexed by (min(i,j,k), max(i,j,k)) for i<j<k
# We need a mapping from (i,j,k) to the three distance indices
def _dist_idx(i, j):
    """Get index into the upper-triangular distance array for pair (i,j)."""
    if i > j:
        i, j = j, i
    # Number of pairs before row i
    return i * N - i*(i+1)//2 + (j - i - 1)

# Pre-compute for all triangles
tri_dist_idx = np.zeros((165, 3), dtype=int)
for t, (i, j, k) in enumerate(tri_indices):
    tri_dist_idx[t, 0] = _dist_idx(i, j)
    tri_dist_idx[t, 1] = _dist_idx(i, k)
    tri_dist_idx[t, 2] = _dist_idx(j, k)

# Number of pairwise distances
num_dists = N * (N - 1) // 2  # 55

def _cayley_menger_areas(d2):
    """
    Compute squared areas of all triangles from squared distances.
    d2: array of shape (55,) containing squared distances.
    Returns array of shape (165,) with squared areas (can be negative due to rounding).
    
    Cayley-Menger for triangle with sides a, b, c:
    16 * area^2 = 2*(a^2*b^2 + b^2*c^2 + c^2*a^2) - (a^4 + b^4 + c^4)
    """
    a2 = d2[tri_dist_idx[:, 0]]
    b2 = d2[tri_dist_idx[:, 1]]
    c2 = d2[tri_dist_idx[:, 2]]
    
    # 16 * area^2 = 2*(a2*b2 + b2*c2 + c2*a2) - (a2^2 + b2^2 + c2^2)
    area2 = 2.0 * (a2 * b2 + b2 * c2 + c2 * a2) - (a2**2 + b2**2 + c2**2)
    
    return area2 / 16.0

def _soft_min_from_d2(d2, beta=100.0):
    """
    Soft minimum of triangle areas computed from squared distances.
    Returns negative soft-min (to minimize).
    """
    area2 = _cayley_menger_areas(d2)
    # Clip to avoid log of negative
    area2 = np.maximum(area2, 1e-15)
    areas = np.sqrt(area2)
    
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -np.min(areas)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    
    return -soft_min

def _embed_to_triangle(d2, n_iters=100, deadline=None):
    """
    Given optimal squared distances, embed points into the unit equilateral triangle.
    Uses iterative least-squares: fix two points, solve for others.
    
    Returns Cartesian coordinates (N, 2).
    """
    if deadline is not None and time.time() > deadline - 2:
        # Fallback: use hexagonal init
        return _hexagonal_cart()
    
    # Target distances
    d_target = np.sqrt(np.maximum(d2, 0))
    
    # Build full distance matrix
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            D[i, j] = d_target[_dist_idx(i, j)]
            D[j, i] = d_target[_dist_idx(i, j)]
    
    # Start with a reasonable initial placement
    # Place points in a grid-like pattern within the triangle
    pts = _initial_embedding()
    
    # Iterative refinement: minimize sum of (dist_ij - D_ij)^2
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline - 2:
            break
        
        improved = False
        for i in range(N):
            # Try to optimize point i to match distances to all other points
            best_pt = pts[i].copy()
            best_err = _dist_error(pts, D)
            
            # Try small perturbations
            for dx in np.linspace(-0.05, 0.05, 11):
                for dy in np.linspace(-0.05, 0.05, 11):
                    trial = pts.copy()
                    trial[i] = pts[i] + np.array([dx, dy])
                    # Check if inside triangle
                    if _inside_triangle(trial[i]):
                        err = _dist_error(trial, D)
                        if err < best_err:
                            best_err = err
                            best_pt = trial[i].copy()
            
            if not np.allclose(best_pt, pts[i]):
                pts[i] = best_pt
                improved = True
        
        if not improved:
            break
    
    # Final check: ensure all points are in the triangle
    for i in range(N):
        if not _inside_triangle(pts[i]):
            pts[i] = _project_to_triangle(pts[i])
    
    return pts

def _dist_error(pts, D_target):
    """Compute sum of squared differences between actual and target distances."""
    err = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d_actual = np.linalg.norm(pts[i] - pts[j])
            err += (d_actual - D_target[i, j])**2
    return err

def _inside_triangle(p, eps=1e-9):
    """Check if point p is inside or on the boundary of the unit equilateral triangle."""
    x, y = p
    if x < -eps or y < -eps or x > 1 + eps:
        return False
    if y > H + eps:
        return False
    # Check below the two upper edges
    # Edge from (0,0) to (0.5, H): y <= sqrt(3)*x
    if y > np.sqrt(3) * x + eps:
        return False
    # Edge from (1,0) to (0.5, H): y <= sqrt(3)*(1-x)
    if y > np.sqrt(3) * (1 - x) + eps:
        return False
    return True

def _project_to_triangle(p):
    """Project point p to the nearest point in the unit equilateral triangle."""
    x, y = p
    # Clip to bounding box
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, H)
    
    # Check and project to edges
    # Edge 1: y <= sqrt(3)*x
    if y > np.sqrt(3) * x:
        # Project to line y = sqrt(3)*x
        # Nearest point on line from (x,y)
        a = np.sqrt(3)
        # Line: a*x - y = 0
        d = (a*x - y) / (a**2 + 1)
        x = x - a * d
        y = y + d
    
    # Edge 2: y <= sqrt(3)*(1-x)
    if y > np.sqrt(3) * (1 - x):
        # Line: sqrt(3)*x + y = sqrt(3)
        a = np.sqrt(3)
        d = (a*x + y - a) / (a**2 + 1)
        x = x - a * d
        y = y - d
    
    # Final clip
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, H)
    
    return np.array([x, y])

def _initial_embedding():
    """Generate an initial embedding of N points in the triangle."""
    # Use a simple grid pattern
    pts = []
    # Try different grid configurations
    for rows in range(3, 6):
        for cols in range(3, 6):
            if rows * cols >= N:
                break
        else:
            continue
        break
    
    # Fallback: evenly spaced
    rng = np.random.default_rng(42)
    for _ in range(N):
        while True:
            x = rng.random()
            y = rng.random() * H
            if _inside_triangle(np.array([x, y])):
                break
        pts.append(np.array([x, y]))
    
    return np.array(pts)

def _hexagonal_cart():
    """Fallback: return hexagonal lattice points in Cartesian."""
    pts = []
    s = 0.25
    for i in range(5):
        for j in range(5):
            x = 0.25 + i * s * 0.5 + j * s * np.sqrt(3)/2 * 0.5
            y = j * s * np.sqrt(3)/2
            if _inside_triangle(np.array([x, y])) and len(pts) < N:
                pts.append(np.array([x, y]))
    while len(pts) < N:
        pts.append(np.array([0.5, H/3]))
    return np.array(pts[:N])

def _d2_to_bary_and_score(d2):
    """
    Convert squared distances to barycentric coords and compute score.
    This is used to evaluate the quality of a distance configuration.
    """
    pts = _embed_to_triangle(d2)
    # Convert to barycentric
    w = pts[:, 1] / H
    v = pts[:, 0] - 0.5 * w
    u = 1 - v - w
    bary = np.column_stack([u, v])
    
    # Compute areas
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    areas = np.abs(det)
    
    return bary, np.min(areas)

def _init_d2_from_bary(bary):
    """Initialize squared distances from a barycentric configuration."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    pts = np.column_stack([x, y])
    
    d2 = np.zeros(num_dists)
    for i in range(N):
        for j in range(i+1, N):
            d2[_dist_idx(i, j)] = np.sum((pts[i] - pts[j])**2)
    return d2

def _optimize_d2(d2_init, beta=100.0, maxiter=500, deadline=None):
    """
    Optimize squared distances to maximize minimum triangle area.
    """
    if deadline is not None and time.time() > deadline - 5:
        return d2_init
    
    def obj(d2):
        return _soft_min_from_d2(d2, beta=beta)
    
    bounds = [(0, 1.0)] * num_dists
    
    try:
        result = minimize(
            obj,
            d2_init,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        return result.x
    except Exception:
        return d2_init

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        u = bary[:, 0]
        v = bary[:, 1]
        w = 1.0 - u - v
        # Check validity
        if np.any(u < -1e-6) or np.any(v < -1e-6) or np.any(w < -1e-6):
            return False
        if np.any(u > 1+1e-6) or np.any(v > 1+1e-6) or np.any(w > 1+1e-6):
            return False
        
        ui = u[tri_idx[:, 0]]
        vi = v[tri_idx[:, 0]]
        wi = w[tri_idx[:, 0]]
        uj = u[tri_idx[:, 1]]
        vj = v[tri_idx[:, 1]]
        wj = w[tri_idx[:, 1]]
        uk = u[tri_idx[:, 2]]
        vk = v[tri_idx[:, 2]]
        wk = w[tri_idx[:, 2]]
        det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
        areas = np.abs(det)
        val = np.min(areas)
        
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # Phase 1: Generate initial barycentric configs and convert to d2
    inits = []
    
    # Hexagonal init
    for s in np.linspace(0.15, 0.45, 10):
        points = []
        for i in range(5):
            for j in range(5):
                u = i * s
                v = j * s
                if u + v <= 1 and len(points) < N:
                    points.append([u, v])
        if len(points) == N:
            inits.append(np.array(points))
    
    # Random inits
    rng = np.random.default_rng(42)
    for _ in range(10):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        inits.append(np.column_stack([u, v]))
    
    # Phase 2: Optimize d2 from each init
    best_d2 = None
    best_d2_score = -np.inf
    
    for init_bary in inits:
        if time.time() > deadline - 30:
            break
        
        d2_init = _init_d2_from_bary(init_bary)
        
        # Optimize with different betas
        for beta in [50, 100, 200]:
            if time.time() > deadline - 10:
                break
            d2_opt = _optimize_d2(d2_init, beta=beta, maxiter=300, deadline=deadline)
            
            # Evaluate
            bary_eval, score = _d2_to_bary_and_score(d2_opt)
            if score > best_d2_score:
                best_d2_score = score
                best_d2 = d2_opt.copy()
            
            # Update best bary
            update_best(bary_eval)
    
    # Phase 3: Further refine best d2
    if best_d2 is not None:
        for beta in [150, 250, 400]:
            if time.time() > deadline - 15:
                break
            d2_refined = _optimize_d2(best_d2, beta=beta, maxiter=400, deadline=deadline)
            bary_eval, score = _d2_to_bary_and_score(d2_refined)
            if score > best_d2_score:
                best_d2_score = score
                best_d2 = d2_refined.copy()
            update_best(bary_eval)
    
    # Phase 4: Local polish in barycentric space
    if best_bary is not None:
        # Coordinate descent polish
        for it in range(50):
            if time.time() > deadline - 10:
                break
            improved = False
            for i in range(N):
                for du in [0.002, -0.002, 0.005, -0.005, 0.01, -0.01]:
                    trial = best_bary.copy()
                    trial[i, 0] += du
                    if update_best(trial):
                        improved = True
                        break
                if improved:
                    continue
                for dv in [0.002, -0.002, 0.005, -0.005, 0.01, -0.01]:
                    trial = best_bary.copy()
                    trial[i, 1] += dv
                    if update_best(trial):
                        improved = True
                        break
            if not improved:
                break
    
    # Phase 5: Nelder-Mead polish
    if best_bary is not None and time.time() < deadline - 10:
        def hard_min_obj(bary_flat):
            bary = bary_flat.reshape(N, 2)
            u = bary[:, 0]
            v = bary[:, 1]
            w = 1.0 - u - v
            if np.any(u < -1e-8) or np.any(v < -1e-8) or np.any(w < -1e-8):
                return 1.0  # Penalty
            ui = u[tri_idx[:, 0]]
            vi = v[tri_idx[:, 0]]
            wi = w[tri_idx[:, 0]]
            uj = u[tri_idx[:, 1]]
            vj = v[tri_idx[:, 1]]
            wj = w[tri_idx[:, 1]]
            uk = u[tri_idx[:, 2]]
            vk = v[tri_idx[:, 2]]
            wk = w[tri_idx[:, 2]]
            det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
            return -np.min(np.abs(det))
        
        try:
            result = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            update_best(result.x.reshape(N, 2))
        except Exception:
            pass
    
    # Ensure we have a valid result
    if best_bary is None:
        # Fallback
        rng = np.random.default_rng(1)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
    
    # Final: convert to Cartesian
    u = best_bary[:, 0]
    v = best_bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    cart_pts = np.column_stack([x, y])
    
    # Project any points outside the triangle
    for i in range(N):
        if not _inside_triangle(cart_pts[i]):
            cart_pts[i] = _project_to_triangle(cart_pts[i])
    
    return cart_pts

# Test run if executed directly
if __name__ == "__main__":
    pts = construct()
    print("Points:")
    print(pts)
    # Calculate score
    H_val = np.sqrt(3) / 2
    w = pts[:, 1] / H_val
    v = pts[:, 0] - 0.5 * w
    u = 1 - v - w
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    areas = np.abs(det)
    print("Min Area (normalized):", np.min(areas))
