import numpy as np
from scipy.spatial import Delaunay
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0  # Safety margin for 150s limit

# Pre-compute triangle indices for all combinations
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def cart_to_bary(cart):
    """Convert Cartesian (x, y) to barycentric coords (u, v)."""
    x = cart[:, 0]
    y = cart[:, 1]
    v = x - 0.5 * (1.0 - y / H)  # Wait, let's derive properly.
    # x = v + 0.5(1-u-v) = v + 0.5 - 0.5u - 0.5v = 0.5 + 0.5v - 0.5u
    # y = (1-u-v)H  => 1-u-v = y/H => u+v = 1 - y/H
    # From x eq: 0.5v - 0.5u = x - 0.5 => v - u = 2x - 1
    # System:
    # u + v = 1 - y/H
    # -u + v = 2x - 1
    # Adding: 2v = 1 - y/H + 2x - 1 = 2x - y/H => v = x - 0.5*(y/H)
    # u = (1 - y/H) - v = 1 - y/H - x + 0.5*y/H = 1 - x - 0.5*(y/H)
    
    v = x - 0.5 * (y / H)
    u = 1.0 - x - 0.5 * (y / H)
    return np.column_stack([u, v])

def calc_areas_cart(cart):
    """Calculate areas of all triangles formed by N points in Cartesian coords."""
    xi = cart[tri_idx[:, 0], 0]
    yi = cart[tri_idx[:, 0], 1]
    
    xj = cart[tri_idx[:, 1], 0]
    yj = cart[tri_idx[:, 1], 1]
    
    xk = cart[tri_idx[:, 2], 0]
    yk = cart[tri_idx[:, 2], 1]
    
    # Area = 0.5 * |det([xj-xi, yj-yi; xk-xi, yk-yi])|
    det = (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)
    return 0.5 * np.abs(det)

def clamp_bary(bary, eps=1e-9):
    """Clamp barycentric coordinates to the strictly valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def objective(cart):
    """
    Objective function combining:
    1. Maximize the smallest triangle area (primary goal).
    2. Penalize "skinny" triangles via circumradius (secondary goal for stability).
    
    We use a weighted sum or a lexicographic approach.
    Since we want to maximize min_area, we minimize -min_area.
    To avoid degenerate configurations where points are collinear or very close,
    we add a penalty term for the sum of squared circumradii of the Delaunay triangles.
    """
    min_area = np.min(calc_areas_cart(cart))
    
    # Compute Delaunay triangulation to identify "structural" triangles
    # Note: Delaunay is only well-defined if points are in general position.
    # For robustness, we compute circumradii for ALL triangles but weight them?
    # No, the directive suggests using Delaunay.
    # However, computing Delaunay inside the objective is expensive.
    # Alternative: Use a smooth surrogate that approximates the "fatness" of the mesh.
    # A common measure is the sum of 1/area^2 for all triangles, which penalizes small areas heavily.
    
    areas = calc_areas_cart(cart)
    # Avoid division by zero
    safe_areas = np.where(areas > 1e-12, areas, 1e-12)
    
    # Penalty: Sum of inverse areas (encourages spreading out)
    # We want to maximize min_area, so we minimize -min_area + lambda * sum(1/area)
    # But sum(1/area) is dominated by the smallest area, so it's redundant with min_area.
    
    # Let's stick to the primary goal: Maximize min_area.
    # The "Delaunay constraint" idea is that we should ensure the triangulation is Delaunay.
    # If the current configuration is not Delaunay, we can try to flip edges?
    # That's complex. Instead, we use a global search (Simulated Annealing) 
    # that explores the space of Cartesian coordinates.
    
    return -min_area

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration in Cartesian.
    """
    best_cart = None
    best_val = -np.inf
    
    for scale in [0.2, 0.25, 0.3]:
        pts = []
        # Hexagonal packing in Cartesian
        # Rows of points
        num_rows = int(1.0 / (scale * np.sqrt(3)/2)) + 1
        for i in range(num_rows):
            y = i * scale * np.sqrt(3)/2
            if y > H:
                break
            # Offset every other row
            offset = (scale / 2) if (i % 2 == 1) else 0
            x_start = offset
            while x_start < 1.0:
                # Check if point is inside triangle
                # x >= 0, y >= 0, y <= H * (1 - x)  (for the specific triangle)
                # Actually, triangle is (0,0), (1,0), (0.5, H)
                # Left edge: x=0? No, (0,0) to (0.5, H) => y = H/(0.5) x = 2H x.
                # Right edge: (1,0) to (0.5, H) => y = -2H (x-1) = 2H(1-x)
                # So y <= min(2H x, 2H (1-x))
                
                if x_start >= 0 and x_start <= 1.0:
                    y_max = min(2*H*x_start, 2*H*(1-x_start))
                    if y <= y_max + 1e-9:
                        pts.append([x_start, y])
                x_start += scale
        
        if len(pts) >= N:
            rng = np.random.default_rng(42)
            for _ in range(5):
                if len(pts) == N:
                    cart = np.array(pts)
                else:
                    idx = rng.choice(len(pts), N, replace=False)
                    cart = np.array(pts)[idx]
                
                # Clamp to triangle
                cart = clamp_to_triangle_cart(cart)
                val = -objective(cart)
                if val > best_val:
                    best_val = val
                    best_cart = cart.copy()
                
    if best_cart is None:
        rng = np.random.default_rng(42)
        cart = rng.random((N, 2))
        cart = clamp_to_triangle_cart(cart)
        best_cart = cart
        
    return best_cart

def clamp_to_triangle_cart(cart, eps=1e-6):
    """Clamp Cartesian points to be inside the triangle."""
    # Triangle vertices: (0,0), (1,0), (0.5, H)
    # Inequalities:
    # 1. y >= 0
    # 2. y <= 2H x  (Left edge)
    # 3. y <= 2H (1-x) (Right edge)
    
    x = cart[:, 0].copy()
    y = cart[:, 1].copy()
    
    y = np.clip(y, eps, H - eps)
    
    # Project onto the triangle if outside
    # This is a bit tricky to do exactly with simple clipping.
    # We convert to barycentric, clamp, and convert back.
    bary = cart_to_bary(cart)
    bary = clamp_bary(bary)
    cart_new = bary_to_cart(bary)
    
    return cart_new

def simulated_annealing(cart_init, time_limit=10.0):
    """
    Simulated Annealing to maximize the minimum triangle area.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    cart_current = cart_init.copy()
    val_current = -objective(cart_current)
    
    cart_best = cart_current.copy()
    val_best = val_current
    
    T = 0.05  # Initial temperature
    cooling_rate = 0.995
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline and T > 1e-6:
        # Generate neighbor
        # Move a random point by a small amount
        idx = rng.integers(N)
        delta = rng.normal(0, T, 2)
        
        cart_new = cart_current.copy()
        cart_new[idx] += delta
        
        # Clamp to triangle
        cart_new = clamp_to_triangle_cart(cart_new)
        
        val_new = -objective(cart_new)
        
        delta_val = val_new - val_current
        
        if delta_val > 0:
            # Accept if better
            cart_current = cart_new
            val_current = val_new
            if val_new > val_best:
                val_best = val_new
                cart_best = cart_new.copy()
        else:
            # Accept if worse with probability exp(delta_val / T)
            if rng.random() < np.exp(delta_val / T):
                cart_current = cart_new
                val_current = val_new
        
        T *= cooling_rate
        
    return cart_best

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_cart = None
    best_val = -np.inf
    
    def update_best(cart):
        nonlocal best_cart, best_val
        if cart is None: return
        cart = clamp_to_triangle_cart(cart)
        val = -objective(cart)
        if val > best_val:
            best_val = val
            best_cart = cart.copy()

    # 1. Hexagonal Init
    if time.time() < deadline - 10:
        hex_init = generate_hexagonal_init()
        update_best(hex_init)
        
    # 2. Multi-start Simulated Annealing
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 5:
        restart_count += 1
        if best_cart is not None and restart_count % 2 == 0:
            # Perturb best
            perturbation = rng.normal(0, 0.05, best_cart.shape)
            init = clamp_to_triangle_cart(best_cart + perturbation)
        else:
            # Random init
            init = rng.random((N, 2))
            init = clamp_to_triangle_cart(init)
            
        # Run SA with remaining time / 2, capped at 10s
        sa_time = min(10.0, max(1.0, (deadline - time.time()) / 2))
        sa_result = simulated_annealing(init, time_limit=sa_time)
        update_best(sa_result)
        
    # 3. Final Local Refinement (Nelder-Mead) on the best result
    if best_cart is not None and time.time() < deadline - 2:
        from scipy.optimize import minimize
        try:
            # Flatten cart
            x0 = best_cart.flatten()
            
            def obj_flat(x):
                cart = x.reshape(N, 2)
                cart = clamp_to_triangle_cart(cart)
                return -objective(cart)
                
            result = minimize(
                obj_flat,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            update_best(clamp_to_triangle_cart(result.x.reshape(N, 2)))
        except Exception:
            pass
            
    if best_cart is None:
        # Fallback
        best_cart = generate_hexagonal_init()
        
    return best_cart

# EVOLVE-BLOCK-END
