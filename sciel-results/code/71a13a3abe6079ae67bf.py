"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of equilateral triangle with side 1

# Safety margin from 150s limit
DEADLINE_MARGIN = 140.0

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
NUM_TRIP = len(_TRIPLETS)  # 165

# Triangle vertices in Cartesian
_VERTS = np.array([[0, 0], [1, 0], [0.5, S3H]])

def _bary_to_cart(bary):
    """Convert barycentric coordinates (N, 3) to Cartesian (N, 2)."""
    return bary[:, 0:1] * _VERTS[0] + bary[:, 1:2] * _VERTS[1] + bary[:, 2:3] * _VERTS[2]

def _cart_to_bary(cart):
    """Convert Cartesian (N, 2) to barycentric coordinates (N, 3)."""
    x = cart[:, 0]
    y = cart[:, 1]
    b2 = y / S3
    b1 = x - 0.5 * b2
    b0 = 1 - b1 - b2
    return np.column_stack([b0, b1, b2])

def _vectorized_min_area_cart(points):
    """Calculate the minimum area of all triangles formed by the points in Cartesian."""
    p1 = points[_T_I]
    p2 = points[_T_J]
    p3 = points[_T_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _soft_min_area_cart(points, alpha=100):
    """Calculate soft minimum of triangle areas using log-sum-exp."""
    p1 = points[_T_I]
    p2 = points[_T_J]
    p3 = points[_T_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    min_a = np.min(areas)
    return min_a - (1.0 / alpha) * np.log(np.sum(np.exp(-alpha * (areas - min_a))))

def _project_cartesian(points):
    """Project points to be inside the triangle."""
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    
    y = np.maximum(0, y)
    y = np.minimum(y, S3 * x)
    y = np.minimum(y, S3 - S3 * x)
    
    x = np.maximum(0, x)
    x = np.minimum(1, x)
    y = np.maximum(0, y)
    y = np.minimum(y, S3 * x)
    y = np.minimum(y, S3 - S3 * x)
    
    return np.column_stack([x, y])

def _is_valid_cart(points):
    """Check if all points are inside the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    tol = 1e-9
    return (np.all(y >= -tol) and 
            np.all(y <= S3 * x + tol) and 
            np.all(S3 * x + y <= S3 + tol))

def _random_bary():
    """Generate random barycentric coordinates."""
    r = np.random.random((N, 3))
    r /= r.sum(axis=1, keepdims=True)
    return r

def _lattice_model(params):
    """
    Quasi-lattice model: 3x4 grid topology with affine transform + perturbations.
    params: [a, b, c, d, e, f, ...perturbations (2*N)]
    - Affine: x = a*ux + b*uy + c, y = d*ux + e*uy + f
    - Perturbations: small offsets for each point
    """
    # 3x4 grid in unit square-like coordinates
    # Rows: 0, 1, 2 (3 rows)
    # Cols: 0, 1, 2, 3 (4 cols) but row 2 has only 3 points
    # Total: 4 + 4 + 3 = 11
    
    # Define base grid positions (in a normalized coordinate system)
    # Row 0 (bottom): 4 points
    # Row 1 (middle): 4 points  
    # Row 2 (top): 3 points
    
    # Base positions in a triangular lattice-like arrangement
    # Use barycentric-like coordinates for the grid
    base = np.zeros((N, 2))
    
    # Row 0: y=0, x = 0, 1/3, 2/3, 1
    base[0] = [0.0, 0.0]
    base[1] = [1/3, 0.0]
    base[2] = [2/3, 0.0]
    base[3] = [1.0, 0.0]
    
    # Row 1: y = 1/3 * S3H, x shifted
    base[4] = [1/6, S3H/3]
    base[5] = [1/2, S3H/3]
    base[6] = [5/6, S3H/3]
    base[7] = [1.0 - 1/6, S3H/3]  # = 5/6, but let me recalculate
    
    # Actually, let's use a proper triangular grid
    # In an equilateral triangle with side 1:
    # Level 0 (bottom, y=0): 4 points at x = 0, 1/3, 2/3, 1
    # Level 1 (y = S3H/3): 3 or 4 points
    # Level 2 (y = 2*S3H/3): 2 or 3 points
    # Level 3 (top, y = S3H): 1 point
    
    # For 11 points, a common arrangement:
    # Bottom row: 4 points
    # Middle row: 4 points  
    # Top row: 3 points
    
    # Let me define a proper triangular lattice
    # Using the fact that the triangle has vertices at (0,0), (1,0), (0.5, S3H)
    
    # A natural 11-point configuration:
    # 4 points on the bottom edge
    # 4 points in the middle
    # 3 points near the top
    
    # Let's use a parametric grid
    # u,v in [0,1] with v representing height fraction
    
    # Base grid (before affine transform)
    # I'll use a 3-row grid with 4, 4, 3 points
    base[0, 0] = 0.0;  base[0, 1] = 0.0
    base[1, 0] = 0.25; base[1, 1] = 0.0
    base[2, 0] = 0.50; base[2, 1] = 0.0
    base[3, 0] = 0.75; base[3, 1] = 0.0
    
    base[4, 0] = 0.125; base[4, 1] = 0.333
    base[5, 0] = 0.375; base[5, 1] = 0.333
    base[6, 0] = 0.625; base[6, 1] = 0.333
    base[7, 0] = 0.875; base[7, 1] = 0.333
    
    base[8, 0] = 0.25;  base[8, 1] = 0.667
    base[9, 0] = 0.50;  base[9, 1] = 0.667
    base[10, 0] = 0.75; base[10, 1] = 0.667
    
    # Affine transform: [x, y] = A @ [u, v] + t
    a, b, c, d, e, f = params[0:6]
    
    # Apply affine
    cart = np.empty((N, 2))
    cart[:, 0] = a * base[:, 0] + b * base[:, 1] + c
    cart[:, 1] = d * base[:, 0] + e * base[:, 1] + f
    
    # Add perturbations
    if len(params) > 6:
        perts = params[6:6+2*N].reshape(N, 2)
        cart += perts
    
    return cart

def _lattice_objective(params, alpha=50):
    """Objective for lattice model."""
    cart = _lattice_model(params)
    cart = _project_cartesian(cart)
    
    soft_min = _soft_min_area_cart(cart, alpha)
    
    # Penalty for points outside triangle
    x = cart[:, 0]
    y = cart[:, 1]
    pen1 = np.sum(np.maximum(0, -y)**2)
    pen2 = np.sum(np.maximum(0, y - S3 * x)**2)
    pen3 = np.sum(np.maximum(0, S3 * x + y - S3)**2)
    pen = pen1 + pen2 + pen3
    
    return -soft_min + 100 * pen

def _bary_objective(flat_bary, alpha=100):
    """Objective function in barycentric coordinates."""
    bary = flat_bary.reshape(N, 3)
    bary = np.maximum(0, bary)
    bary /= bary.sum(axis=1, keepdims=True)
    
    cart = _bary_to_cart(bary)
    cart = _project_cartesian(cart)
    
    soft_min = _soft_min_area_cart(cart, alpha)
    
    x = cart[:, 0]
    y = cart[:, 1]
    pen1 = np.sum(np.maximum(0, -y)**2)
    pen2 = np.sum(np.maximum(0, y - S3 * x)**2)
    pen3 = np.sum(np.maximum(0, S3 * x + y - S3)**2)
    pen = pen1 + pen2 + pen3
    
    return -soft_min + 100 * pen

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    np.random.seed(42)
    
    # Initial best from random
    best_bary = _random_bary()
    best_cart = _bary_to_cart(best_bary)
    best_cart = _project_cartesian(best_cart)
    best_score = _vectorized_min_area_cart(best_cart)
    
    # Phase 1: Optimize the lattice model
    # Parameters: 6 affine + 22 perturbations = 28 params
    # Start with reasonable affine: map [0,1]x[0,1] to the triangle
    # The triangle has vertices (0,0), (1,0), (0.5, S3H)
    # A simple affine: x = u, y = v * S3H * (1 - 0.5*|u-0.5|*2) ... complex
    # Simpler: use barycentric-like mapping
    
    # Initial affine that maps the unit square to approximately the triangle
    # x = u + 0.5*(v-0.5)  ... no
    # Let's just use: x = u, y = v * S3H (this maps to a parallelogram, not triangle)
    # Better: start with identity-ish and let optimization fix it
    
    init_params = np.array([
        1.0, 0.0, 0.0,    # a, b, c for x
        0.0, S3H, 0.0,    # d, e, f for y
    ])
    
    # Small random perturbations
    init_params = np.concatenate([init_params, np.random.randn(2*N) * 0.01])
    
    # Optimize lattice model with multiple alphas
    for alpha in [10, 50, 100, 200, 500]:
        if time.time() > deadline:
            break
        
        # Multiple restarts
        for restart in range(5):
            if time.time() > deadline:
                break
            
            if restart == 0:
                x0 = init_params.copy()
            else:
                x0 = init_params + np.random.randn(len(init_params)) * 0.05
            
            res = minimize(_lattice_objective, x0, args=(alpha,),
                          method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.fun < _lattice_objective(init_params, alpha):
                init_params = res.x
            
            # Evaluate
            cart = _lattice_model(init_params)
            cart = _project_cartesian(cart)
            score = _vectorized_min_area_cart(cart)
            if score > best_score:
                best_score = score
                best_cart = cart
    
    # Phase 2: Use the lattice result as initial guess for full barycentric optimization
    if time.time() < deadline - 20:
        best_bary = _cart_to_bary(best_cart)
        
        for alpha in [50, 100, 200, 500]:
            if time.time() > deadline:
                break
            
            for restart in range(3):
                if time.time() > deadline:
                    break
                
                if restart == 0:
                    x0 = best_bary.flatten()
                else:
                    x0 = _random_bary().flatten()
                
                res = minimize(_bary_objective, x0, args=(alpha,),
                              method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-7})
                
                if res.fun < _bary_objective(best_bary.flatten(), alpha):
                    new_bary = res.x.reshape(N, 3)
                    new_bary = np.maximum(0, new_bary)
                    new_bary /= new_bary.sum(axis=1, keepdims=True)
                    
                    new_cart = _bary_to_cart(new_bary)
                    new_cart = _project_cartesian(new_cart)
                    
                    new_score = _vectorized_min_area_cart(new_cart)
                    if new_score > best_score:
                        best_score = new_score
                        best_bary = new_bary
                        best_cart = new_cart
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5:
        x0 = best_cart.flatten()
        
        def cart_objective(flat_cart):
            cart = flat_cart.reshape(N, 2)
            cart = _project_cartesian(cart)
            area = _vectorized_min_area_cart(cart)
            return -area
        
        res = minimize(cart_objective, x0, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-9})
        
        if res.fun < -best_score:
            final_cart = res.x.reshape(N, 2)
            final_cart = _project_cartesian(final_cart)
            final_score = _vectorized_min_area_cart(final_cart)
            if final_score > best_score:
                best_cart = final_cart
                best_score = final_score
    
    # Ensure validity
    best_cart = _project_cartesian(best_cart)
    
    return best_cart
# EVOLVE-BLOCK-END
