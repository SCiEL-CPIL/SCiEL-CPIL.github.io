import numpy as np
from scipy.optimize import differential_evolution, minimize, basinhopping
from itertools import combinations
import time

N = 11
TIME_LIMIT = 115.0  # Safety margin for 120s limit

# Pre-compute triangle indices
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

def calc_areas_bary(bary):
    """Calculate areas of all triangles formed by N points in barycentric coords."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
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
    
    return np.abs(det)

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

def generate_hexagonal_template():
    """
    Generate a base hexagonal lattice pattern for 11 points.
    We use a 4-row hexagonal packing: 1-2-3-3-2? No, 1+2+3+3+2 = 11? 
    Standard hex packing in triangle:
    Row 0 (top): 1
    Row 1: 2
    Row 2: 3
    Row 3: 3
    Row 4: 2
    Total: 11.
    
    Let's define relative coordinates in a local frame centered at the centroid.
    """
    # Centroid of equilateral triangle is at (0.5, H/3) in Cartesian, or (1/3, 1/3) in Bary.
    # Let's work in Barycentric coordinates directly for the template.
    # Center is (1/3, 1/3).
    
    # Define a hexagonal grid step size 's'
    # Points are generated in a local Cartesian-like space then mapped?
    # Easier: Define points relative to center (1/3, 1/3) in Bary.
    
    # Hexagonal neighbors in Bary:
    # (1/3, 1/3)
    # (1/3 + s, 1/3)
    # (1/3 - s, 1/3)
    # (1/3, 1/3 + s)
    # (1/3, 1/3 - s)
    # (1/3 + s/2, 1/3 + s*sqrt(3)/2) ... wait, Bary axes are 60 deg apart.
    # Distance in Bary space: d^2 = du^2 + dv^2 - du*dv is not Euclidean distance.
    # Euclidean distance squared between (u1,v1) and (u2,v2) in unit triangle:
    # dx = (v2 - v1) + 0.5*(w2 - w1) ... it's complex.
    
    # Let's stick to Cartesian template and transform to Bary.
    # Center: (0.5, H/3)
    # Hex lattice vectors in Cartesian:
    # a1 = (s, 0)
    # a2 = (s/2, s*sqrt(3)/2)
    
    # We want 11 points.
    # Center (0,0)
    # 6 neighbors:
    # (s, 0), (-s, 0)
    # (s/2, s*sqrt(3)/2), (-s/2, -s*sqrt(3)/2)
    # (s/2, -s*sqrt(3)/2), (-s/2, s*sqrt(3)/2)
    # 4 more?
    # Next ring has 12. We need 4 more to make 11.
    # Maybe a 2-3-3-2-1 structure?
    # Let's use a known good 11-point config structure if possible, or just generate a dense cluster.
    
    # Alternative: Use the "hexagonal lattice" subset.
    # Let's generate a 5x5 grid of hex points and pick the 11 closest to center.
    
    pts_cart = []
    s = 1.0 # Scale factor will be optimized
    # Generate points in a hex grid
    for i in range(-2, 3):
        for j in range(-2, 3):
            # Hex coords to Cartesian
            x = s * (i + j/2.0)
            y = s * (j * np.sqrt(3)/2.0)
            pts_cart.append([x, y])
            
    pts_cart = np.array(pts_cart)
    # Distance from center
    dists = np.linalg.norm(pts_cart, axis=1)
    idx = np.argsort(dists)
    # Take 11 closest
    pts_cart = pts_cart[idx[:11]]
    
    # Translate to center of triangle
    center = np.array([0.5, H/3.0])
    pts_cart = pts_cart + center
    
    # Convert to Barycentric
    # x = v + 0.5*w
    # y = w*H
    # w = y/H
    # v = x - 0.5*w
    # u = 1 - v - w
    w = pts_cart[:, 1] / H
    v = pts_cart[:, 0] - 0.5 * w
    u = 1.0 - v - w
    
    bary_template = np.column_stack([u, v])
    return bary_template

def lattice_to_bary(params, template):
    """
    Apply global transformations to the template.
    params: [scale, shear_x, shear_y, offset_u, offset_v]
    We apply an affine transformation in Barycentric space?
    Barycentric affine transformations preserve collinearity and ratios.
    If we scale the template, we might push points out of bounds.
    
    Let's define the transformation in Cartesian space for geometric intuition,
    then map to Bary.
    
    Actually, simpler: 
    1. Start with template in Bary.
    2. Center it at (1/3, 1/3).
    3. Apply scaling factor 's' around the center.
    4. Apply shear? Shear in Bary is tricky.
    5. Apply offset.
    
    Let's use Cartesian for the deformation.
    1. Template in Cart.
    2. Center at (0.5, H/3).
    3. Scale by 's'.
    4. Shear: x' = x + k1*y, y' = y + k2*x (relative to center).
    5. Offset: x' += du, y' += dv.
    6. Map to Bary.
    """
    # Template in Cart
    u_t = template[:, 0]
    v_t = template[:, 1]
    w_t = 1.0 - u_t - v_t
    x_t = v_t + 0.5 * w_t
    y_t = w_t * H
    
    center_x, center_y = 0.5, H/3.0
    
    # Relative to center
    dx = x_t - center_x
    dy = y_t - center_y
    
    s = params[0]
    k1 = params[1]
    k2 = params[2]
    du = params[3]
    dv = params[4]
    
    # Scale
    dx = dx * s
    dy = dy * s
    
    # Shear (simple linear mix)
    # dx_new = dx + k1 * dy
    # dy_new = dy + k2 * dx
    # Note: using original dx, dy for shear to keep it simple linear
    dx_new = dx + k1 * dy
    dy_new = dy + k2 * dx
    
    # Offset
    x_new = center_x + dx_new + du
    y_new = center_y + dy_new + dv
    
    # Convert to Bary
    w_new = y_new / H
    v_new = x_new - 0.5 * w_new
    u_new = 1.0 - v_new - w_new
    
    return np.column_stack([u_new, v_new])

def objective(params):
    """
    Objective for the lattice parameters.
    Returns -min_area (to be minimized).
    """
    template = generate_hexagonal_template()
    bary = lattice_to_bary(params, template)
    
    # Check validity
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # If any point is significantly outside, penalize heavily
    min_coord = np.min(np.vstack([u, v, w]))
    if min_coord < -1e-4:
        return 1e6 + abs(min_coord)
        
    bary_clamped = clamp_bary(bary)
    areas = calc_areas_bary(bary_clamped)
    min_area = np.min(areas)
    
    return -min_area

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = np.min(calc_areas_bary(bary))
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Global Search using Differential Evolution on Lattice Parameters
    # params: [scale, shear_x, shear_y, offset_u, offset_v]
    # Scale: 0.1 to 1.0 (template is roughly size 1)
    # Shear: -0.5 to 0.5
    # Offset: -0.2 to 0.2
    
    bounds = [
        (0.2, 1.2),   # Scale
        (-0.5, 0.5),  # Shear X
        (-0.5, 0.5),  # Shear Y
        (-0.2, 0.2),  # Offset U
        (-0.2, 0.2)   # Offset V
    ]
    
    # Run DE with time limit
    # DE can be slow, so we limit popsize and maxiter
    # We use a wrapper to check time
    
    de_result = None
    try:
        # Use a small popsize to keep it fast
        de_result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=200,
            popsize=10,
            tol=1e-8,
            polish=False, # We will polish manually
            # Update function to check time? DE doesn't support easy early stopping in scipy.
            # We will rely on maxiter being small enough.
            # 10 * 200 * 5 params * eval cost.
            # Eval cost is ~165 determinants. Very fast.
            # Should finish in < 10s.
        )
        
        if de_result.success:
            best_params = de_result.x
            best_bary = lattice_to_bary(best_params, generate_hexagonal_template())
            update_best(best_bary)
    except Exception as e:
        print(f"DE failed: {e}")
    
    # 2. Local Refinement of the best lattice solution
    if best_bary is not None:
        # We can refine the lattice parameters further
        # Or, we can switch to full 22-dim optimization starting from this good init
        # Given the stagnation warning, let's try a hybrid:
        # Use the lattice solution as a strong init for a full NLP or SLSQP on all 22 coords.
        
        # Define full 22-dim objective
        def full_obj(x):
            bary = x.reshape(N, 2)
            bary = clamp_bary(bary)
            areas = calc_areas_bary(bary)
            return -np.min(areas)
        
        # Use SLSQP on full coords
        # This might be slow, but we have time.
        # Let's try a few restarts from the lattice solution + noise
        
        rng = np.random.default_rng(123)
        
        for i in range(5):
            if time.time() > deadline - 20:
                break
                
            # Perturb the lattice solution
            noise = rng.normal(0, 0.05, best_bary.shape)
            init_bary = clamp_bary(best_bary + noise)
            
            x0 = init_bary.flatten()
            
            # Bounds for bary coords
            b_bounds = [(0, 1)] * (N * 2)
            
            # SLSQP doesn't handle the non-differentiable min well directly without constraints.
            # Let's use the epigraph formulation or just minimize -min_area with smooth surrogate?
            # Actually, SLSQP on -min_area is non-smooth.
            # Let's use a soft-min surrogate for the local polish.
            
            def soft_min_obj(x, beta=100.0):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                areas = calc_areas_bary(bary)
                # Log-sum-exp trick for -min
                # min(areas) = -max(-areas)
                # max(-areas) approximated by (1/beta) * log(sum(exp(-beta * areas)))
                # So -min(areas) approx - (1/beta) * log(sum(exp(-beta * areas)))
                # Wait. We want to MINIMIZE -min(areas).
                # So we want to MAXIMIZE min(areas).
                # Equivalent to MINIMIZING -min(areas).
                
                # Soft min: m_beta = (1/beta) * log(sum(exp(-beta * a_i)))
                # This is a smooth approximation of min(a_i).
                # As beta -> inf, m_beta -> min(a_i).
                # We want to maximize m_beta.
                # So objective is -m_beta.
                
                scaled = -beta * areas
                max_scaled = np.max(scaled)
                if max_scaled > 700:
                    return -np.min(areas)
                log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
                soft_min_val = log_sum / beta
                return -soft_min_val
            
            try:
                res = minimize(
                    soft_min_obj,
                    x0,
                    method='SLSQP',
                    bounds=b_bounds,
                    options={'maxiter': 100, 'ftol': 1e-10}
                )
                
                if res.success:
                    update_best(res.x.reshape(N, 2))
            except Exception as e:
                pass

    # 3. Final Polish with high-beta soft min on the absolute best
    if best_bary is not None and time.time() < deadline - 5:
        def high_beta_obj(x):
            bary = x.reshape(N, 2)
            bary = clamp_bary(bary)
            areas = calc_areas_bary(bary)
            beta = 500.0
            scaled = -beta * areas
            max_scaled = np.max(scaled)
            if max_scaled > 700:
                return -np.min(areas)
            log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
            soft_min_val = log_sum / beta
            return -soft_min_val
            
        x0 = best_bary.flatten()
        b_bounds = [(0, 1)] * (N * 2)
        
        try:
            res = minimize(
                high_beta_obj,
                x0,
                method='SLSQP',
                bounds=b_bounds,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if res.success:
                update_best(res.x.reshape(N, 2))
        except Exception as e:
            pass

    # Convert to cartesian
    if best_bary is None:
        # Fallback to random
        rng = np.random.default_rng(0)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
        
    cart = bary_to_cart(best_bary)
    return cart

# EVOLVE-BLOCK-END
