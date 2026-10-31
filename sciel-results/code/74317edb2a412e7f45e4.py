import numpy as np
from scipy.optimize import minimize
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
AREA_EQUI = 0.5 * H  # Area of the unit equilateral triangle

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
    v = bary[:, 1]. copy()
    
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

def get_initial_guess():
    """
    Generate a symmetric initial guess.
    We place one point at the centroid and others in a ring around it,
    or use a known good configuration from literature if available.
    Here we use a heuristic grid-based start which is often better than random.
    """
    # A common good start for Heilbronn is a triangular grid
    # For N=11, we can try a 3x3 grid plus center plus corners? 
    # Actually, let's use a perturbed regular polygon-like structure inside.
    
    # Start with a 3x3 grid in barycentric coords
    # Points (i/3, j/3) where i,j >= 0 and i+j <= 3
    pts = []
    for i in range(4):
        for j in range(4 - i):
            pts.append([i/3.0, j/3.0])
    
    # We have 10 points from the 3x3 grid (0,0), (1/3,0), (2/3,0), (1,0),
    # (0,1/3), (1/3,1/3), (2/3,1/3), (0,2/3), (1/3,2/3), (0,1)
    # Wait, the grid i+j<=3 has 10 points:
    # (0,0), (1/3,0), (2/3,0), (1,0)
    # (0,1/3), (1/3,1/3), (2/3,1/3)
    # (0,2/3), (1/3,2/3)
    # (0,1)
    # Total 10 points. We need 11.
    # Let's add the centroid (1/3, 1/3) is already there.
    # Let's add a point near the center, e.g., (1/3, 1/3) is occupied.
    # Let's just add a random point or shift one.
    # Better: Use a 4x4 grid subset?
    
    # Let's try a different init: Random but constrained to be "spread out"
    rng = np.random.default_rng(42)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    bary = np.column_stack([u, v])
    
    # Force one point to be the centroid to break symmetry and provide a anchor
    bary[0] = [1/3, 1/3]
    
    return clamp_bary(bary)

def solve_feasibility(target_area, init_bary, time_limit=5.0):
    """
    Try to find a configuration where all triangle areas >= target_area.
    Returns (success, best_bary).
    """
    start_time = time.time()
    
    # Objective: Minimize the sum of squared violations (smooth)
    # Violation = max(0, target_area - area)
    # We use a soft penalty: sum((target_area - area)^2) for all areas < target_area
    # Or simply minimize -min(areas) but with constraints?
    # SLSQP handles constraints well.
    
    # Constraints: area_ijk >= target_area for all 165 triangles
    # This is too many constraints for SLSQP (165 constraints, 22 variables).
    # SLSQP might struggle or be slow.
    
    # Alternative: Minimize the "violation" function:
    # f(x) = sum_{k} max(0, target_area - area_k)^2
    # This is smooth (C1) and 0 if feasible.
    
    def obj(x):
        bary = x.reshape(N, 2)
        bary = clamp_bary(bary)
        areas = calc_areas_bary(bary)
        violations = np.maximum(0, target_area - areas)
        return np.sum(violations**2)
    
    # We don't strictly need constraints if the objective penalizes violations heavily.
    # But to ensure we stay inside the triangle, we use bounds.
    bounds = [(0, 1)] * (N * 2)
    
    # We also need to ensure barycentric sum <= 1.
    # SLSQP doesn't handle non-linear equality/inequality constraints in bounds easily.
    # We must use the 'constraints' argument.
    
    # Constraint: u_i + v_i <= 1 for all i
    # This is linear.
    # c_i(x) = 1 - u_i - v_i >= 0
    
    def con_linear(x):
        bary = x.reshape(N, 2)
        return 1.0 - bary[:, 0] - bary[:, 1]
    
    # SLSQP expects a single function returning an array for multiple constraints?
    # Or a list of dicts.
    # Let's use a single non-linear constraint function returning an array of 11 values.
    # scipy SLSQP supports this if 'type': 'ineq'.
    
    constraints = [{'type': 'ineq', 'fun': con_linear}]
    
    try:
        result = minimize(
            obj,
            init_bary.flatten(),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        
        if result.success or obj(result.x) < 1e-10:
            return True, clamp_bary(result.x.reshape(N, 2))
        else:
            # Check if it's "close enough"
            if obj(result.x) < 1e-6:
                return True, clamp_bary(result.x.reshape(N, 2))
            return False, clamp_bary(result.x.reshape(N, 2))
            
    except Exception:
        return False, init_bary

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = get_initial_guess()
    # Calculate initial best value
    best_val = np.min(calc_areas_bary(best_bary))
    
    # We want to maximize the min area.
    # Let's use Bisection on the target area.
    # Lower bound: 0
    # Upper bound: Area of equilateral triangle / N ? No, that's an upper bound for average.
    # Max possible min area is roughly Area_Total / N * some_factor.
    # For N=11, the seed score is ~31.7. The raw value is score/seed?
    # The prompt says "search score is value/seed".
    # The evaluator measures "raw value".
    # The seed program returns a value.
    # Let's estimate the range.
    # Area of unit triangle is ~0.433.
    # If we divide into 11 roughly equal areas, each is ~0.04.
    # The min area is likely smaller than the average.
    # Let's set upper bound to 0.1.
    
    lo = 0.0
    hi = 0.1
    
    # First, try to find a good starting point for bisection
    # by doing a few local optimizations on the "min area" objective directly
    # to get a baseline.
    
    def hard_min_obj(x):
        bary = x.reshape(N, 2)
        bary = clamp_bary(bary)
        areas = calc_areas_bary(bary)
        return -np.min(areas)
    
    # Quick local polish on initial guess
    try:
        bounds = [(0, 1)] * (N * 2)
        res = minimize(hard_min_obj, best_bary.flatten(), method='Nelder-Mead', options={'maxiter': 200})
        best_bary = clamp_bary(res.x.reshape(N, 2))
        best_val = -hard_min_obj(best_bary.flatten())
        # Update bounds for bisection
        lo = best_val
        hi = best_val * 1.5 if best_val > 0 else 0.1
    except Exception:
        pass

    # Bisection Loop
    # We have limited time. Each feasibility check takes ~1-2s.
    # We can afford ~20-30 checks.
    
    while time.time() < deadline - 10:
        mid = (lo + hi) / 2.0
        
        success, candidate_bary = solve_feasibility(mid, best_bary, time_limit=5.0)
        
        if success:
            # We found a config where all areas >= mid
            # Update best if this config is better (it should be, as min area >= mid)
            candidate_val = np.min(calc_areas_bary(candidate_bary))
            if candidate_val > best_val:
                best_val = candidate_val
                best_bary = candidate_bary
            lo = mid
        else:
            hi = mid
            
        # If the interval is small, we can stop or switch to final polish
        if hi - lo < 1e-5:
            break
            
    # Final Polish:
    # Use the best_bary found and do a high-precision local optimization
    # on the hard min objective to squeeze out any remaining gain.
    if time.time() < deadline - 2:
        try:
            bounds = [(0, 1)] * (N * 2)
            # Use SLSQP for final polish as it handles the non-smoothness better with tight tolerances?
            # Actually, Nelder-Mead is robust for the final push on non-smooth functions.
            res = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            candidate_bary = clamp_bary(res.x.reshape(N, 2))
            candidate_val = -hard_min_obj(candidate_bary.flatten())
            if candidate_val > best_val:
                best_bary = candidate_bary
                best_val = candidate_val
        except Exception:
            pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
