"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
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

def calc_areas_bary(bary):
    """
    Calculate areas of all triangles formed by N points given in barycentric coords (u, v).
    Returns array of shape (165,) containing the area ratios.
    """
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

def clamp_bary(bary, eps=1e-6):
    """
    Clamp barycentric coordinates to the strictly valid triangle region.
    u >= eps, v >= eps, u+v <= 1 - eps
    """
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    # Clip to strict interior
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    # Ensure u + v <= 1 - eps
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    # Re-check bounds after scaling
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def soft_min_value(areas, beta):
    """Compute soft-min of areas array."""
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return np.min(areas)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -log_sum / beta

def composite_obj(bary_flat, beta=200.0, lam=0.05, frac=0.3):
    """
    Composite objective: soft-min minus variance regularization on smallest areas.
    This encourages multiple triangles to be at the minimum simultaneously.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    sm = soft_min_value(areas, beta)
    
    # Variance on the smallest fraction of areas
    n_small = max(3, int(frac * len(areas)))
    sorted_areas = np.sort(areas)
    small_areas = sorted_areas[:n_small]
    variance_term = np.var(small_areas)
    
    return -(sm - lam * variance_term)

def hard_min_obj(bary_flat):
    """Objective using hard min (for final polish)."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def cart_to_bary(cart):
    """Convert Cartesian coords to barycentric (u, v)."""
    w = cart[:, 1] / H
    v = cart[:, 0] - 0.5 * w
    u = 1.0 - v - w
    return np.column_stack([u, v])

def generate_symmetric_init():
    """
    Generate a symmetric initial configuration based on the known structure
    of the optimal Heilbronn configuration for 11 points.
    """
    # Known approximate structure: 3 corners, 3 edge-midpoints, 3 near-midpoints, 2 interior
    # Using barycentric coordinates with approximate symmetry
    
    # Corner points (slightly inset)
    eps = 0.02
    corners = np.array([
        [eps, eps],           # near V0
        [eps, 1 - eps],       # near V1
        [1 - eps, eps],       # near V2
    ])
    
    # Edge midpoints (slightly inset)
    edge_mid = np.array([
        [0.5 * eps, 0.5 - eps],      # mid V0-V1
        [0.5 - eps, 0.5 * eps],      # mid V1-V2
        [0.5 * eps, 0.5 * eps],      # mid V2-V0
    ])
    
    # Interior points with approximate 3-fold symmetry
    # Center is at (1/3, 1/3) in barycentric
    center = 1.0 / 3.0
    r = 0.15
    # Three points at 120-degree intervals around center
    interior1 = np.array([
        [center + r * np.cos(0), center + r * np.sin(0) * 0.5],
        [center + r * np.cos(2*np.pi/3), center + r * np.sin(2*np.pi/3) * 0.5],
        [center + r * np.cos(4*np.pi/3), center + r * np.sin(4*np.pi/3) * 0.5],
    ])
    
    # Two more interior points
    r2 = 0.08
    interior2 = np.array([
        [center, center + r2],
        [center, center - r2],
    ])
    
    all_pts = np.vstack([corners, edge_mid, interior1, interior2])
    
    # Clamp to valid region
    all_pts = clamp_bary(all_pts)
    
    return all_pts

def generate_hexagonal_init():
    """Generate hexagonal lattice-like initial configuration."""
    best_bary = None
    best_val = -np.inf
    
    for s in np.linspace(0.1, 0.5, 20):
        points = []
        for i in range(5):
            for j in range(5):
                u = i * s
                v = j * s * np.sqrt(3) / 2
                if u >= 0 and v >= 0 and u + v <= 1 and len(points) < N:
                    points.append([u, v])
        
        if len(points) == N:
            bary = np.array(points)
            bary = clamp_bary(bary)
            val = -hard_min_obj(bary.flatten())
            if val > best_val:
                best_val = val
                best_bary = bary.copy()
    
    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
    
    return best_bary

def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_init(base, noise_scale=0.05, seed=None):
    """Generate perturbed version of a base configuration."""
    rng = np.random.default_rng(seed)
    bary = base.copy()
    bary += rng.normal(0, noise_scale, bary.shape)
    return clamp_bary(bary)

def _delaunay_smooth_obj(bary_flat, target=None, lam=1.0):
    """
    Smooth surrogate: minimize sum of squared deviations of Delaunay triangle areas
    from a target value (default: current min area). This creates a differentiable
    landscape that encourages uniform small-triangle areas.
    """
    from scipy.spatial import Delaunay
    
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    cart = bary_to_cart(bary)
    
    try:
        tri = Delaunay(cart)
        simplices = tri.simplices  # (K, 3)
        
        if len(simplices) == 0:
            return hard_min_obj(bary_flat)
        
        # Compute areas of Delaunay triangles
        areas = np.zeros(len(simplices))
        for i, (a, b, c) in enumerate(simplices):
            # Area = 0.5 * |cross(b-a, c-a)|
            ab = cart[b] - cart[a]
            ac = cart[c] - cart[a]
            areas[i] = 0.5 * abs(ab[0]*ac[1] - ab[1]*ac[0])
        
        # Target: use the minimum area as reference
        if target is None:
            target = np.min(areas)
        
        # Objective: sum of squared deviations from target + penalty for min area
        deviations = areas - target
        obj = lam * np.sum(deviations**2) - np.min(areas)
        
        return obj
    except Exception:
        return hard_min_obj(bary_flat)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    best_bary = None
    best_score = -np.inf
    
    def update_best(bary):
        """Update best if improved."""
        nonlocal best_bary, best_score
        bary = clamp_bary(bary)
        min_area = np.min(calc_areas_bary(bary))
        
        if min_area > best_score:
            best_score = min_area
            best_bary = bary.copy()
            return True
        return False

    def try_composite_optimize(bary_init, beta=200.0, lam=0.05, frac=0.3, maxiter=500):
        """
        Optimize using composite objective with variance regularization.
        """
        if time.time() > deadline - 15:
            return
        
        try:
            def obj(x):
                return composite_obj(x, beta=beta, lam=lam, frac=frac)
            
            bounds = [(0, 1)] * (N * 2)
            
            result = minimize(
                obj,
                bary_init.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            update_best(result.x.reshape(N, 2))
                
        except Exception:
            pass

    def try_two_stage(bary_init, beta1=100.0, beta2=400.0, lam1=0.1, lam2=0.02, maxiter1=300, maxiter2=400):
        """
        Two-stage: first with loose variance balance, then tight.
        """
        if time.time() > deadline - 20:
            return
        
        try:
            # Stage 1: Broader search with variance balancing
            def obj1(x):
                return composite_obj(x, beta=beta1, lam=lam1, frac=0.4)
            
            bounds = [(0, 1)] * (N * 2)
            
            result1 = minimize(
                obj1,
                bary_init.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter1, 'ftol': 1e-13, 'gtol': 1e-9}
            )
            
            if time.time() > deadline - 10:
                update_best(result1.x.reshape(N, 2))
                return

            # Stage 2: Sharper with less variance penalty
            def obj2(x):
                return composite_obj(x, beta=beta2, lam=lam2, frac=0.2)
            
            result2 = minimize(
                obj2,
                result1.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter2, 'ftol': 1e-15, 'gtol': 1e-11}
            )
            
            update_best(result2.x.reshape(N, 2))
                
        except Exception:
            pass

    def try_delaunay_optimize(bary_init, lam=1.0, maxiter=300):
        """
        Optimize using Delaunay-based smooth objective.
        """
        if time.time() > deadline - 15:
            return
        
        try:
            def obj(x):
                return _delaunay_smooth_obj(x, lam=lam)
            
            bounds = [(0, 1)] * (N * 2)
            
            result = minimize(
                obj,
                bary_init.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            update_best(result.x.reshape(N, 2))
                
        except Exception:
            pass

    def try_nm_polish(bary_init, maxiter=5000):
        """Nelder-Mead polish on hard min."""
        if time.time() > deadline - 8:
            return
        
        try:
            result = minimize(
                hard_min_obj,
                bary_init.flatten(),
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True}
            )
            update_best(result.x.reshape(N, 2))
        except Exception:
            pass

    def try_slsqp_polish(bary_init, maxiter=500):
        """SLSQP polish on hard min with explicit bounds."""
        if time.time() > deadline - 8:
            return
        
        try:
            bounds = [(0, 1)] * (N * 2)
            
            # SLSQP constraints: u+v <= 1 for each point
            constraints = []
            for i in range(N):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i: 1.0 - x[i*2] - x[i*2+1]
                })
            
            result = minimize(
                hard_min_obj,
                bary_init.flatten(),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': maxiter, 'ftol': 1e-14}
            )
            
            update_best(result.x.reshape(N, 2))
        except Exception:
            pass

    # --- Phase 1: Multi-start with composite objective ---
    
    # 1. Symmetric init (exploits known structure)
    sym_init = generate_symmetric_init()
    try_two_stage(sym_init, beta1=100, beta2=400, lam1=0.1, lam2=0.02)
    
    # 2. Hexagonal init
    hex_init = generate_hexagonal_init()
    try_two_stage(hex_init, beta1=80, beta2=300, lam1=0.08, lam2=0.03)
    
    # 3. Multiple random starts
    num_random = 8
    for i in range(num_random):
        if time.time() > deadline - 40:
            break
        rand_init = generate_random_init(seed=200 + i * 13)
        try_composite_optimize(rand_init, beta=200, lam=0.05, frac=0.3, maxiter=400)
        
    # 4. Delaunay-based optimization from best so far
    if best_bary is not None:
        try_delaunay_optimize(best_bary, lam=1.0, maxiter=300)
        try_delaunay_optimize(best_bary, lam=0.5, maxiter=300)
    
    # 5. Perturb best and re-optimize
    if best_bary is not None:
        for i in range(6):
            if time.time() > deadline - 25:
                break
            perturbed = generate_perturbed_init(best_bary, noise_scale=0.03, seed=700 + i * 7)
            try_composite_optimize(perturbed, beta=300, lam=0.03, frac=0.25, maxiter=300)
    
    # --- Phase 2: Nelder-Mead and SLSQP polish on hard min ---
    if best_bary is not None and time.time() < deadline - 15:
        try_nm_polish(best_bary, maxiter=4000)
        try_slsqp_polish(best_bary, maxiter=500)
        
        # Re-perturb and polish
        for i in range(4):
            if time.time() > deadline - 12:
                break
            perturbed = generate_perturbed_init(best_bary, noise_scale=0.015, seed=1100 + i * 11)
            try_nm_polish(perturbed, maxiter=3000)
            if time.time() < deadline - 10:
                try_slsqp_polish(perturbed, maxiter=300)
    
    # --- Phase 3: Final tight polish ---
    if best_bary is not None and time.time() < deadline - 5:
        try:
            result = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 6000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
            )
            update_best(result.x.reshape(N, 2))
        except Exception:
            pass
        
        # Final SLSQP polish
        if time.time() < deadline - 3:
            try_slsqp_polish(best_bary, maxiter=400)

    # Ensure we have a valid result
    if best_bary is None:
        best_bary = generate_random_init(seed=1)
        
    # Final clamp
    best_bary = clamp_bary(best_bary)
    
    # Convert to Cartesian
    cart_pts = bary_to_cart(best_bary)
    
    return cart_pts

# Test run if executed directly
if __name__ == "__main__":
    pts = construct()
    print("Points:")
    print(pts)
    # Calculate score
    bary = cart_to_bary(pts)
    areas = calc_areas_bary(bary)
    print("Min Area (normalized):", np.min(areas))
    print("Max Area (normalized):", np.max(areas))
