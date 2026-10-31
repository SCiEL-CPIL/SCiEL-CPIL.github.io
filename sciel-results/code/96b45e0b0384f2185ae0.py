import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations
import time

N = 11
TIME_LIMIT = 148.0  # Safety margin for 150s limit

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)
N_TRI = len(tri_indices)

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

def clamp_bary(bary, eps=1e-10):
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

def hard_min_value(bary):
    """Calculate the minimum triangle area."""
    areas = calc_areas_bary(bary)
    return float(np.min(areas))

def epigraph_obj(x):
    """
    Objective function for epigraph formulation.
    x = [u0, v0, ..., u10, v10, t]
    Minimize: sum(max(0, t - area_k)^2) - t
    """
    bary = x[:2*N].reshape(N, 2)
    t = x[2*N]
    
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Violation: t > area
    viol = np.maximum(0, t - areas)
    penalty = np.sum(viol**2)
    
    # If t is negative, we don't want it, but the -t term handles maximizing t.
    # However, if t is very negative, penalty is 0, and -t is large positive? No, we minimize.
    # If t < 0, -t > 0. We want to maximize t, so minimize -t.
    # If t < 0, it's a bad solution.
    # Actually, if t < min(areas), viol=0, obj = -t. Minimizing -t means maximizing t.
    # So if t < min(areas), it will push t up until t = min(areas).
    
    return penalty - t

def epigraph_grad(x):
    """
    Gradient of the epigraph objective.
    """
    bary = x[:2*N].reshape(N, 2).copy()
    t = x[2*N]
    
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    viol = np.maximum(0, t - areas)
    
    # Gradient w.r.t t
    grad_t = -1.0 + 2.0 * np.sum(viol)
    
    # Gradient w.r.t barycentric coordinates
    grad_bary = np.zeros((N, 2))
    
    if np.any(viol > 0):
        u = bary[:, 0]
        v = bary[:, 1]
        w = 1.0 - u - v
        
        # Iterate only over violating triangles to save time
        violating_indices = np.where(viol > 0)[0]
        
        for k in violating_indices:
            coeff = 2.0 * viol[k]
            i, j, kk = tri_idx[k]
            
            # Area A_k = |det|. Assuming interior points, det > 0.
            # If det < 0, the gradient sign flips. 
            # For robustness, we can compute the sign.
            # However, in a valid configuration, most triangles should have positive orientation 
            # if points are in convex position or generally well-distributed.
            # Let's check the sign of the determinant for this specific triangle.
            ui, vi, wi = u[i], v[i], w[i]
            uj, vj, wj = u[j], v[j], w[j]
            uk, vk, wk = u[kk], v[kk], w[kk]
            
            det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
            sign = 1.0 if det >= 0 else -1.0
            
            # Gradient of area w.r.t u_i, v_i
            # A = u_i(v_j w_k - v_k w_j) - v_i(u_j w_k - u_k w_j) + (1-u_i-v_i)(u_j v_k - u_k v_j)
            # dA/du_i = (v_j w_k - v_k w_j) - (u_j v_k - u_k v_j)
            # dA/dv_i = -(u_j w_k - u_k w_j) - (u_j v_k - u_k v_j)
            
            A_i = v[j]*w[kk] - v[kk]*w[j]
            B_i = u[j]*w[kk] - u[kk]*w[j]
            C_i = u[j]*v[kk] - u[kk]*v[j]
            
            grad_bary[i, 0] += sign * coeff * (A_i - C_i)
            grad_bary[i, 1] += sign * coeff * (-B_i - C_i)
            
            # For point j
            A_j = v[i]*w[kk] - v[kk]*w[i]
            B_j = u[i]*w[kk] - u[kk]*w[i]
            C_j = u[i]*v[kk] - u[kk]*v[i]
            
            grad_bary[j, 0] += sign * coeff * (A_j - C_j)
            grad_bary[j, 1] += sign * coeff * (-B_j - C_j)
            
            # For point k
            A_k = v[i]*w[j] - v[j]*w[i]
            B_k = u[i]*w[j] - u[j]*w[i]
            C_k = u[i]*v[j] - u[j]*v[i]
            
            grad_bary[kk, 0] += sign * coeff * (A_k - C_k)
            grad_bary[kk, 1] += sign * coeff * (-B_k - C_k)
            
    grad = np.zeros(2*N + 1)
    grad[:2*N] = grad_bary.flatten()
    grad[2*N] = grad_t
    return grad

def generate_lattice_init():
    """
    Generate a triangular lattice initial configuration.
    K=3 grid has 10 points. We add an 11th point near the center.
    """
    pts = []
    for i in range(4):
        for j in range(4-i):
            pts.append([i/3.0, j/3.0])
    
    # The 10 points are:
    # (0,0), (1,0), (2,0), (3,0)
    # (0,1), (1,1), (2,1)
    # (0,2), (1,2)
    # (0,3)
    # Index 5 is (1,1) which is the center (1/3, 1/3).
    # We replace the center with a slightly offset point or just add one?
    # The problem requires exactly 11 points.
    # Let's take the 10 lattice points and add one more.
    # Where to add? Maybe near the center but offset to break symmetry and allow area growth.
    # Or perturb the center.
    # Let's try perturbing the center (1,1) and keeping it as 1 point, so we have 10.
    # We need 11. So we add a new point.
    # A good spot might be (0.5, 0.5) which is midpoint of edge, but that's already (1,1) in bary? No.
    # (0.5, 0.5) in bary is midpoint of V1-V2.
    # Lattice points:
    # (0,0) -> V0
    # (1,0) -> 1/3 V0 + 2/3 V1? No.
    # Barycentric (u,v) corresponds to u*V0 + v*V1 + (1-u-v)*V2.
    # (0,0) -> V2.
    # (1,0) -> V1.
    # (0,1) -> V1? No.
    # Let's stick to the standard mapping:
    # x = u*0 + v*1 + (1-u-v)*0.5
    # y = u*0 + v*0 + (1-u-v)*H
    # (0,0) -> (0.5, H) = V2
    # (1,0) -> (1, 0) = V1
    # (0,1) -> (0, H) ... wait.
    # If u=0, v=1, w=0. Point is V1?
    # x = 0 + 1 + 0 = 1. y = 0. Yes, V1.
    # If u=1, v=0, w=0. Point is V0?
    # x = 0 + 0 + 0.5 = 0.5. y = H. No, that's V2.
    # My bary_to_cart: x = v + 0.5*w, y = w*H.
    # u=1, v=0, w=0 -> x=0, y=0. V0. Correct.
    # u=0, v=1, w=0 -> x=1, y=0. V1. Correct.
    # u=0, v=0, w=1 -> x=0.5, y=H. V2. Correct.
    
    # So lattice points (i/3, j/3):
    # i=0, j=0 -> (0,0) -> V0
    # i=1, j=0 -> (1/3, 0)
    # i=2, j=0 -> (2/3, 0)
    # i=3, j=0 -> (1, 0) -> V1
    # i=0, j=1 -> (0, 1/3)
    # i=1, j=1 -> (1/3, 1/3) -> Center
    # i=2, j=1 -> (2/3, 1/3)
    # i=0, j=2 -> (0, 2/3)
    # i=1, j=2 -> (1/3, 2/3)
    # i=0, j=3 -> (0, 1) -> V2
    
    # These 10 points are a valid packing.
    # We need an 11th.
    # Let's add a point near the center, e.g., (0.35, 0.35)
    pts.append([0.35, 0.35])
    
    return np.array(pts)

def generate_symmetric_init():
    """
    Generate a symmetric initial configuration.
    3 vertices, 3 edge midpoints, 3 inner ring, 1 center, 1 extra?
    """
    pts = []
    # Vertices
    pts.append([0.0, 0.0])
    pts.append([1.0, 0.0])
    pts.append([0.0, 1.0])
    
    # Edge midpoints
    pts.append([0.5, 0.0])
    pts.append([0.5, 0.5])
    pts.append([0.0, 0.5])
    
    # Center
    pts.append([1/3, 1/3])
    
    # Inner ring (radius ~0.2 from center in barycentric)
    # Center is (1/3, 1/3).
    # Directions in barycentric are tricky. Let's use Cartesian.
    # Convert center to Cartesian
    cx, cy = bary_to_cart(np.array([[1/3, 1/3]]))[0]
    
    for k in range(3):
        angle = 2 * np.pi * k / 3
        r = 0.15
        dx = r * np.cos(angle)
        dy = r * np.sin(angle)
        # Convert back to barycentric? 
        # x = v + 0.5w, y = w*H
        # w = y/H
        # v = x - 0.5w
        # u = 1 - v - w
        new_x = cx + dx
        new_y = cy + dy
        
        w = new_y / H
        v = new_x - 0.5 * w
        u = 1 - v - w
        
        if u >= 0 and v >= 0 and w >= 0:
            pts.append([u, v])
            
    # We have 10 points. Add 1 more.
    # Maybe offset the center?
    pts.append([1/3 + 0.05, 1/3])
    
    return np.array(pts)

def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def refine_epigraph(bary_init, t_init=None, time_limit=10.0):
    """
    Refine using epigraph formulation with L-BFGS-B.
    """
    start = time.time()
    bary = clamp_bary(bary_init)
    
    if t_init is None:
        t_init = hard_min_value(bary) * 0.95
    else:
        t_init = max(t_init, 1e-6)
        
    x0 = np.concatenate([bary.flatten(), [t_init]])
    
    bounds = [(0, 1)] * (2*N) + [(0, 0.05)] # t is small
    
    best_bary = bary.copy()
    best_val = hard_min_value(bary)
    
    # Stage 1: L-BFGS-B with gradient
    try:
        res = minimize(
            epigraph_obj,
            x0,
            jac=epigraph_grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        if res.x is not None:
            b = clamp_bary(res.x[:2*N].reshape(N, 2))
            val = hard_min_value(b)
            if val > best_val:
                best_val = val
                best_bary = b
    except Exception:
        pass
    
    # Stage 2: SLSQP with gradient (often better for constrained problems)
    if time.time() < start + time_limit * 0.7:
        try:
            x0_slsqp = np.concatenate([best_bary.flatten(), [best_val * 0.95]])
            res = minimize(
                epigraph_obj,
                x0_slsqp,
                jac=epigraph_grad,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14}
            )
            if res.x is not None:
                b = clamp_bary(res.x[:2*N].reshape(N, 2))
                val = hard_min_value(b)
                if val > best_val:
                    best_val = val
                    best_bary = b
        except Exception:
            pass
            
    # Stage 3: Nelder-Mead on hard min for final polish
    if time.time() < start + time_limit * 0.95:
        try:
            res = minimize(
                lambda x: -hard_min_value(clamp_bary(x.reshape(N, 2))),
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            b = clamp_bary(res.x.reshape(N, 2))
            val = hard_min_value(b)
            if val > best_val:
                best_val = val
                best_bary = b
        except Exception:
            pass
            
    return best_bary

def coordinate_descent(bary, max_iter=500, step_size=0.005):
    """
    Coordinate descent refinement for the hard min objective.
    """
    bary = bary.copy()
    current_val = hard_min_value(bary)
    
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            for d in range(2):
                original = bary[i, d]
                for step in [step_size, -step_size, step_size/2, -step_size/2, step_size/10, -step_size/10]:
                    new_val = bary[i, d] + step
                    if new_val < 0 or new_val > 1:
                        continue
                    bary[i, d] = new_val
                    new_bary = clamp_bary(bary)
                    val = hard_min_value(new_bary)
                    if val > current_val:
                        current_val = val
                        improved = True
                        break
                    else:
                        bary[i, d] = original
                # Reset if not improved in this direction
                bary[i, d] = original if not improved else bary[i, d]
        
        if not improved:
            break
    
    return bary

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = hard_min_value(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Try multiple structured initializations
    inits = [
        generate_lattice_init(),
        generate_symmetric_init(),
    ]
    
    for init in inits:
        if time.time() > deadline - 30:
            break
        refined = refine_epigraph(init, time_limit=8.0)
        update_best(refined)
        
    # 2. Global Search with Differential Evolution
    if time.time() < deadline - 40:
        # Use hard min for DE fitness, as it's the true objective
        def de_obj(x):
            bary = clamp_bary(x.reshape(N, 2))
            return -hard_min_value(bary)
            
        bounds = [(0, 1)] * (N * 2)
        
        # Run DE with a few seeds
        for de_seed in [42, 123]:
            if time.time() > deadline - 25:
                break
            try:
                result_de = differential_evolution(
                    de_obj,
                    bounds,
                    seed=de_seed,
                    maxiter=60,
                    popsize=15,
                    tol=1e-8,
                    polish=False
                )
                
                de_bary = clamp_bary(result_de.x.reshape(N, 2))
                # Refine the DE result
                refined_de = refine_epigraph(de_bary, time_limit=5.0)
                update_best(refined_de)
            except Exception:
                pass

    # 3. Multi-start: random + perturbed restarts
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 15:
        restart_count += 1
        if best_bary is not None and restart_count % 2 == 0:
            # Perturbation of best solution
            sigma = 0.03
            noise = rng.normal(0, sigma, best_bary.shape)
            init = clamp_bary(best_bary + noise)
        else:
            # Fresh random start
            init = generate_random_init(seed=3000 + restart_count)
        
        refined = refine_epigraph(init, time_limit=3.0)
        update_best(refined)

    # 4. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 5:
        try:
            # One more SLSQP pass on the best
            x0 = np.concatenate([best_bary.flatten(), [best_val * 0.95]])
            bounds = [(0, 1)] * (2*N) + [(0, 0.05)]
            res = minimize(
                epigraph_obj,
                x0,
                jac=epigraph_grad,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-16}
            )
            update_best(clamp_bary(res.x[:2*N].reshape(N, 2)))
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 2:
            try:
                bary_cd = coordinate_descent(best_bary, max_iter=500, step_size=0.001)
                update_best(bary_cd)
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

if __name__ == "__main__":
    # Quick test
    pts = construct()
    print("Points:", pts)
    bary = np.zeros((11, 2))
    # Inverse of bary_to_cart for debugging if needed
    # But construct returns cart.
    print("Score:", hard_min_value(clamp_bary(bary))) # Placeholder, actual score is in evaluator
