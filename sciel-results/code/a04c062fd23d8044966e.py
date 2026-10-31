import numpy as np
import time
from scipy.optimize import minimize, root, brent
from scipy.linalg import eigh

_DEADLINE = time.time() + 195

# Precompute I[n] = integral_{-inf}^{inf} x^{2n} exp(-x^2) dx
I = np.zeros(25)
I[0] = np.sqrt(np.pi)
for n in range(1, 25):
    I[n] = I[n-1] * (2.0*n - 1.0) / (2.0*n)

# Hermite polynomial coefficients in power basis x^{2k}
H0_c = np.array([1, 0, 0, 0, 0, 0, 0])
H4_c = np.array([3, -6, 1, 0, 0, 0, 0])
H8_c = np.array([105, -420, 210, -28, 1, 0, 0])
H12_c = np.array([945, -7560, 13860, -6930, 1155, -66, 1])

c12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

col0 = H0_c + c12[0]*H12_c
col1 = H4_c + c12[1]*H12_c
col2 = H8_c + c12[2]*H12_c

A_mat = np.column_stack((col0, col1, col2))  # Shape (7, 3)

# Precompute the quadratic form matrices
# M_x[0] = a^T Bx a, M_x[1] = a^T Cx a
# M_xi[0] = a^T Bxi a, M_xi[1] = a^T Cxi a
# where M[n] = sum_{i,j} p_i p_j I[i+j+n], p = A_mat @ a
# So M[n] = a^T (A_mat^T @ I_mat[n] @ A_mat) a
# where I_mat[n][i,j] = I[i+j+n]

I_mat0 = np.zeros((7, 7))
I_mat1 = np.zeros((7, 7))
for i in range(7):
    for j in range(7):
        I_mat0[i, j] = I[i + j]
        I_mat1[i, j] = I[i + j + 1]

# q = p * (-1)^k, so q = D @ p where D = diag([1,-1,1,-1,1,-1,1])
D = np.diag(np.array([1, -1, 1, -1, 1, -1, 1], dtype=float))

Bx = A_mat.T @ I_mat0 @ A_mat       # 3x3
Cx = A_mat.T @ I_mat1 @ A_mat       # 3x3
Bxi = A_mat.T @ (D @ I_mat0 @ D) @ A_mat  # 3x3
Cxi = A_mat.T @ (D @ I_mat1 @ D) @ A_mat  # 3x3

# Symmetrize to avoid numerical issues
Bx = (Bx + Bx.T) / 2
Cx = (Cx + Cx.T) / 2
Bxi = (Bxi + Bxi.T) / 2
Cxi = (Cxi + Cxi.T) / 2


def _eval_bound(a):
    """Fast evaluation of the uncertainty bound using precomputed matrices."""
    a0, a1, a2 = a[0], a[1], a[2]
    # Quadratic forms
    mx0 = a0*(Bx[0,0]*a0 + Bx[0,1]*a1 + Bx[0,2]*a2) + a1*(Bx[1,0]*a0 + Bx[1,1]*a1 + Bx[1,2]*a2) + a2*(Bx[2,0]*a0 + Bx[2,1]*a1 + Bx[2,2]*a2)
    mx1 = a0*(Cx[0,0]*a0 + Cx[0,1]*a1 + Cx[0,2]*a2) + a1*(Cx[1,0]*a0 + Cx[1,1]*a1 + Cx[1,2]*a2) + a2*(Cx[2,0]*a0 + Cx[2,1]*a1 + Cx[2,2]*a2)
    mxi0 = a0*(Bxi[0,0]*a0 + Bxi[0,1]*a1 + Bxi[0,2]*a2) + a1*(Bxi[1,0]*a0 + Bxi[1,1]*a1 + Bxi[1,2]*a2) + a2*(Bxi[2,0]*a0 + Bxi[2,1]*a1 + Bxi[2,2]*a2)
    mxi1 = a0*(Cxi[0,0]*a0 + Cxi[0,1]*a1 + Cxi[0,2]*a2) + a1*(Cxi[1,0]*a0 + Cxi[1,1]*a1 + Cxi[1,2]*a2) + a2*(Cxi[2,0]*a0 + Cxi[2,1]*a1 + Cxi[2,2]*a2)
    
    if mx0 <= 1e-15 or mxi0 <= 1e-15:
        return 1e18
    vx = mx1 / mx0
    vxi = mxi1 / mxi0
    if vx <= 1e-15 or vxi <= 1e-15:
        return 1e18
    bound = np.sqrt(vx * vxi)
    if not np.isfinite(bound):
        return 1e18
    return bound


def _compute_bound_and_grad(a):
    """
    Compute the uncertainty bound and its analytic gradient.
    a: array of shape (3,)
    Returns: (bound, grad)
    """
    a0, a1, a2 = a[0], a[1], a[2]
    
    # Quadratic forms and their gradients
    # f(a) = a^T M a => grad = 2*M*a
    mx0 = a @ Bx @ a
    mx1 = a @ Cx @ a
    mxi0 = a @ Bxi @ a
    mxi1 = a @ Cxi @ a
    
    if mx0 <= 1e-15 or mxi0 <= 1e-15:
        return 1e18, np.zeros(3)
    
    vx = mx1 / mx0
    vxi = mxi1 / mxi0
    
    if vx <= 1e-15 or vxi <= 1e-15:
        return 1e18, np.zeros(3)
    
    bound = np.sqrt(vx * vxi)
    
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)
    
    # Gradients of quadratic forms: 2*M*a
    gmx0 = 2.0 * (Bx @ a)
    gmx1 = 2.0 * (Cx @ a)
    gmxi0 = 2.0 * (Bxi @ a)
    gmxi1 = 2.0 * (Cxi @ a)
    
    # dvx = (gmx1 * mx0 - mx1 * gmx0) / mx0^2
    dvx = (gmx1 * mx0 - mx1 * gmx0) / (mx0 * mx0)
    dvxi = (gmxi1 * mxi0 - mxi1 * gmxi0) / (mxi0 * mxi0)
    
    # d(bound) = (1/(2*bound)) * (vxi*dvx + vx*dvxi)
    factor = 1.0 / (2.0 * bound)
    grad = factor * (vxi * dvx + vx * dvxi)
    
    return bound, grad



def construct():
    """
    Find three coefficients (a0, a4, a8) that minimize the uncertainty product.
    Exploits homogeneity: the objective is degree-0 homogeneous in a, so we can
    normalize and search over 2D (or 1D) parameter space.
    Returns array of shape (3,).
    """
    
    # Seed value
    seed_params = np.array([-1.13, 0.04, 0.000308642])
    seed_val = _eval_bound(seed_params)
    
    best_val = seed_val
    best_params = seed_params.copy()
    
    deadline_margin = 10  # seconds safety margin
    
    # The objective is homogeneous of degree 0, so scale doesn't matter.
    # We can fix one component and search over ratios.
    # Strategy: parameterize as a = [1, u, v] and search over (u, v) in 2D,
    # plus handle the case a0=0 by searching [0, 1, w] and [0, 0, 1] boundaries.
    # But more robustly: do a 2D grid + local optimization on the sphere-like parameterization.
    
    # Approach 1: 2D search with a0 = 1
    # Search over (a4, a8) with a0 fixed at 1
    # Since the problem is homogeneous, the optimal direction is what matters.
    
    # High-resolution 2D grid
    n_grid = 80
    u_range = np.linspace(-0.5, 0.5, n_grid)
    v_range = np.linspace(-0.02, 0.02, n_grid)
    
    grid_best_val = 1e18
    grid_best_u = 0.0
    grid_best_v = 0.0
    
    for u in u_range:
        for v in v_range:
            a_test = np.array([1.0, u, v])
            val = _eval_bound(a_test)
            if val < grid_best_val:
                grid_best_val = val
                grid_best_u = u
                grid_best_v = v
    
    if grid_best_val < best_val:
        best_val = grid_best_val
        best_params = np.array([1.0, grid_best_u, grid_best_v])
    
    if time.time() > _DEADLINE - deadline_margin:
        return best_params
    
    # Also search with a0 = -1 (different quadrant)
    grid_best_val2 = 1e18
    grid_best_u2 = 0.0
    grid_best_v2 = 0.0
    for u in u_range:
        for v in v_range:
            a_test = np.array([-1.0, u, v])
            val = _eval_bound(a_test)
            if val < grid_best_val2:
                grid_best_val2 = val
                grid_best_u2 = u
                grid_best_v2 = v
    
    if grid_best_val2 < best_val:
        best_val = grid_best_val2
        best_params = np.array([-1.0, grid_best_u2, grid_best_v2])
    
    if time.time() > _DEADLINE - deadline_margin:
        return best_params
    
    # Search with a0 = 0 (boundary)
    # a = [0, 1, w]
    w_range = np.linspace(-0.1, 0.1, 50)
    for w in w_range:
        a_test = np.array([0.0, 1.0, w])
        val = _eval_bound(a_test)
        if val < best_val:
            best_val = val
            best_params = a_test.copy()
    
    if time.time() > _DEADLINE - deadline_margin:
        return best_params
    
    # Local optimization from the best grid point using L-BFGS-B on the 3D problem
    # (the optimizer will handle the homogeneity naturally)
    if time.time() < _DEADLINE - deadline_margin:
        try:
            res_min = minimize(_compute_bound_and_grad, best_params, method='L-BFGS-B', jac=True, options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            if res_min.success and res_min.fun < best_val:
                best_val = res_min.fun
                best_params = res_min.x.copy()
        except Exception:
            pass
    
    if time.time() > _DEADLINE - deadline_margin:
        return best_params
    
    # Multi-start local optimization from multiple points
    rng = np.random.default_rng(123)
    n_starts = 100
    
    for i in range(n_starts):
        if time.time() > _DEADLINE - deadline_margin:
            break
        
        # Generate diverse starting points
        r = rng.random(3)
        # Mix of strategies: near seed, random directions, near grid best
        if i < 30:
            # Near the current best
            a_start = best_params + rng.normal(0, 0.1, 3)
        elif i < 60:
            # Random direction, normalized
            a_start = rng.normal(0, 1, 3)
            norm = np.linalg.norm(a_start)
            if norm > 1e-10:
                a_start = a_start / norm
                # Scale to be in a reasonable range
                a_start = a_start * rng.uniform(0.5, 2.0)
            else:
                a_start = seed_params + rng.normal(0, 0.2, 3)
        else:
            # Near seed
            a_start = seed_params + rng.normal(0, 0.3, 3)
        
        try:
            res_min = minimize(_compute_bound_and_grad, a_start, method='L-BFGS-B', jac=True, options={'maxiter': 150, 'ftol': 1e-13, 'gtol': 1e-11})
            if res_min.success and res_min.fun < best_val:
                best_val = res_min.fun
                best_params = res_min.x.copy()
        except Exception:
            pass
    
    if time.time() > _DEADLINE - deadline_margin:
        return best_params
    
    # Final high-precision polish
    if time.time() < _DEADLINE - deadline_margin - 5:
        try:
            res_final = minimize(_compute_bound_and_grad, best_params, method='L-BFGS-B', jac=True, options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14})
            if res_final.success and res_final.fun < best_val:
                best_val = res_final.fun
                best_params = res_final.x.copy()
        except Exception:
            pass
    
    # Trust-constr polish for even higher precision
    if time.time() < _DEADLINE - deadline_margin - 3:
        try:
            res_tc = minimize(
                lambda x: _compute_bound_and_grad(x)[0],
                best_params,
                jac=lambda x: _compute_bound_and_grad(x)[1],
                method='trust-constr',
                bounds=[(-10, 10), (-5, 5), (-1, 1)],
                options={'maxiter': 300, 'gtol': 1e-15, 'xtol': 1e-15, 'ctol': 1e-15}
            )
            if res_tc.success and res_tc.fun < best_val:
                best_val = res_tc.fun
                best_params = res_tc.x.copy()
        except Exception:
            pass
    
    if not np.all(np.isfinite(best_params)):
        return seed_params

    return np.array(best_params)
