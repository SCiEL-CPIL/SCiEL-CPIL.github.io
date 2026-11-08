import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh
import time

# Precompute Hermite polynomial coefficients (physicists' convention)
# H0 = 1
# H4 = 16x^4 - 48x^2 + 12
# H8 = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 3360
# H12 = 4096x^12 - 98304x^10 + 846720x^8 - 3932160x^6 + 8064000x^4 - 6048000x^2 + 1039500

H0_coeffs = np.array([1.0])
H4_coeffs = np.array([12.0, -48.0, 16.0])
H8_coeffs = np.array([3360.0, -13440.0, 11520.0, -3072.0, 256.0])
H12_coeffs = np.array([1039500.0, -6048000.0, 8064000.0, -3932160.0, 846720.0, -98304.0, 4096.0])

# Precompute integral tables
def _precompute_integrals():
    """Precompute integral tables for moments."""
    I1 = np.zeros(25)
    I2 = np.zeros(25)
    for n in range(25):
        if n == 0:
            I1[n] = np.sqrt(np.pi)
            I2[n] = np.sqrt(np.pi / 2.0)
        else:
            double_fact = 1.0
            for k in range(n):
                double_fact *= (2*k + 1)
            I1[n] = double_fact * np.sqrt(np.pi) / (2.0 ** n)
            I2[n] = double_fact * np.sqrt(np.pi) / (2.0 ** (2*n + 0.5))
    return I1, I2

I1_table, I2_table = _precompute_integrals()

# Precompute index tensors for vectorized operations
_n = 7
_idx2 = np.add.outer(np.arange(_n), np.arange(_n))  # 7x7
_idx4 = np.zeros((_n, _n, _n, _n), dtype=int)
for i in range(_n):
    for j in range(_n):
        _idx4[i, j, :, :] = i + j + np.add.outer(np.arange(_n), np.arange(_n))

def _compute_P_coeffs(c0, c1, c2, c3):
    """Compute the monomial coefficients of P(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12."""
    P = np.zeros(7)
    P[0] += c0 * H0_coeffs[0]
    P[0] += c1 * H4_coeffs[0]
    P[1] += c1 * H4_coeffs[1]
    P[2] += c1 * H4_coeffs[2]
    for i in range(5):
        P[i] += c2 * H8_coeffs[i]
    for i in range(7):
        P[i] += c3 * H12_coeffs[i]
    return P

def _poly_square_moment_vec(P):
    """Compute ∫ P(x)² * exp(-x²) dx using precomputed table."""
    return np.sum(P[:, None] * P[None, :] * I1_table[_idx2])

def _poly_fourth_moment_vec(P):
    """Compute ∫ P(x)⁴ * exp(-2x²) dx using precomputed table."""
    P4 = P[:, None, None, None] * P[None, :, None, None] * P[None, None, :, None] * P[None, None, None, :]
    return np.sum(P4 * I2_table[_idx4])

def _objective_and_grad(c0, c1, c2):
    """Compute the objective (upper bound on C4) and its gradient."""
    c3 = -(c0 + 12*c1 + 3360*c2) / 1039500.0
    
    P = _compute_P_coeffs(c0, c1, c2, c3)
    
    L2_sq = _poly_square_moment_vec(P)
    L4_4th = _poly_fourth_moment_vec(P)
    
    if L2_sq <= 0 or L4_4th <= 0:
        return np.inf, np.zeros(3)
    
    C4 = np.sqrt(L4_4th) / L2_sq
    
    # Derivatives of P with respect to c0, c1, c2
    dP_dc0 = H0_coeffs.copy()
    dP_dc0 = np.zeros(7)
    dP_dc0[0] += 1.0
    dP_dc0 += (-1/1039500.0) * H12_coeffs
    
    dP_dc1 = np.zeros(7)
    dP_dc1[0] += 12.0
    dP_dc1[1] += -48.0
    dP_dc1[2] += 16.0
    dP_dc1 += (-12/1039500.0) * H12_coeffs
    
    dP_dc2 = np.zeros(7)
    dP_dc2[0] += 3360.0
    dP_dc2[1] += -13440.0
    dP_dc2[2] += 11520.0
    dP_dc2[3] += -3072.0
    dP_dc2[4] += 256.0
    dP_dc2 += (-3360/1039500.0) * H12_coeffs
    
    # dL2_sq/dc = 2 * sum_{i,j} dP[i] * P[j] * I1[i+j]
    def _dL2_sq(dP):
        return 2.0 * np.sum(dP[:, None] * P[None, :] * I1_table[_idx2])
    
    # dL4_4th/dc = 4 * sum_{i,j,k,l} dP[i] * P[j]*P[k]*P[l] * I2[i+j+k+l]
    P3 = P[:, None, None] * P[None, :, None] * P[None, None, :]
    
    def _dL4_4th(dP):
        term = dP[:, None, None, None] * P3[None, :, :, :] * I2_table[_idx4]
        return 4.0 * np.sum(term)
    
    dL2_sq_dc0 = _dL2_sq(dP_dc0)
    dL2_sq_dc1 = _dL2_sq(dP_dc1)
    dL2_sq_dc2 = _dL2_sq(dP_dc2)
    
    dL4_4th_dc0 = _dL4_4th(dP_dc0)
    dL4_4th_dc1 = _dL4_4th(dP_dc1)
    dL4_4th_dc2 = _dL4_4th(dP_dc2)
    
    sqrt_L4 = np.sqrt(L4_4th)
    dC4_dc0 = (0.5 * dL4_4th_dc0 / sqrt_L4 - dL2_sq_dc0 / L2_sq) * C4
    dC4_dc1 = (0.5 * dL4_4th_dc1 / sqrt_L4 - dL2_sq_dc1 / L2_sq) * C4
    dC4_dc2 = (0.5 * dL4_4th_dc2 / sqrt_L4 - dL2_sq_dc2 / L2_sq) * C4
    
    return C4, np.array([dC4_dc0, dC4_dc1, dC4_dc2])

def _objective_2d(theta, phi):
    """
    2D objective on unit circle: fix L2 norm = 1, minimize ∫P⁴e^{-2x²}.
    Parameterize the 3D coefficient space (c0, c1, c2) with c3 determined by g(0)=0.
    Use spherical coordinates: c = r * [sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]
    with r chosen so that L2(P) = 1.
    """
    s = np.sin(theta)
    c = np.cos(theta)
    sp = np.sin(phi)
    cp = np.cos(phi)
    
    # Direction vector in (c0, c1, c2) space
    v = np.array([s * cp, s * sp, c])
    
    # Compute P for unit direction
    c3_dir = -(v[0] + 12*v[1] + 3360*v[2]) / 1039500.0
    P_dir = _compute_P_coeffs(v[0], v[1], v[2], c3_dir)
    
    L2_sq_dir = _poly_square_moment_vec(P_dir)
    
    if L2_sq_dir <= 0:
        return np.inf
    
    # Scale so that L2(P) = 1: P = P_dir / sqrt(L2_sq_dir)
    scale = 1.0 / np.sqrt(L2_sq_dir)
    P = P_dir * scale
    
    L4_4th = _poly_fourth_moment_vec(P)
    
    if L4_4th <= 0:
        return np.inf
    
    return L4_4th

def _objective_2d_grad(theta, phi):
    """
    Compute gradient of the 2D objective w.r.t. (theta, phi) using finite differences
    for the angular derivatives, combined with the chain rule.
    """
    h = 1e-7
    
    f0 = _objective_2d(theta, phi)
    if f0 >= np.inf:
        return np.inf, np.array([0.0, 0.0])
    
    # Central differences
    f_tp = _objective_2d(theta + h, phi)
    f_tm = _objective_2d(theta - h, phi)
    f_pp = _objective_2d(theta, phi + h)
    f_pm = _objective_2d(theta, phi - h)
    
    if any(f >= np.inf for f in [f_tp, f_tm, f_pp, f_pm]):
        return f0, np.array([0.0, 0.0])
    
    df_dtheta = (f_tp - f_tm) / (2 * h)
    df_dphi = (f_pp - f_pm) / (2 * h)
    
    return f0, np.array([df_dtheta, df_dphi])

def _newton_refine_2d(theta0, phi0, max_iter=50, tol=1e-14):
    """
    Newton's method refinement in 2D angular space.
    Uses numerical Hessian via finite differences.
    """
    theta, phi = theta0, phi0
    
    for _ in range(max_iter):
        f, grad = _objective_2d_grad(theta, phi)
        if f >= np.inf:
            break
        
        if np.linalg.norm(grad) < tol:
            break
        
        # Numerical Hessian
        h = 1e-5
        f0, g0 = _objective_2d_grad(theta, phi)
        
        # H[0,0]
        f_tp, g_tp = _objective_2d_grad(theta + h, phi)
        f_tm, g_tm = _objective_2d_grad(theta - h, phi)
        H00 = (g_tp[0] - g_tm[0]) / (2 * h) if f_tp < np.inf and f_tm < np.inf else 1.0
        
        # H[0,1]
        f_pp, g_pp = _objective_2d_grad(theta, phi + h)
        f_pm, g_pm = _objective_2d_grad(theta, phi - h)
        H01 = (g_pp[0] - g_pm[0]) / (2 * h) if f_pp < np.inf and f_pm < np.inf else 0.0
        
        # H[1,0]
        H10 = (g_tp[1] - g_tm[1]) / (2 * h) if f_tp < np.inf and f_tm < np.inf else 0.0
        
        # H[1,1]
        H11 = (g_pp[1] - g_pm[1]) / (2 * h) if f_pp < np.inf and f_pm < np.inf else 1.0
        
        H = np.array([[H00, H01], [H10, H11]])
        
        # Regularize if needed
        eigvals = np.linalg.eigvalsh(H)
        if np.min(eigvals) < 1e-10:
            H += 1e-10 * np.eye(2)
        
        try:
            step = np.linalg.solve(H, grad)
        except:
            break
        
        # Damping
        step_norm = np.linalg.norm(step)
        if step_norm > 0.1:
            step = step / step_norm * 0.1
        
        theta_new = theta - step[0]
        phi_new = phi - step[1]
        
        # Keep angles in valid range
        theta_new = np.clip(theta_new, 1e-6, np.pi - 1e-6)
        
        f_new, _ = _objective_2d_grad(theta_new, phi_new)
        
        if f_new < f:
            theta, phi = theta_new, phi_new
        else:
            # Try smaller step
            theta_new = theta - 0.5 * step[0]
            phi_new = phi - 0.5 * step[1]
            theta_new = np.clip(theta_new, 1e-6, np.pi - 1e-6)
            f_new, _ = _objective_2d_grad(theta_new, phi_new)
            if f_new < f:
                theta, phi = theta_new, phi_new
            else:
                break
    
    return theta, phi

def _recover_3d_from_2d(theta, phi):
    """Recover the 3D coefficients (c0, c1, c2) from 2D angles."""
    s = np.sin(theta)
    c = np.cos(theta)
    sp = np.sin(phi)
    cp = np.cos(phi)
    
    v = np.array([s * cp, s * sp, c])
    
    c3_dir = -(v[0] + 12*v[1] + 3360*v[2]) / 1039500.0
    P_dir = _compute_P_coeffs(v[0], v[1], v[2], c3_dir)
    
    L2_sq_dir = _poly_square_moment_vec(P_dir)
    
    if L2_sq_dir <= 0:
        return np.array([0.0, 0.0, 0.0])
    
    scale = 1.0 / np.sqrt(L2_sq_dir)
    
    return np.array([v[0] * scale, v[1] * scale, v[2] * scale])

def construct():
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin before 210s limit
    
    best_C4 = np.inf
    best_c = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    # Phase 1: Dense global grid search in 2D angular space
    n_theta = 200
    n_phi = 400
    
    theta_vals = np.linspace(1e-4, np.pi - 1e-4, n_theta)
    phi_vals = np.linspace(-np.pi, np.pi, n_phi)
    
    grid_best_f = np.inf
    grid_best_theta = 0
    grid_best_phi = 0
    
    # Vectorized grid evaluation
    for i_t, theta in enumerate(theta_vals):
        if time.time() > deadline - 60.0:
            break
        
        s = np.sin(theta)
        c = np.cos(theta)
        
        # Vectorize over phi
        phi_arr = phi_vals
        sp_arr = np.sin(phi_arr)
        cp_arr = np.cos(phi_arr)
        
        # v = [s*cp, s*sp, c] for each phi
        v0 = s * cp_arr  # shape (n_phi,)
        v1 = s * sp_arr
        v2 = np.full_like(phi_arr, c)
        
        c3_arr = -(v0 + 12*v1 + 3360*v2) / 1039500.0
        
        # Compute P for each phi (vectorized)
        P_arr = np.zeros((n_phi, 7))
        P_arr[:, 0] += v0 * 1.0
        P_arr[:, 0] += v1 * 12.0
        P_arr[:, 1] += v1 * (-48.0)
        P_arr[:, 2] += v1 * 16.0
        P_arr[:, 0] += v2 * 3360.0
        P_arr[:, 1] += v2 * (-13440.0)
        P_arr[:, 2] += v2 * 11520.0
        P_arr[:, 3] += v2 * (-3072.0)
        P_arr[:, 4] += v2 * 256.0
        P_arr[:, 0] += c3_arr * 1039500.0
        P_arr[:, 1] += c3_arr * (-6048000.0)
        P_arr[:, 2] += c3_arr * 8064000.0
        P_arr[:, 3] += c3_arr * (-3932160.0)
        P_arr[:, 4] += c3_arr * 846720.0
        P_arr[:, 5] += c3_arr * (-98304.0)
        P_arr[:, 6] += c3_arr * 4096.0
        
        # L2_sq for each: sum_{i,j} P[i]*P[j]*I1[i+j]
        L2_sq_arr = np.sum(P_arr[:, :, None] * P_arr[:, None, :] * I1_table[_idx2], axis=(1, 2))
        
        valid = L2_sq_arr > 0
        if not np.any(valid):
            continue
        
        scale_arr = np.zeros(n_phi)
        scale_arr[valid] = 1.0 / np.sqrt(L2_sq_arr[valid])
        
        P_scaled = P_arr * scale_arr[:, None]
        
        # L4_4th for each: sum_{i,j,k,l} P[i]*P[j]*P[k]*P[l]*I2[i+j+k+l]
        # This is expensive to vectorize fully; do it in chunks
        L4_arr = np.zeros(n_phi)
        chunk_size = 50
        for start_idx in range(0, n_phi, chunk_size):
            end_idx = min(start_idx + chunk_size, n_phi)
            Pc = P_scaled[start_idx:end_idx]
            # Pc shape: (chunk, 7)
            # Need sum over i,j,k,l
            # Expand: Pc[:, i, None, None, None] * Pc[:, None, j, None, None] * ...
            terms = Pc[:, :, None, None, None] * Pc[:, None, :, None, None] * Pc[:, None, None, :, None] * Pc[:, None, None, None, :]
            L4_arr[start_idx:end_idx] = np.sum(terms * I2_table[_idx4][None, :, :, :, :], axis=(1, 2, 3, 4))
        
        valid2 = valid & (L4_arr > 0)
        if np.any(valid2):
            local_min = np.min(L4_arr[valid2])
            if local_min < grid_best_f:
                idx = np.where(valid2)[0][np.argmin(L4_arr[valid2])]
                grid_best_f = local_min
                grid_best_theta = theta
                grid_best_phi = phi_vals[idx]
    
    # Convert best grid point to 3D
    if grid_best_f < np.inf:
        c_from_grid = _recover_3d_from_2d(grid_best_theta, grid_best_phi)
        obj_grid, _ = _objective_and_grad(c_from_grid[0], c_from_grid[1], c_from_grid[2])
        if obj_grid < best_C4:
            best_C4 = obj_grid
            best_c = c_from_grid
    
    # Phase 2: Newton refinement from best grid point and several nearby points
    candidates = []
    if grid_best_f < np.inf:
        candidates.append((grid_best_theta, grid_best_phi))
        
        # Add neighbors on the grid
        idx_t = np.argmin(np.abs(theta_vals - grid_best_theta))
        idx_p = np.argmin(np.abs(phi_vals - grid_best_phi))
        
        for dt in [-2, -1, 0, 1, 2]:
            for dp in [-2, -1, 0, 1, 2]:
                nt = idx_t + dt
                np_ = idx_p + dp
                if 0 <= nt < n_theta and 0 <= np_ < n_phi:
                    candidates.append((theta_vals[nt], phi_vals[np_]))
    
    # Also add the seed point converted to angles
    seed_c = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    seed_norm = np.linalg.norm(seed_c)
    if seed_norm > 0:
        seed_dir = seed_c / seed_norm
        # Convert to spherical: theta = arccos(z/r), phi = arctan2(y, x)
        theta_seed = np.arccos(np.clip(seed_dir[2], -1, 1))
        phi_seed = np.arctan2(seed_dir[1], seed_dir[0])
        candidates.append((theta_seed, phi_seed))
    
    # Newton refine each candidate
    for theta0, phi0 in candidates:
        if time.time() > deadline - 30.0:
            break
        
        try:
            theta_opt, phi_opt = _newton_refine_2d(theta0, phi0, max_iter=30, tol=1e-14)
            c_opt = _recover_3d_from_2d(theta_opt, phi_opt)
            obj_opt, _ = _objective_and_grad(c_opt[0], c_opt[1], c_opt[2])
            if obj_opt < best_C4:
                best_C4 = obj_opt
                best_c = c_opt
        except:
            pass
    
    # Phase 3: Final polish with BFGS in 3D from best 2D result
    if time.time() < deadline - 10.0:
        try:
            result = minimize(
                lambda c: _objective_and_grad(c[0], c[1], c[2])[0],
                best_c,
                jac=lambda c: _objective_and_grad(c[0], c[1], c[2])[1],
                method='BFGS',
                options={'maxiter': 300, 'gtol': 1e-14}
            )
            if result.success and result.fun < best_C4:
                best_C4 = result.fun
                best_c = result.x
        except Exception:
            pass
    
    # Phase 4: Additional 2D Newton from random points if time remains
    if time.time() < deadline - 20.0:
        rng = np.random.RandomState(123)
        for _ in range(50):
            if time.time() > deadline - 15.0:
                break
            theta_rand = rng.uniform(0.01, np.pi - 0.01)
            phi_rand = rng.uniform(-np.pi, np.pi)
            try:
                theta_opt, phi_opt = _newton_refine_2d(theta_rand, phi_rand, max_iter=20, tol=1e-13)
                c_opt = _recover_3d_from_2d(theta_opt, phi_opt)
                obj_opt, _ = _objective_and_grad(c_opt[0], c_opt[1], c_opt[2])
                if obj_opt < best_C4:
                    best_C4 = obj_opt
                    best_c = c_opt
            except:
                pass
    
    # Final high-precision polish
    if time.time() < deadline - 3.0:
        try:
            result = minimize(
                lambda c: _objective_and_grad(c[0], c[1], c[2])[0],
                best_c,
                jac=lambda c: _objective_and_grad(c[0], c[1], c[2])[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-16, 'ftol': 1e-16}
            )
            if result.success and result.fun < best_C4:
                best_C4 = result.fun
                best_c = result.x
        except Exception:
            pass
    
    return np.array(best_c)
