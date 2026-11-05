import time
import numpy as np
from numpy.linalg import eigh

def _compute_gap_fast(cell, n_k=15, pw_radius=12):
    """Fast gap computation using precomputed Fourier coefficients."""
    N = len(cell)
    eps = 1.0 + 11.0 * np.array(cell, dtype=float)
    inv_eps = 1.0 / eps
    fft_inv = np.fft.fft2(inv_eps) / (N * N)
    
    # Build G-vector list
    G_list = []
    for Gx in range(-pw_radius, pw_radius + 1):
        for Gy in range(-pw_radius, pw_radius + 1):
            if Gx * Gx + Gy * Gy <= pw_radius * pw_radius:
                G_list.append((Gx, Gy))
    n_pw = len(G_list)
    G_arr = np.array(G_list, dtype=float)
    
    # Precompute inv_eps matrix
    size_d = 4 * pw_radius + 1
    offset_d = 2 * pw_radius
    inv_eps_full = np.zeros((size_d, size_d), dtype=complex)
    
    Gx_diff = G_arr[:, 0][:, None] - G_arr[:, 0][None, :]
    Gy_diff = G_arr[:, 1][:, None] - G_arr[:, 1][None, :]
    
    # Vectorized fill
    dqx = Gx_diff.astype(int)
    dqy = Gy_diff.astype(int)
    inv_eps_full[dqx + offset_d, dqy + offset_d] = fft_inv[dqx % N, dqy % N]
    
    # K-points in irreducible BZ (first quadrant, kx >= ky)
    k_points = []
    # Along Gamma-X
    for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        k_points.append((t, 0.0))
    # Along X-M
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        k_points.append((0.5, t))
    # Along Gamma-M (diagonal)
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        k_points.append((t, t))
    # Interior
    k_points.extend([(0.3, 0.1), (0.4, 0.2), (0.2, 0.3), (0.35, 0.15)])
    
    max_w4 = 0.0
    min_w5 = float('inf')
    
    for k in k_points:
        kvec = np.array(k, dtype=float)
        kG = kvec + G_arr
        dots = np.sum(kG * kG, axis=1)
        
        inv_eps_mat = inv_eps_full[
            dqx + offset_d, 
            dqy + offset_d
        ]
        
        M = np.outer(dots, 1.0) * inv_eps_mat  # diagonal * matrix
        M = (M + M.conj().T) / 2
        
        try:
            eigvals = eigh(M, eigvals_only=True)
            eigvals = np.sqrt(np.maximum(eigvals, 0))
            if len(eigvals) >= 5:
                max_w4 = max(max_w4, eigvals[3])
                min_w5 = min(min_w5, eigvals[4])
        except:
            pass
    
    if min_w5 == float('inf') or max_w4 <= 0:
        return -1.0
    return 2.0 * (min_w5 - max_w4) / (min_w5 + max_w4)


def _make_cell_from_params(params):
    """
    Create a 32x32 cell from parameters.
    params: [mode, radius, aspect, smooth]
      mode: 0 = rod (dielectric in center), 1 = hole (air in center)
      radius: base radius in pixels
      aspect: y-axis stretch (1.0 = circular)
      smooth: corner rounding factor (0 = square, 1 = circle)
    """
    size = 32
    mode = params[0] if len(params) > 0 else 0
    radius = params[1] if len(params) > 1 else 8.0
    aspect = params[2] if len(params) > 2 else 1.0
    smooth = params[3] if len(params) > 3 else 0.5
    
    cell = np.zeros((size, size), dtype=int)
    half = size // 2
    
    for i in range(size):
        for j in range(size):
            # Center coordinates (0 at center)
            x = j - half + 0.5
            y = i - half + 0.5
            
            # Normalize to radius
            nx = x / radius
            ny = y / (radius * aspect)
            
            # Superellipse: |nx|^(2p) + |ny|^(2p) < 1 where p = 1/(1-smooth)
            # smooth=0 -> p=1 (diamond/square rotated 45deg)
            # smooth=1 -> p->inf (circle)
            p = 1.0 / max(0.1, 1.0 - smooth)
            val = abs(nx) ** (2 * p) + abs(ny) ** (2 * p)
            
            is_inside = val < 1.0
            
            if mode < 0.5:  # Rod: dielectric inside
                cell[i, j] = 1 if is_inside else 0
            else:  # Hole: air inside
                cell[i, j] = 0 if is_inside else 1
    
    return cell.tolist()


def _nelder_mead(objective, x0, bounds, max_iter=60, tol=1e-4):
    """Nelder-Mead simplex optimization with bounds."""
    n = len(x0)
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    # Build initial simplex
    simplex = [list(x0)]
    fvals = [objective(x0)]
    
    for i in range(n):
        x_new = list(x0)
        step = 0.05 * (bounds[1][i] - bounds[0][i]) + 0.01
        x_new[i] += step
        x_new[i] = np.clip(x_new[i], bounds[0][i], bounds[1][i])
        simplex.append(x_new)
        fvals.append(objective(x_new))
    
    for iteration in range(max_iter):
        if time.time() > _deadline - 40:
            break
        
        # Sort by function value
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean([simplex[i] for i in range(n)], axis=0)
        
        # Reflection
        x_r = centroid + alpha * (centroid - simplex[-1])
        x_r = np.clip(x_r, bounds[0], bounds[1])
        f_r = objective(x_r)
        
        if f_r < fvals[0]:
            # Expansion
            x_e = centroid + gamma * (x_r - centroid)
            x_e = np.clip(x_e, bounds[0], bounds[1])
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = list(x_e)
                fvals[-1] = f_e
            else:
                simplex[-1] = list(x_r)
                fvals[-1] = f_r
        elif f_r < fvals[-2]:
            simplex[-1] = list(x_r)
            fvals[-1] = f_r
        else:
            # Contraction
            x_c = centroid + rho * (simplex[-1] - centroid)
            x_c = np.clip(x_c, bounds[0], bounds[1])
            f_c = objective(x_c)
            if f_c < fvals[-1]:
                simplex[-1] = list(x_c)
                fvals[-1] = f_c
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                    simplex[i] = np.clip(simplex[i], bounds[0], bounds[1])
                    fvals[i] = objective(simplex[i])
    
    # Return best
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]


_deadline = 0

def design(problem):
    global _deadline
    start_time = time.time()
    _deadline = start_time + 190.0  # Safety margin
    
    size = problem["pixels"]
    assert size == 32
    
    # Parameter bounds
    bounds = ([0.0, 4.0, 0.5, 0.0], [1.0, 14.0, 2.0, 1.0])
    
    def objective(params):
        cell = _make_cell_from_params(params)
        return _compute_gap_fast(cell, n_k=12, pw_radius=10)
    
    # Multi-start: try both rod and hole modes
    initial_points = [
        [0.0, 7.0, 1.0, 0.5],   # Circular rod
        [0.0, 8.0, 1.0, 0.7],   # Rounded rod
        [0.0, 6.0, 1.2, 0.3],   # Elliptical rod
        [0.0, 9.0, 0.8, 0.5],   # Wide rod
        [1.0, 7.0, 1.0, 0.5],   # Circular hole
        [1.0, 8.0, 1.0, 0.7],   # Rounded hole
        [1.0, 6.0, 1.3, 0.3],   # Elliptical hole
        [1.0, 5.0, 1.0, 0.5],   # Small hole
        [0.0, 10.0, 1.0, 0.9],  # Large rounded rod
        [1.0, 9.0, 1.0, 0.9],   # Large rounded hole
    ]
    
    best_score = -1.0
    best_params = initial_points[0]
    
    # Quick screen
    for p0 in initial_points:
        if time.time() > _deadline - 80:
            break
        try:
            f0 = objective(p0)
            if f0 > best_score:
                best_score = f0
                best_params = p0
        except:
            pass
    
    # Run Nelder-Mead from best and a few others
    for p0 in [best_params] + initial_points[:3]:
        if time.time() > _deadline - 30:
            break
        try:
            p_opt, f_opt = _nelder_mead(
                objective,
                p0,
                bounds,
                max_iter=40,
                tol=1e-3
            )
            if f_opt > best_score:
                best_score = f_opt
                best_params = p_opt
        except:
            pass
    
    # Final refinement with higher resolution if time allows
    if time.time() < _deadline - 20 and best_params is not None:
        def objective_high_res(params):
            cell = _make_cell_from_params(params)
            return _compute_gap_fast(cell, n_k=15, pw_radius=12)
        
        try:
            p_final, f_final = _nelder_mead(
                objective_high_res,
                best_params,
                bounds,
                max_iter=20,
                tol=1e-4
            )
            if f_final > best_score:
                best_score = f_final
                best_params = p_final
        except:
            pass
    
    # Generate final cell
    final_cell = _make_cell_from_params(best_params)
    
    # Ensure symmetry (should already be symmetric by construction)
    c = np.array(final_cell, dtype=float)
    c_sym = (c + c.T + c[::-1, :] + c[:, ::-1] + c[::-1, ::-1] + c.T[::-1, :] + c.T[:, ::-1] + c[::-1, ::-1].T) / 8.0
    final_cell = (c_sym > 0.5).astype(int).tolist()
    
    return final_cell
