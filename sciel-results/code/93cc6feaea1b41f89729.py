import numpy as np
import scipy.optimize
import time

def design(problem):
    size = problem["pixels"]
    eps_high = problem["eps_high"]
    eps_low = problem["eps_low"]
    
    # 1. Generate K-points in the Irreducible Brillouin Zone (IBZ)
    # IBZ is the triangle with vertices (0,0), (0.5, 0), (0.5, 0.5)
    # We need ~30 points.
    k_points = []
    # High symmetry points
    k_points.append((0.0, 0.0))
    k_points.append((0.5, 0.0))
    k_points.append((0.5, 0.5))
    k_points.append((0.25, 0.0))
    k_points.append((0.5, 0.25))
    k_points.append((0.25, 0.25))
    k_points.append((0.0, 0.0)) # Gamma
    
    # Add points along the boundary kx=0, ky=kx, ky=0
    # and interior
    for i in range(1, 10):
        k_points.append((0.0, i * 0.05))
        k_points.append((i * 0.05, 0.0))
        k_points.append((i * 0.05, i * 0.05))
    for i in range(1, 5):
        k_points.append((0.5 - i * 0.1, i * 0.1))
        
    # Deduplicate and limit to 30
    unique_k = []
    for k in k_points:
        if k not in unique_k:
            unique_k.append(k)
    k_points = unique_k[:30]
    
    # Subset for optimization
    k_points_opt = k_points[:10]

    def make_cell(params):
        d, r = params
        cell = np.zeros((size, size), dtype=int)
        xs = (np.arange(size) + 0.5) / size - 0.5
        ys = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        
        d1 = (X - d)**2 + (Y - d)**2
        d2 = (X + d)**2 + (Y - d)**2
        d3 = (X - d)**2 + (Y + d)**2
        d4 = (X + d)**2 + (Y + d)**2
        
        min_d_sq = np.minimum(np.minimum(d1, d2), np.minimum(d3, d4))
        cell[min_d_sq < r**2] = 1
        return cell

    def compute_fourier_coeffs(inv_eps):
        coeffs = np.zeros((29, 29), dtype=complex)
        I, J = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        for nx in range(-14, 15):
            for ny in range(-14, 15):
                phase = -1j * 2 * np.pi * (nx * I + ny * J) / size
                coeffs[nx+14, ny+14] = np.sum(inv_eps * np.exp(phase)) / (size*size)
        return coeffs

    def evaluate(cell, k_points_list):
        inv_eps = np.where(cell == 1, 1.0/eps_high, 1.0/eps_low)
        coeffs = compute_fourier_coeffs(inv_eps)
        
        n_vals = np.arange(-14, 15)
        Gx, Gy = np.meshgrid(n_vals, n_vals, indexing='ij')
        Gx_flat = Gx.flatten()
        Gy_flat = Gy.flatten()
        
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points_list:
            kg_sq = (kx + Gx_flat)**2 + (ky + Gy_flat)**2
            H = np.zeros((289, 289), dtype=complex)
            
            # Precompute indices for coeffs
            # G index: idx -> (Gx_flat[idx], Gy_flat[idx])
            # We need c_{G'-G}
            # Loop over G' (row) and G (col)
            for idx_gp in range(289):
                gxp = Gx_flat[idx_gp]
                gyp = Gy_flat[idx_gp]
                for idx_g in range(289):
                    gx = Gx_flat[idx_g]
                    gy = Gy_flat[idx_g]
                    diff_x = int(gxp - gx)
                    diff_y = int(gyp - gy)
                    if -14 <= diff_x <= 14 and -14 <= diff_y <= 14:
                        c_val = coeffs[diff_x+14, diff_y+14]
                        # M_{G', G} = c_{G'-G} (k+G)^2
                        H[idx_gp, idx_g] = c_val * kg_sq[idx_g]
            
            evals = np.linalg.eigvals(H)
            evals_real = np.sort(np.real(evals))
            
            # 4th and 5th bands (indices 3 and 4)
            omega_4 = np.sqrt(evals_real[3])
            omega_5 = np.sqrt(evals_real[4])
            
            max_omega_4 = max(max_omega_4, omega_4)
            min_omega_5 = min(min_omega_5, omega_5)
        
        gap = 2 * (min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)
        return gap

    start_time = time.time()
    best_params = [0.25, 0.15]
    best_score = -1
    
    def objective(params):
        cell = make_cell(params)
        score = evaluate(cell, k_points_opt)
        return -score

    # Optimization
    try:
        res = scipy.optimize.minimize(objective, best_params, method='Nelder-Mead', 
                                       options={'maxiter': 40, 'xatol': 1e-4, 'fatol': 1e-4})
        if res.x[0] > 0 and res.x[1] > 0:
            best_params = res.x
    except Exception:
        pass

    # Refinement with full k-points if time permits
    if time.time() - start_time < 100:
        # Do a few local steps with full k-points
        # Just evaluate the current best and maybe neighbors
        current_cell = make_cell(best_params)
        current_score = evaluate(current_cell, k_points)
        
        # Try small perturbations
        for dd in [-0.01, 0.01]:
            for dr in [-0.01, 0.01]:
                p = [best_params[0]+dd, best_params[1]+dr]
                if p[0] > 0.1 and p[1] > 0.05:
                    c = make_cell(p)
                    s = evaluate(c, k_points)
                    if s > current_score:
                        current_score = s
                        best_params = p
                        current_cell = c
                        if time.time() - start_time > 110: break
            
            if time.time() - start_time > 110: break

    final_cell = make_cell(best_params)
    return final_cell.tolist()
