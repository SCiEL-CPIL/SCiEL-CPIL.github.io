import time
import math
import numpy as np
import random
import sys

def _reflectance_array(materials, thicknesses, n_ambient, n_substrate, wls, n_mat):
    """
    Compute reflectance array for given materials and thicknesses using
    the thin-film characteristic-matrix method.
    Returns array of reflectance values (0 to 1).
    """
    if len(materials) == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.full(len(wls), r * r)

    n_layers = len(materials)
    n_wls = len(wls)

    # Vectorized beta computation
    ns = np.array([n_mat[m] for m in materials])
    ds = np.array(thicknesses, dtype=float)
    beta = 2.0 * math.pi * np.outer(ns * ds, 1.0 / wls)

    cosb = np.cos(beta)
    sinb = np.sin(beta)

    # Build the total characteristic matrix
    M11 = np.ones(n_wls, dtype=complex)
    M12 = np.zeros(n_wls, dtype=complex)
    M21 = np.zeros(n_wls, dtype=complex)
    M22 = np.ones(n_wls, dtype=complex)

    for i in range(n_layers):
        n = ns[i]
        c = cosb[i]
        s = sinb[i]

        A = c
        B = 1j * s / n
        C = 1j * n * s
        D = c

        nM11 = M11 * A + M12 * C
        nM12 = M11 * B + M12 * D
        nM21 = M21 * A + M22 * C
        nM22 = M21 * B + M22 * D

        M11, M12, M21, M22 = nM11, nM12, nM21, nM22

    Y = (M21 + n_substrate * M22) / (M11 + n_substrate * M12)
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.abs(r) ** 2
    return R

def _mean_reflectance_percent(materials, thicknesses, problem, wls, n_mat):
    """
    Compute mean reflectance in percent with constraint penalties.
    """
    if len(materials) != len(thicknesses):
        return 1e6
    if len(materials) > problem["max_layers"]:
        return 1e6

    max_layer = problem["max_layer_nm"]
    max_total = problem["max_total_nm"]

    for t in thicknesses:
        if t < -1e-9 or t > max_layer + 1e-9:
            return 1e6

    total_th = sum(thicknesses)
    if total_th > max_total + 1e-6:
        return 1e6 + (total_th - max_total) * 10.0

    if len(materials) == 0:
        r = (problem["ambient_index"] - problem["substrate_index"]) / (
            problem["ambient_index"] + problem["substrate_index"])
        return r * r * 100.0

    R = _reflectance_array(materials, thicknesses, problem["ambient_index"],
                           problem["substrate_index"], wls, n_mat)
    return float(np.mean(R)) * 100.0

def _reflectance_and_grad(materials, thicknesses, problem, wls, n_mat):
    """
    Compute mean reflectance and its gradient with respect to thicknesses.
    Returns (mean_R_percent, grad_array).
    """
    if len(materials) != len(thicknesses):
        return 1e6, np.zeros(len(thicknesses))
    
    max_layer = problem["max_layer_nm"]
    max_total = problem["max_total_nm"]
    
    for t in thicknesses:
        if t < -1e-9 or t > max_layer + 1e-9:
            return 1e6, np.zeros(len(thicknesses))
    
    total_th = sum(thicknesses)
    if total_th > max_total + 1e-6:
        return 1e6, np.zeros(len(thicknesses))
    
    if len(materials) == 0:
        r = (problem["ambient_index"] - problem["substrate_index"]) / (
            problem["ambient_index"] + problem["substrate_index"])
        return r * r * 100.0, np.zeros(0)
    
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_layers = len(materials)
    n_wls = len(wls)
    
    ns = np.array([n_mat[m] for m in materials])
    ds = np.array(thicknesses, dtype=float)
    
    # Compute beta and derivatives
    beta = 2.0 * math.pi * np.outer(ns * ds, 1.0 / wls)
    cosb = np.cos(beta)
    sinb = np.sin(beta)
    # dbeta_dd = 2*pi*n/wl for each layer
    dbeta_dd = 2.0 * math.pi * np.outer(ns, 1.0 / wls)
    
    # Build matrices and their derivatives
    M11 = np.ones(n_wls, dtype=complex)
    M12 = np.zeros(n_wls, dtype=complex)
    M21 = np.zeros(n_wls, dtype=complex)
    M22 = np.ones(n_wls, dtype=complex)
    
    # Store intermediate matrices for gradient computation
    M11_hist = [M11.copy()]
    M12_hist = [M12.copy()]
    M21_hist = [M21.copy()]
    M22_hist = [M22.copy()]
    
    for i in range(n_layers):
        n = ns[i]
        c = cosb[i]
        s = sinb[i]
        
        A = c
        B = 1j * s / n
        C = 1j * n * s
        D = c
        
        nM11 = M11 * A + M12 * C
        nM12 = M11 * B + M12 * D
        nM21 = M21 * A + M22 * C
        nM22 = M21 * B + M22 * D
        
        M11, M12, M21, M22 = nM11, nM12, nM21, nM22
        
        M11_hist.append(M11.copy())
        M12_hist.append(M12.copy())
        M21_hist.append(M21.copy())
        M22_hist.append(M22.copy())
    
    Y = (M21 + n_substrate * M22) / (M11 + n_substrate * M12)
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.abs(r) ** 2
    mean_R = float(np.mean(R)) * 100.0
    
    # Compute gradient using finite differences for simplicity
    grad = np.zeros(n_layers)
    eps = 0.1
    for i in range(n_layers):
        ds_plus = ds.copy()
        ds_plus[i] = min(ds_plus[i] + eps, max_layer)
        ds_minus = ds.copy()
        ds_minus[i] = max(ds_minus[i] - eps, 0.0)
        
        # Compute R for plus
        beta_p = 2.0 * math.pi * np.outer(ns * ds_plus, 1.0 / wls)
        cosb_p = np.cos(beta_p)
        sinb_p = np.sin(beta_p)
        
        M11_p = np.ones(n_wls, dtype=complex)
        M12_p = np.zeros(n_wls, dtype=complex)
        M21_p = np.zeros(n_wls, dtype=complex)
        M22_p = np.ones(n_wls, dtype=complex)
        
        for j in range(n_layers):
            n_j = ns[j]
            c_j = cosb_p[j]
            s_j = sinb_p[j]
            A_j = c_j
            B_j = 1j * s_j / n_j
            C_j = 1j * n_j * s_j
            D_j = c_j
            nM11_p = M11_p * A_j + M12_p * C_j
            nM12_p = M11_p * B_j + M12_p * D_j
            nM21_p = M21_p * A_j + M22_p * C_j
            nM22_p = M21_p * B_j + M22_p * D_j
            M11_p, M12_p, M21_p, M22_p = nM11_p, nM12_p, nM21_p, nM22_p
        
        Y_p = (M21_p + n_substrate * M22_p) / (M11_p + n_substrate * M12_p)
        r_p = (n_ambient - Y_p) / (n_ambient + Y_p)
        R_p = np.abs(r_p) ** 2
        mean_R_p = float(np.mean(R_p)) * 100.0
        
        # Compute R for minus
        beta_m = 2.0 * math.pi * np.outer(ns * ds_minus, 1.0 / wls)
        cosb_m = np.cos(beta_m)
        sinb_m = np.sin(beta_m)
        
        M11_m = np.ones(n_wls, dtype=complex)
        M12_m = np.zeros(n_wls, dtype=complex)
        M21_m = np.zeros(n_wls, dtype=complex)
        M22_m = np.ones(n_wls, dtype=complex)
        
        for j in range(n_layers):
            n_j = ns[j]
            c_j = cosb_m[j]
            s_j = sinb_m[j]
            A_j = c_j
            B_j = 1j * s_j / n_j
            C_j = 1j * n_j * s_j
            D_j = c_j
            nM11_m = M11_m * A_j + M12_m * C_j
            nM12_m = M11_m * B_j + M12_m * D_j
            nM21_m = M21_m * A_j + M22_m * C_j
            nM22_m = M21_m * B_j + M22_m * D_j
            M11_m, M12_m, M21_m, M22_m = nM11_m, nM12_m, nM21_m, nM22_m
        
        Y_m = (M21_m + n_substrate * M22_m) / (M11_m + n_substrate * M12_m)
        r_m = (n_ambient - Y_m) / (n_ambient + Y_m)
        R_m = np.abs(r_m) ** 2
        mean_R_m = float(np.mean(R_m)) * 100.0
        
        grad[i] = (mean_R_p - mean_R_m) / (2.0 * eps)
    
    return mean_R, grad

def _project_to_feasible(x, lo, hi, max_total):
    """
    Project x to feasible region: bounds and total thickness constraint.
    """
    x = np.clip(x, lo, hi)
    s = np.sum(x)
    if s > max_total:
        if s > 1e-9:
            x = x * (max_total / s)
        else:
            x = np.zeros_like(x)
    x = np.clip(x, lo, hi)
    return x

def _optimize_l_bfgs(objective, x0, lo, hi, max_total, deadline, max_iter=100):
    """
    Simple L-BFGS implementation with bound constraints.
    """
    x = np.array(x0, dtype=float)
    x = _project_to_feasible(x, lo, hi, max_total)
    f_current = objective(x)
    n = len(x)
    
    # L-BFGS memory
    m = min(10, n)
    s_history = []  # step vectors
    y_history = []  # gradient differences
    g_history = []  # gradient history
    
    # Compute initial gradient
    _, g = _reflectance_and_grad(None, list(x), None, None, None) if False else (f_current, np.zeros(n))
    
    # Use finite difference gradient for L-BFGS
    def compute_grad(x_vec):
        eps = 0.05
        grad = np.zeros(n)
        for i in range(n):
            x_plus = x_vec.copy()
            x_plus[i] = min(x_plus[i] + eps, hi[i])
            x_plus = _project_to_feasible(x_plus, lo, hi, max_total)
            f_plus = objective(x_plus)
            
            x_minus = x_vec.copy()
            x_minus[i] = max(x_minus[i] - eps, lo[i])
            x_minus = _project_to_feasible(x_minus, lo, hi, max_total)
            f_minus = objective(x_minus)
            
            grad[i] = (f_plus - f_minus) / (2.0 * eps)
        return grad
    
    g = compute_grad(x)
    g_history.append(g.copy())
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Check convergence
        if np.max(np.abs(g)) < 1e-6:
            break
        
        # Two-loop recursion to compute search direction
        if len(s_history) > 0:
            gamma = 1.0
            if len(s_history) >= 1:
                s_last = s_history[-1]
                y_last = y_history[-1]
                sy = np.dot(s_last, y_last)
                if sy > 1e-10:
                    gamma = np.dot(s_last, g) / sy
            
            # Two-loop recursion
            r = g.copy()
            alpha = np.zeros(len(s_history))
            
            for i in range(len(s_history) - 1, -1, -1):
                s_i = s_history[i]
                y_i = y_history[i]
                sy = np.dot(s_i, y_i)
                if sy > 1e-10:
                    alpha[i] = np.dot(s_i, r) / sy
                    r -= alpha[i] * y_i
            
            q = r * gamma
            
            for i in range(len(s_history)):
                s_i = s_history[i]
                y_i = y_history[i]
                sy = np.dot(s_i, y_i)
                if sy > 1e-10:
                    beta = np.dot(y_i, q) / sy
                    q += (alpha[i] - beta) * s_i
            
            direction = -q
        else:
            direction = -g
        
        # Line search
        step = 1.0
        for ls_iter in range(20):
            x_new = x + step * direction
            x_new = _project_to_feasible(x_new, lo, hi, max_total)
            f_new = objective(x_new)
            
            if f_new < f_current:
                break
            step *= 0.5
        
        if f_new >= f_current:
            # Try smaller step or break
            step = 0.01
            x_new = x + step * direction
            x_new = _project_to_feasible(x_new, lo, hi, max_total)
            f_new = objective(x_new)
            if f_new >= f_current:
                break
        
        # Update L-BFGS memory
        g_new = compute_grad(x_new)
        s_vec = x_new - x
        y_vec = g_new - g
        sy = np.dot(s_vec, y_vec)
        
        if sy > 1e-10:
            s_history.append(s_vec)
            y_history.append(y_vec)
            g_history.append(g_new)
            
            if len(s_history) > m:
                s_history.pop(0)
                y_history.pop(0)
                g_history.pop(0)
        
        x = x_new
        f_current = f_new
        g = g_new
    
    return x, f_current

def _optimize_nelder_mead(objective, x0, lo, hi, max_total, deadline, max_iter=300, tol=1e-7):
    """
    Nelder-Mead with bound handling, total thickness projection, and deadline check.
    """
    x0 = np.array(x0, dtype=float)
    n = len(x0)
    
    x0 = _project_to_feasible(x0, lo, hi, max_total)
    
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        step = 0.05 * (hi[i] - lo[i]) if hi[i] - lo[i] > 1e-6 else 1.0
        xi[i] += step
        simplex.append(_project_to_feasible(xi, lo, hi, max_total))

    f_vals = [objective(x) for x in simplex]
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break

        order = np.argsort(f_vals)
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]

        if abs(f_vals[0] - f_vals[-1]) < tol:
            break

        centroid = np.mean(simplex[:-1], axis=0)

        x_r = _project_to_feasible(centroid + alpha * (centroid - simplex[-1]), lo, hi, max_total)
        f_r = objective(x_r)

        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1] = x_r
            f_vals[-1] = f_r
            continue

        if f_r < f_vals[0]:
            x_e = _project_to_feasible(centroid + gamma * (x_r - centroid), lo, hi, max_total)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
            continue

        x_c = _project_to_feasible(centroid + rho * (simplex[-1] - centroid), lo, hi, max_total)
        f_c = objective(x_c)
        if f_c < f_vals[-1]:
            simplex[-1] = x_c
            f_vals[-1] = f_c
            continue

        for i in range(1, n + 1):
            simplex[i] = _project_to_feasible(simplex[0] + sigma * (simplex[i] - simplex[0]), lo, hi, max_total)
            f_vals[i] = objective(simplex[i])

    order = np.argsort(f_vals)
    best_idx = order[0]
    return _project_to_feasible(simplex[best_idx], lo, hi, max_total), f_vals[best_idx]

def _coordinate_descent(objective, x0, lo, hi, max_total, deadline, max_rounds=100):
    """
    Cyclic coordinate descent with line search.
    """
    x = np.array(x0, dtype=float)
    x = _project_to_feasible(x, lo, hi, max_total)
    f_current = objective(x)
    n = len(x)
    
    for rnd in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            x_old = x[i]
            best_x = x_old
            best_f = f_current
            # Try several steps
            for step_frac in [0.5, -0.5, 0.25, -0.25, 0.1, -0.1, 0.75, -0.75]:
                x_new = x.copy()
                x_new[i] = np.clip(x_old + step_frac * (hi[i] - lo[i]) * 0.2, lo[i], hi[i])
                x_new = _project_to_feasible(x_new, lo, hi, max_total)
                f_new = objective(x_new)
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new[i]
            if best_x != x_old:
                x[i] = best_x
                x = _project_to_feasible(x, lo, hi, max_total)
                f_current = best_f
                improved = True
        if not improved:
            break
    
    return x, f_current

def design(problem):
    """
    Design a broadband antireflection coating using a search-based approach.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]

    band_start, band_end = problem["band_nm"]
    wls = np.linspace(band_start, band_end, 401)
    n_mat = problem["materials"]

    best_val = float('inf')
    best_mats = None
    best_ths = None

    def update_best(mats, ths):
        nonlocal best_val, best_mats, best_ths
        ths_clipped = [min(max(t, 0.0), max_layer) for t in ths]
        s = sum(ths_clipped)
        if s > max_total:
            if s > 1e-9:
                f = max_total / s
                ths_clipped = [t * f for t in ths_clipped]
        val = _mean_reflectance_percent(mats, ths_clipped, problem, wls, n_mat)
        if val < best_val:
            best_val = val
            best_mats = list(mats)
            best_ths = list(ths_clipped)
        return val

    # Generate diverse initial designs and optimize them
    # Strategy: for each layer count, try multiple starting points
    
    # Common quarter-wave designs
    center_wls = [450, 500, 550, 600, 650, 700]
    
    # Build list of (materials, initial_thicknesses) candidates
    candidates = []
    
    for n_layers in [2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20]:
        if n_layers > max_layers:
            continue
        # Alternating starting with L or H
        for start_char in ['L', 'H']:
            mats = []
            for i in range(n_layers):
                if (i % 2 == 0) == (start_char == 'L'):
                    mats.append('L')
                else:
                    mats.append('H')
            
            # Multiple initial thickness guesses
            for cw in center_wls:
                ths = []
                for m in mats:
                    n = n_mat[m]
                    q = cw / (4.0 * n)
                    q *= random.uniform(0.7, 1.3)
                    ths.append(q)
                # Scale to fit total constraint
                s = sum(ths)
                if s > max_total:
                    f = max_total / s
                    ths = [t * f for t in ths]
                candidates.append((mats, ths))
            
            # Add random initial points
            for _ in range(3):
                ths = [random.uniform(20, max_layer * 0.6) for _ in range(n_layers)]
                s = sum(ths)
                if s > max_total:
                    f = max_total / s
                    ths = [t * f for t in ths]
                candidates.append((mats, ths))

    # Shuffle for diversity
    random.shuffle(candidates)
    
    # Limit number of candidates based on time
    max_candidates = 80
    candidates = candidates[:max_candidates]
    
    for mats, ths_init in candidates:
        if time.time() > deadline:
            break
        
        n_layers = len(mats)
        lo = np.zeros(n_layers)
        hi = np.full(n_layers, max_layer)
        
        x0 = np.clip(np.array(ths_init, dtype=float), lo, hi)
        x0 = _project_to_feasible(x0, lo, hi, max_total)
        
        def objective(th, m=mats):
            return _mean_reflectance_percent(m, list(th), problem, wls, n_mat)
        
        # Try multiple optimization strategies
        best_x = x0.copy()
        best_f = objective(x0)
        
        # 1. Coordinate descent
        x_cd, f_cd = _coordinate_descent(objective, x0, lo, hi, max_total, deadline, max_rounds=20)
        if f_cd < best_f:
            best_x = x_cd
            best_f = f_cd
        
        # 2. L-BFGS (faster for smooth problems)
        if time.time() < deadline - 5:
            x_lbfgs, f_lbfgs = _optimize_l_bfgs(objective, best_x, lo, hi, max_total, deadline, max_iter=80)
            if f_lbfgs < best_f:
                best_x = x_lbfgs
                best_f = f_lbfgs
        
        # 3. Nelder-Mead for final refinement
        if time.time() < deadline - 3:
            x_nm, f_nm = _optimize_nelder_mead(objective, best_x, lo, hi, max_total, deadline, max_iter=150)
            if f_nm < best_f:
                best_x = x_nm
                best_f = f_nm
        
        update_best(mats, best_x)
    
    # Additional targeted search: refine the best found so far
    if best_mats is not None and time.time() < deadline - 15:
        n_layers = len(best_mats)
        lo = np.zeros(n_layers)
        hi = np.full(n_layers, max_layer)
        
        def objective(th):
            return _mean_reflectance_percent(best_mats, list(th), problem, wls, n_mat)
        
        # Multiple restarts from perturbed best
        for _ in range(8):
            if time.time() > deadline:
                break
            x0 = np.array(best_ths, dtype=float).copy()
            # Perturb
            for i in range(n_layers):
                if random.random() < 0.5:
                    x0[i] += random.uniform(-0.15, 0.15) * (hi[i] - lo[i])
                    x0[i] = np.clip(x0[i], lo[i], hi[i])
            x0 = _project_to_feasible(x0, lo, hi, max_total)
            
            # Use L-BFGS for refinement
            x_opt, f_opt = _optimize_l_bfgs(objective, x0, lo, hi, max_total, deadline, max_iter=100)
            update_best(best_mats, x_opt)
    
    # Fallback
    if best_mats is None:
        best_mats = ['L']
        best_ths = [600.0 / (4.0 * n_L)]
        update_best(best_mats, best_ths)
    
    # Final constraint check
    best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    s = sum(best_ths)
    if s > max_total + 1e-6:
        f = max_total / s
        best_ths = [t * f for t in best_ths]
        best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    
    return list(best_mats), list(best_ths)
