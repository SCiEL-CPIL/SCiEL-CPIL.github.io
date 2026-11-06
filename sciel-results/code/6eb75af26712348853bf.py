"""Broadband antireflection coating design via global stochastic search and local polish."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

# EVOLVE-BLOCK-START

def _mean_R(labels, thicknesses, n_amb, n_sub, n_L, n_H, band_nm, n_wl=401):
    """Mean reflectance (%) over n_wl equally spaced wavelengths in band_nm."""
    lam_min, lam_max = band_nm
    lams = np.linspace(lam_min, lam_max, n_wl)
    if len(labels) == 0:
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return float(np.abs(r) ** 2 * 100.0)
    
    n_mat = np.array([n_L if l == 'L' else n_H for l in labels], dtype=float)
    d = np.array(thicknesses, dtype=float)
    beta = 2.0 * np.pi * n_mat * d
    
    total = 0.0
    chunk = 64
    for s in range(0, n_wl, chunk):
        e = min(s + chunk, n_wl)
        L = lams[s:e]
        m = L.shape[0]
        M00 = np.ones(m, dtype=complex); M01 = np.zeros(m, dtype=complex)
        M10 = np.zeros(m, dtype=complex); M11 = np.ones(m, dtype=complex)
        for i in range(len(labels)):
            delta = beta[i] / L
            c = np.cos(delta); sn = np.sin(delta)
            b = 1j * sn / n_mat[i]
            cc = 1j * n_mat[i] * sn
            n00 = M00 * c + M01 * cc
            n01 = M00 * b + M01 * c
            n10 = M10 * c + M11 * cc
            n11 = M10 * b + M11 * c
            M00, M01, M10, M11 = n00, n01, n10, n11
        Y = M10 / M00
        r = (n_amb - Y) / (n_amb + Y)
        total += np.sum(np.abs(r) ** 2 * 100.0)
    return total / n_wl

def design(problem):
    start = time.time()
    deadline = start + 130.0  # Safety margin

    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    centre = 0.5 * (band_nm[0] + band_nm[1])
    
    # Best solution tracking
    best_labels = ["L"]
    best_thick = [centre / (4.0 * n_L)]
    best_score = _mean_R(best_labels, best_thick, n_amb, n_sub, n_L, n_H, band_nm)
    
    def update_best(labels, thick, score):
        nonlocal best_labels, best_thick, best_score
        if score < best_score:
            best_labels = labels[:]
            best_thick = thick[:]
            best_score = score

    # Helper to encode/decode a variable-length stack from a fixed-size vector
    # x = [n_layers, t1, t2, ..., t_max, m1, m2, ..., m_max]
    # n_layers is integer 1..max_layers
    # t_i are thicknesses
    # m_i are material indices (0 for L, 1 for H)
    
    def decode(x):
        n_layers = int(np.round(x[0]))
        n_layers = np.clip(n_layers, 1, max_layers)
        t = x[1 : 1 + max_layers][:n_layers]
        m = x[1 + max_layers : 1 + max_layers + max_layers][:n_layers]
        
        # Enforce constraints
        t = np.clip(t, 0.0, max_layer_nm)
        total_t = t.sum()
        if total_t > max_total_nm:
            t = t * (max_total_nm / total_t)
            
        labels = ['L' if mi < 0.5 else 'H' for mi in m]
        return labels, t.tolist()

    def objective(x):
        labels, t = decode(x)
        return _mean_R(labels, t, n_amb, n_sub, n_L, n_H, band_nm)

    # Initial population for DE
    # Dimension: 1 (n_layers) + max_layers (t) + max_layers (m)
    dim = 1 + 2 * max_layers
    bounds = [(1, max_layers)] + [(0, max_layer_nm)] * max_layers + [(0, 1)] * max_layers
    
    # We need a custom initialization or strategy to ensure good starting points
    # DE with 'best1bin' or 'best1exp' mutation
    
    # Run Differential Evolution
    # maxiter and popsize tuned for 130s budget
    # Each function eval is ~401 wavelengths * n_layers ops. 
    # n_layers max 20. 401 * 20 * ~10 flops = ~80k flops. Very fast.
    # We can afford many evaluations.
    
    # Strategy: Run DE for a few seconds, then L-BFGS-B on the best result.
    de_start = time.time()
    max_de_time = 60.0
    
    try:
        # Use a custom callback to stop DE if time is up
        def de_callback(xk, convergence):
            if time.time() > start + max_de_time:
                return True
            return False

        de_res = differential_evolution(
            objective, 
            bounds, 
            seed=42,
            maxiter=1000,
            popsize=20,
            tol=1e-5,
            callback=de_callback,
            polish=False,  # We will do our own polish
            init='latinhypercube'
        )
        
        if de_res.success or de_res.fun < best_score:
            labels, t = decode(de_res.x)
            update_best(labels, t, de_res.fun)
            
    except Exception:
        pass

    # Now, perform local refinement on the best candidate found
    # We will try to refine the specific sequence and thicknesses found.
    # L-BFGS-B is good for smooth, continuous optimization.
    # However, the sequence (labels) is discrete.
    # We can fix the sequence from DE and optimize thicknesses with L-BFGS-B.
    
    if best_labels and time.time() < deadline - 10:
        # Optimize thicknesses for the best sequence found
        n = len(best_labels)
        
        def obj_thick(x):
            return _mean_R(best_labels, x, n_amb, n_sub, n_L, n_H, band_nm)
        
        x0 = np.array(best_thick)
        bounds_t = [(0, max_layer_nm) for _ in range(n)]
        
        # L-BFGS-B with total thickness constraint is tricky directly.
        # We can use a penalty or just project. 
        # Let's use a simple projection method or just trust L-BFGS-B with bounds 
        # and check total thickness afterwards. 
        # Actually, L-BFGS-B doesn't handle linear equality constraints like sum(t) <= max_total.
        # We can use SLSQP which does.
        
        try:
            cons = ({'type': 'ineq', 'fun': lambda x: max_total_nm - np.sum(x)})
            res = minimize(
                obj_thick, 
                x0, 
                method='SLSQP', 
                bounds=bounds_t, 
                constraints=cons,
                options={'maxiter': 100, 'ftol': 1e-6}
            )
            if res.success:
                t_new = res.x
                # Final constraint check
                t_new = np.clip(t_new, 0, max_layer_nm)
                if t_new.sum() > max_total_nm:
                    t_new = t_new * (max_total_nm / t_new.sum())
                s_new = obj_thick(t_new)
                if s_new < best_score:
                    update_best(best_labels, t_new.tolist(), s_new)
        except Exception:
            pass

    # Additionally, try a few random sequence swaps on the best solution to see if a different
    # material arrangement helps, followed by SLSQP thickness optimization.
    if time.time() < deadline - 15:
        import random
        rng = random.Random(123)
        for _ in range(10):
            if time.time() > deadline - 10:
                break
            cand_labels = best_labels[:]
            if len(cand_labels) < 2:
                continue
            i = rng.randint(0, len(cand_labels) - 1)
            j = rng.randint(0, len(cand_labels) - 1)
            if i == j:
                continue
            cand_labels[i], cand_labels[j] = cand_labels[j], cand_labels[i]
            
            # Quick check with current thicknesses
            s_quick = _mean_R(cand_labels, best_thick, n_amb, n_sub, n_L, n_H, band_nm, n_wl=101)
            if s_quick < best_score * 1.1: # Promising
                # Refine
                n = len(cand_labels)
                def obj_t2(x):
                    return _mean_R(cand_labels, x, n_amb, n_sub, n_L, n_H, band_nm)
                try:
                    cons2 = ({'type': 'ineq', 'fun': lambda x: max_total_nm - np.sum(x)})
                    res2 = minimize(
                        obj_t2, 
                        np.array(best_thick), 
                        method='SLSQP', 
                        bounds=[(0, max_layer_nm) for _ in range(n)], 
                        constraints=cons2,
                        options={'maxiter': 50, 'ftol': 1e-5}
                    )
                    if res2.success:
                        t_new = np.clip(res2.x, 0, max_layer_nm)
                        if t_new.sum() > max_total_nm:
                            t_new = t_new * (max_total_nm / t_new.sum())
                        s_new = obj_t2(t_new)
                        if s_new < best_score:
                            update_best(cand_labels, t_new.tolist(), s_new)
                except Exception:
                    pass

    # Final constraint enforcement
    best_thick = [max(0.0, min(t, max_layer_nm)) for t in best_thick]
    tot = sum(best_thick)
    if tot > max_total_nm:
        sc = max_total_nm / tot
        best_thick = [t * sc for t in best_thick]
        
    return best_labels, best_thick

# EVOLVE-BLOCK-END
