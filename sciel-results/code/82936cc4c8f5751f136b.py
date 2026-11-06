"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _char_matrix(n, d, lam):
    """Return the characteristic matrix for a single layer (vectorized over lam)."""
    beta = 2.0 * np.pi * n * d / lam
    c = np.cos(beta)
    s = np.sin(beta)
    # For normal incidence, B = n, A = 1
    # M = [[A*cos(beta), i*B*sin(beta)], [i*(1/B)*sin(beta), A*cos(beta)]]
    # Since A=1, B=n for normal incidence
    M11 = c
    M12 = 1j * n * s
    M21 = 1j * s / n
    M22 = c
    return np.stack([np.stack([M11, M12], axis=-1),
                     np.stack([M21, M22], axis=-1)], axis=-2)

def _reflectance(materials, thicknesses, wavelengths, n_ambient, n_substrate):
    """Compute reflectance (fraction) at normal incidence for given layers."""
    # Build total characteristic matrix
    # M_total = M1 * M2 * ... * M_N
    # For each wavelength, we multiply 2x2 matrices
    # We'll do this vectorized: each element is (N, len(wavelengths))
    # Start with identity
    total = np.zeros((2, 2, len(wavelengths)), dtype=complex)
    total[0, 0] = 1.0
    total[1, 1] = 1.0
    
    for mat, th in zip(materials, thicknesses):
        n = mat
        M = _char_matrix(n, th, wavelengths)
        # total = total @ M (element-wise matrix multiplication over last axis)
        # total is (2,2,N), M is (2,2,N)
        new_total = np.zeros_like(total)
        new_total[0, 0] = total[0, 0] * M[0, 0] + total[0, 1] * M[1, 0]
        new_total[0, 1] = total[0, 0] * M[0, 1] + total[0, 1] * M[1, 1]
        new_total[1, 0] = total[1, 0] * M[0, 0] + total[1, 1] * M[1, 0]
        new_total[1, 1] = total[1, 0] * M[0, 1] + total[1, 1] * M[1, 1]
        total = new_total
    
    # Admittance at the air side: Y = total[1, 0] / total[0, 0] ... wait
    # The characteristic matrix relates (E, H) at input to (E, H) at output
    # [E_in; H_in] = M_total [E_sub; H_sub]
    # At the substrate, E_sub/H_sub = n_sub (for normal incidence, going into substrate)
    # Actually, the admittance seen from the air side is Y = (M21 + M22 * n_sub) / (M11 + M12 * n_sub)
    # Wait, let me re-derive.
    # The transfer matrix M satisfies:
    # [E; H]_0 = M [E; H]_N
    # where 0 is the input side (air) and N is the output side (substrate)
    # At the substrate side, the admittance is Y_N = n_sub (since it's semi-infinite)
    # So H_N = Y_N * E_N = n_sub * E_N
    # From the matrix:
    # E_0 = M11 * E_N + M12 * H_N = (M11 + M12 * n_sub) * E_N
    # H_0 = M21 * E_N + M22 * H_N = (M21 + M22 * n_sub) * E_N
    # So the admittance at the air side is Y_0 = H_0 / E_0 = (M21 + M22 * n_sub) / (M11 + M12 * n_sub)
    
    Y = (total[1, 0] + total[1, 1] * n_substrate) / (total[0, 0] + total[0, 1] * n_substrate)
    
    # Reflectance: R = |(Y - n_ambient) / (Y + n_ambient)|^2
    r = (Y - n_ambient) / (Y + n_ambient)
    R = np.abs(r)**2
    return R

def _mean_reflectance_percent(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band."""
    lam0, lam1 = problem["band_nm"]
    wavelengths = np.linspace(lam0, lam1, 401)
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    
    R = _reflectance(materials, thicknesses, wavelengths, n_ambient, n_substrate)
    return np.mean(R) * 100.0

def _nelder_mead(f, x0, args=(), max_iter=200, tol=1e-6, deadline=None):
    """Simple Nelder-Mead implementation with deadline checking."""
    n = len(x0)
    # Initial simplex
    simplex = [list(x0)]
    for i in range(n):
        xi = list(x0)
        xi[i] += 0.1 if xi[i] == 0 else 0.1 * abs(xi[i])
        simplex.append(xi)
    
    values = [f(np.array(s), *args) for s in simplex]
    best_idx = np.argmin(values)
    best_val = values[best_idx]
    best_x = np.array(simplex[best_idx])
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Sort simplex by function value
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        best_val = values[0]
        best_x = np.array(simplex[0])
        
        # Check convergence
        if abs(values[-1] - values[0]) < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean([np.array(s) for s in simplex[:-1]], axis=0)
        
        # Reflection
        alpha = 1.0
        x_r = centroid + alpha * (centroid - np.array(simplex[-1]))
        f_r = f(x_r, *args)
        
        if values[0] <= f_r < values[-2]:
            simplex[-1] = x_r
            values[-1] = f_r
            continue
        
        # Expansion
        gamma = 2.0
        x_e = centroid + gamma * (x_r - centroid)
        f_e = f(x_e, *args)
        
        if f_e < f_r:
            simplex[-1] = x_e
            values[-1] = f_e
        else:
            # Contraction
            rho = 0.5
            if f_r < values[-1]:
                x_c = centroid + rho * (x_r - centroid)
            else:
                x_c = centroid - rho * (np.array(simplex[-1]) - centroid)
            f_c = f(x_c, *args)
            
            if f_c < min(f_r, values[-1]):
                simplex[-1] = x_c
                values[-1] = f_c
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = [simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                    values[i] = f(np.array(simplex[i]), *args)
    
    return best_x, best_val

def design(problem):
    """Design an antireflection coating using multi-layer Nelder-Mead optimization."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    materials_dict = problem["materials"]
    n_L = materials_dict["L"]
    n_H = materials_dict["H"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    band = problem["band_nm"]
    lam0, lam1 = band
    
    # Precompute wavelengths for fast evaluation
    wavelengths = np.linspace(lam0, lam1, 401)
    
    def fast_eval(x):
        """Evaluate mean reflectance from parameter vector x.
        x contains: [n_layers, t1, t2, ..., tn] where first element is number of layers (int)
        Actually, let's use a fixed structure per optimization run.
        """
        n_layers = int(round(x[0]))
        n_layers = max(1, min(n_layers, max_layers))
        thicknesses = x[1:1+n_layers]
        
        # Ensure constraints
        if np.any(thicknesses < 0) or np.any(thicknesses > max_layer_nm):
            return 100.0
        if np.sum(thicknesses) > max_total_nm:
            return 100.0
        
        # Determine material pattern: alternate L, H, L, H, ... starting with L
        # Or we could optimize the pattern too, but let's start with alternating
        materials = []
        for i in range(n_layers):
            if i % 2 == 0:
                materials.append(n_L)
            else:
                materials.append(n_H)
        
        R = _reflectance(materials, list(thicknesses), wavelengths, n_ambient, n_substrate)
        return np.mean(R) * 100.0
    
    best_score = float('inf')
    best_result = None
    
    # Try different numbers of layers with alternating L/H pattern
    for n_layers in range(1, min(max_layers + 1, 15)):
        if time.time() > deadline:
            break
        
        # Initial guess: quarter-wave at center wavelength
        centre = 0.5 * (lam0 + lam1)
        initial_th = []
        for i in range(n_layers):
            if i % 2 == 0:
                initial_th.append(centre / (4.0 * n_L))
            else:
                initial_th.append(centre / (4.0 * n_H))
        
        x0 = np.array([n_layers] + initial_th)
        
        # Run Nelder-Mead
        try:
            x_opt, score = _nelder_mead(fast_eval, x0, max_iter=100, tol=1e-5, deadline=deadline)
            if score < best_score:
                best_score = score
                n_opt = int(round(x_opt[0]))
                n_opt = max(1, min(n_opt, max_layers))
                thicknesses_opt = x_opt[1:1+n_opt].tolist()
                
                # Ensure constraints
                if all(0 <= t <= max_layer_nm for t in thicknesses_opt) and sum(thicknesses_opt) <= max_total_nm:
                    materials_opt = []
                    for i in range(n_opt):
                        if i % 2 == 0:
                            materials_opt.append('L')
                        else:
                            materials_opt.append('H')
                    best_result = (materials_opt, thicknesses_opt)
        except Exception:
            continue
    
    if best_result is None:
        # Fallback: single quarter-wave layer
        centre = 0.5 * (lam0 + lam1)
        return ["L"], [centre / (4.0 * n_L)]
    
    return best_result

# EVOLVE-BLOCK-END
