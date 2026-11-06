"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def _compute_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band using characteristic matrix method."""
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band = problem["band_nm"]
    
    # 401 equally spaced wavelengths from 400 to 800 nm
    wavelengths = np.linspace(band[0], band[1], 401)
    
    n_dict = {'L': nL, 'H': nH}
    
    reflectances = []
    for lam in wavelengths:
        k = 2 * np.pi / lam
        # Build characteristic matrix
        # For each layer: [[cos(delta), i*sin(delta)/n], [i*n*sin(delta), cos(delta)]]
        # where delta = k * n * d
        
        M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        
        for mat, d in zip(materials, thicknesses):
            n = n_dict[mat]
            delta = k * n * d
            cos_d = np.cos(delta)
            sin_d = np.sin(delta)
            layer_M = np.array([
                [cos_d, 1j * sin_d / n],
                [1j * n * sin_d, cos_d]
            ], dtype=complex)
            M = M @ layer_M
        
        # Reflectance at normal incidence
        # R = | (n0*M11 + n0*M12 - ns*M21 - ns*M22) / (n0*M11 + n0*M12 + ns*M21 + ns*M22) |^2
        numerator = n0 * M[0,0] + n0 * M[0,1] - ns * M[1,0] - ns * M[1,1]
        denominator = n0 * M[0,0] + n0 * M[0,1] + ns * M[1,0] + ns * M[1,1]
        r = numerator / denominator
        R = np.abs(r) ** 2
        reflectances.append(R)
    
    return 100.0 * np.mean(reflectances)

def _quarter_wave_stack(n_layers, problem):
    """Generate a quarter-wave stack design."""
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    
    # For a quarter-wave stack, we alternate L and H starting with L
    # Thickness of each quarter-wave layer at center wavelength
    dL = centre / (4.0 * nL)
    dH = centre / (4.0 * nH)
    
    materials = []
    thicknesses = []
    
    for i in range(n_layers):
        if i % 2 == 0:
            materials.append('L')
            thicknesses.append(dL)
        else:
            materials.append('H')
            thicknesses.append(dH)
    
    return materials, thicknesses

def _nelder_mead(objective, x0, max_iter=200, tol=1e-4):
    """Simple Nelder-Mead implementation using only NumPy."""
    n = len(x0)
    # Initialize simplex
    simplex = [x0]
    for i in range(n):
        new_point = x0.copy()
        new_point[i] += 0.1 if x0[i] != 0 else 1.0
        simplex.append(new_point)
    
    f_vals = [objective(x) for x in simplex]
    
    for _ in range(max_iter):
        # Sort simplex
        order = np.argsort(f_vals)
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]
        
        # Check convergence
        if abs(f_vals[0] - f_vals[-1]) < tol:
            break
            
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        alpha = 1.0
        x_r = centroid + alpha * (centroid - simplex[-1])
        f_r = objective(x_r)
        
        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1] = x_r
            f_vals[-1] = f_r
            continue
            
        # Expansion
        if f_r < f_vals[0]:
            gamma = 2.0
            x_e = centroid + gamma * (x_r - centroid)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
            continue
            
        # Contraction
        rho = 0.5
        x_c = centroid + rho * (simplex[-1] - centroid)
        f_c = objective(x_c)
        
        if f_c < f_vals[-1]:
            simplex[-1] = x_c
            f_vals[-1] = f_c
        else:
            # Shrink
            for i in range(1, n + 1):
                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                f_vals[i] = objective(simplex[i])
                
    return simplex[0]

def _optimize_stack(materials, problem, max_iter=500):
    """Optimize thicknesses for a given material sequence using custom Nelder-Mead."""
    n_layers = len(materials)
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    # Initial guess: quarter-wave thicknesses
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    
    x0 = []
    for mat in materials:
        n = nL if mat == 'L' else nH
        x0.append(centre / (4.0 * n))
    
    def objective(x):
        # Check constraints
        if any(d < 0 or d > max_layer_nm for d in x):
            return 100.0
        if sum(x) > max_total_nm:
            return 100.0
        return _compute_reflectance(materials, x, problem)
    
    # Use custom Nelder-Mead
    x_opt = _nelder_mead(objective, x0, max_iter=max_iter)
    
    # Project back to feasible region
    x_opt = np.clip(x_opt, 0, max_layer_nm)
    total = np.sum(x_opt)
    if total > max_total_nm:
        x_opt = x_opt * (max_total_nm / total)
    
    return list(x_opt)

def design(problem):
    """Design an antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    max_layers = problem["max_layers"]
    best_score = float('inf')
    best_design = None
    
    # Try different numbers of layers
    # Start with smaller stacks and increase
    for n_layers in range(1, min(max_layers + 1, 11)):
        if time.time() > deadline:
            break
        
        # Try alternating L-H stack starting with L
        materials, thicknesses = _quarter_wave_stack(n_layers, problem)
        
        # Check if it fits within constraints
        max_layer_nm = problem["max_layer_nm"]
        max_total_nm = problem["max_total_nm"]
        
        if all(0 <= d <= max_layer_nm for d in thicknesses) and sum(thicknesses) <= max_total_nm:
            score = _compute_reflectance(materials, thicknesses, problem)
            if score < best_score:
                best_score = score
                best_design = (list(materials), list(thicknesses))
        
        # Optimize the thicknesses
        if time.time() > deadline:
            break
        
        opt_thicknesses = _optimize_stack(materials, problem, max_iter=200)
        
        # Check constraints
        if all(0 <= d <= max_layer_nm for d in opt_thicknesses) and sum(opt_thicknesses) <= max_total_nm:
            score = _compute_reflectance(materials, opt_thicknesses, problem)
            if score < best_score:
                best_score = score
                best_design = (list(materials), list(opt_thicknesses))
    
    # If no valid design found, return a simple quarter-wave layer
    if best_design is None:
        centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
        return ["L"], [centre / (4.0 * problem["materials"]["L"])]
    
    return best_design
# EVOLVE-BLOCK-END
