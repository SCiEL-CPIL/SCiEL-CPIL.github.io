"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _reflectance_vectorized(n_layers, thicknesses, materials, n_air, n_sub, wavelengths):
    """
    Compute normal-incidence reflectance (fraction) for a stack using the
    characteristic-matrix method, vectorized over wavelengths.
    
    n_layers: int
    thicknesses: array-like of shape (n_layers,) in nm
    materials: dict mapping 'L','H' to indices; but here we assume thicknesses
               are paired with indices passed as an array? 
               Wait, the interface requires returning labels and thicknesses.
               For internal optimization, we need to know the indices.
               Let's pass an array of indices corresponding to the layers.
    """
    # n_layers is the count
    # thicknesses: (n_layers,)
    # indices: (n_layers,)
    # wavelengths: (n_w,)
    
    # Convert to arrays
    t = np.asarray(thicknesses, dtype=float)
    n_idx = np.asarray(indices, dtype=float)
    lam = np.asarray(wavelengths, dtype=float)
    
    # Optical thickness in radians: 2 * pi * n * d / lambda
    # Shape: (n_layers, n_w)
    delta = 2.0 * np.pi * n_idx[:, None] * t[:, None] / lam[None, :]
    
    # Cosine of delta
    c = np.cos(delta)
    s = np.sin(delta)
    
    # Initialize matrices
    # M = [[cos(delta), i*sin(delta)/n], [i*n*sin(delta), cos(delta)]]
    # For normal incidence, the characteristic matrix for a layer is:
    # M = [[cos(delta), 1j * sin(delta) / n],
    #      [1j * n * sin(delta), cos(delta)]]
    
    # We multiply M_1 * M_2 * ... * M_N
    # Start with identity
    # To vectorize, we can iterate layers and update the matrix product.
    # M_total = M_1 @ M_2 @ ... @ M_N
    # Each M_k is 2x2.
    # We can store M_total as a 2x2 matrix where each element is an array of shape (n_w,)
    
    M_total = np.eye(2, dtype=complex)
    M_total = M_total[:, :, None] * np.ones((1, 1, len(lam)), dtype=complex)
    
    for k in range(n_layers):
        n_k = n_idx[k]
        # Construct M_k
        # M_k[0,0] = c
        # M_k[0,1] = 1j * s / n_k
        # M_k[1,0] = 1j * n_k * s
        # M_k[1,1] = c
        
        M_k = np.zeros((2, 2, len(lam)), dtype=complex)
        M_k[0, 0] = c[k]
        M_k[0, 1] = 1j * s[k] / n_k
        M_k[1, 0] = 1j * n_k * s[k]
        M_k[1, 1] = c[k]
        
        # M_total = M_total @ M_k
        # Element-wise multiplication with broadcasting
        # (2,2,n) @ (2,2,n) is not standard matmul because of the last axis.
        # We can do it manually or use einsum.
        # M_total_new[i,j] = sum_k M_total[i,k] * M_k[k,j]
        M_total_new = np.zeros((2, 2, len(lam)), dtype=complex)
        for i in range(2):
            for j in range(2):
                M_total_new[i, j] = M_total[i, 0] * M_k[0, j] + M_total[i, 1] * M_k[1, j]
        M_total = M_total_new
    
    # Now M_total is 2x2 matrix of arrays (n_w,)
    # Let M11, M12, M21, M22 be the elements
    M11 = M_total[0, 0]
    M12 = M_total[0, 1]
    M21 = M_total[1, 0]
    M22 = M_total[1, 1]
    
    # Refractive index of ambient and substrate
    n0 = n_air
    ns = n_sub
    
    # Reflectance formula:
    # r = (n0 * M11 + n0 * ns * M12 - M21 - ns * M22) / (n0 * M11 + n0 * ns * M12 + M21 + ns * M22)
    # Wait, standard formula:
    # Y = (M11 + M12 / ns) / (M21 + M22 / ns)  <-- This is for admittance Y
    # Actually, let's use the standard characteristic matrix relation:
    # [B, C]^T = M * [1, 1/ns]^T ? No.
    
    # Let's stick to the standard derivation:
    # The characteristic matrix M is defined such that:
    # E = M11 * E_s + M12 * H_s
    # H = M21 * E_s + M22 * H_s
    # Where E_s, H_s are fields at the substrate interface.
    # At the substrate, H_s = E_s / ns (for forward wave? No, H = E/n for forward)
    # Wait, H = (n / (i*omega*mu)) * k x E. For normal incidence, H = E/n.
    # So H_s = E_s / ns.
    # Therefore:
    # E = M11 * E_s + M12 * (E_s / ns) = E_s * (M11 + M12/ns)
    # H = M21 * E_s + M22 * (E_s / ns) = E_s * (M21 + M22/ns)
    # Admittance Y = H / E = (M21 + M22/ns) / (M11 + M12/ns)
    
    # Reflection coefficient r = (Y - n0) / (Y + n0)
    # Or equivalently:
    # r = (n0 * (M11 + M12/ns) - (M21 + M22/ns)) / (n0 * (M11 + M12/ns) + (M21 + M22/ns))
    
    Y = (M21 + M22 / ns) / (M11 + M12 / ns)
    r = (Y - n0) / (Y + n0)
    
    # Reflectance R = |r|^2
    R = np.abs(r) ** 2
    return R

def _mean_reflectance(labels, thicknesses, problem):
    """Calculate mean reflectance in percent."""
    n_air = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    materials = problem["materials"]
    band = problem["band_nm"]
    
    # Generate wavelengths
    wavelengths = np.linspace(band[0], band[1], 401)
    
    # Build index array
    n_idx = np.array([materials[l] for l in labels])
    
    # Calculate reflectance
    R = _reflectance_vectorized(len(labels), thicknesses, n_idx, n_air, n_sub, wavelengths)
    
    # Mean reflectance in percent
    return np.mean(R) * 100.0

def _is_valid(labels, thicknesses, problem):
    """Check constraints."""
    if len(labels) != len(thicknesses):
        return False
    if len(labels) > problem["max_layers"]:
        return False
    if sum(thicknesses) > problem["max_total_nm"] + 1e-9:
        return False
    for t in thicknesses:
        if t < -1e-9 or t > problem["max_layer_nm"] + 1e-9:
            return False
    return True

def design(problem):
    """
    Design an antireflection coating using Nelder-Mead optimization.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n_air = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    materials = problem["materials"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Pre-compute wavelengths
    wavelengths = np.linspace(band[0], band[1], 401)
    
    # Strategy: Start with a 4-layer stack: L H L H
    # This is a common broadband AR design.
    # Or maybe H L H L?
    # Let's try L H L H.
    # Initial guess: quarter wave at center wavelength for each layer?
    # Center wavelength = 600 nm
    # L: 600 / (4 * 1.38) ~ 108 nm
    # H: 600 / (4 * 2.35) ~ 63.8 nm
    # Total ~ 343 nm. Well within 1000 nm.
    
    # Let's try a 6-layer stack for better broadband performance: L H L H L H
    # Or maybe just optimize a 4-layer stack first, it's simpler.
    # Let's use 4 layers: L, H, L, H
    labels = ['L', 'H', 'L', 'H']
    n_layers = len(labels)
    n_idx = np.array([materials[l] for l in labels])
    
    # Initial thicknesses
    center_wl = 0.5 * (band[0] + band[1])
    t0 = center_wl / (4.0 * n_idx)
    # Ensure t0 is within bounds
    t0 = np.clip(t0, 0.0, max_layer_nm)
    
    # Nelder-Mead implementation (simple version)
    # We will minimize _mean_reflectance
    
    def objective(x):
        # x: array of thicknesses
        # Ensure non-negative
        x = np.clip(x, 0.0, max_layer_nm)
        # Check total thickness
        if np.sum(x) > max_total_nm:
            return 100.0  # Penalty
        return _mean_reflectance(labels, x, problem)
    
    # Initialize simplex
    # Simplex size: 10% of max thickness or something?
    # Let's use 10 nm perturbation.
    simplex = np.zeros((n_layers + 1, n_layers))
    simplex[0] = t0
    for i in range(1, n_layers + 1):
        simplex[i] = t0.copy()
        simplex[i][i-1] += 10.0  # Perturb one dimension at a time
    
    # Evaluate initial simplex
    f_simplex = np.array([objective(x) for x in simplex])
    
    # Nelder-Mead parameters
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    best_x = simplex[0]
    best_f = f_simplex[0]
    
    max_iter = 1000
    
    for i_iter in range(max_iter):
        if time.time() > deadline:
            break
            
        # Sort simplex by function value
        order = np.argsort(f_simplex)
        simplex = simplex[order]
        f_simplex = f_simplex[order]
        
        # Update best
        if f_simplex[0] < best_f:
            best_f = f_simplex[0]
            best_x = simplex[0].copy()
            
        # Centroid of all points except the worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = objective(xr)
        
        if fr < f_simplex[0]:
            # Expansion
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                f_simplex[-1] = fe
            else:
                simplex[-1] = xr
                f_simplex[-1] = fr
        elif fr < f_simplex[-2]:
            # Reflection accepted
            simplex[-1] = xr
            f_simplex[-1] = fr
        else:
            # Contraction
            if fr < f_simplex[-1]:
                # Outer contraction
                xc = centroid + rho * (simplex[-1] - centroid)
                fc = objective(xc)
                if fc <= fr:
                    simplex[-1] = xc
                    f_simplex[-1] = fc
                else:
                    # Shrink
                    for j in range(1, n_layers + 1):
                        simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                        f_simplex[j] = objective(simplex[j])
            else:
                # Inner contraction
                xc = centroid - rho * (centroid - simplex[-1])
                fc = objective(xc)
                if fc <= f_simplex[-1]:
                    simplex[-1] = xc
                    f_simplex[-1] = fc
                else:
                    # Shrink
                    for j in range(1, n_layers + 1):
                        simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                        f_simplex[j] = objective(simplex[j])
        
        # Check convergence
        if np.max(f_simplex) - np.min(f_simplex) < 1e-6:
            break
    
    # Final result
    t_opt = best_x
    # Clip to ensure valid
    t_opt = np.clip(t_opt, 0.0, max_layer_nm)
    
    # Check total thickness constraint
    if np.sum(t_opt) > max_total_nm:
        # Scale down
        scale = max_total_nm / np.sum(t_opt)
        t_opt = t_opt * scale
    
    return labels, list(t_opt)
# EVOLVE-BLOCK-END
