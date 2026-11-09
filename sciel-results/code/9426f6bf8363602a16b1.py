"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def _reflectance_spectrum(thicknesses, materials, ambient_index, substrate_index, band_nm):
    """
    Compute mean reflectance (in percent) over the band using the characteristic-matrix method.
    thicknesses: list of physical thicknesses in nm
    materials: dict mapping 'L'->n_L, 'H'->n_H
    Returns mean R in percent.
    """
    n_L = materials['L']
    n_H = materials['H']
    
    # Wavelengths: 401 equally spaced points from 400 to 800 nm
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    # Refractive indices for each layer
    # We need to know the layer sequence. But wait, the input to this helper 
    # only has thicknesses. The caller must pass the sequence or we assume a fixed sequence?
    # In the design function, we fix the sequence. So this helper should take the sequence too 
    # or be integrated. Let's make it take the full stack definition.
    pass

def _compute_reflectance(stack_labels, thicknesses, materials, ambient_index, substrate_index, wavelengths):
    """
    Compute reflectance at given wavelengths for a specific stack.
    Returns array of R values (0 to 1).
    """
    n_L = materials['L']
    n_H = materials['H']
    
    # Pre-compute optical thicknesses? No, depends on wavelength.
    # M = product of layer matrices
    # Each layer matrix:
    # [[cos(delta), i*sin(delta)/n], [i*n*sin(delta), cos(delta)]]
    # where delta = 2*pi*n*d / lambda
    
    n_layers = len(stack_labels)
    n_indices = np.array([materials[label] for label in stack_labels])
    d = np.array(thicknesses)
    
    R = np.zeros(len(wavelengths))
    
    for i, lam in enumerate(wavelengths):
        # Start with identity matrix
        M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        
        for j in range(n_layers):
            n_j = n_indices[j]
            d_j = d[j]
            delta_j = 2.0 * math.pi * n_j * d_j / lam
            
            c = math.cos(delta_j)
            s = math.sin(delta_j)
            
            # Matrix for layer j
            M_j = np.array([
                [c, 1j * s / n_j],
                [1j * n_j * s, c]
            ], dtype=complex)
            
            M = M @ M_j
        
        # Substrate matrix? No, the characteristic matrix M is for the coating stack only.
        # The boundary condition at the substrate interface is handled by the admittance.
        # Y_sub = substrate_index (real)
        # The characteristic matrix relates the fields at the air-coating interface to the substrate.
        # Actually, the standard formula is:
        # [E]   = M * [E_sub]
        # [H]       [H_sub]
        # Where E_sub and H_sub are related by Y_sub = n_sub (for normal incidence, s-polarization? 
        # Wait, for normal incidence, s and p are identical. The admittance is n).
        # The ratio of E to H at the substrate is 1/n_sub.
        # So [E_sub, H_sub]^T = [1, 1/n_sub]^T * E0_sub
        # Then [E_air, H_air]^T = M * [1, 1/n_sub]^T * E0_sub
        # The admittance at the air side is Y_air = H_air / E_air
        # Y_air = (M[1,0] + M[1,1]/n_sub) / (M[0,0] + M[0,1]/n_sub)
        # Reflection coefficient r = (n_air - Y_air) / (n_air + Y_air)
        # R = |r|^2
        
        Y_sub = 1.0 / substrate_index
        # Vector at substrate: [1, Y_sub]
        # Multiply by M
        E_air = M[0, 0] + M[0, 1] * Y_sub
        H_air = M[1, 0] + M[1, 1] * Y_sub
        
        Y_air = H_air / E_air
        
        n_air = ambient_index
        r = (n_air - Y_air) / (n_air + Y_air)
        R[i] = np.abs(r)**2
        
    return R

def _mean_reflectance_percent(stack_labels, thicknesses, materials, ambient_index, substrate_index, band_nm):
    """
    Compute mean reflectance in percent.
    """
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    R = _compute_reflectance(stack_labels, thicknesses, materials, ambient_index, substrate_index, wavelengths)
    return 100.0 * np.mean(R)

def design(problem):
    """
    Design a stack of at most 20 layers minimizing mean reflectance.
    """
    start_time = time.time()
    deadline = start_time + 150.0  # Safety margin for 150s limit
    
    materials = problem["materials"]
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    n_L = materials['L']
    n_H = materials['H']
    
    # Strategy:
    # 1. Fix a good layer sequence (graded index profile).
    #    A common good profile for broadband AR is:
    #    L H L H L H ... or H L H L ...
    #    Since n_air (1.0) < n_sub (1.52), and we have L=1.38, H=2.35.
    #    Ideally, the index should step down from substrate to air.
    #    Substrate: 1.52
    #    If we start with L (1.38), it's closer to 1.52 than H (2.35).
    #    So starting with L on the substrate side is good.
    #    But the stack is ordered from air to substrate.
    #    So the last layer is closest to the substrate.
    #    Let's try a sequence that alternates, starting with L (air side) or H?
    #    Actually, a "graded" profile means the index changes gradually.
    #    Air (1.0) -> ... -> Substrate (1.52)
    #    If we use L (1.38) and H (2.35), H is quite high.
    #    Maybe L H L H ... ending with L near the substrate?
    #    Let's try: L, H, L, H, ..., L (odd number of layers, ending in L)
    #    Or H, L, H, L, ..., H?
    #    Let's test a few fixed sequences and optimize thicknesses for each.
    
    # Candidate sequences:
    # 1. L H L H ... (start L, end depends on N)
    # 2. H L H L ... (start H)
    # 3. L H H L L H H L ... (double layers)
    # 4. L L H H L L H H ...
    
    # Let's define a few promising sequences.
    # Sequence A: Alternating L, H, L, H, ...
    # Sequence B: Alternating H, L, H, L, ...
    # Sequence C: L, L, H, H, L, L, H, H, ...
    # Sequence D: H, H, L, L, H, H, L, L, ...
    
    # We will optimize the thicknesses for a fixed number of layers N.
    # Let's try N = 4, 6, 8, 10, 12, 14, 16, 18, 20.
    
    # To save time, we'll use Nelder-Mead or a simple local search.
    # Since we have up to 20 variables, Nelder-Mead is feasible if fast.
    # But 401 wavelengths * 20 layers * matrix mult is ~ 401*20 = 8020 matrix mults per eval.
    # NumPy is fast. Let's estimate: 8020 * (2x2 complex mult) is very fast.
    # Let's say 1 eval takes 1ms. 1000 evals = 1s.
    # We have 150s. So we can do ~100,000 evals.
    
    # Let's implement a robust local optimizer.
    # We'll try multiple starting points for each sequence.
    
    best_score = float('inf')
    best_labels = None
    best_thick = None
    
    # Define sequences
    def gen_seq(pattern, N):
        # pattern is a string like "LH" or "LLHH"
        seq = []
        for i in range(N):
            seq.append(pattern[i % len(pattern)])
        return seq
    
    patterns = ["LH", "HL", "LLHH", "HHL L", "LHH", "HLL"]
    # Remove spaces
    patterns = [p.replace(" ", "") for p in patterns]
    
    # We'll iterate over N from 4 to 20, step 2.
    # For each N, try each pattern.
    # For each (N, pattern), run a local optimization.
    
    # To speed up, we can use a coarse grid first?
    # Or just use Nelder-Mead with a reasonable initial guess.
    # Initial guess: quarter wave at center wavelength for each layer.
    center_wl = 0.5 * (band_nm[0] + band_nm[1])
    
    # Let's use scipy.optimize.minimize (Nelder-Mead) if available, else custom.
    # Standard library and NumPy/SciPy are available.
    
    from scipy.optimize import minimize
    
    for N in range(4, max_layers + 1, 2):
        if time.time() > deadline - 10:
            break
            
        for pat in patterns:
            labels = gen_seq(pat, N)
            
            # Initial thicknesses: quarter wave
            init_thick = []
            for label in labels:
                n = materials[label]
                d = center_wl / (4.0 * n)
                init_thick.append(d)
            
            # Bounds: [0, max_layer_nm]
            bounds = [(0.0, max_layer_nm)] * N
            
            # Objective function
            def objective(thick):
                # Check total thickness constraint
                if sum(thick) > max_total_nm:
                    return 1000.0  # Penalty
                # Check individual thickness
                # Bounds handle individual, but let's be safe
                for t in thick:
                    if t < 0 or t > max_layer_nm:
                        return 1000.0
                return _mean_reflectance_percent(labels, thick, materials, ambient_index, substrate_index, band_nm)
            
            # Run optimizer
            try:
                res = minimize(objective, init_thick, method='Nelder-Mead', 
                               bounds=bounds, 
                               options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 1000, 'maxfev': 500})
                
                if res.success:
                    score = res.fun
                    if score < best_score:
                        best_score = score
                        best_labels = labels
                        best_thick = list(res.x)
                        
            except Exception:
                pass
                
            # Check time
            if time.time() > deadline - 10:
                break
                
    # If no optimizer found a good solution, fallback to a simple analytical guess
    if best_labels is None:
        # Single layer L
        labels = ["L"]
        thick = [center_wl / (4.0 * n_L)]
        best_labels = labels
        best_thick = thick
        
    # Final check on constraints
    if sum(best_thick) > max_total_nm:
        # Scale down
        scale = max_total_nm / sum(best_thick)
        best_thick = [t * scale for t in best_thick]
        
    for i in range(len(best_thick)):
        if best_thick[i] > max_layer_nm:
            best_thick[i] = max_layer_nm
        if best_thick[i] < 0:
            best_thick[i] = 0
            
    return best_labels, best_thick

# EVOLVE-BLOCK-END
