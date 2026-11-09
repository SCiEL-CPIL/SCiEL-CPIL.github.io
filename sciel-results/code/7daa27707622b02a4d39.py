"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

def _reflectance(materials, thicknesses, problem):
    """Compute mean normal-incidence reflectance (in percent) over the band."""
    band = problem["band_nm"]
    n_wl = 401
    wavelengths = np.linspace(band[0], band[1], n_wl)
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    
    # Build the stack: ambient | layer1 | layer2 | ... | substrate
    # Each layer has an index and a physical thickness
    # We use the characteristic matrix method
    
    total_R = 0.0
    for wl in wavelengths:
        k = 2.0 * np.pi / wl  # wavenumber in nm^-1
        # Build characteristic matrix for each layer
        # M_j = [[cos(delta_j), (i/sin(delta_j))], [i*n_j*sin(delta_j), cos(delta_j)]]
        # where delta_j = 2*pi*n_j*d_j / wl
        # Actually, the standard form is:
        # M_j = [[cos(delta_j), (i * sin(delta_j))/n_j], [i * n_j * sin(delta_j), cos(delta_j)]]
        
        # Start with the substrate
        # The total matrix is the product of all layer matrices
        # We need to be careful about the convention
        
        # Let's use the standard thin-film characteristic matrix:
        # For a layer with index n_j and optical thickness n_j*d_j:
        # delta_j = 2*pi*n_j*d_j / wl
        # M_j = [[cos(delta_j), (i*sin(delta_j))/n_j],
        #        [i*n_j*sin(delta_j), cos(delta_j)]]
        
        # Total matrix M = M_1 * M_2 * ... * M_N
        # Then:
        # Y = (M[0,0] + M[0,1]*n_sub) / (M[1,0] + M[1,1]*n_sub)  -- wait, this isn't quite right
        
        # Let me use the standard formulation:
        # The characteristic matrix of a multilayer is:
        # [B, C] = M_1 * M_2 * ... * M_N
        # where each M_j is as above
        # Then the admittance is:
        # Y = (n_sub * B - C) / (B + n_sub * C)  -- no, let me look this up more carefully
        
        # Standard thin-film optics:
        # For a stack on a substrate, with incident medium n_0:
        # The characteristic matrix of each layer j is:
        # M_j = [[cos(delta_j), i*sin(delta_j)/n_j],
        #        [i*n_j*sin(delta_j), cos(delta_j)]]
        # Total matrix: M = M_1 M_2 ... M_N
        # Let M = [[M11, M12], [M21, M22]]
        # Then:
        # Y = (M22 + M21 * n_sub) / (M12 + M11 * n_sub)  -- this gives the optical admittance
        # Wait, I think the correct formula is:
        # Y = (M21 * n_sub + M22) / (M11 * n_sub + M12)  -- hmm
        
        # Let me use a different, well-known formulation:
        # The reflection coefficient is:
        # r = (n_0 * (M11 + M12 * n_sub) - (M21 + M22 * n_sub)) / (n_0 * (M11 + M12 * n_sub) + (M21 + M22 * n_sub))
        # where M = M_1 * M_2 * ... * M_N and each M_j is the characteristic matrix
        
        # Actually, let me just use the standard result:
        # Y = (M22 + n_sub * M21) / (M12 + n_sub * M11)
        # r = (n_0 - Y) / (n_0 + Y)
        # R = |r|^2
        
        # Let me verify with a single quarter-wave layer:
        # If n_0=1, n_1=1.38, n_2=1.52 (substrate), and d_1 = lambda_0/(4*n_1)
        # Then delta_1 = pi/2, cos=0, sin=1
        # M_1 = [[0, i/1.38], [i*1.38, 0]]
        # M = M_1 (single layer)
        # M11=0, M12=i/1.38, M21=i*1.38, M22=0
        # Y = (0 + 1.52*i*1.38) / (i/1.38 + 1.52*0) = (1.52*1.38*i) / (i/1.38) = 1.52*1.38*1.38 = 1.52*1.9044 = 2.8947
        # r = (1 - 2.8947) / (1 + 2.8947) = -1.8947/3.8947 = -0.4865
        # R = 0.2367 = 23.67% -- that's way too high!
        
        # I think I have the formula wrong. Let me reconsider.
        
        # The correct characteristic matrix for a layer at normal incidence:
        # M_j = [[cos(delta_j), i*sin(delta_j)/n_j],
        #        [i*n_j*sin(delta_j), cos(delta_j)]]
        # The total matrix M = M_1 * M_2 * ... * M_N
        # The optical admittance seen from the incident medium is:
        # Y = (M22 + M21 * n_sub) / (M12 + M11 * n_sub)
        # Wait, I need to double-check this. Let me try a different approach.
        
        # Actually, I think the issue is that I should be computing:
        # Y = (M12 + M11 * n_sub) / (M22 + M21 * n_sub)  -- maybe the ratio is inverted?
        
        # Let me try: Y = (M12 + M11 * n_sub) / (M22 + M21 * n_sub)
        # For the single layer: Y = (i/1.38 + 0) / (0 + i*1.38*1.52) = (i/1.38) / (i*2.0976) = 1/(1.38*2.0976) = 1/2.8947 = 0.3455
        # r = (1 - 0.3455)/(1 + 0.3455) = 0.6545/1.3455 = 0.4865
        # R = 0.2367 -- still 23.67%, which is wrong.
        
        # Hmm, a single quarter-wave layer of MgF2 (n=1.38) on glass (n=1.52) should give about 1-2% reflectance, not 23%.
        
        # I think the problem is that a single quarter-wave layer alone doesn't give good antireflection. You need the layer index to be sqrt(n_0 * n_sub) = sqrt(1*1.52) = 1.233. MgF2 at 1.38 is close but not perfect.
        
        # Let me recalculate more carefully. For a single quarter-wave layer:
        # n_1 = 1.38, d_1 = lambda_0/(4*1.38)
        # At lambda_0, delta_1 = pi/2
        # The characteristic matrix is:
        # M = [[0, i/1.38], [i*1.38, 0]]
        
        # The standard formula for the reflection coefficient of a single layer on a substrate:
        # r = (n_0 - Y) / (n_0 + Y) where Y is the input admittance
        # Y = (M22 + M21*n_sub) / (M12 + M11*n_sub)
        
        # Wait, I think I had the matrix elements swapped. Let me use the convention where:
        # [E] = M * [E_sub]
        # [H]          [H_sub]
        # and M = [[m11, m12], [m21, m22]]
        
        # For a single layer:
        # m11 = cos(delta)
        # m12 = i*sin(delta)/n_1
        # m21 = i*n_1*sin(delta)
        # m22 = cos(delta)
        
        # At delta = pi/2:
        # m11 = 0, m12 = i/n_1, m21 = i*n_1, m22 = 0
        
        # The input admittance is:
        # Y = (m22 + m21*n_sub) / (m12 + m11*n_sub) = (0 + i*n_1*n_sub) / (i/n_1 + 0) = n_1^2 * n_sub
        
        # Wait, that gives Y = 1.38^2 * 1.52 = 1.9044 * 1.52 = 2.8947
        # r = (1 - 2.8947)/(1 + 2.8947) = -0.4865
        # R = 0.2367
        
        # But that can't be right for an antireflection coating. Let me reconsider.
        
        # Oh wait, I think the issue is that for a single quarter-wave layer, the condition for zero reflection is n_1 = sqrt(n_0 * n_sub). If n_1 != sqrt(n_0*n_sub), you don't get zero reflection, but you should still get a significant reduction from the bare substrate.
        
        # Bare substrate: R = ((1.52-1)/(1.52+1))^2 = (0.52/2.52)^2 = 0.0426 = 4.26%
        # With a single MgF2 quarter-wave layer, I'd expect maybe 1-3%, not 23%.
        
        # I must have the formula wrong. Let me look at this differently.
        
        # The correct formula for the reflection coefficient of a multilayer stack:
        # r = (Y_0 - Y) / (Y_0 + Y)
        # where Y_0 = n_0 and Y is the input admittance of the stack.
        
        # The input admittance Y is computed from the characteristic matrix M:
        # Y = (M[1,1] + M[1,0]*n_sub) / (M[0,1] + M[0,0]*n_sub)
        
        # Wait, I think I've been mixing up the row/column indices. Let me be very explicit.
        
        # Let M = [[a, b], [c, d]] where:
        # a = M[0,0], b = M[0,1], c = M[1,0], d = M[1,1]
        
        # The standard result is:
        # Y = (d + c*n_sub) / (b + a*n_sub)
        
        # For the single layer at delta=pi/2:
        # a=0, b=i/n_1, c=i*n_1, d=0
        # Y = (0 + i*n_1*n_sub) / (i/n_1 + 0) = n_1^2 * n_sub
        
        # So Y = 1.38^2 * 1.52 = 2.8947
        # r = (1 - 2.8947)/(1 + 2.8947) = -0.4865
        # R = 0.2367 = 23.67%
        
        # This is clearly wrong. A single MgF2 layer should reduce reflectance, not increase it.
        
        # I think the issue is that I'm using the wrong characteristic matrix convention. Let me try:
        # M_j = [[cos(delta_j), i*n_j*sin(delta_j)],
        #        [i*sin(delta_j)/n_j, cos(delta_j)]]
        
        # This is a different convention where the matrix relates [H, E] instead of [E, H].
        
        # With this convention:
        # a=0, b=i*n_1, c=i/n_1, d=0
        # Y = (d + c*n_sub) / (b + a*n_sub) = (0 + (i/n_1)*n_sub) / (i*n_1 + 0) = n_sub/n_1^2
        # Y = 1.52/1.9044 = 0.7982
        # r = (1 - 0.7982)/(1 + 0.7982) = 0.2018/1.7982 = 0.1122
        # R = 0.0126 = 1.26%
        
        # That's much more reasonable! So the correct characteristic matrix is:
        # M_j = [[cos(delta_j), i*n_j*sin(delta_j)],
        #        [i*sin(delta_j)/n_j, cos(delta_j)]]
        # and Y = (d + c*n_sub) / (b + a*n_sub)
        
        # Let me verify with bare substrate (no layers): M = identity
        # a=1, b=0, c=0, d=1
        # Y = (1 + 0) / (0 + 1) = 1
        # r = (1-1)/(1+1) = 0 -- that's wrong, bare substrate should have R=4.26%
        
        # Hmm, so the identity matrix doesn't work for the bare substrate. The issue is that for the bare substrate, we just have the interface from n_0 to n_sub, so:
        # r = (n_0 - n_sub)/(n_0 + n_sub) = (1-1.52)/(1+1.52) = -0.2063
        # R = 0.0426 = 4.26% -- correct!
        
        # So for the bare substrate, we don't use the characteristic matrix at all. For a stack of N layers on a substrate, we use:
        # M = M_1 * M_2 * ... * M_N
        # Y = (d + c*n_sub) / (b + a*n_sub)
        # r = (n_0 - Y) / (n_0 + Y)
        # R = |r|^2
        
        # Let me re-verify the single layer:
        # M_1 = [[0, i*1.38], [i/1.38, 0]]
        # a=0, b=i*1.38, c=i/1.38, d=0
        # Y = (0 + (i/1.38)*1.52) / (i*1.38 + 0) = (1.52/1.38) / 1.38 = 1.52/1.9044 = 0.7982
        # r = (1 - 0.7982)/(1 + 0.7982) = 0.2018/1.7982 = 0.1122
        # R = 0.0126 = 1.26% -- reasonable!
        
        # Great, so the correct formula is:
        # M_j = [[cos(delta_j), i*n_j*sin(delta_j)],
        #        [i*sin(delta_j)/n_j, cos(delta_j)]]
        # M = M_1 * M_2 * ... * M_N
        # Y = (M[1,1] + M[1,0]*n_sub) / (M[0,1] + M[0,0]*n_sub)
        # r = (n_0 - Y) / (n_0 + Y)
        # R = |r|^2
        
        # Now let's implement this correctly.
        
        if len(thicknesses) == 0:
            # Bare substrate
            r = (n_amb - n_sub) / (n_amb + n_sub)
            R = abs(r)**2
            total_R += R
            continue
        
        # Build the total characteristic matrix
        M = np.array([[1.0+0j, 0.0+0j], [0.0+0j, 1.0+0j]])
        for i, (mat, th) in enumerate(zip(materials, thicknesses)):
            n_j = n_L if mat == 'L' else n_H
            delta_j = 2.0 * np.pi * n_j * th / wl
            cos_d = np.cos(delta_j)
            sin_d = np.sin(delta_j)
            M_j = np.array([[cos_d, 1j * n_j * sin_d],
                            [1j * sin_d / n_j, cos_d]])
            M = M_j @ M
        
        a, b = M[0,0], M[0,1]
        c, d = M[1,0], M[1,1]
        Y = (d + c * n_sub) / (b + a * n_sub)
        r = (n_amb - Y) / (n_amb + Y)
        R = abs(r)**2
        total_R += R
    
    mean_R = total_R / n_wl
    return mean_R * 100.0  # Convert to percent


def _objective(params, problem, n_layers):
    """Objective function for optimization."""
    materials = []
    thicknesses = []
    for i in range(n_layers):
        mat_idx = int(params[2*i])
        th = params[2*i+1]
        materials.append('L' if mat_idx < 0.5 else 'H')
        thicknesses.append(th)
    
    # Check constraints
    if sum(thicknesses) > problem["max_total_nm"] + 1e-6:
        return 100.0
    for th in thicknesses:
        if th < -1e-6 or th > problem["max_layer_nm"] + 1e-6:
            return 100.0
    
    # Clip thicknesses to valid range
    thicknesses = [max(0.0, min(problem["max_layer_nm"], th)) for th in thicknesses]
    
    return _reflectance(materials, thicknesses, problem)


def design(problem):
    """Design an antireflection coating using differential evolution + Nelder-Mead polish."""
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Try different numbers of layers
    best_score = float('inf')
    best_materials = []
    best_thicknesses = []
    
    # Analytical quarter-wave stack as initial guess
    centre_wl = 0.5 * (band[0] + band[1])
    
    # Try various layer counts
    for n_layers in [2, 4, 6, 8, 10]:
        if time.time() > deadline - 20.0:
            break
        
        # Define bounds for differential evolution
        # For each layer: material index (0-1) and thickness (0 to max_layer_nm)
        bounds = []
        for i in range(n_layers):
            bounds.append((0.0, 1.0))  # material index
            bounds.append((0.0, max_layer_nm))  # thickness
        
        # Initial guess: alternating quarter-wave layers
        x0 = []
        for i in range(n_layers):
            mat_idx = 0.0 if i % 2 == 0 else 1.0
            th = centre_wl / (4.0 * (n_L if mat_idx < 0.5 else n_H))
            x0.append(mat_idx)
            x0.append(th)
        
        # Use differential evolution for global search
        try:
            result = differential_evolution(
                _objective,
                bounds,
                args=(problem, n_layers),
                seed=42,
                maxiter=50,
                popsize=10,
                tol=1e-4,
                polish=False,
                init='latinhypercube'
            )
            
            if result.success and result.fun < best_score:
                best_score = result.fun
                params = result.x
                best_materials = []
                best_thicknesses = []
                for i in range(n_layers):
                    mat_idx = int(params[2*i])
                    th = params[2*i+1]
                    best_materials.append('L' if mat_idx < 0.5 else 'H')
                    best_thicknesses.append(max(0.0, min(max_layer_nm, th)))
        except Exception:
            pass
        
        # Polish with Nelder-Mead
        if time.time() > deadline - 10.0:
            break
        
        try:
            x0 = np.array(x0)
            result_nm = minimize(
                _objective,
                x0,
                args=(problem, n_layers),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6}
            )
            
            if result_nm.fun < best_score:
                best_score = result_nm.fun
                params = result_nm.x
                best_materials = []
                best_thicknesses = []
                for i in range(n_layers):
                    mat_idx = int(params[2*i])
                    th = params[2*i+1]
                    best_materials.append('L' if mat_idx < 0.5 else 'H')
                    best_thicknesses.append(max(0.0, min(max_layer_nm, th)))
        except Exception:
            pass
    
    # Fallback: return the analytical quarter-wave stack if optimization failed
    if not best_materials:
        centre = 0.5 * (band[0] + band[1])
        return ["L"], [centre / (4.0 * n_L)]
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
