"""Broadband antireflection coating design via multi-start Powell hybrid with dynamic layer count pruning."""
import numpy as np
import time
from scipy.optimize import minimize

def _compute_reflectance(wavelengths_nm, indices, thicknesses_nm, ambient_idx, substrate_idx):
    """
    Compute reflectance (fraction, 0-1) at each wavelength using the characteristic matrix method.
    """
    n_layers = len(indices)
    if n_layers == 0:
        r0 = (ambient_idx - substrate_idx) / (ambient_idx + substrate_idx)
        return np.full(len(wavelengths_nm), r0 * r0)
    
    k = 2.0 * np.pi / wavelengths_nm
    
    M00 = np.ones(len(wavelengths_nm), dtype=complex)
    M01 = np.zeros(len(wavelengths_nm), dtype=complex)
    M10 = np.zeros(len(wavelengths_nm), dtype=complex)
    M11 = np.ones(len(wavelengths_nm), dtype=complex)
    
    for i in range(n_layers):
        n_i = indices[i]
        d_i = thicknesses_nm[i]
        delta = k * n_i * d_i
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        B00 = cos_d
        B01 = 1j * sin_d / n_i
        B10 = 1j * n_i * sin_d
        B11 = cos_d
        
        new_M00 = M00 * B00 + M01 * B10
        new_M01 = M00 * B01 + M01 * B11
        new_M10 = M10 * B00 + M11 * B10
        new_M11 = M10 * B01 + M11 * B11
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    Y_out = substrate_idx
    Y_in = (M10 + M11 * Y_out) / (M00 + M01 * Y_out)
    
    r = (ambient_idx - Y_in) / (ambient_idx + Y_in)
    R = np.abs(r) ** 2
    
    return R

def _objective(thicknesses, indices, wavelengths_nm, ambient_idx, substrate_idx, 
               max_layer_nm, max_total_nm):
    """
    Objective function: mean reflectance in percent.
    thicknesses: array of physical thicknesses
    """
    thicknesses = np.asarray(thicknesses, dtype=float)
    
    # Hard constraint check
    if np.any(thicknesses < -1e-9) or np.any(thicknesses > max_layer_nm + 1e-9):
        return 100.0 + np.sum(np.abs(thicknesses - np.clip(thicknesses, 0, max_layer_nm)))
    
    total = np.sum(thicknesses)
    if total > max_total_nm + 1e-9:
        return 100.0 + 10.0 * (total - max_total_nm)
        
    # Clip for safety in evaluation
    thicknesses = np.clip(thicknesses, 0.0, max_layer_nm)
    
    R = _compute_reflectance(wavelengths_nm, indices, thicknesses, 
                             ambient_idx, substrate_idx)
    
    return np.mean(R) * 100.0

def design(problem):
    """
    Design a broadband antireflection coating.
    Returns: (labels, thicknesses)
    """
    start_time = time.time()
    time_limit = 145.0 # Safety margin for 150s limit
    
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    
    best_result = None
    best_score = np.inf
    
    # Strategy: Multi-start Powell hybrid.
    # We will try different layer counts and different initial patterns.
    # Powell is deterministic and fast per run, allowing many restarts.
    
    # Layer counts to try. We prioritize higher counts for better broadband performance.
    # But we must balance with the dimensionality of the optimization problem.
    # Powell scales well up to ~20 dimensions.
    
    # We'll iterate through a curated list of layer counts.
    # For each count, we'll run several multi-starts.
    
    # Order: Higher counts first as they generally yield lower reflectance if optimized well.
    # But we need to ensure we have time for lower counts if high ones fail.
    # Actually, let's interleave or just go high to low, breaking if time is low.
    
    # Let's try 18, 16, 14, 12, 10, 20, 19, 17, 15, 13, 11
    # Why this order? 18-20 are the "heavy hitters" for broadband.
    # 10-12 are safer/faster.
    
    counts_to_try = [18, 16, 14, 12, 10, 20, 19, 17, 15, 13, 11]
    
    # Pre-compute indices for common patterns? No, labels depend on count.
    # We assume alternating L/H starting with L (air side).
    # This is standard for AR coatings on high-index substrates.
    
    # Alternative pattern: Start with H?
    # For air (1.0) -> Substrate (1.52), with L=1.38, H=2.35.
    # L is closer to air, H is closer to substrate?
    # Usually, the first layer should be low index to match air.
    # So L H L H ... is a good default.
    
    # Let's also try a few different starting patterns for diversity?
    # The prompt says "labels ordered from air side".
    # We can choose the labels.
    # Common patterns:
    # 1. L H L H ... (Start L)
    # 2. H L H L ... (Start H) - Less likely to be optimal for air side, but worth a shot?
    #    Actually, starting with H (2.35) from air (1.0) creates a large impedance mismatch.
    #    Starting with L (1.38) is much better.
    #    So we stick to L-starting patterns.
    
    # However, we can vary the *sequence* slightly?
    # E.g., L L H H L L H H ... ?
    # The problem allows any sequence of L/H.
    # But alternating is the most efficient use of layers for broadband.
    # Let's stick to alternating L/H but try different counts.
    
    # To add diversity, we can also try "superstrate" effects? No, just layers.
    
    # Let's define a helper to generate labels
    def get_labels(n):
        return ['L' if i % 2 == 0 else 'H' for i in range(n)]
    
    # Multi-starts per count
    n_starts = 5
    
    # We will allocate time dynamically.
    # Total time available: time_limit
    # We want to spend most of it on the most promising counts.
    
    # Let's estimate time per Powell run.
    # Powell with 20 vars, 401 wavelengths.
    # Each eval is ~0.1-0.5ms?
    # Powell might take 1-10 seconds per run.
    # 5 starts * 10s = 50s per count.
    # 11 counts * 50s = 550s. Too much.
    
    # We need to be smarter.
    # Reduce starts for higher counts? Or reduce counts?
    # Let's reduce counts to try: [16, 14, 12, 10, 18]
    # And reduce starts to 3.
    
    # Revised plan:
    # Try counts: 16, 14, 12, 10, 18, 20
    # Starts: 4 for each.
    # If time runs out, stop.
    
    counts_to_try = [16, 14, 12, 10, 18, 20]
    n_starts = 4
    
    for n_layers in counts_to_try:
        if n_layers > max_layers:
            continue
            
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed - 5.0
        
        if remaining < 5.0:
            break
        
        labels = get_labels(n_layers)
        indices = [materials[label] for label in labels]
        
        # Bounds for each layer thickness
        bounds = [(0.0, max_layer_nm) for _ in range(n_layers)]
        
        # Initial guesses
        # 1. Quarter wave at center wavelength
        center_wl = 0.5 * (band_nm[0] + band_nm[1])
        init_qw = np.array([center_wl / (4.0 * n) for n in indices])
        if np.sum(init_qw) > max_total_nm:
            init_qw = init_qw * (max_total_nm / np.sum(init_qw))
            
        # 2. Random valid initialization
        # 3. Scaled quarter wave
        # 4. Another random
        
        # We will generate initial points and run Powell on each.
        
        for s in range(n_starts):
            elapsed = time.time() - start_time
            remaining = time_limit - elapsed - 2.0
            if remaining < 2.0:
                break
            
            # Generate initial point
            if s == 0:
                x0 = init_qw.copy()
            else:
                # Random point within bounds, scaled to satisfy total thickness
                x0 = np.random.uniform(0.1, 1.0, size=n_layers) * max_layer_nm
                # Scale to satisfy total thickness constraint
                total = np.sum(x0)
                if total > max_total_nm:
                    x0 = x0 * (max_total_nm / total)
                elif total < max_total_nm * 0.5:
                    # Maybe scale up?
                    x0 = x0 * (max_total_nm / total)
                
                # Ensure individual bounds
                x0 = np.clip(x0, 0.0, max_layer_nm)
                
                # Re-check total after clip
                total = np.sum(x0)
                if total > max_total_nm:
                    x0 = x0 * (max_total_nm / total)
            
            # Define objective for scipy
            def obj_func(x, indices=indices, wavelengths_nm=wavelengths_nm, 
                         ambient_idx=ambient_idx, substrate_idx=substrate_idx,
                         max_layer_nm=max_layer_nm, max_total_nm=max_total_nm):
                return _objective(x, indices, wavelengths_nm, ambient_idx, substrate_idx, 
                                  max_layer_nm, max_total_nm)
            
            # Run Powell
            # maxiter: 1000 is usually enough for Powell if it's converging.
            # We rely on time limit.
            # scipy.optimize.minimize with method='Powell' doesn't have a direct time limit,
            # so we wrap it or just let it run and check time after.
            # But we can't interrupt it easily.
            # We can use a callback to check time? No, Powell doesn't support callbacks well for early stop in all versions.
            # Better: set a reasonable maxiter and rely on the fact that Powell is fast.
            # Or use a wrapper that raises an exception on timeout? Risky.
            
            # Let's set maxiter based on remaining time.
            # Assume 1s per 100 iterations?
            # Let's just set maxiter=200 for safety.
            
            try:
                result = minimize(obj_func, x0, method='Powell', bounds=bounds, 
                                  options={'maxiter': 200, 'maxfev': 5000, 'disp': False})
                
                x_opt = result.x
                f_opt = result.fun
                
                # Post-process: ensure constraints are strictly met
                x_opt = np.clip(x_opt, 0.0, max_layer_nm)
                total = np.sum(x_opt)
                if total > max_total_nm:
                    x_opt = x_opt * (max_total_nm / total)
                
                # Final evaluation
                f_final = obj_func(x_opt)
                
                if f_final < best_score:
                    best_score = f_final
                    best_result = (labels, x_opt.tolist())
                    
            except Exception:
                continue
    
    # Fallback
    if best_result is None:
        labels = ['L']
        thicknesses = [600.0 / (4.0 * materials["L"])]
        return labels, thicknesses
    
    labels, thicknesses = best_result
    
    # Final constraint enforcement
    thicknesses = np.clip(thicknesses, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
    
    return labels, thicknesses.tolist()
