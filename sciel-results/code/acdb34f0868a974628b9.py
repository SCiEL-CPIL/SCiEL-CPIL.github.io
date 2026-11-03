import numpy as np
import time
import sys

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)

# Precompute for fast evaluation
# Using a fine grid to accurately capture the peak of the autoconvolution
# which is piecewise linear for step functions.
FINE_N = 8000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
# Map fine grid points to coarse step function intervals
# (x + 0.25) / W gives the index in [0, N)
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
    """
    Computes the normalized maximum of the autoconvolution.
    h: array of step heights (length N)
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    
    # Check for trivial zero function
    if np.sum(h) < 1e-15:
        return float('inf')
        
    # Expand to fine grid
    f = h[CELL]
    
    # Compute autoconvolution via FFT
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    
    # Compute L2 norm squared
    l2_sq = np.sum(f * f) * DX
    if l2_sq < 1e-15:
        return float('inf')
        
    # The metric is max(conv) / L2_norm? 
    # Let's re-read the task: "minimize the normalized maximum of its autoconvolution"
    # Usually "normalized" implies dividing by some norm.
    # The seed program used: max(conv) / l2 where l2 = sum(f^2)*DX. 
    # Wait, the seed code says:
    # l2 = np.sum(f * f) * DX
    # return float(np.max(conv) / l2)
    # This is max(autoconv) / L2^2. 
    # Let's stick to the seed's definition to ensure comparability, 
    # although "normalized maximum" could mean max(conv) / max(conv_max_possible) or similar.
    # However, the seed score is 0.9328. If the definition changes, the score scale changes.
    # The prompt says "Keep the function names and signatures... score is the upper bound on C1".
    # C1 in the context of "First autocorrelation inequality" usually refers to:
    # max(f*f) <= C1 * ||f||_2^2 ? Or something similar.
    # If the seed uses max(conv)/L2^2, then C1 is that ratio.
    
    return float(np.max(conv) / l2_sq)

def param_to_h(params):
    """
    Converts a set of parameters to a step function height array.
    params: array of [center, width, height, exponent]
    We use a super-Gaussian (or generalized Gaussian) bump:
    h(x) = height * exp(-((x - center) / width) ** exponent)
    
    We ensure non-negativity and compact support effectively by the exponential decay.
    To make it a "step function on 600 intervals", we evaluate this continuous
    function at the centers of the intervals (or just X) and use those as heights.
    """
    c, w, h_max, p = params
    # Ensure width is positive
    w = abs(w) + 1e-6
    # Ensure exponent is reasonable
    p = max(1.0, p)
    
    # Calculate heights at X points
    # Using np.exp might underflow for large p and x, which is fine (becomes 0)
    h = h_max * np.exp(-np.abs((X - c) / w) ** p)
    
    # Clip to ensure non-negative (exp is always positive, but just in case)
    h = np.clip(h, 0.0, None)
    
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Parameters: [center, width, height, exponent]
    # Initial guesses based on intuition:
    # 1. Standard Gaussian (p=2)
    # 2. Compact bump (p=4, small width)
    # 3. Very flat (p=1, large width) - likely worse
    # 4. Centered at 0
    
    initial_params_list = [
        np.array([0.0, 0.05, 1.0, 2.0]),
        np.array([0.0, 0.03, 1.0, 4.0]),
        np.array([0.0, 0.08, 1.0, 2.0]),
        np.array([0.0, 0.02, 1.0, 8.0]),
        np.array([0.0, 0.10, 1.0, 1.5]),
    ]
    
    best_params = initial_params_list[0].copy()
    best_val = float('inf')
    
    # Evaluate initial candidates
    for p in initial_params_list:
        h = param_to_h(p)
        v = ratio_from_h(h)
        if v < best_val:
            best_val = v
            best_params = p.copy()
            
    # Barycentric Coordinate Descent (BCD)
    # This method maintains a simplex of points and moves towards the best vertex
    # by barycentric coordinates, then shrinks the simplex if no improvement is found.
    # It is effective for low-dimensional, smooth functions.
    
    dim = 4
    # Initialize simplex around best_params
    simplex = np.zeros((dim + 1, dim))
    simplex[0] = best_params
    
    # Perturbations for other vertices
    # Center: [-0.1, 0.1]
    # Width: [0.01, 0.15]
    # Height: [0.1, 5.0] (Scale doesn't matter for the ratio? 
    # Wait, ratio = max(conv) / L2^2. 
    # If h -> k*h, conv -> k^2 conv, L2^2 -> k^2 L2^2. Ratio is invariant to scaling!
    # So height parameter is redundant for the objective value, but helps with numerical stability
    # or if we add constraints. Let's fix height to 1.0 to reduce dimensionality to 3.
    # Actually, let's keep it but realize it's a flat direction. 
    # Better: Remove height from optimization to save dimensions and avoid noise.
    # Let's redefine param_to_h to take [center, width, exponent] and set height=1.
    
    # REDEFINING STRATEGY FOR DIMENSION REDUCTION:
    # Since the ratio is scale-invariant, we can fix the height (or L2 norm) to 1.
    # Let's fix the peak height to 1.0.
    
    def param_to_h_scaled(params):
        c, w, p = params
        w = abs(w) + 1e-6
        p = max(1.0, p)
        h = np.exp(-np.abs((X - c) / w) ** p)
        # Normalize? No, ratio is invariant. But for numerical precision in FFT,
        # keeping values in [0,1] is good.
        return h

    # Update initial params to 3D
    initial_params_list_3d = [
        np.array([0.0, 0.05, 2.0]),
        np.array([0.0, 0.03, 4.0]),
        np.array([0.0, 0.08, 2.0]),
        np.array([0.0, 0.02, 8.0]),
        np.array([0.0, 0.10, 1.5]),
    ]
    
    best_params = initial_params_list_3d[0].copy()
    best_val = float('inf')
    
    for p in initial_params_list_3d:
        h = param_to_h_scaled(p)
        v = ratio_from_h(h)
        if v < best_val:
            best_val = v
            best_params = p.copy()
            
    dim = 3
    simplex = np.zeros((dim + 1, dim))
    simplex[0] = best_params
    
    # Define step sizes for simplex initialization
    # Center: 0.1
    # Width: 0.05
    # Exponent: 1.0
    steps = np.array([0.1, 0.05, 1.0])
    
    for i in range(1, dim + 1):
        simplex[i] = best_params.copy()
        simplex[i][i-1] += steps[i-1]
        
    fvals = np.array([ratio_from_h(param_to_h_scaled(simplex[i])) for i in range(dim + 1)])
    
    # BCD Parameters
    # We will perform a limited number of BCD iterations.
    # Each iteration involves sorting, calculating barycentric coords, and evaluating.
    
    max_bcd_iters = 100
    for it in range(max_bcd_iters):
        if time.time() > deadline - 10.0:
            break
            
        # Sort simplex by objective value
        idx = np.argsort(fvals)
        simplex = simplex[idx]
        fvals = fvals[idx]
        
        # Check convergence
        if fvals[0] < best_val:
            best_val = fvals[0]
            best_params = simplex[0].copy()
            
        if np.max(fvals) - np.min(fvals) < 1e-8:
            break
            
        # Calculate barycentric coordinates of the centroid of the best d points
        # For 3D, centroid of the best 3 vertices.
        # Actually, standard BCD moves the worst vertex towards the centroid of the others.
        # Or, it finds the point in the simplex with minimum function value by
        # solving a quadratic program or using barycentric coordinates of the centroid.
        
        # Simpler approach: Nelder-Mead is a variant of this. 
        # But let's try a specific BCD step:
        # 1. Identify worst point (simplex[-1]).
        # 2. Calculate centroid of the other points (simplex[:-1]).
        # 3. Move worst point to centroid (Contraction) or reflect it.
        # This is essentially Nelder-Mead.
        
        # To make it "Materially Different", let's use **Global** search via 
        # **Random Restart BCD** or **Multistart**.
        # The current program uses a single local search.
        # This program will use MULTIPLE random starts of the BCD/Nelder-Mead process
        # across the parameter space, leveraging the low dimensionality (3D) to afford
        # many restarts within the time budget.
        
        # Restart logic:
        # If we are in the first 10 iterations, we are just refining the initial best.
        # After that, we generate a new random start and run a few iterations.
        
        # Let's restructure:
        # Loop over restarts.
        # Each restart gets a time budget.
        
        # Break out of the inner optimization loop to handle restarts properly.
        # We will implement a helper function that runs Nelder-Mead for a fixed number of steps.
        
        # Actually, let's just implement the restarts in the main loop.
        
        # If it < 50, continue refining current best.
        # If it >= 50, start a new random search.
        
        if it >= 20:
            # Generate new random parameters
            # Center: U(-0.1, 0.1)
            # Width: U(0.02, 0.15)
            # Exponent: U(1.5, 6.0)
            rng = np.random.default_rng(it)
            new_p = np.array([
                rng.uniform(-0.1, 0.1),
                rng.uniform(0.02, 0.15),
                rng.uniform(1.5, 6.0)
            ])
            
            # Initialize new simplex around new_p
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = new_p
            for i in range(1, dim + 1):
                simplex[i] = new_p.copy()
                simplex[i][i-1] += steps[i-1]
                
            fvals = np.array([ratio_from_h(param_to_h_scaled(simplex[i])) for i in range(dim + 1)])
            
            # Check if this new start is better
            if fvals[0] < best_val:
                best_val = fvals[0]
                best_params = simplex[0].copy()
                
            # Continue to next iteration of the outer loop, which will now
            # perform a Nelder-Mead step on this new simplex.
            continue
            
        # --- Standard Nelder-Mead Step on current simplex ---
        x_avg = np.mean(simplex[:-1], axis=0)
        x_worst = simplex[-1]
        f_worst = fvals[-1]
        f_best = fvals[0]
        
        # Reflection
        alpha = 1.0
        x_r = x_avg + alpha * (x_avg - x_worst)
        f_r = ratio_from_h(param_to_h_scaled(x_r))
        
        if f_r < f_best:
            # Expansion
            gamma = 2.0
            x_e = x_avg + gamma * (x_r - x_avg)
            f_e = ratio_from_h(param_to_h_scaled(x_e))
            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
        elif f_r < fvals[-2]:
            simplex[-1] = x_r
            fvals[-1] = f_r
        else:
            # Contraction
            rho = 0.5
            if f_r < f_worst:
                x_c = x_avg + rho * (x_r - x_avg)
            else:
                x_c = x_avg - rho * (x_worst - x_avg)
            f_c = ratio_from_h(param_to_h_scaled(x_c))
            if f_c < f_worst:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                # Shrink
                sigma = 0.5
                for i in range(1, dim + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = ratio_from_h(param_to_h_scaled(simplex[i]))
        
        # Update best
        if fvals[0] < best_val:
            best_val = fvals[0]
            best_params = simplex[0].copy()

    # Final refinement: Coordinate descent on the 3 parameters
    if time.time() < deadline - 5.0:
        current_p = best_params.copy()
        current_v = best_val
        rng = np.random.default_rng(99)
        
        # Fine-tune each parameter
        for _ in range(100):
            if time.time() > deadline - 2.0:
                break
            idx = rng.integers(0, dim)
            p_try = current_p.copy()
            # Small perturbation
            p_try[idx] += rng.normal(0, 0.01)
            # Ensure valid ranges
            p_try[1] = abs(p_try[1]) + 1e-6
            p_try[2] = max(1.0, p_try[2])
            
            v_try = ratio_from_h(param_to_h_scaled(p_try))
            if v_try < current_v:
                current_p = p_try
                current_v = v_try
                if current_v < best_val:
                    best_val = current_v
                    best_params = current_p.copy()

    # Convert best parameters to final step function
    best_h = param_to_h_scaled(best_params)
    
    return list(best_h)

if __name__ == "__main__":
    # Quick self-test
    h = construct()
    print(f"Score: {ratio_from_h(h)}")
