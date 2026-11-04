import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    deadline = time.time() + 115  # Safety margin for 120s limit
    
    def compute_score(h):
        s = np.sum(h)
        if s == 0 or not np.isfinite(s):
            return float('inf')
        # np.convolve(h, h) is linear convolution
        z = np.convolve(h, h)
        # The metric takes max of signed entries
        # We want to minimize this value.
        # Note: If max(z) is negative, abs() makes it positive. 
        # Ideally we want max(z) to be close to 0.
        return abs(2 * N * np.max(z) / (s * s))

    def generate_signal(params):
        """
        Generates a signal from parameters.
        params: [center_width, side_width, side_distance, center_amp, side_amp, skew]
        Structure: Positive center bump, negative side bumps.
        Using cosine tapers for smoothness and better sidelobe properties.
        """
        c_w, s_w, s_d, c_a, s_a, skew = params
        
        # Ensure positive widths and amps to avoid weird artifacts, 
        # though skew can be negative
        c_w = abs(c_w) + 0.01
        s_w = abs(s_w) + 0.01
        s_d = abs(s_d) + 0.01
        c_a = abs(c_a) + 0.1
        s_a = abs(s_a) + 0.1
        
        x = np.linspace(0, 1, N)
        
        # Center bump: Cosine taper
        # Support: [0.5 - c_w/2 + skew, 0.5 + c_w/2 + skew]
        center_start = 0.5 - c_w/2 + skew
        center_end = 0.5 + c_w/2 + skew
        h_center = np.zeros(N)
        mask_c = (x >= center_start) & (x <= center_end)
        if np.any(mask_c):
            x_c = x[mask_c]
            # Normalize x_c to [0, 1]
            x_norm = (x_c - center_start) / (center_end - center_start)
            # Cosine taper: 0.5 * (1 + cos(pi * x_norm))
            taper = 0.5 * (1 + np.cos(np.pi * x_norm))
            h_center[mask_c] = c_a * taper

        # Side bumps: Negative, symmetric around center, but shifted by skew? 
        # Let's keep sides symmetric relative to the center of the signal (0.5) 
        # but allow independent width/amp.
        # Side 1 (left)
        side1_start = 0.5 - s_d - s_w/2
        side1_end = 0.5 - s_d + s_w/2
        h_side1 = np.zeros(N)
        mask_s1 = (x >= side1_start) & (x <= side1_end)
        if np.any(mask_s1):
            x_s1 = x[mask_s1]
            x_norm_s1 = (x_s1 - side1_start) / (side1_end - side1_start)
            taper_s1 = 0.5 * (1 + np.cos(np.pi * x_norm_s1))
            h_side1[mask_s1] = -s_a * taper_s1

        # Side 2 (right)
        side2_start = 0.5 + s_d - s_w/2
        side2_end = 0.5 + s_d + s_w/2
        h_side2 = np.zeros(N)
        mask_s2 = (x >= side2_start) & (x <= side2_end)
        if np.any(mask_s2):
            x_s2 = x[mask_s2]
            x_norm_s2 = (x_s2 - side2_start) / (side2_end - side2_start)
            taper_s2 = 0.5 * (1 + np.cos(np.pi * x_norm_s2))
            h_side2[mask_s2] = -s_a * taper_s2

        h = h_center + h_side1 + h_side2
        
        # Optional: Add a small global offset or noise? 
        # No, keep it clean for the optimizer.
        return h

    def objective(params):
        if time.time() > deadline:
            return float('inf')
        try:
            h = generate_signal(params)
            if not np.all(np.isfinite(h)):
                return float('inf')
            return compute_score(h)
        except Exception:
            return float('inf')

    # Initial guesses for parameters: [c_w, s_w, s_d, c_a, s_a, skew]
    # Typical good structures: center width ~0.3, side width ~0.2, distance ~0.3
    initial_params_list = [
        [0.3, 0.2, 0.3, 1.0, 0.5, 0.0],
        [0.4, 0.3, 0.2, 1.0, 0.6, 0.0],
        [0.2, 0.1, 0.4, 1.0, 0.4, 0.0],
        [0.5, 0.4, 0.1, 1.0, 0.8, 0.0],
        [0.3, 0.2, 0.3, 1.0, 0.5, 0.1], # Skewed
        [0.3, 0.2, 0.3, 1.0, 0.5, -0.1], # Skewed
    ]
    
    best_h = np.ones(N)
    best_score = compute_score(best_h)
    
    # Try multiple starts
    for p0 in initial_params_list:
        if time.time() > deadline - 5:
            break
            
        # Use Powell or Nelder-Mead
        # Powell is often more robust for box-like constraints implicitly
        # We don't have strict bounds, but the generator handles abs() for widths/amps
        
        result = minimize(objective, p0, method='Powell', options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if result.fun < best_score:
            best_score = result.fun
            best_h = generate_signal(result.x)
            
        # Also try Nelder-Mead on the best result found so far if time permits
        if time.time() < deadline - 10:
            try:
                result2 = minimize(objective, result.x, method='Nelder-Mead', options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-8})
                if result2.fun < best_score:
                    best_score = result2.fun
                    best_h = generate_signal(result2.x)
            except Exception:
                pass

    # Final check
    if time.time() > deadline:
        return best_h
        
    # One last random restart if we have time, maybe with different structure?
    # The parametric family is fixed. We could add a random noise injection 
    # to the best_h and do a quick local polish?
    # No, the parametric family is the strategy. Stick to it.
    
    return best_h
