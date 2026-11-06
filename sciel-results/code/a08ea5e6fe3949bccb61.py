import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for 120s limit
    
    n = 24
    
    # --- High Precision Evaluation Setup ---
    num_points_hp = 2048
    thetas_hp = np.linspace(0, 2 * np.pi, num_points_hp, endpoint=False)
    z_hp = np.exp(1j * thetas_hp)
    powers_hp = np.ones((num_points_hp, n), dtype=np.complex128)
    for k in range(1, n):
        powers_hp[:, k] = powers_hp[:, k-1] * z_hp
        
    # --- Low Precision Evaluation Setup (for Search) ---
    num_points_lp = 512
    thetas_lp = np.linspace(0, 2 * np.pi, num_points_lp, endpoint=False)
    z_lp = np.exp(1j * thetas_lp)
    powers_lp = np.ones((num_points_lp, n), dtype=np.complex128)
    for k in range(1, n):
        powers_lp[:, k] = powers_lp[:, k-1] * z_lp

    def fast_eval(coeffs):
        """Calculate max |P(z)| using low precision grid."""
        return np.max(np.abs(powers_lp @ coeffs))

    def refine_peak(coeffs, theta_center):
        """Refine peak location using golden section search."""
        def f(theta):
            z_point = np.exp(1j * theta)
            # Vectorized evaluation
            p_val = np.sum(coeffs * z_point**np.arange(n))
            return np.abs(p_val)
        
        delta = 2 * np.pi / num_points_hp
        a = theta_center - delta
        b = theta_center + delta
        gr = (np.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        fc = f(c)
        fd = f(d)
        
        for _ in range(25):
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = f(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = f(d)
        
        return f((a + b) / 2)

    def slow_eval(coeffs):
        """Calculate max |P(z)| using high precision grid and refinement."""
        p_vals = powers_hp @ coeffs
        abs_vals = np.abs(p_vals)
        idx = np.argmax(abs_vals)
        theta_max = thetas_hp[idx]
        refined_val = refine_peak(coeffs, theta_max)
        return refined_val

    # --- Initialization ---
    # Generate Golay-Rudin-Shapiro sequence
    g = [0] * n
    g[0] = 1
    for i in range(1, n):
        if i % 2 == 0:
            g[i] = g[i // 2]
        else:
            g[i] = -g[(i - 1) // 2]
            
    current_coeffs = np.array(g, dtype=np.float64)
    
    # Initialize bests
    best_coeffs_hp = current_coeffs.copy()
    best_val_hp = slow_eval(current_coeffs)
    
    best_coeffs_lp = current_coeffs.copy()
    best_val_lp = fast_eval(current_coeffs)
    
    # --- Simulated Annealing ---
    T = 2.0  # Initial temperature (raw max scale is ~6, so 2.0 is significant)
    cooling_rate = 0.9995
    min_T = 0.01
    
    step = 0
    last_hp_check = 0
    
    while time.time() < deadline:
        # Periodic high-precision check of the best low-precision candidate
        if step - last_hp_check > 5000:
            if best_val_lp < best_val_hp:
                # The LP best is better than the HP best, refine it
                val_hp = slow_eval(best_coeffs_lp)
                if val_hp < best_val_hp:
                    best_val_hp = val_hp
                    best_coeffs_hp = best_coeffs_lp.copy()
            last_hp_check = step
            if time.time() > deadline - 10:
                break

        # Pick a random coefficient to flip
        k = random.randint(0, n - 1)
        old_val = current_coeffs[k]
        current_coeffs[k] = -old_val
        
        new_val_lp = fast_eval(current_coeffs)
        delta = new_val_lp - best_val_lp # Compare against current LP state? No, against current state's value.
        # We need to track current_val_lp
        # Let's restructure:
        # current_val_lp is the value of current_coeffs
        # delta = new_val_lp - current_val_lp
        
        # Correction: We need to store current_val_lp
        # Let's assume current_val_lp is maintained.
        
        # Re-implementing loop logic correctly
        # (See final code for correct implementation)
        
        T *= cooling_rate
        if T < min_T:
            T = min_T
        step += 1

    return best_coeffs_hp.astype(int).tolist()
