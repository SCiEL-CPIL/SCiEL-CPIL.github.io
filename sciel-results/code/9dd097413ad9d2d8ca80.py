import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a multiplicative structure with a Fourier basis for temp/pH.
    
    Model: y = A * Monod(s) * Inhibit(b) * (1 + F_temp + F_ph) + B
    
    The Monod term handles substrate saturation.
    The Inhibit term handles density inhibition.
    The Fourier terms (sines) handle the non-monotonic optimal ranges for temperature and pH.
    
    Parameters (10):
    p0: A (Global Amplitude)
    p1: K_s (Monod half-saturation)
    p2: K_b (Density Inhibition Constant)
    p3: w_t (Frequency for Temp sine)
    p4: A_t (Amplitude for Temp sine)
    p5: phi_t (Phase for Temp sine)
    p6: w_p (Frequency for pH sine)
    p7: A_p (Amplitude for pH sine)
    p8: phi_p (Phase for pH sine)
    p9: B (Additive Bias)
    """
    t0 = time.time()
    
    # Extract parameters
    A = params[0]
    K_s = np.abs(params[1]) + 1e-6
    K_b = np.abs(params[2]) + 1e-6
    w_t = params[3]
    A_t = params[4]
    phi_t = params[5]
    w_p = params[6]
    A_p = params[7]
    phi_p = params[8]
    B = params[9]
    
    # Normalize inputs to help BFGS converge (center and scale)
    # temp range approx [1, 40], center 20.5, span ~10
    t_norm = (temp - 20.5) / 10.0
    # pH range approx [1, 12], center 6.5, span ~2.5
    h_norm = (pH - 6.5) / 2.5
    
    # Monod kinetics for substrate
    # s range [1, 40]
    monod = s / (K_s + s + 1e-12)
    
    # Density inhibition
    # b range [1, 5000] (based on current program comment, though task says [1, 5] in inspiration 1? 
    # Task says: [1, 4.998] for first input which is b? No, inputs are [b, s, temp, pH].
    # Task text: "Inputs in training range, in order: [1, 4.998], [1, 4.999], [1.002, 39.98], [2, 12]."
    # This lists ranges for b, s, temp, pH respectively.
    # So b is in [1, 5], s is in [1, 5], temp is in [1, 40], pH is in [1, 12].
    # Wait, the current program comment says b range [1, 5000]. The task contract says [1, 4.998].
    # I must trust the task contract: b is [1, 5].
    # If b is [1, 5], inhibition might be mild or specific.
    dens_term = 1.0 / (1.0 + b / K_b)
    
    # Fourier terms for temperature
    # Using normalized temp to keep frequencies small
    f_temp = A_t * np.sin(w_t * t_norm + phi_t)
    
    # Fourier terms for pH
    f_ph = A_p * np.sin(w_p * h_norm + phi_p)
    
    # Combine modulation
    # The base growth is 1.0 (when A=1, others 0). The sine terms modulate it.
    # We allow the modulation to go negative if amplitudes are large, but clip to avoid 
    # extreme negative growth rates which are non-physical and might hurt optimization.
    modulation = 1.0 + f_temp + f_ph
    
    # Clip modulation to a reasonable range to prevent explosion during optimization
    # Growth rate should generally be positive, but let's allow small negatives 
    # if the fit demands it, but cap the magnitude.
    modulation = np.clip(modulation, -2.0, 5.0)
    
    # Final result
    result = A * monod * dens_term * modulation + B
    
    # Ensure finite output
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Deadline guard
    if time.time() - t0 > 140:
        return result
        
    return result
