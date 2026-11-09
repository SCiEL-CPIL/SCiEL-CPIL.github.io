import numpy as np
import time

_START = time.time()
_DEADLINE = 150.0 - 10.0

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check deadline guard
    if time.time() > _DEADLINE:
        return np.zeros_like(b, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Safe parameters to avoid numerical issues
    # p1: Monod K (keep positive via abs, min 0.1)
    K = np.maximum(np.abs(p1), 0.1)
    
    # p2, p3: Temp center and width
    t_center = p2
    t_width = np.maximum(np.abs(p3), 0.5)
    
    # p4, p5: pH center and width
    ph_center = p4
    ph_width = np.maximum(np.abs(p5), 0.5)
    
    # p6: Frequency for Fourier term in temp
    freq_t = p6
    
    # p7: Phase for Fourier term in temp
    phase_t = p7
    
    # p8: Frequency for Fourier term in pH
    freq_p = p8
    
    # p9: Phase for Fourier term in pH
    phase_p = p9
    
    # 1. Monod substrate kinetics
    monod = s / (K + s)
    
    # 2. Temperature response: Gaussian core + Fourier correction
    # The Gaussian captures the main bell shape.
    # The Fourier term (sin) allows for asymmetry or fine-tuning of the peak shape.
    temp_gauss = np.exp(-((temp - t_center) / t_width) ** 2)
    temp_fourier = np.sin(freq_t * temp + phase_t)
    # Combine: The Fourier term is scaled to be a small correction to the Gaussian
    # We use a fixed small scale (0.1) to prevent the Fourier term from dominating
    # and making the landscape too rugged for BFGS, but large enough to shape.
    temp_term = temp_gauss * (1.0 + 0.1 * temp_fourier)
    
    # 3. pH response: Gaussian core + Fourier correction
    ph_gauss = np.exp(-((pH - ph_center) / ph_width) ** 2)
    ph_fourier = np.sin(freq_p * pH + phase_p)
    ph_term = ph_gauss * (1.0 + 0.1 * ph_fourier)
    
    # 4. Density inhibition
    # Linear inhibition is standard. We fix the coefficient to 1.0 scaled by monod?
    # Or just a simple factor. Let's use a simple linear factor (1 - 0.1*b) ?
    # Actually, let's keep it simple: multiplicative factor that decreases with b.
    # Since we have no free params left for density slope, we fix it to a reasonable value.
    # Or, we can absorb density into the main coefficient p0 if we assume it's linear?
    # No, b is an input. Let's use a fixed mild inhibition: (1 - 0.05 * b).
    # b ranges [1, 5]. 1-0.05=0.95, 1-0.25=0.75.
    density_term = 1.0 - 0.05 * b
    
    # 5. Main term
    # p0 is the scale factor for the whole model.
    result = p0 * monod * temp_term * ph_term * density_term
    
    # Ensure finite output
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
