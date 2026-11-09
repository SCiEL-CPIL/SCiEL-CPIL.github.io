# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate from population density, substrate, temperature and pH.
    
    Uses a sum-of-cosines basis with learnable frequencies.
    Cosine terms with baked-in centers model unimodal optima more naturally,
    improving BFGS convergence by reducing the number of free phase parameters.
    
    params layout (10 total):
      params[0]: base offset
      params[1]: amplitude for temp (1st harmonic)
      params[2]: frequency for temp
      params[3]: amplitude for temp (2nd harmonic)
      params[4]: amplitude for pH (1st harmonic)
      params[5]: frequency for pH
      params[6]: amplitude for pH (2nd harmonic)
      params[7]: amplitude for s
      params[8]: frequency for s
      params[9]: amplitude for b
    """
    offset = params[0]
    
    # Baked-in centers for temperature and pH to help optimization
    t_center = 37.0
    ph_center = 7.0
    
    # Temperature: two cosine harmonics
    amp_t1 = params[1]
    freq_t = params[2]
    amp_t2 = params[3]
    
    dt = temp - t_center
    term_t = amp_t1 * np.cos(freq_t * dt) + amp_t2 * np.cos(2.0 * freq_t * dt)
    
    # pH: two cosine harmonics
    amp_ph1 = params[4]
    freq_ph = params[5]
    amp_ph2 = params[6]
    
    dp = pH - ph_center
    term_ph = amp_ph1 * np.cos(freq_ph * dp) + amp_ph2 * np.cos(2.0 * freq_ph * dp)
    
    # Substrate: monotonic, use sine with learnable frequency
    amp_s = params[7]
    freq_s = params[8]
    term_s = amp_s * np.sin(freq_s * s)
    
    # Density: gentle effect, fixed frequency
    amp_b = params[9]
    term_b = amp_b * np.sin(0.2 * b)
    
    return offset + term_b + term_s + term_t + term_ph
# EVOLVE-BLOCK-END
