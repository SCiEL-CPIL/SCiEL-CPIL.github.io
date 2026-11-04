"""
Strategy: Low-dimensional parametric optimization using structured sequences.
Instead of optimizing 400 free variables, we construct signals from a linear 
combination of a few basis sequences (e.g., m-sequences, Gold codes, or chirps) 
with optimized coefficients. This reduces the dimensionality significantly, 
allowing for a global search over the parameters rather than a local search 
in high-dimensional space.
"""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_max_abs(x):
    """Max absolute value of the discrete autoconvolution of x."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))

def _integral(x):
    """Integral of the equal-width step function = sum of the heights."""
    return float(np.sum(x))

def _metric(x):
    """max |autoconv| / (integral)^2.  Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _generate_msequence(n):
    """
    Generate a Maximum Length Sequence (m-sequence) of length 2^n - 1.
    We use a simple LFSR implementation.
    """
    # Primitive polynomials for small n
    polys = {
        1: 0b11, # x^2 + x + 1 (len 3) - actually x^2+x+1 is degree 2, len 3? 2^2-1=3.
        2: 0b1011, # x^4 + x + 1 (len 15)
        3: 0b10011, # x^5 + x^2 + 1 (len 31)
        4: 0b100011, # x^6 + x + 1 (len 63)
        5: 0b1000011, # x^7 + x + 1 (len 127)
        6: 0b10000011, # x^8 + x^4 + x^3 + x^2 + 1 (len 255)
        7: 0b100000011, # x^9 + x^4 + x^3 + x^2 + 1 (len 511) - might be too long
    }
    
    # We need length close to 400. 
    # 2^8 - 1 = 255. 2^9 - 1 = 511.
    # We can pad or truncate. Let's use n=8 (len 255) and n=9 (len 511).
    
    # For n=8, poly x^8 + x^4 + x^3 + x^2 + 1
    # Tap positions: 8, 4, 3, 2, 0 (0-indexed from LSB of poly)
    # Actually, let's just use a known good m-sequence generator.
    
    def gen_mseq(n, poly):
        # poly is integer representation of primitive polynomial
        # e.g., 0b10000011 for x^8 + x + 1
        # We implement a standard LFSR
        # State is n bits.
        # Feedback tap at position corresponding to poly.
        
        # This is getting complicated. Let's use a simpler structured sequence:
        # Chirps or Sines.
        pass
        
    # Let's stick to simple parametric families:
    # 1. Sine waves: sin(2*pi*k*t/N)
    # 2. Chirps: sin(2*pi*(k*t + 0.5*alpha*t^2)/N)
    # 3. Random binary sequences (optimized)
    
    return None # Placeholder, we will use explicit basis functions instead

def _basis_sine(k, phase):
    t = np.arange(N)
    return np.sin(2 * np.pi * k * t / N + phase)

def _basis_chirp(k, alpha, phase):
    t = np.arange(N)
    return np.sin(2 * np.pi * (k * t + 0.5 * alpha * t**2) / N + phase)

def _parametric_signal(params):
    """
    Construct a signal from a linear combination of basis functions.
    params: [a1, a2, a3, a4, a5, k1, k2, k3, k4, k5, alpha1, alpha2, alpha3, alpha4, alpha5, phase1...phase5]
    We will fix the basis types to a mix of sines and chirps to keep dimension low.
    """
    # Let's use 5 components.
    # Params: 5 amplitudes, 5 frequencies, 5 phases. (15 params)
    amps = params[:5]
    freqs = params[5:10]
    phases = params[10:15]
    
    t = np.arange(N)
    x = np.zeros(N)
    
    for i in range(5):
        # Mix of sine and chirp based on index? 
        # Let's just use sines for simplicity first, then add chirps if needed.
        # Sines have perfectly flat autocorrelation (delta at 0) in continuous sense.
        # In discrete, they have low sidelobes if windowed, but here we have hard edges.
        # Hard edges cause spectral leakage.
        # Chirps have better time-frequency localization.
        
        # Let's use a generic form: sin(2*pi*(f_i*t + 0.5*chirp_i*t^2)/N + phi_i)
        # We can fix chirp_i to 0 for sines, non-zero for chirps.
        # To keep it simple, let's just use sines with integer frequencies.
        # Integer frequencies ensure periodicity, reducing edge effects.
        
        f = int(round(freqs[i]))
        if f == 0:
            f = 1 # Avoid DC for now, or allow it? DC contributes to integral.
            # If we allow DC, we can control the integral directly.
        
        x += amps[i] * np.sin(2 * np.pi * f * t / N + phases[i])
        
    return x

def _objective(params):
    x = _parametric_signal(params)
    return _metric(x)

def construct():
    deadline = time.time() + 145.0
    
    # Strategy:
    # 1. Generate a set of diverse starting points using random integer frequencies.
    # 2. Use L-BFGS-B to polish each starting point.
    # 3. Keep the best.
    
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    rng = np.random.default_rng(2024)
    
    # Number of starting points
    n_starts = 20
    
    for i in range(n_starts):
        if time.time() > deadline - 10:
            break
            
        # Random starting parameters
        # Frequencies: random integers 1..N/2
        freqs = rng.integers(1, N//2 + 1, size=5).astype(float)
        amps = rng.uniform(0.5, 1.5, size=5)
        phases = rng.uniform(0, 2*np.pi, size=5)
        
        params0 = np.concatenate([amps, freqs, phases])
        
        # Optimize using L-BFGS-B
        # Bounds: 
        # Amps: [0.1, 3.0]
        # Freqs: [1, N/2] (integer constraint is hard for L-BFGS, but we round in _parametric_signal)
        #       Actually, rounding makes the function non-smooth. 
        #       Better to allow real frequencies and let the optimizer handle it, 
        #       or use a different basis that is smooth in frequency.
        #       Let's allow real frequencies. The signal will not be periodic, 
        #       causing edge discontinuities. This might be okay if amplitudes are small.
        #       Or, we can use a window function? No, the task implies the step function is the signal.
        #       Let's try real frequencies first.
        
        bounds = [
            (0.1, 3.0), (0.1, 3.0), (0.1, 3.0), (0.1, 3.0), (0.1, 3.0), # Amps
            (1.0, N/2), (1.0, N/2), (1.0, N/2), (1.0, N/2), (1.0, N/2), # Freqs
            (0.0, 2*np.pi), (0.0, 2*np.pi), (0.0, 2*np.pi), (0.0, 2*np.pi), (0.0, 2*np.pi) # Phases
        ]
        
        try:
            res = minimize(_objective, params0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 100, 'maxfun': 1000})
            
            if res.success or res.fun < best_val:
                x_opt = _parametric_signal(res.x)
                val_opt = _metric(x_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_x = x_opt
        except Exception:
            pass

    # Final polish with local search on the best candidate found
    if time.time() < deadline - 5:
        # Simple local search on the continuous variables of the best parametric signal
        # We can extract the parameters of best_x? No, best_x is the signal.
        # We should keep the parameters of the best signal.
        # Let's refactor to keep params.
        
        # For now, do a quick random perturbation on the signal itself.
        step_scale = 0.1
        no_improve = 0
        x_cur = best_x.copy()
        
        for _ in range(10000):
            if time.time() > deadline - 2:
                break
            if no_improve > 100:
                step_scale *= 0.9
                no_improve = 0
                if step_scale < 0.001:
                    break
                    
            idx = rng.integers(0, N)
            x_test = x_cur.copy()
            x_test[idx] += rng.normal(0, step_scale)
            v = _metric(x_test)
            if v < best_val:
                best_val = v
                x_cur = x_test
                no_improve = 0
            else:
                no_improve += 1
        
        best_x = x_cur

    return best_x
# EVOLVE-BLOCK-END
