import numpy as np
import scipy.optimize
import time

N = 400

def _compute_bound(f):
    """
    Computes the peak sidelobe magnitude of the linear autocorrelation.
    """
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    PSD = F * np.conj(F)
    R = np.fft.irfft(PSD, n=n_fft)
    # R[0] is the peak (energy)
    # R[1:N] are the sidelobes
    return np.max(np.abs(R[1:N]))

def _objective_high_power(f):
    """
    Objective function: Sum of high-power sidelobes.
    This approximates the minimization of the maximum sidelobe (L-infinity)
    while remaining differentiable.
    """
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    PSD = F * np.conj(F)
    R = np.fft.irfft(PSD, n=n_fft)
    sidelobes = R[1:N]
    # Use power 20 to strongly penalize large sidelobes
    return np.sum(np.abs(sidelobes)**20)

def _generate_frank_code(N, M):
    """
    Generate a real-valued Frank code of length N = M^2.
    """
    n = np.arange(N)
    # Frank code formula: exp(i * 2*pi * n * floor(n/M) / M^2)
    # Real part: cos(2*pi * n * floor(n/M) / M^2)
    phase = 2 * np.pi * n * np.floor(n / M) / N
    return np.cos(phase)

def _generate_chirp(N):
    """
    Generate a real-valued chirp.
    """
    n = np.arange(N)
    # Linear chirp: cos(2*pi * n^2 / (2*N))
    phase = 2 * np.pi * n**2 / (2 * N)
    return np.cos(phase)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_f = None
    best_score = float('inf')
    
    # Initializations
    inits = []
    
    # 1. Frank Code (M=20)
    inits.append(_generate_frank_code(N, 20))
    
    # 2. Chirp
    inits.append(_generate_chirp(N))
    
    # 3. Kaiser Window (beta=10)
    inits.append(scipy.signal.windows.kaiser(N, beta=10))
    
    # 4. Gaussian
    x = np.linspace(-1, 1, N)
    inits.append(np.exp(-2 * x**2))
    
    # 5. Random (fixed seed for reproducibility in this block, though not required)
    # np.random.seed(42)
    # inits.append(np.random.randn(N))
    
    for f_init in inits:
        if time.time() > deadline - 10:
            break
            
        # Normalize to unit energy
        f_init = f_init / np.linalg.norm(f_init)
        
        try:
            # Optimize the high-power objective
            res = scipy.optimize.minimize(
                _objective_high_power, 
                f_init, 
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            f_opt = res.x
            # Re-normalize
            f_opt = f_opt / np.linalg.norm(f_opt)
            
            # Evaluate the actual metric (Peak Sidelobe)
            score = _compute_bound(f_opt)
            
            if score < best_score:
                best_score = score
                best_f = f_opt.copy()
                
        except Exception:
            continue
            
    # Fallback
    if best_f is None:
        best_f = inits[0] / np.linalg.norm(inits[0])
        
    return best_f
