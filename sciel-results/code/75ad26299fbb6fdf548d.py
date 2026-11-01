"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_acf(h):
    """Compute linear autocorrelation using FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:N]
    return acf

def _compute_c3_bound_from_acf(acf):
    """Compute C3 bound from precomputed autocorrelation."""
    sidelobes = acf[1:]
    return np.sum(np.abs(sidelobes) ** 3) / (N ** 3)

def _compute_c3_bound(h):
    """Compute C3 bound from sequence."""
    acf = _compute_acf(h)
    return _compute_c3_bound_from_acf(acf)

def _spectral_init(rng):
    """
    Initialize sequence from inverse FFT of a target low-peak magnitude spectrum.
    We want flat autocorrelation sidelobes, which corresponds to a flat power spectrum
    (for the sidelobes). We can start with a random phase spectrum and take the
    inverse FFT, then threshold to +/-1.
    """
    # Generate random phases
    phases = rng.uniform(0, 2 * np.pi, size=N)
    # Create a flat magnitude spectrum (all ones) with random phases
    spectrum = np.exp(1j * phases)
    # Inverse FFT
    h_cont = np.fft.ifft(spectrum).real
    # Threshold to +/-1
    h = np.sign(h_cont)
    # Ensure no zeros
    h[h == 0] = 1.0
    return h

def _local_search_bitflip(h, acf, current_val, deadline, rng):
    """
    Custom bit-flip heuristic that minimizes the C3 bound.
    Uses greedy improvement with some randomization to escape local minima.
    """
    best_h = h.copy()
    best_val = current_val
    best_acf = acf.copy()
    
    # Greedy phase: try flipping each bit, keep if it improves
    for _ in range(5):  # Multiple passes
        if time.time() > deadline:
            break
        improved = False
        # Random order to break symmetry
        indices = rng.permutation(N)
        
        for i in indices:
            if time.time() > deadline:
                return best_h, best_val
            
            # Calculate change in acf if we flip h[i]
            # d_acf[k] = -2 * h[i] * (h[i+k] + h[i-k]) for k >= 1
            d_acf = np.zeros(N)
            
            if i < N - 1:
                right_len = N - 1 - i
                d_acf[1:right_len+1] += -2 * h[i] * h[i+1:]
            
            if i > 0:
                left_len = i
                d_acf[1:left_len+1] += -2 * h[i] * h[:i][::-1]
            
            new_acf = acf + d_acf
            new_val = _compute_c3_bound_from_acf(new_acf)
            
            if new_val < current_val:
                h[i] = -h[i]
                acf = new_acf
                current_val = new_val
                improved = True
                
                if current_val < best_val:
                    best_val = current_val
                    best_h = h.copy()
                    best_acf = acf.copy()
        
        if not improved:
            break
    
    # Simulated annealing phase to escape local minima
    T = 0.5  # Initial temperature
    cooling = 0.998
    
    while time.time() < deadline and T > 1e-6:
        indices = rng.permutation(N)
        
        for i in indices:
            if time.time() > deadline:
                return best_h, best_val
            
            d_acf = np.zeros(N)
            
            if i < N - 1:
                right_len = N - 1 - i
                d_acf[1:right_len+1] += -2 * h[i] * h[i+1:]
            
            if i > 0:
                left_len = i
                d_acf[1:left_len+1] += -2 * h[i] * h[:i][::-1]
            
            new_acf = acf + d_acf
            new_val = _compute_c3_bound_from_acf(new_acf)
            
            delta = new_val - current_val
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                h[i] = -h[i]
                acf = new_acf
                current_val = new_val
                
                if current_val < best_val:
                    best_val = current_val
                    best_h = h.copy()
                    best_acf = acf.copy()
        
        T *= cooling
    
    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    rng = np.random.default_rng(123)
    
    best_h = None
    best_val = np.inf
    best_acf = None
    
    # Phase 1: Spectral initialization with multiple random phases
    for restart in range(50):
        if time.time() > deadline - 10:
            break
            
        h = _spectral_init(rng)
        acf = _compute_acf(h)
        val = _compute_c3_bound_from_acf(acf)
        
        if val < best_val:
            best_h = h.copy()
            best_val = val
            best_acf = acf.copy()
    
    # Phase 2: Random initializations
    for restart in range(30):
        if time.time() > deadline - 10:
            break
            
        h = rng.choice([-1.0, 1.0], size=N)
        acf = _compute_acf(h)
        val = _compute_c3_bound_from_acf(acf)
        
        if val < best_val:
            best_h = h.copy()
            best_val = val
            best_acf = acf.copy()
    
    # Phase 3: Local search on the best initializations
    if best_h is not None:
        # Run local search on the best found so far
        res_h, res_val = _local_search_bitflip(best_h, best_acf, best_val, deadline, rng)
        if res_val < best_val:
            best_h = res_h
            best_val = res_val
            best_acf = _compute_acf(res_h)
        
        # Run local search on a few other good initializations
        # Re-run spectral init and random init, keep top few, and polish
        candidates = []
        
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            h = _spectral_init(rng)
            acf = _compute_acf(h)
            val = _compute_c3_bound_from_acf(acf)
            candidates.append((val, h, acf))
        
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            h = rng.choice([-1.0, 1.0], size=N)
            acf = _compute_acf(h)
            val = _compute_c3_bound_from_acf(acf)
            candidates.append((val, h, acf))
        
        candidates.sort(key=lambda x: x[0])
        
        # Polish top 10 candidates
        for val, h, acf in candidates[:10]:
            if time.time() > deadline - 2:
                break
            res_h, res_val = _local_search_bitflip(h, acf, val, deadline, rng)
            if res_val < best_val:
                best_h = res_h
                best_val = res_val
                best_acf = _compute_acf(res_h)
    
    return best_h

# EVOLVE-BLOCK-END
