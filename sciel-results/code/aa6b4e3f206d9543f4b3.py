"""
Spectral Design Strategy for Erdos Minimum Overlap.
Instead of optimizing h[i] directly, we optimize the Fourier domain representation.
We model h as a sum of complex exponentials with controlled amplitudes and phases.
The goal is to minimize the peak sidelobe level of the autocorrelation, which is
equivalent to maximizing the minimum autocorrelation value.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
TARGET_SUM = N / 2.0

def _autocorr_min(h):
    """Compute the minimum circular autocorrelation value."""
    H = np.fft.fft(h)
    power = H * np.conj(H)
    autocorr = np.real(np.fft.ifft(power))
    return np.min(autocorr)

def _max_overlap(h):
    """Compute the maximum overlap: N/2 - min_autocorr."""
    return N / 2.0 - _autocorr_min(h)

def _project_to_feasible(h):
    """
    Project h onto the feasible set: 0 <= h[i] <= 1, sum(h) = N/2.
    Uses a simple iterative water-filling style projection.
    """
    h = h.copy()
    h = np.real(h)
    
    # 1. Clip to [0, 1]
    h = np.clip(h, 0.0, 1.0)
    
    # 2. Adjust sum to exactly N/2
    # We need to distribute the difference (target - current) among elements
    # that have room (for positive diff) or value (for negative diff).
    
    for _ in range(10):
        current_sum = np.sum(h)
        diff = TARGET_SUM - current_sum
        if abs(diff) < 1e-12:
            break
        
        if diff > 0:
            # Need to increase sum. Find indices where h < 1
            mask = h < 1.0 - 1e-9
            if not np.any(mask):
                # If all are 1, we can't increase. But sum should be N/2, so this shouldn't happen if initial sum was close.
                # If it does, just scale down? No, that reduces sum.
                # Actually, if all are 1, sum is N. We need N/2. So we must decrease.
                # This implies the previous step over-corrected or initial state was bad.
                # Let's just handle the general case:
                mask = h > 0
                if not np.any(mask):
                    h = np.full(N, 0.5)
                    break
                # If we need to increase but no room, something is wrong.
                # Fallback: uniform
                h = np.full(N, 0.5)
                break
            
            # Distribute diff proportionally to room
            room = 1.0 - h
            room[~mask] = 0
            total_room = np.sum(room)
            if total_room < 1e-9:
                break
            h += diff * room / total_room
        else:
            # Need to decrease sum. Find indices where h > 0
            mask = h > 1e-9
            if not np.any(mask):
                h = np.full(N, 0.5)
                break
            
            # Distribute -diff proportionally to current value
            val = h
            val[~mask] = 0
            total_val = np.sum(val)
            if total_val < 1e-9:
                break
            h += diff * val / total_val
        
        h = np.clip(h, 0.0, 1.0)
    
    # Final exact correction
    diff = TARGET_SUM - np.sum(h)
    if abs(diff) > 1e-10:
        # Add/subtract evenly, then clip and hope for the best?
        # Better: find one element to adjust
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
        
        # If still not exact, fix one element
        diff = TARGET_SUM - np.sum(h)
        if abs(diff) > 1e-12:
            # Find an index that can absorb the diff
            if diff > 0:
                idx = np.argmin(h)
            else:
                idx = np.argmax(h)
            h[idx] += diff
            h[idx] = np.clip(h[idx], 0.0, 1.0)
            
            # If that failed, brute force fix
            if abs(np.sum(h) - TARGET_SUM) > 1e-12:
                # This is a last resort, might break [0,1] constraints slightly if not careful
                # But usually the projection loop handles it.
                pass
                
    return h

def _spectral_to_time(mags, phases):
    """
    Convert spectral magnitudes and phases to time-domain signal.
    We only use positive frequencies (1 to N//2) and DC (0).
    DC is fixed to ensure mean 0.5 (H[0] = N/2).
    """
    # Initialize H
    H = np.zeros(N, dtype=complex)
    H[0] = TARGET_SUM  # DC component ensures sum is N/2
    
    # Set positive frequencies
    # mags and phases are for k=1 to N//2
    k_indices = np.arange(1, N//2 + 1)
    H[k_indices] = mags * np.exp(1j * phases)
    
    # Set negative frequencies (conjugate symmetry for real signal)
    # H[N-k] = conj(H[k])
    neg_indices = N - k_indices
    H[neg_indices] = np.conj(H[k_indices])
    
    # Inverse FFT
    h = np.fft.ifft(H).real
    return h

def _objective_spectral(params):
    """
    Objective function for spectral optimization.
    params: [mags (N//2), phases (N//2)]
    """
    half_N = N // 2
    mags = params[:half_N]
    phases = params[half_N:]
    
    # Ensure non-negative magnitudes
    mags = np.abs(mags)
    
    h = _spectral_to_time(mags, phases)
    h_proj = _project_to_feasible(h)
    
    # We want to minimize max overlap, which is N/2 - min_autocorr
    # So we minimize (N/2 - min_autocorr)
    return _max_overlap(h_proj)

def _random_spectral_init():
    """Generate random initial spectral parameters."""
    half_N = N // 2
    # Random magnitudes, small to start to avoid huge peaks
    mags = np.random.uniform(0, 5, half_N)
    phases = np.random.uniform(0, 2*np.pi, half_N)
    return np.concatenate([mags, phases])

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # 1. Try a few random spectral initializations
    # We use Nelder-Mead on the spectral parameters
    # This is a different search space than time-domain
    
    # Number of restarts depends on time budget
    # Each Nelder-Mead call might take a few seconds.
    # Let's try 5-10 restarts.
    
    np.random.seed(42)
    
    # Initial candidate: uniform
    current_h = np.full(N, 0.5)
    current_score = _max_overlap(current_h)
    best_h = current_h.copy()
    best_score = current_score
    
    # Also try a simple binary alternating pattern as a baseline
    h_alt = np.zeros(N)
    h_alt[::2] = 1.0
    # Sum is 50, so it's valid.
    score_alt = _max_overlap(h_alt)
    if score_alt < best_score:
        best_score = score_alt
        best_h = h_alt.copy()
    
    # Spectral Optimization
    # We optimize in the domain of (mags, phases)
    # To help the optimizer, we might want to fix the DC and maybe the first few low frequencies
    # to something reasonable, but let's start random.
    
    max_restarts = 10
    
    for i in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        params0 = _random_spectral_init()
        
        # Use Nelder-Mead
        # Note: The objective involves a projection which is non-smooth.
        # Nelder-Mead handles this okay-ish.
        
        try:
            result = minimize(
                _objective_spectral, 
                params0, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True}
            )
            
            if result.success or result.fun < best_score:
                h_opt = _spectral_to_time(result.x[:N//2], result.x[N//2:])
                h_opt = _project_to_feasible(h_opt)
                score_opt = _max_overlap(h_opt)
                
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt.copy()
        except Exception as e:
            pass

    # 2. Local refinement in time domain from the best spectral solution
    # The spectral solution might have high-frequency artifacts or projection issues.
    # A local search in time domain can polish it.
    
    if time.time() < deadline - 5:
        # Simple coordinate descent / local search
        # We try to swap values or adjust small neighbors
        # To be fast, we only check a few random moves per iteration
        
        h = best_h.copy()
        score = best_score
        N_local = N
        
        # Precompute FFT for fast overlap? 
        # _max_overlap uses FFT which is O(N log N). 
        # For N=100, it's very fast.
        
        for it in range(1000):
            if time.time() > deadline - 2:
                break
                
            # Pick two random indices
            i1 = np.random.randint(N_local)
            i2 = np.random.randint(N_local)
            if i1 == i2:
                continue
                
            # Try swapping or averaging
            # Move 1: Swap
            h_new = h.copy()
            h_new[i1], h_new[i2] = h[i2], h[i1]
            # Sum is preserved
            s_new = _max_overlap(h_new)
            if s_new < score - 1e-9:
                h = h_new
                score = s_new
                continue
                
            # Move 2: Perturb i1, adjust i2 to keep sum
            delta = np.random.uniform(-0.1, 0.1)
            h_new = h.copy()
            val1 = h[i1] + delta
            val2 = h[i2] - delta
            
            if 0 <= val1 <= 1 and 0 <= val2 <= 1:
                h_new[i1] = val1
                h_new[i2] = val2
                s_new = _max_overlap(h_new)
                if s_new < score - 1e-9:
                    h = h_new
                    score = s_new
        
        if score < best_score:
            best_score = score
            best_h = h
    
    return best_h

# EVOLVE-BLOCK-END
