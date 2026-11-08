"""Combined approach: FFT-based overlap + Fourier-mode parameterization + smooth optimization."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 140.0  # seconds, with 10s safety margin before 150s limit

def _compute_max_overlap(h):
    """
    Compute the maximum overlap of h with shifted 1-h over all circular shifts.
    
    h is an array of N values on [0, 2], so dx = 2/N.
    sum(h) * dx = 1, so sum(h) = N/2.
    
    overlap(k) = sum_j h[j] * (1 - h[(j+k)%N]) * dx
               = (sum_j h[j] - sum_j h[j]*h[(j+k)%N]) * dx
               = (N/2 - C[k]) * dx
    where C[k] = sum_j h[j]*h[(j+k)%N]
    """
    n = len(h)
    dx = 2.0 / n
    sum_h = np.sum(h)
    
    # Compute circular cross-correlation C[k] = sum_j h[j]*h[(j+k)%n]
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    C = np.zeros(n)
    C[0] = autocorr[0]
    if n > 1:
        C[1:] = autocorr[n-1:0:-1]
    
    overlaps = (sum_h - C) * dx
    return np.max(overlaps)

def _soft_max(values, temperature=0.01):
    """Soft maximum using log-sum-exp for smooth optimization."""
    shifted = values - np.max(values)
    log_sum = np.log(np.sum(np.exp(shifted / temperature))) * temperature
    return np.max(values) + log_sum

def _build_h_from_fourier(coeffs, alpha=5.0):
    """
    Build h from Fourier coefficients using smooth tanh mapping.
    coeffs = [a_1, b_1, a_2, b_2, ..., a_m, b_m]
    h(x) = 0.5 + sum_{k=1}^{m} (a_k * cos(pi*k*x) + b_k * sin(pi*k*x))
    Then apply smooth mapping: h_mapped = 0.5 + 0.5 * tanh(alpha * (h - 0.5))
    This keeps h in (0,1) smoothly without hard clipping.
    """
    x = np.linspace(0, 2, N, endpoint=False)
    h_raw = np.full(N, 0.5)
    n_modes = len(coeffs) // 2
    for k in range(1, n_modes + 1):
        a_k = coeffs[2*(k-1)]
        b_k = coeffs[2*(k-1)+1]
        h_raw += a_k * np.cos(np.pi * k * x) + b_k * np.sin(np.pi * k * x)
    
    # Smooth mapping to (0,1): 0.5 + 0.5*tanh(alpha*(h_raw - 0.5))
    h = 0.5 + 0.5 * np.tanh(alpha * (h_raw - 0.5))
    
    # Normalize to sum = N/2
    target_sum = N / 2.0
    current_sum = h.sum()
    if current_sum > 1e-10:
        # Use a smooth affine correction: h = h * s + d where s and d chosen so mean = 0.5
        # Simpler: scale then shift
        s = target_sum / current_sum
        h = h * s
        # Now adjust to ensure mean is exactly 0.5 (sum = N/2)
        # Since tanh keeps values in (0,1), small adjustments are safe
        diff = target_sum - h.sum()
        if abs(diff) > 1e-12:
            # Distribute evenly (safe since values are in (0,1) with margin from tanh)
            h = h + diff / N
            h = np.clip(h, 0.0, 1.0)
    
    return h

def _objective_fourier_smooth(coeffs, temperature=0.02, alpha=5.0):
    """Smooth objective using soft-max for the overlap."""
    h = _build_h_from_fourier(coeffs, alpha=alpha)
    n = len(h)
    dx = 2.0 / n
    sum_h = np.sum(h)
    
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    C = np.zeros(n)
    C[0] = autocorr[0]
    if n > 1:
        C[1:] = autocorr[n-1:0:-1]
    
    overlaps = (sum_h - C) * dx
    return _soft_max(overlaps, temperature)

def _objective_fourier(coeffs, alpha=5.0):
    """Objective function: max overlap from Fourier coefficients."""
    h = _build_h_from_fourier(coeffs, alpha=alpha)
    return _compute_max_overlap(h)

def _try_candidate(h):
    """Try a candidate h, return (h, score) if valid and better, else (None, inf)."""
    h = np.clip(h, 0.0, 1.0)
    target_sum = N / 2.0
    current_sum = h.sum()
    if current_sum > 1e-10:
        scale = target_sum / current_sum
        h = h * scale
        h = np.clip(h, 0.0, 1.0)
        # Exact sum correction
        diff = target_sum - h.sum()
        if abs(diff) > 1e-12:
            mask = (h > 1e-8) & (h < 1.0 - 1e-8)
            if np.sum(mask) > 0:
                h[mask] += diff / np.sum(mask)
                h = np.clip(h, 0.0, 1.0)
    score = _compute_max_overlap(h)
    return h, score

def _spectral_penalty(coeffs, n_modes, weight=0.001):
    """Penalty for high-frequency components to encourage smooth solutions.
    Reduced weight to allow higher frequencies if needed for minimax."""
    total = 0.0
    for k in range(1, n_modes + 1):
        a_k = coeffs[2*(k-1)]
        b_k = coeffs[2*(k-1)+1]
        # Weight higher frequencies more heavily, but less aggressively
        w = weight * k
        total += w * (a_k**2 + b_k**2)
    return total

def construct():
    start_time = time.time()
    
    # Baseline: uniform 0.5
    best_h = np.full(N, 0.5)
    best_score = _compute_max_overlap(best_h)  # should be 0.5
    
    x = np.linspace(0, 2, N, endpoint=False)
    
    # Try analytic candidates first
    candidates = []
    
    # h = 0.5 + 0.5*sin(pi*x)
    h1 = 0.5 + 0.5 * np.sin(np.pi * x)
    candidates.append(h1)
    
    # h = 0.5 + 0.5*cos(pi*x)
    h2 = 0.5 + 0.5 * np.cos(np.pi * x)
    candidates.append(h2)
    
    # Square wave: 1 on [0,1), 0 on [1,2)
    h3 = (x < 1.0).astype(float)
    candidates.append(h3)
    
    # h = 0.5 + 0.3*sin(2*pi*x)
    h4 = 0.5 + 0.3 * np.sin(2 * np.pi * x)
    candidates.append(h4)
    
    # h = 0.5 + 0.4*cos(2*pi*x)
    h5 = 0.5 + 0.4 * np.cos(2 * np.pi * x)
    candidates.append(h5)
    
    # h = 0.5 + 0.3*sin(3*pi*x)
    h6 = 0.5 + 0.3 * np.sin(3 * np.pi * x)
    candidates.append(h6)
    
    # h = 0.5 + 0.25*sin(4*pi*x)
    h7 = 0.5 + 0.25 * np.sin(4 * np.pi * x)
    candidates.append(h7)
    
    # h = 0.5 + 0.2*cos(5*pi*x)
    h8 = 0.5 + 0.2 * np.cos(5 * np.pi * x)
    candidates.append(h8)
    
    # Additional candidates: smoother transitions
    # h = 0.5 + 0.4*sin(pi*x) + 0.1*sin(3*pi*x)
    h9 = 0.5 + 0.4 * np.sin(np.pi * x) + 0.1 * np.sin(3 * np.pi * x)
    candidates.append(h9)
    
    # h = 0.5 + 0.35*cos(pi*x) + 0.15*cos(3*pi*x)
    h10 = 0.5 + 0.35 * np.cos(np.pi * x) + 0.15 * np.cos(3 * np.pi * x)
    candidates.append(h10)
    
    # h = 0.5 + 0.3*sin(pi*x) + 0.2*sin(2*pi*x)
    h11 = 0.5 + 0.3 * np.sin(np.pi * x) + 0.2 * np.sin(2 * np.pi * x)
    candidates.append(h11)
    
    for h_cand in candidates:
        h_new, score = _try_candidate(h_cand)
        if score < best_score:
            best_score = score
            best_h = h_new
    
    # Multi-start optimization with Fourier coefficients
    # Increased modes to allow more complex shapes
    n_modes = 25 
    n_coeffs = 2 * n_modes
    bounds = [(-0.5, 0.5)] * n_coeffs
    
    rng = np.random.default_rng(42)
    
    # Increased starts to explore the higher-dimensional space
    n_starts = 100
    for start_idx in range(n_starts):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Random initial point
        if start_idx == 0:
            x0 = np.zeros(n_coeffs)
        elif start_idx == 1:
            # Start from sine wave
            x0 = np.zeros(n_coeffs)
            x0[1] = 0.5  # b_1 for sin(pi*x)
        elif start_idx == 2:
            # Start from cosine wave
            x0 = np.zeros(n_coeffs)
            x0[0] = 0.5  # a_1 for cos(pi*x)
        elif start_idx == 3:
            # Start from mixed
            x0 = np.zeros(n_coeffs)
            x0[0] = 0.3
            x0[1] = 0.3
        elif start_idx == 4:
            # Start from higher frequency
            x0 = np.zeros(n_coeffs)
            x0[4] = 0.4  # a_2
        elif start_idx == 5:
            # Start from 3rd harmonic
            x0 = np.zeros(n_coeffs)
            x0[5] = 0.3  # b_3
        elif start_idx == 6:
            # Start from combination
            x0 = np.zeros(n_coeffs)
            x0[0] = 0.2
            x0[1] = 0.2
            x0[4] = 0.15
            x0[5] = 0.15
        else:
            # Random with bias toward lower frequencies, but less decay to allow high freq
            x0 = rng.uniform(-0.4, 0.4, n_coeffs)
            # Decay high frequencies
            for k in range(n_modes):
                decay = 1.0 / (1.0 + 0.1 * k)
                x0[2*k] *= decay
                x0[2*k+1] *= decay
        
        try:
            # Use smooth objective with spectral penalty for better gradient behavior
            # Lower temperature for sharper max approximation
            def obj_with_penalty(c, temp=0.01, alpha=5.0):
                base = _objective_fourier_smooth(c, temperature=temp, alpha=alpha)
                penalty = _spectral_penalty(c, n_modes, weight=0.001)
                return base + penalty
            
            result = minimize(
                obj_with_penalty, x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            h_new = _build_h_from_fourier(result.x, alpha=5.0)
            score = _compute_max_overlap(h_new)
            
            if score < best_score:
                best_score = score
                best_h = h_new.copy()
        except Exception:
            pass
    
    # Polish phase: use exact (non-smooth) objective with tight tolerances
    if time.time() - start_time < DEADLINE - 10:
        try:
            # Get Fourier coefficients from best_h
            # Use more modes for refinement
            n_modes_refine = min(30, n_modes)
            A = np.zeros((N, 2 * n_modes_refine))
            for k in range(1, n_modes_refine + 1):
                A[:, 2*(k-1)] = np.cos(np.pi * k * x)
                A[:, 2*(k-1)+1] = np.sin(np.pi * k * x)
            target = best_h - 0.5
            coeffs, _, _, _ = np.linalg.lstsq(A, target, rcond=None)
            
            # Refine with smooth objective first (multiple temperatures)
            for temp in [0.01, 0.005, 0.002]:
                if time.time() - start_time > DEADLINE - 10:
                    break
                try:
                    result_smooth = minimize(
                        lambda c, t=temp: _objective_fourier_smooth(c, temperature=t, alpha=5.0), coeffs,
                        method='L-BFGS-B',
                        bounds=[(-0.5, 0.5)] * (2 * n_modes_refine),
                        options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13}
                    )
                    
                    h_smooth = _build_h_from_fourier(result_smooth.x, alpha=5.0)
                    score_smooth = _compute_max_overlap(h_smooth)
                    
                    if score_smooth < best_score:
                        best_score = score_smooth
                        best_h = h_smooth.copy()
                        coeffs = result_smooth.x
                    else:
                        coeffs = result_smooth.x
                except Exception:
                    pass
            
            # Then refine with exact objective
            result_exact = minimize(
                lambda c: _objective_fourier(c, alpha=5.0), coeffs,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * n_modes_refine),
                options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            
            h_exact = _build_h_from_fourier(result_exact.x, alpha=5.0)
            score_exact = _compute_max_overlap(h_exact)
            
            if score_exact < best_score:
                best_score = score_exact
                best_h = h_exact.copy()
        except Exception:
            pass
    
    # Final Nelder-Mead polish if time permits
    if time.time() - start_time < DEADLINE - 5:
        try:
            n_modes_nm = min(10, n_modes)
            A = np.zeros((N, 2 * n_modes_nm))
            for k in range(1, n_modes_nm + 1):
                A[:, 2*(k-1)] = np.cos(np.pi * k * x)
                A[:, 2*(k-1)+1] = np.sin(np.pi * k * x)
            target = best_h - 0.5
            coeffs_nm, _, _, _ = np.linalg.lstsq(A, target, rcond=None)
            
            result_nm = minimize(
                lambda c: _objective_fourier(c, alpha=5.0), coeffs_nm,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            h_nm = _build_h_from_fourier(result_nm.x, alpha=5.0)
            score_nm = _compute_max_overlap(h_nm)
            
            if score_nm < best_score:
                best_score = score_nm
                best_h = h_nm.copy()
        except Exception:
            pass
    
    # Final validation: ensure exact sum
    best_h = np.clip(best_h, 0.0, 1.0)
    target_sum = N / 2.0
    diff = target_sum - best_h.sum()
    if abs(diff) > 1e-10:
        mask = (best_h > 1e-8) & (best_h < 1.0 - 1e-8)
        if np.sum(mask) > 0:
            best_h[mask] += diff / np.sum(mask)
            best_h = np.clip(best_h, 0.0, 1.0)
    
    # Final check
    if abs(np.sum(best_h) - N/2) > 1e-6:
        best_h = np.full(N, 0.5)
    
    return best_h
