import numpy as np
import time
import scipy.optimize

N = 400

def _autoconv_lin(x):
    """Compute linear autoconvolution using FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    return np.fft.irfft(fx * fx, size)[:2 * n - 1]

def _metric_std(x):
    """Standard metric: max |c_k| / S^2 including k=0."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    return float(np.max(np.abs(c))) / (s * s)

def _flat_spectrum_with_phase_sum_constraint(magnitude_target, phases, sum_target):
    """
    Given a target magnitude and phases, compute the time-domain sequence
    and adjust to meet the sum constraint.
    """
    # Reconstruct from magnitude and phase
    n = len(magnitude_target)
    X = magnitude_target * np.exp(1j * phases)
    x = np.fft.ifft(X).real
    
    # Adjust to meet sum constraint
    s = np.sum(x)
    if abs(s) > 1e-12:
        x = x * (sum_target / s)
    
    return x

def _optimize_phases_for_flat_spectrum(magnitude_target, sum_target, initial_phases, max_iters=200, tol=1e-8):
    """
    Optimize the phases to achieve a flat magnitude spectrum while maintaining
    the sum constraint. We use gradient descent on the phases.
    """
    phases = initial_phases.copy()
    n = len(magnitude_target)
    
    best_metric = float('inf')
    best_phases = phases.copy()
    
    for it in range(max_iters):
        # Compute current x from magnitude and phase
        X = magnitude_target * np.exp(1j * phases)
        x = np.fft.ifft(X).real
        
        # Enforce sum constraint
        s = np.sum(x)
        if abs(s) > 1e-12:
            x = x * (sum_target / s)
        
        # Compute metric
        metric = _metric_std(x)
        if metric < best_metric:
            best_metric = metric
            best_phases = phases.copy()
        
        if metric < tol:
            break
        
        # Compute gradient w.r.t. phases
        # We want to minimize the peak of |autoconv|, which in frequency domain
        # is related to |X|^2. For flat spectrum, we want |X_k| to be constant.
        # The gradient of the peak w.r.t. phase involves the autoconvolution structure.
        
        # Alternative: use the fact that for a given magnitude, the phases that
        # minimize the time-domain peak are those that make x as "flat" as possible.
        # This is equivalent to making the DFT phases such that x is close to constant.
        
        # Compute the current DFT
        X_current = np.fft.fft(x)
        
        # The ideal flat spectrum has all magnitudes equal to sum_target / sqrt(n)
        # (since |X_0| = sum, and for flat spectrum |X_k| = sum/sqrt(n) for k != 0)
        
        # Gradient descent on phases: we want to minimize the variance of |X_k|
        # which is equivalent to making the spectrum flat.
        
        # For a complex number X_k = |X_k| e^{i phi_k}, the gradient of |X_k|^2
        # w.r.t. phi_k is 0 (magnitude doesn't depend on phase directly).
        # But the constraint is on the time-domain signal.
        
        # Let's use a different approach: minimize the peak of the autoconvolution
        # directly using subgradient methods on the phases.
        
        # Compute autoconvolution
        c = _autoconv_lin(x)
        abs_c = np.abs(c)
        m = np.max(abs_c)
        
        if m < 1e-15:
            break
        
        # Find active lags
        threshold = m * 0.999
        active_mask = abs_c >= threshold
        
        # Compute gradient w.r.t. phases
        # x = ifft(X), where X_k = |X_k| e^{i phi_k}
        # dx/dphi_k = i * |X_k| e^{i phi_k} * ifft_basis
        # This is complex; let's use a simpler approach.
        
        # Perturbation-based gradient estimation
        eps = 1e-4
        grad_phases = np.zeros(n)
        
        for k in range(n):
            phases_plus = phases.copy()
            phases_plus[k] += eps
            X_plus = magnitude_target * np.exp(1j * phases_plus)
            x_plus = np.fft.ifft(X_plus).real
            s_plus = np.sum(x_plus)
            if abs(s_plus) > 1e-12:
                x_plus = x_plus * (sum_target / s_plus)
            c_plus = _autoconv_lin(x_plus)
            m_plus = np.max(np.abs(c_plus))
            
            grad_phases[k] = (m_plus - m) / eps
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad_phases)
        if grad_norm < 1e-15:
            break
        
        # Adaptive step size
        step_size = 0.01 / grad_norm
        
        # Update phases
        phases = phases - step_size * grad_phases
        
        # Keep phases in [-pi, pi]
        phases = (phases + np.pi) % (2 * np.pi) - np.pi
    
    return best_phases, best_metric

def _generate_flat_spectrum_candidate(sum_target, rng, max_phase_iters=100):
    """
    Generate a candidate with flat magnitude spectrum and optimized phases.
    """
    n = N
    # Target magnitude: for a flat spectrum, |X_0| = sum_target, and for k != 0,
    # |X_k| = sum_target / sqrt(n) approximately.
    # More precisely, if x is constant, X_0 = n*x, X_k = 0 for k != 0.
    # For a "flat" spectrum in the sense of equal magnitude, we want |X_k| = C for all k.
    # Then sum(x) = X_0 = C (real), and the energy is n * C^2 / n = C^2 by Parseval.
    # But sum(x)^2 = C^2, so the metric would be max|c_k| / C^2.
    # For a flat spectrum, the autoconvolution in frequency is |X|^2 = C^2 (constant),
    # so the time-domain autoconvolution is C^2 * delta, meaning c_0 = C^2 and c_k = 0 for k != 0.
    # This gives metric = C^2 / C^2 = 1.
    
    # To do better, we need a spectrum that is NOT flat but has lower peak autoconvolution.
    # The key is that max|c_k| is related to the peak of |X|^2 in frequency.
    # By Parseval, sum|c_k|^2 = (1/2pi) * integral |X|^4 domega (continuous) or
    # sum|c_k|^2 = (1/N) sum|X_k|^4 (discrete, with appropriate scaling).
    
    # For linear autoconvolution, the relationship is more complex.
    # Let's just try to minimize the peak directly.
    
    # Start with random phases
    phases = rng.uniform(-np.pi, np.pi, size=n)
    
    # Target magnitude: try different values
    # For a sequence with sum S, X_0 = S. For other frequencies, we want |X_k| to be
    # as small as possible to minimize the autoconvolution peak.
    
    # Actually, let's think differently. The autoconvolution c_k = sum_i x_i x_{i-k}.
    # In frequency: C(omega) = |X(omega)|^2.
    # The peak of c_k is related to the peak of |X(omega)|^2.
    # To minimize max|c_k|, we want |X(omega)|^2 to be as flat as possible.
    # But X_0 = sum(x) is fixed. So we want |X_k| for k != 0 to be as small as possible,
    # but this is constrained by the time-domain signal being real.
    
    # For a real signal, X_{N-k} = conj(X_k). So we have N/2 independent complex values.
    # The constraint is that the inverse DFT is real.
    
    # Let's try: set X_0 = sum_target, and for k = 1, ..., N/2, set |X_k| = sum_target / sqrt(N-1)
    # and random phases. Then check if the inverse DFT is real (it should be if we enforce
    # conjugate symmetry).
    
    # Enforce conjugate symmetry
    X = np.zeros(n, dtype=complex)
    X[0] = sum_target  # Real
    for k in range(1, n // 2 + 1):
        if k < n - k:  # Not the Nyquist frequency
            mag = sum_target / np.sqrt(n - 1)
            X[k] = mag * np.exp(1j * phases[k])
            X[n - k] = np.conj(X[k])
        else:  # Nyquist frequency (only if n is even)
            mag = sum_target / np.sqrt(n - 1)
            X[k] = mag * np.exp(1j * phases[k])
            # For Nyquist, we need it to be real for x to be real
            X[k] = np.real(X[k])
    
    x = np.fft.ifft(X).real
    
    # Check if x is real (it should be by construction)
    # Enforce sum constraint
    s = np.sum(x)
    if abs(s) > 1e-12:
        x = x * (sum_target / s)
    
    metric = _metric_std(x)
    
    # Optimize phases
    if metric > 1.0:  # Only optimize if we can do better than the trivial bound
        phases_opt, metric_opt = _optimize_phases_for_flat_spectrum(
            np.abs(X), sum_target, phases, max_phase_iters=max_phase_iters
        )
        if metric_opt < metric:
            X = np.abs(X) * np.exp(1j * phases_opt)
            # Re-enforce conjugate symmetry
            for k in range(1, n // 2 + 1):
                if k < n - k:
                    X[n - k] = np.conj(X[k])
                else:
                    X[k] = np.real(X[k])
            x = np.fft.ifft(X).real
            s = np.sum(x)
            if abs(s) > 1e-12:
                x = x * (sum_target / s)
            metric = _metric_std(x)
    
    return x, metric

def _try_barker_like_sequences():
    """
    Try sequences inspired by Barker codes, which have low sidelobes in autocorrelation.
    Barker codes have the property that |c_k| <= 1 for k != 0 when the code has length N.
    """
    candidates = []
    
    # Known Barker sequences
    barker_codes = [
        [1, 1, 1, 1, 1, -1, 1],  # Length 7
        [1, 1, 1, -1, 1, -1, -1],  # Length 7
        [1, 1, -1, 1, 1, 1, -1, 1, -1, 1],  # Length 10
        [1, 1, 1, 1, -1, -1, -1, 1, -1, 1, -1, 1],  # Length 12
        [1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1],  # Length 13
    ]
    
    # Extend Barker codes to length N by repetition or padding
    for code in barker_codes:
        # Tile the code to reach length N
        x = np.tile(code, N // len(code) + 1)[:N]
        s = np.sum(x)
        if abs(s) > 1e-6:
            x = x / s
        candidates.append(x)
    
    # Also try random sign sequences with low autocorrelation
    rng = np.random.default_rng(42)
    for _ in range(10):
        x = rng.choice([-1.0, 1.0], size=N)
        # Check autocorrelation
        c = _autoconv_lin(x)
        max_offdiag = np.max(np.abs(c[1:]))
        if max_offdiag < 20:  # Low sidelobes
            s = np.sum(x)
            if abs(s) > 1e-6:
                x = x / s
            candidates.append(x)
    
    return candidates

def _try_fourier_design():
    """
    Design a sequence in the frequency domain to minimize the autoconvolution peak.
    """
    n = N
    candidates = []
    rng = np.random.default_rng(123)
    
    # Try different magnitude profiles
    for trial in range(20):
        # Random phases with conjugate symmetry
        phases = rng.uniform(-np.pi, np.pi, size=n)
        
        # Try different magnitude profiles
        if trial % 3 == 0:
            # Flat magnitude
            mags = np.ones(n)
        elif trial % 3 == 1:
            # Low-pass: larger magnitudes at low frequencies
            freqs = np.fft.fftfreq(n)
            mags = np.exp(-10 * np.abs(freqs))
        else:
            # High-pass: larger magnitudes at high frequencies
            freqs = np.fft.fftfreq(n)
            mags = np.exp(-10 * (1 - np.abs(freqs)))
        
        # Normalize so that X_0 = 1 (sum = 1)
        X = mags * np.exp(1j * phases)
        X[0] = abs(X[0])  # Make X_0 real and positive
        
        # Enforce conjugate symmetry
        for k in range(1, n // 2 + 1):
            if k < n - k:
                X[n - k] = np.conj(X[k])
            else:
                X[k] = np.real(X[k])
        
        x = np.fft.ifft(X).real
        s = np.sum(x)
        if abs(s) > 1e-6:
            x = x / s
        
        candidates.append(x)
    
    return candidates

def _subgradient_refine(x, max_iters=30, tol=1e-8):
    """
    Subgradient refinement for the non-smooth max|c_k| objective.
    """
    x = x.copy()
    n = len(x)
    
    for _ in range(max_iters):
        s = np.sum(x)
        if abs(s) < 1e-12:
            return x
        
        c = _autoconv_lin(x)
        abs_c = np.abs(c)
        m = np.max(abs_c)
        
        if m < 1e-15:
            return x
        
        # Find which lags are at the maximum
        threshold = m * 0.999
        active_mask = abs_c >= threshold
        
        # Compute subgradient w.r.t. x
        grad = np.zeros(n)
        
        for k in range(len(c)):
            if not active_mask[k]:
                continue
            
            for i in range(n):
                contrib = 0.0
                if i >= k and i - k < n:
                    contrib += x[i - k]
                if k > 0 and i + k < n:
                    contrib += x[i + k]
                grad[i] += contrib
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            return x
        
        step_size = 0.05 / grad_norm
        x_new = x - step_size * grad
        
        # Project: preserve sum
        s_new = np.sum(x_new)
        if abs(s_new) > 1e-12:
            x_new = x_new * (s / s_new)
        
        c_new = _autoconv_lin(x_new)
        m_new = np.max(np.abs(c_new))
        
        if abs(m_new - m) < tol:
            return x_new
        
        x = x_new
    
    return x

def construct():
    deadline = time.time() + 140.0
    rng = np.random.default_rng(42)
    
    best_x = np.ones(N)
    best_val = _metric_std(best_x)
    
    # Candidate 1: Constant sequence
    cand_const = np.ones(N)
    v_const = _metric_std(cand_const)
    if v_const < best_val:
        best_val = v_const
        best_x = cand_const
    
    # Candidate 2: Barker-like sequences
    if time.time() < deadline - 60:
        try:
            barker_cands = _try_barker_like_sequences()
            for cand in barker_cands:
                v = _metric_std(cand)
                if v < best_val:
                    best_val = v
                    best_x = cand
        except Exception:
            pass
    
    # Candidate 3: Fourier design
    if time.time() < deadline - 40:
        try:
            fourier_cands = _try_fourier_design()
            for cand in fourier_cands:
                v = _metric_std(cand)
                if v < best_val:
                    best_val = v
                    best_x = cand
        except Exception:
            pass
    
    # Candidate 4: Flat spectrum with optimized phases
    if time.time() < deadline - 30:
        try:
            for trial in range(5):
                if time.time() > deadline - 20:
                    break
                x_flat, v_flat = _generate_flat_spectrum_candidate(1.0, rng, max_phase_iters=50)
                if v_flat < best_val:
                    best_val = v_flat
                    best_x = x_flat
        except Exception:
            pass
    
    # Candidate 5: Subgradient refinement from best candidate
    if time.time() < deadline - 15:
        try:
            x_sub = _subgradient_refine(best_x, max_iters=30, tol=1e-8)
            v_sub = _metric_std(x_sub)
            if v_sub < best_val:
                best_val = v_sub
                best_x = x_sub
        except Exception:
            pass
    
    # Final: more subgradient iterations if time permits
    if time.time() < deadline - 5:
        try:
            x_final = _subgradient_refine(best_x, max_iters=20, tol=1e-10)
            v_final = _metric_std(x_final)
            if v_final < best_val:
                best_val = v_final
                best_x = x_final
        except Exception:
            pass
    
    return best_x
