import numpy as np
import time

N = 50

def _autoconvolution(h):
    """Compute the full autoconvolution of h with itself."""
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)
    return conv[:2 * N - 1]

def _score(h):
    """Compute the score: ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h ** 2)
    conv = _autoconvolution(h)
    l_inf = np.max(np.abs(conv))
    if l_inf == 0:
        return 0.0
    return l2_sq / (l1 * l_inf)

def _h_from_complex(z):
    """
    Given a complex vector z of length N, compute h_k = |IDFT(z)|^2.
    This ensures h is nonnegative. We work with the full complex DFT.
    """
    # z is the DFT of some sequence. h = |IDFT(z)|^2
    # But we want h to be the time-domain signal whose DFT has magnitude squared related to z.
    # Actually, let's be precise:
    # Let x be a complex vector of length N.
    # Let h_k = |x_k|^2 ? No, that's just element-wise.
    # The strategy says: parameterize h as squared magnitude of DFT of a complex vector.
    # Let v be a complex vector of length N.
    # Let V = DFT(v).
    # Let h = |V|^2 ? No, h must be length N.
    # If V is length N, |V|^2 is length N.
    # So h_k = |V_k|^2 where V = DFT(v).
    # This means h is the power spectrum of v.
    # But the autoconvolution of h is what we care about.
    # Let's check: if h = |DFT(v)|^2, then h is nonnegative.
    # The autoconvolution of h is (h*h)[t] = sum_j h_j h_{j+t}.
    # In frequency domain, H(f) = DFT(h) = DFT(|V|^2).
    # This doesn't simplify the autoconvolution directly to a simple operation on V.
    
    # Alternative interpretation:
    # Let h be the time-domain signal.
    # Let H = DFT(h).
    # We know that autoconvolution h*h has DFT H^2.
    # The peak of h*h is related to the inverse DFT of H^2.
    # If we parameterize H directly as a complex vector, then h = IDFT(H).
    # But h must be real and nonnegative.
    # So we can parameterize H as a complex vector, compute h = IDFT(H),
    # and then take h = Re(h) or h = |h| or h = max(Re(h), 0).
    # Taking h = max(Re(IDFT(H)), 0) is non-smooth.
    # Taking h = |IDFT(H)| is non-smooth at zeros.
    # Taking h = Re(IDFT(H)) and projecting to nonnegative is non-smooth.
    
    # Let's try a different parameterization:
    # Let v be a complex vector of length N.
    # Let h = |v|^2 (element-wise squared magnitude).
    # This is smooth and nonnegative.
    # But this doesn't exploit the DFT structure of autoconvolution.
    
    # Let's re-read the directive: "Parameterize the 50 heights as the squared magnitude of the Discrete Fourier Transform (DFT) of a complex vector"
    # So h_k = |DFT(v)_k|^2 where v is a complex vector of length N.
    # This means h is the periodogram of v.
    # Let's stick with this.
    
    V = np.fft.fft(v)
    h = np.abs(V) ** 2
    return h

def _objective_spectral(v):
    """
    v is a complex vector of length N (represented as 2N real numbers).
    h = |DFT(v)|^2.
    We want to minimize log(||h||_1) + log(||h*h||_inf) subject to ||h||_2 = 1.
    Since h = |DFT(v)|^2, ||h||_2 = ||DFT(v)||_4^2? No.
    ||h||_2^2 = sum_k h_k^2 = sum_k |DFT(v)_k|^4.
    By Parseval, sum_k |DFT(v)_k|^2 = N * sum_k |v_k|^2.
    But we need sum |DFT(v)_k|^4.
    
    Actually, the scale invariance means we can fix ||h||_2 = 1.
    ||h||_2 = (sum_k h_k^2)^{1/2} = (sum_k |DFT(v)_k|^4)^{1/2}.
    This is not a simple norm of v.
    
    Let's just compute h from v, normalize h to have L2 norm 1, then compute the objective.
    """
    v = v.reshape(N, 2)
    v = v[:, 0] + 1j * v[:, 1]
    h = _h_from_complex(v)
    norm = np.linalg.norm(h)
    if norm < 1e-15:
        return 1e10
    h = h / norm
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 1e10
    conv = _autoconvolution(h)
    max_conv = np.max(conv)
    if max_conv < 1e-15:
        return 1e10
    return np.log(l1) + np.log(max_conv)

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    rng = np.random.RandomState(42)
    
    # Generate diverse starting points for the complex vector v
    starts = []
    
    # 1. Random complex
    for _ in range(20):
        v = rng.randn(N) + 1j * rng.randn(N)
        starts.append(v.flatten())
        
    # 2. Real positive
    for _ in range(10):
        v = rng.rand(N) + 0j
        starts.append(v.flatten())
        
    # 3. Structured: low frequency dominant
    for _ in range(10):
        v = np.zeros(N, dtype=complex)
        v[:5] = rng.randn(5) + 1j * rng.randn(5)
        starts.append(v.flatten())
        
    # 4. Structured: high frequency dominant
    for _ in range(10):
        v = np.zeros(N, dtype=complex)
        v[-5:] = rng.randn(5) + 1j * rng.randn(5)
        starts.append(v.flatten())
        
    # 5. Cosine phase modulation
    for _ in range(10):
        mag = rng.rand(N)
        phase = np.cos(2 * np.pi * rng.rand() * np.arange(N) / N)
        v = mag * np.exp(1j * phase)
        starts.append(v.flatten())
        
    # 6. Sinc-like in frequency
    for _ in range(10):
        x = np.arange(N) - N//2
        v = np.exp(-x**2 / (2 * 5**2)) + 0j
        starts.append(v.flatten())
        
    rng.shuffle(starts)
    
    for s in starts:
        if time.time() > deadline - 20:
            break
            
        # Use scipy minimize on the real representation
        from scipy.optimize import minimize
        
        def obj(x):
            return _objective_spectral(x)
        
        try:
            res = minimize(
                obj,
                s,
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-12}
            )
        except Exception:
            continue
            
        v_opt = res.x.reshape(N, 2)
        v_opt = v_opt[:, 0] + 1j * v_opt[:, 1]
        h_opt = _h_from_complex(v_opt)
        norm = np.linalg.norm(h_opt)
        if norm < 1e-15:
            continue
        h_opt = h_opt / norm
        
        score = _score(h_opt)
        if score > best_score:
            best_score = score
            best_h = h_opt.copy()
        
        # Local refinement
        if time.time() > deadline - 10:
            break
        try:
            res2 = minimize(
                obj,
                res.x,
                method='L-BFGS-B',
                options={'maxiter': 80, 'ftol': 1e-14}
            )
            v_opt2 = res2.x.reshape(N, 2)
            v_opt2 = v_opt2[:, 0] + 1j * v_opt2[:, 1]
            h_opt2 = _h_from_complex(v_opt2)
            norm2 = np.linalg.norm(h_opt2)
            if norm2 > 1e-15:
                h_opt2 = h_opt2 / norm2
                score2 = _score(h_opt2)
                if score2 > best_score:
                    best_score = score2
                    best_h = h_opt2.copy()
        except Exception:
            pass
    
    # Final polish
    if best_score > 0 and time.time() < deadline - 5:
        # Try a few more random restarts if time permits
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            v = rng.randn(N) + 1j * rng.randn(N)
            s = v.flatten()
            from scipy.optimize import minimize
            def obj(x):
                return _objective_spectral(x)
            try:
                res = minimize(obj, s, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-15})
                v_opt = res.x.reshape(N, 2)
                v_opt = v_opt[:, 0] + 1j * v_opt[:, 1]
                h_opt = _h_from_complex(v_opt)
                norm = np.linalg.norm(h_opt)
                if norm > 1e-15:
                    h_opt = h_opt / norm
                    score = _score(h_opt)
                    if score > best_score:
                        best_score = score
                        best_h = h_opt.copy()
            except Exception:
                pass
    
    best_h = np.maximum(best_h, 0)
    return list(best_h)

if __name__ == "__main__":
    result = construct()
    print("Best score:", _score(np.array(result)))
