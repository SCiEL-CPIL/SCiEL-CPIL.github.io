"""
Program for the Erdos minimum-overlap task using spectral amplitude optimization.

Strategy:
1. Represent h[i] = 0.5 + u[i].
2. The overlap profile C[k] = N/4 - autocorr_u[k], where autocorr_u is the circular autocorrelation of u.
3. autocorr_u[k] = IFFT(|FFT(u)|^2).real.
4. We want to minimize max_k C[k] = N/4 - min_k autocorr_u[k].
   This is equivalent to maximizing min_k autocorr_u[k].
5. We parameterize u by its DFT amplitudes. Let U[m] = A[m] * exp(i * phi[m]).
   We fix random phases phi[m] and optimize amplitudes A[m] >= 0.
   To ensure h[i] in [0,1], we add a soft penalty to the objective if h goes out of bounds.
   The constraint sum(h) = N/2 is satisfied if U[0] = 0, which implies A[0]=0.
   
   Objective: max_k C[k] + lambda * sum(softplus(-h[i]) + softplus(h[i]-1))
   
   We optimize log(A[m]) for m in 1..N/2 (and symmetric part) to keep A positive.
   Since u is real, U is Hermitian. We only need to optimize the first half of the spectrum.
   
   This is a structurally different approach: it directly shapes the frequency content 
   (power spectrum) to control the spatial autocorrelation, rather than moving h in space.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = 140.0
_t0 = time.time()

def _overlap_profile(h):
    h = np.asarray(h, dtype=float)
    one_minus_h = 1.0 - h
    H = np.fft.fft(h)
    OMH = np.fft.fft(one_minus_h)
    r = np.fft.ifft(H * np.conj(OMH)).real
    return r

def _score(h):
    h = np.asarray(h, dtype=float)
    return float(np.max(_overlap_profile(h)))

def _fix_sum(h):
    h = np.clip(np.asarray(h, dtype=float), 0.0, 1.0)
    diff = N / 2.0 - h.sum()
    if abs(diff) < 1e-12:
        return h
    step = diff / N
    h2 = h + step
    if h2.min() >= -1e-12 and h2.max() <= 1.0 + 1e-12:
        h2 = np.clip(h2, 0.0, 1.0)
        residual = N / 2.0 - h2.sum()
        if abs(residual) < 1e-9:
            h2 += residual / N
        return h2
    h2 = np.clip(h + step, 0.0, 1.0)
    return h2

def construct():
    global _t0
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)
    
    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE

    rng = np.random.default_rng(42)
    
    # We optimize the log-amplitudes of the DFT of u = h - 0.5
    # U[0] = 0 (DC term is 0 because mean(u)=0)
    # For real u, U[N-m] = conj(U[m]).
    # We optimize A[1..N/2-1] and A[N/2] (if N even).
    # Let's work with the full spectrum amplitudes for simplicity in reconstruction,
    # but constrain the optimization variables to the independent half.
    
    # Number of independent frequency bins: N//2
    # Indices: 1 to N//2 (inclusive)
    # Note: index N//2 is real-valued in DFT for even N.
    
    n_freqs = N // 2
    # Variables: log(A[m]) for m in 1..n_freqs
    # Initial amplitudes: small random values
    # We will use a "soft" constraint on h bounds via penalty.
    
    # Precompute phase vectors for several random phase configurations
    # We will try a few different random phase sets and optimize amplitudes for each.
    
    # Objective function for spectral optimization
    def spectral_objective(log_A_flat, phases):
        # log_A_flat corresponds to A[1..n_freqs]
        log_A = log_A_flat
        A = np.exp(log_A)
        
        # Construct U
        U = np.zeros(N, dtype=complex)
        # U[0] = 0
        for m in range(1, n_freqs + 1):
            U[m] = A[m-1] * np.exp(1j * phases[m])
            # Hermitian symmetry
            U[-m] = np.conj(U[m])
            
        # Reconstruct u
        u = np.fft.ifft(U).real
        h = 0.5 + u
        
        # Calculate score
        C = _overlap_profile(h)
        max_C = np.max(C)
        
        # Penalty for h out of [0, 1]
        # We use a large penalty to push h into bounds
        # softplus(x) is smooth approximation of max(0, x)
        # penalty = sum(softplus(-h) + softplus(h-1))
        # Using np.log1p(np.exp(x)) for stability
        def softplus(x):
            return np.where(x > 30, x, np.log1p(np.exp(x)))
        
        pen_low = np.sum(softplus(-h))
        pen_high = np.sum(softplus(h - 1.0))
        penalty = 10.0 * (pen_low + pen_high)
        
        return max_C + penalty

    # Gradient of spectral objective
    def spectral_gradient(log_A_flat, phases):
        # Numerical gradient is safer here given the complexity, 
        # but L-BFGS-B needs gradients. 
        # Let's try finite difference or use a trick.
        # Actually, we can derive it, but it's messy.
        # Given the time limit, let's use 'BFGS' or 'Nelder-Mead' which don't need gradients,
        # or use finite difference gradients if L-BFGS-B is too slow without them.
        # L-BFGS-B with finite diff might be okay for ~50 variables.
        return None # Signal to use finite diff if needed, or we implement it.

    # Let's implement a simple forward difference gradient for L-BFGS-B
    # Or just use 'Nelder-Mead' which is gradient-free and robust for small dims.
    # n_freqs = 50. Nelder-Mead might be slow.
    # Let's use 'BFGS' with numerical gradient.
    
    def opt_spectral(phases, init_log_A):
        # BFGS requires gradient. We'll provide a numerical one.
        eps = 1e-7
        def num_grad(x):
            grad = np.zeros_like(x)
            f0 = spectral_objective(x, phases)
            for i in range(len(x)):
                x_p = x.copy()
                x_p[i] += eps
                f_p = spectral_objective(x_p, phases)
                grad[i] = (f_p - f0) / eps
            return grad
        
        res = minimize(spectral_objective, init_log_A, args=(phases,), 
                       method='L-BFGS-B', 
                       jac=num_grad,
                       options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-6})
        return res.x

    # Try multiple random phase configurations
    # The optimal solution likely has specific phase relationships, but random phases 
    # allow the amplitudes to shape the spectrum. 
    # A known good structure for "flat" autocorrelation is a "chirp" or specific binary sequence,
    # but we are in continuous domain.
    
    # Let's try a few different random seeds for phases
    for phase_seed in range(20):
        if not deadline_ok():
            break
            
        rng_local = np.random.default_rng(phase_seed)
        # Random phases
        phases = rng_local.uniform(0, 2*np.pi, size=N)
        
        # Initial amplitudes: uniform small
        # We want to start with a "white noise" like spectrum
        init_A = rng_local.uniform(0.01, 0.1, size=n_freqs)
        init_log_A = np.log(init_A)
        
        try:
            opt_log_A = opt_spectral(phases, init_log_A)
            A = np.exp(opt_log_A)
            
            # Reconstruct h
            U = np.zeros(N, dtype=complex)
            for m in range(1, n_freqs + 1):
                U[m] = A[m-1] * np.exp(1j * phases[m])
                U[-m] = np.conj(U[m])
            
            u = np.fft.ifft(U).real
            h = 0.5 + u
            h = _fix_sum(h) # Hard project to ensure validity
            
            sc = _score(h)
            if sc < best_score - 1e-12:
                best_score = sc
                best_h = h.copy()
                
        except Exception as e:
            pass

    # Final polish: Local search in spatial domain on the best spectral result
    if deadline_ok():
        h = best_h.copy()
        eta = 0.005
        no_improve = 0
        while deadline_ok() and no_improve < 30:
            C = _overlap_profile(h)
            k_max = int(np.argmax(C)) # We want to minimize max C, so focus on the peak
            # Gradient of C[k_max] w.r.t h
            # C[k] = sum_i h[i] (1-h[i-k])
            # d C[k] / d h[j] = (1-h[(j-k)%N]) - h[(j+k)%N]
            shifted_minus = np.roll(h, k_max)
            shifted_plus = np.roll(h, -k_max)
            g = (1.0 - shifted_minus) - shifted_plus
            
            h_new = h - eta * g # Descent direction
            h_new = _fix_sum(h_new)
            sc_new = _score(h_new)
            
            if sc_new < best_score - 1e-10:
                best_score = sc_new
                best_h = h_new.copy()
                h = h_new.copy()
                no_improve = 0
            else:
                no_improve += 1
                eta *= 0.9
                # Perturb
                h = best_h + rng.normal(0, 0.005, size=N)
                h = _fix_sum(h)

    return best_h
