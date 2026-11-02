"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_fft(sig, n_fft=None):
    """Compute autoconvolution using FFT."""
    if n_fft is None:
        n_fft = 1
        while n_fft < 2 * N - 1:
            n_fft *= 2
    f = np.fft.rfft(sig, n_fft)
    ac = np.fft.irfft(f * np.conj(f), n_fft)
    return ac[:2*N-1]

def _objective(sig):
    """Compute C3' = max|autoconv| / (integral)^2."""
    integral = np.sum(sig)
    if abs(integral) < 1e-12:
        return 1e10
    ac = _autoconv_fft(sig)
    max_abs = np.max(np.abs(ac))
    return max_abs / (integral * integral)

def _soft_max_and_grad(sig, beta=1000.0):
    """Smooth approximation of max|ac| with gradient."""
    ac = _autoconv_fft(sig)
    abs_ac = np.abs(ac)
    # Soft max using log-sum-exp
    m = np.max(abs_ac)
    soft_max = m + np.log(np.sum(np.exp(-beta * (abs_ac - m)))) / beta
    
    # Gradient of soft_max w.r.t. sig
    weights = np.exp(-beta * (abs_ac - m))
    weights /= np.sum(weights)
    
    # d|ac[k]|/d sig[i] involves the derivative of ac[k] w.r.t. sig[i]
    # ac[k] = sum_j sig[j] * sig[j+k]
    # For k >= 0: dac/dsig[i] = sig[i-k] + sig[i+k] (with bounds)
    # For k < 0: use symmetry
    
    # Build gradient matrix implicitly
    grad_soft = np.zeros(N)
    for k in range(2*N-1):
        w = weights[k]
        if k < N:
            # ac[k] = sum_{j=0}^{N-k-1} sig[j]*sig[j+k]
            # d(ac[k])/d(sig[i]) = sig[i-k] (if i>=k) + sig[i+k] (if i+k<N)
            for i in range(N):
                if i >= k:
                    grad_soft[i] += w * sig[i-k]
                if i + k < N:
                    grad_soft[i] += w * sig[i+k]
        else:
            # Use symmetry: ac[k] = ac[2N-2-k]
            k_sym = 2*N-2-k
            for i in range(N):
                if i >= k_sym:
                    grad_soft[i] += w * sig[i-k_sym]
                if i + k_sym < N:
                    grad_soft[i] += w * sig[i+k_sym]
    
    # Normalize by integral^2 and add derivative of 1/integral^2
    integral = np.sum(sig)
    obj = soft_max / (integral * integral)
    grad_obj = grad_soft / (integral * integral) - 2 * soft_max * np.ones(N) / (integral**3)
    
    return obj, grad_obj

def _coordinate_descent(sig, max_iters=5000, deadline=None):
    """Coordinate descent: flip individual bits to minimize objective."""
    best_sig = sig.copy()
    best_score = _objective(best_sig)
    
    for iteration in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        # Try flipping each position
        for i in range(N):
            test_sig = best_sig.copy()
            test_sig[i] *= -1
            test_score = _objective(test_sig)
            if test_score < best_score - 1e-12:
                best_sig = test_sig
                best_score = test_score
                improved = True
        
        if not improved:
            break
    
    return best_sig, best_score

def _spectral_init():
    """Initialize with a spectral structure to avoid trivial constant."""
    # Start with a low-frequency cosine pattern
    t = np.arange(N)
    sig = np.sign(np.cos(2 * np.pi * t / (N/4)))
    return sig

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Multiple initialization strategies
    candidates = []
    
    # 1. Spectral initialization
    sig1 = _spectral_init()
    candidates.append(sig1)
    
    # 2. Random initialization
    np.random.seed(42)
    sig2 = np.random.choice([-1, 1], size=N)
    candidates.append(sig2)
    
    # 3. Block structure
    sig3 = np.zeros(N)
    block_size = N // 8
    for b in range(8):
        sig3[b*block_size:(b+1)*block_size] = 1 if b % 2 == 0 else -1
    candidates.append(sig3)
    
    best_sig = None
    best_score = float('inf')
    
    for sig in candidates:
        if time.time() > deadline:
            break
        
        # Coordinate descent refinement
        sig_cd, score_cd = _coordinate_descent(sig, max_iters=2000, deadline=deadline)
        if score_cd < best_score:
            best_score = score_cd
            best_sig = sig_cd
    
    # If we still have time, try L-BFGS on a relaxed version
    if time.time() < deadline - 20:
        # Relax to continuous, optimize, then round
        x0 = np.ones(N)
        
        def objective_continuous(x):
            ac = _autoconv_fft(x)
            max_abs = np.max(np.abs(ac))
            integral = np.sum(x)
            if abs(integral) < 1e-12:
                return 1e10
            return max_abs / (integral * integral)
        
        def gradient_continuous(x):
            ac = _autoconv_fft(x)
            abs_ac = np.abs(ac)
            m = np.max(abs_ac)
            beta = 500.0
            weights = np.exp(-beta * (abs_ac - m))
            weights /= np.sum(weights)
            
            grad = np.zeros(N)
            for k in range(2*N-1):
                w = weights[k]
                k_eff = k if k < N else 2*N-2-k
                for i in range(N):
                    if i >= k_eff:
                        grad[i] += w * x[i-k_eff]
                    if i + k_eff < N:
                        grad[i] += w * x[i+k_eff]
            
            integral = np.sum(x)
            obj = m + np.log(np.sum(np.exp(-beta * (abs_ac - m)))) / beta
            grad_obj = grad / (integral * integral) - 2 * obj * np.ones(N) / (integral**3)
            return grad_obj
        
        try:
            result = minimize(objective_continuous, x0, jac=gradient_continuous, 
                             method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-10})
            sig_lbgfs = np.sign(result.x)
            score_lbgfs = _objective(sig_lbgfs)
            if score_lbgfs < best_score:
                best_score = score_lbgfs
                best_sig = sig_lbgfs
        except:
            pass
    
    # Final coordinate descent polish
    if best_sig is not None and time.time() < deadline:
        best_sig, best_score = _coordinate_descent(best_sig, max_iters=1000, deadline=deadline)
    
    return best_sig
# EVOLVE-BLOCK-END
