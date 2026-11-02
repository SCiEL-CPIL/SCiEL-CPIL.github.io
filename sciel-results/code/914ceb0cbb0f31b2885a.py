"""
Combined spectral magnitude shaping + refined time-domain optimization.
Strategy:
1. Shape the Power Spectral Density (PSD) to minimize L-infinity norm of its inverse FFT (autoconv).
   This decouples phase from amplitude, allowing us to find a 'good' magnitude spectrum first.
2. Recover the time-domain sequence via spectral phase retrieval (Gerchberg-Saxton style) 
   projected to the constraint set (continuous values, or signed if needed, but here we allow 
   continuous heights as per "signed step heights" which usually implies real values, 
   though the task says "signed step functions" which might imply +/-1. 
   However, the current program uses continuous values and gets 1.001. 
   The inspirations try +/-1. 
   The task says "sequence of signed step heights". 
   Let's stick to continuous values but enforce the structure via spectral shaping.
   Actually, the current program optimizes continuous x. 
   The prompt says "signed step heights". 
   If x is continuous, "signed" just means it can be negative. 
   If x is binary, it's +/-1. 
   The current score is 1.001. 
   Inspiration 1 (score 0.0001) tried +/-1 and failed badly. 
   Inspiration 2 (score 0.0000) tried spectral phase retrieval with binary projection and failed. 
   The current program is the best. 
   
   The directive says: "Formulate the sequence design as a non-convex spectral magnitude shaping problem...".
   
   So, we will:
   1. Optimize the magnitude spectrum |X_k|^2 to minimize max|IFFT(|X_k|^2)|. 
      Note: IFFT of PSD is the autoconvolution. 
      Let P_k = |X_k|^2. Then ac[k] = IFFT(P)[k] (with scaling).
      We want to minimize max_k |ac[k]| subject to sum(ac) = (sum x)^2 = S^2.
      Also P_k >= 0.
      This is a convex-ish problem in the domain of P? 
      max |IFFT(P)| is convex in P? 
      IFFT is linear. |.| is convex. max of convex is convex. 
      So minimizing a convex function over a convex set (P>=0, sum P = const) is a convex problem!
      Wait, the constraint is on the sum of x, not directly on P.
      sum(x) = X_0. So |X_0|^2 = P_0.
      If we fix P_0 (and thus S), we can minimize max |IFFT(P)| subject to P >= 0 and P_0 fixed?
      No, we can also vary S. 
      Objective: max |IFFT(P)| / S^2.
      Since S^2 = P_0 (for real x, X_0 is real and equals sum x), 
      Objective: max |IFFT(P)| / P_0.
      This is a homogeneous function. We can fix P_0 = 1 and minimize max |IFFT(P)|.
      Then scale x at the end? 
      If we find P*, we get ac* = IFFT(P*). 
      We need to find x such that |FFT(x)|^2 = P*. 
      This is phase retrieval. 
      
   2. Once we have a good P*, we recover x using phase retrieval.
   
   Let's implement this.
"""

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 115
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _psd_to_autoconv(P):
    """
    P is the power spectral density (length N or N_fft).
    Returns the autoconvolution.
    Note: For real x, P is symmetric.
    ac[k] = (1/L_fft) * sum_k P_k exp(i 2 pi k m / L_fft) ?
    Actually, if X = fft(x), then X*X is the DFT of the circular autoconvolution.
    For linear autoconvolution, we need zero-padding.
    Let's work in the frequency domain of the linear convolution length L_fft.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    # P is defined on N points? Or L_fft points?
    # The PSD of x (length N) has N points in FFT.
    # The autoconvolution is length 2N-1.
    # The DFT of the autoconvolution (zero-padded to L_fft) is X^2.
    # So if we optimize P (length N), we need to map it to X^2 (length L_fft).
    # This is tricky because X^2 is the product of X and X.
    # It's easier to optimize the magnitude of X directly, or the PSD of the zero-padded signal.
    
    # Let's optimize the magnitude of the zero-padded FFT.
    # Let X_pad be the FFT of x zero-padded to L_fft.
    # We want to minimize max |IFFT(|X_pad|^2)|.
    # |X_pad|^2 is the PSD of the zero-padded signal.
    # Let P_pad = |X_pad|^2.
    # ac = IFFT(P_pad).
    # We want P_pad to be non-negative.
    # Also, x is the first N samples of IFFT(X_pad).
    # And X_pad is the FFT of x zero-padded.
    # This implies X_pad is the DTFT of x evaluated at discrete frequencies.
    
    # Alternative: Optimize the magnitude spectrum of x (length N) directly?
    # If we fix |X_k| for k=0..N-1, can we find x?
    # Phase retrieval.
    
    # Let's try optimizing P_pad (length L_fft) directly.
    # Constraint: P_pad must be the square of the magnitude of the FFT of a zero-padded signal of length N.
    # This is a strong constraint.
    
    # Simpler approach:
    # 1. Optimize a "target" PSD P_target (length L_fft) to minimize max |IFFT(P_target)|.
    #    Subject to P_target >= 0 and P_target[0] = 1 (normalize).
    #    This is a convex optimization problem.
    # 2. Use P_target to initialize a phase retrieval algorithm to find x.
    
    return None

def _optimize_psd():
    """
    Optimize the PSD P (length L_fft) to minimize max |IFFT(P)|.
    P >= 0.
    P[0] = 1 (fix scale).
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    # Variables: P[1:L_fft] (P[0] fixed to 1)
    # Objective: max |IFFT(P)|
    # We can use a smooth approximation: mu * log(sum(exp(|IFFT(P)|/mu)))
    
    def psd_objective_and_grad(P_vars, mu=0.1):
        P = np.zeros(L_fft)
        P[0] = 1.0
        P[1:] = P_vars
        
        # Ensure non-negativity (projection or penalty)
        # For gradient descent, we can project P >= 0 after step.
        # Or use a barrier.
        
        # Compute IFFT
        ac = np.fft.irfft(P, n=L_fft)
        abs_ac = np.abs(ac)
        
        # LogSumExp
        max_abs = np.max(abs_ac)
        if max_abs < 1e-12:
            return 1e-12, np.zeros_like(P_vars)
        
        shifted = (abs_ac - max_abs) / mu
        exp_terms = np.exp(shifted)
        sum_exp = np.sum(exp_terms)
        log_sum_exp = max_abs / mu + np.log(sum_exp)
        
        M_val = mu * log_sum_exp
        
        # Gradient
        weights = np.sign(ac) * exp_terms / sum_exp
        
        # d(M)/dP[k] = (1/L_fft) * sum_m weights[m] exp(i 2 pi m k / L_fft)
        # This is the FFT of weights (with scaling).
        # ac[m] = (1/L_fft) sum_k P[k] exp(i 2 pi m k / L_fft)
        # d ac[m] / d P[k] = (1/L_fft) exp(i 2 pi m k / L_fft)
        # grad_M[k] = sum_m weights[m] d ac[m] / d P[k]
        #            = (1/L_fft) sum_m weights[m] exp(i 2 pi m k / L_fft)
        # This is (1/L_fft) * FFT(weights)[k]
        
        W_fft = np.fft.fft(weights, n=L_fft)
        grad_M = (1.0 / L_fft) * W_fft.real # P is real, so grad should be real
        
        # P[0] is fixed, so we only return grad for P[1:]
        grad_M_vars = grad_M[1:]
        
        return M_val, grad_M_vars
    
    # Initial P: Flat
    P_vars_init = np.ones(L_fft - 1)
    
    # Use L-BFGS-B
    result = minimize(
        psd_objective_and_grad,
        P_vars_init,
        jac=True,
        method='L-BFGS-B',
        bounds=[(0, None)] * (L_fft - 1),
        options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    P_opt = np.zeros(L_fft)
    P_opt[0] = 1.0
    P_opt[1:] = np.maximum(0, result.x)
    
    return P_opt

def _phase_retrieval(P_target, num_iters=100):
    """
    Recover x from target PSD P_target.
    Use Gerchberg-Saxton.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    # Initialize random phase
    rng = np.random.RandomState(42)
    phase = rng.uniform(0, 2 * np.pi, size=L_fft)
    # Enforce Hermitian symmetry for real x?
    # X[k] = P_target[k]^(1/2) * exp(i phase[k])
    # For real x, X[k] = conj(X[L_fft-k]).
    # So phase[L_fft-k] = -phase[k].
    # And magnitude must be symmetric.
    # P_target from PSD optimization should be symmetric?
    # IFFT of real P is real and symmetric? No, IFFT of real P is real but not necessarily symmetric.
    # Wait, if P is real, IFFT(P) is real.
    # But X = sqrt(P) * exp(i phi).
    # If we want x to be real, X must be Hermitian.
    # This requires P to be symmetric: P[k] = P[L_fft-k].
    # And phase to be antisymmetric: phi[k] = -phi[L_fft-k].
    
    # Check if P_target is symmetric.
    # If not, symmetrize it.
    P_sym = 0.5 * (P_target + P_target[::-1].copy())
    # Ensure P_sym[0] is correct.
    
    mag = np.sqrt(P_sym)
    
    # Initialize phase
    phi = rng.uniform(0, 2 * np.pi, size=L_fft // 2 + 1)
    phi[0] = 0.0 # DC phase is 0 (x is real, sum x > 0 ideally)
    
    X = np.zeros(L_fft, dtype=complex)
    X[:L_fft//2+1] = mag[:L_fft//2+1] * np.exp(1j * phi)
    # Fill the rest via Hermitian symmetry
    X[L_fft//2+1:] = np.conj(X[1:L_fft//2+1][::-1])
    
    x_best = None
    best_val = 1e10
    
    for it in range(num_iters):
        if _check_deadline():
            break
            
        # Inverse FFT
        x_full = np.fft.irfft(X, n=L_fft)
        x = x_full[:N]
        
        # Project to "signed step function"?
        # The task says "signed step heights".
        # If we project to +/-1, we might break the PSD match.
        # Let's keep continuous values for now, as the current program does.
        # But we can try to sharpen them.
        
        # Compute current objective
        ac = _autoconv_fft(x)
        s = np.sum(x)
        if abs(s) < 1e-10:
            val = 1e10
        else:
            val = np.max(np.abs(ac)) / (s * s)
            
        if val < best_val:
            best_val = val
            x_best = x.copy()
            
        # Update Phase
        X_new = np.fft.rfft(x, n=L_fft)
        # New phase
        phi_new = np.angle(X_new[:L_fft//2+1])
        phi_new[0] = 0.0
        # New magnitude (should be close to mag, but x changed)
        # We force magnitude to be mag (target)
        X_new_full = np.zeros(L_fft, dtype=complex)
        X_new_full[:L_fft//2+1] = mag[:L_fft//2+1] * np.exp(1j * phi_new)
        X_new_full[L_fft//2+1:] = np.conj(X_new_full[1:L_fft//2+1][::-1])
        
        X = X_new_full
        
    return x_best, best_val

def _refine_time_domain(x_init):
    """
    Refine x using L-BFGS on the smooth objective.
    """
    def obj_func(x):
        ac = _autoconv_fft(x)
        s = np.sum(x)
        if abs(s) < 1e-10:
            return 1e10, np.zeros(N)
        
        abs_ac = np.abs(ac)
        max_abs = np.max(abs_ac)
        if max_abs < 1e-12:
            return 1e-12, np.zeros(N)
        
        mu = 0.1
        shifted = (abs_ac - max_abs) / mu
        exp_terms = np.exp(shifted)
        sum_exp = np.sum(exp_terms)
        log_sum_exp = max_abs / mu + np.log(sum_exp)
        
        M_val = mu * log_sum_exp
        F_val = M_val / (s * s)
        
        weights = np.sign(ac) * exp_terms / sum_exp
        
        L = 2 * N - 1
        L_fft = 1
        while L_fft < L:
            L_fft *= 2
        
        W_fft = np.fft.rfft(weights, n=L_fft)
        X_fft = np.fft.rfft(x, n=L_fft)
        
        conv1 = np.fft.irfft(W_fft * X_fft, n=L_fft)[:L]
        conv2 = np.fft.irfft(np.conj(W_fft) * X_fft, n=L_fft)[:L]
        
        grad_M = (conv1 + conv2)[:N]
        
        grad_F = grad_M / (s * s) - 2.0 * M_val / (s * s * s) * np.ones(N)
        
        return F_val, grad_F

    result = minimize(
        obj_func,
        x_init,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    return result.x, result.fun

def construct():
    # 1. Optimize PSD
    P_target = _optimize_psd()
    
    # 2. Phase Retrieval
    x_init, val_init = _phase_retrieval(P_target, num_iters=50)
    
    if x_init is None:
        x_init = np.ones(N)
    
    # 3. Refine
    x_final, val_final = _refine_time_domain(x_init)
    
    # Check if refinement improved
    ac = _autoconv_fft(x_final)
    s = np.sum(x_final)
    val_final_true = np.max(np.abs(ac)) / (s * s)
    
    ac_init = _autoconv_fft(x_init)
    s_init = np.sum(x_init)
    val_init_true = np.max(np.abs(ac_init)) / (s_init * s_init)
    
    if val_final_true < val_init_true:
        return x_final
    else:
        return x_init

# EVOLVE-BLOCK-END
