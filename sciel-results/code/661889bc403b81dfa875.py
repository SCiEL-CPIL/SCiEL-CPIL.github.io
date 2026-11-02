"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _autoconv_fft(sig):
    """Compute autoconvolution of a signal using FFT."""
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    f = np.fft.rfft(sig, n_fft)
    ac = np.fft.irfft(f * np.conj(f), n_fft)[:2*N-1]
    return ac


def _exact_score(sig):
    """Compute C3' = max|autoconv| / (integral)^2."""
    integral = np.sum(sig)
    if abs(integral) < 1e-12:
        return float('inf')
    ac = _autoconv_fft(sig)
    max_abs = np.max(np.abs(ac))
    return max_abs / (integral * integral)


def _logsumexp_surrogate(sig, beta=50.0):
    """
    Smooth surrogate for max|autoconv| using log-sum-exp.
    Returns (surrogate_value, gradient_wrt_sig)
    """
    ac = _autoconv_fft(sig)
    # Use log-sum-exp of |ac|^2 to approximate max|ac|^2
    # Then sqrt to get max|ac|
    ac_sq = ac * ac
    # Shift for numerical stability
    max_val = np.max(ac_sq)
    if max_val < 1e-15:
        return 0.0, np.zeros(N)
    shifted = ac_sq - max_val
    exp_shifted = np.exp(beta * shifted)
    logsumexp_val = np.log(np.sum(exp_shifted)) / beta + max_val / beta
    # surrogate = sqrt(logsumexp_val) approximates max|ac|
    surrogate = np.sqrt(logsumexp_val)
    
    # Gradient: d/dx logsumexp(x^2) where x = autoconv(x)
    # Using chain rule through FFT
    # This is complex; use numerical gradient or analytic
    # For now, use the fact that autoconv is quadratic
    # grad of logsumexp(ac^2) w.r.t. sig:
    # = sum_j (2*ac_j * exp(beta*(ac_j^2 - max)) / sum(exp(...))) * d(ac_j)/d(sig)
    # d(ac_j)/d(sig) is the convolution kernel
    
    # Simpler: use automatic differentiation via finite differences for gradient
    # But that's expensive. Let's use analytic gradient.
    
    # ac = sig * sig (convolution)
    # d(ac_k)/d(sig_i) = sig_{k-i} + sig_{k-N+1-i} (for valid range)
    # This is essentially the convolution with sig
    
    # For the log-sum-exp surrogate:
    # L = log(sum_j exp(beta * ac_j^2)) / beta
    # dL/d(ac_j) = 2 * beta * ac_j * exp(beta * ac_j^2) / (beta * sum_k exp(beta * ac_k^2))
    #             = 2 * ac_j * exp(beta * ac_j^2) / sum_k exp(beta * ac_k^2)
    
    weights = 2.0 * ac * exp_shifted / np.sum(exp_shifted)
    
    # Now dL/d(sig_i) = sum_j weights_j * d(ac_j)/d(sig_i)
    # d(ac_j)/d(sig_i) = sum_m sig_{j-m} * delta(m=i) + ... 
    # Actually ac = sig * sig, so d(ac_j)/d(sig_i) = (sig * delta_i)_j + (delta_i * sig)_j
    # = sig_{j-i} (with appropriate bounds)
    
    # We can compute this as: grad = weights * sig + flip(weights) * sig (correlated)
    # More precisely: if ac = conv(sig, sig), then d(ac)/d(sig) applied to weights
    # is conv(weights, sig) + conv(flip(weights), sig) but we need to be careful
    
    # Simpler: grad_i = sum_j weights_j * (sig_{j-i} + sig_{j-N+1-i})
    # This is: grad = conv(weights, sig) + conv(flip(weights), sig)
    # But we need the full convolution, not just the part we keep
    
    # Let's compute the full convolution
    n_fft_full = 1
    while n_fft_full < 2 * N - 1:
        n_fft_full *= 2
    
    w_fft = np.fft.rfft(weights, n_fft_full)
    s_fft = np.fft.rfft(sig, n_fft_full)
    grad_full = np.fft.irfft(w_fft * s_fft, n_fft_full)
    
    # Also need the other term: d(ac_j)/d(sig_i) also involves sig convolved with delta
    # Actually for ac = sig * sig:
    # ac_k = sum_i sig_i * sig_{k-i}
    # d(ac_k)/d(sig_i) = sig_{k-i} + sig_{k-N+1-i} (for linear convolution)
    # 
    # So grad_i = sum_k weights_k * (sig_{k-i} + sig_{k-N+1-i})
    # = (weights * sig)_i + (flip(weights) * sig)_i (approximately)
    
    # Let's just compute it directly using the FFT approach
    # The gradient of the convolution quadratic form:
    # If f(x) = g(conv(x, x)), then grad_x f = 2 * conv(grad_g(conv(x,x)), x)
    # where conv is the full linear convolution
    
    # So: grad = 2 * conv(weights, sig) but we only keep the first N elements
    # Actually we need to be careful about the indexing
    
    # Let me just do it numerically for correctness, but that's O(N^2)
    # For N=400, that's 160k operations per gradient, which is fine
    
    # Actually, let's use the FFT-based approach properly
    # conv(weights, sig) gives us one part
    # We need the "cross" term
    
    # For linear convolution c = a * b:
    # c_k = sum_i a_i b_{k-i}
    # d(c_k)/d(a_i) = b_{k-i}
    # 
    # For autoconv ac = sig * sig:
    # ac_k = sum_i sig_i sig_{k-i}
    # d(ac_k)/d(sig_i) = sig_{k-i} + sig_{k-N+1-i} (the second term for when k-i < 0)
    
    # Let's compute grad as:
    # grad_i = sum_k weights_k * sig_{k-i} + sum_k weights_k * sig_{k-N+1-i}
    
    # First term: (weights * sig)_i where * is cross-correlation
    # Second term: (flip(weights) * sig)_i
    
    # Using FFT:
    # cross_corr(a, b)_i = sum_k a_k b_{k-i} = (a * flip(b))_i
    
    # Let me just compute it directly for N=400
    grad = np.zeros(N)
    for i in range(N):
        # sum_k weights_k * sig_{k-i} for valid k, i
        # k-i must be in [0, N-1], so k in [i, N-1]
        if i < N:
            grad[i] += np.sum(weights[i:] * sig[:N-i])
        # Second term: sum_k weights_k * sig_{k-N+1-i}
        # k-N+1-i in [0, N-1] => k in [N-1+i, 2N-2]
        # But k only goes to 2N-2, and weights has length 2N-1
        # k-N+1-i >= 0 => k >= N-1+i
        # k-N+1-i <= N-1 => k <= 2N-2-i
        if i < N:
            lo = max(0, N-1+i)
            hi = min(2*N-2, 2*N-2-i)
            if lo <= hi:
                sig_idx_lo = lo - N + 1 - i
                sig_idx_hi = hi - N + 1 - i
                grad[i] += np.sum(weights[lo:hi+1] * sig[sig_idx_lo:sig_idx_hi+1])
    
    return surrogate, grad


def _construct_continuous():
    """Use L-BFGS-B to optimize the continuous signal."""
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    # Objective: minimize logsumexp_surrogate / integral^2
    # We'll minimize the surrogate ratio
    
    def objective(sig):
        if time.time() > deadline:
            raise Exception("timeout")
        integral = np.sum(sig)
        if abs(integral) < 1e-10:
            return 1e10
        ac = _autoconv_fft(sig)
        # Log-sum-exp of |ac|^2
        ac_sq = ac * ac
        max_val = np.max(ac_sq)
        if max_val < 1e-15:
            return 0.0
        shifted = ac_sq - max_val
        exp_shifted = np.exp(50.0 * shifted)
        logsumexp_val = np.log(np.sum(exp_shifted)) / 50.0 + max_val / 50.0
        surrogate = np.sqrt(logsumexp_val)
        return surrogate / (integral * integral)
    
    # Start with a good initial guess: two-block structure
    best_sig = None
    best_score = float('inf')
    
    # Try several initializations
    initializations = []
    
    # 1. Two-block: +1 first half, -1 second half
    sig1 = np.zeros(N)
    sig1[:N//2] = 1.0
    sig1[N//2:] = -1.0
    initializations.append(sig1.copy())
    
    # 2. Three-block: +1, -1, +1
    sig2 = np.zeros(N)
    sig2[:N//3] = 1.0
    sig2[N//3:2*N//3] = -1.0
    sig2[2*N//3:] = 1.0
    initializations.append(sig2.copy())
    
    # 3. Random signed
    np.random.seed(42)
    sig3 = np.random.choice([-1.0, 1.0], size=N)
    initializations.append(sig3.copy())
    
    # 4. +1 then -1 with different split
    sig4 = np.zeros(N)
    sig4[:N//4] = 1.0
    sig4[N//4:] = -1.0
    initializations.append(sig4.copy())
    
    for init_sig in initializations:
        if time.time() > deadline:
            break
        
        # Normalize to have integral ~1 for better conditioning
        integral = np.sum(init_sig)
        if abs(integral) > 1e-10:
            init_sig = init_sig / integral
        
        try:
            result = minimize(
                objective,
                init_sig,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50}
            )
            
            if time.time() > deadline:
                break
            
            opt_sig = result.x
            # Round to nearest signed values
            rounded = np.sign(opt_sig)
            # Handle zeros
            rounded[rounded == 0] = 1.0
            
            score = _exact_score(rounded)
            if score < best_score:
                best_score = score
                best_sig = rounded.copy()
                
        except Exception:
            continue
    
    return best_sig, best_score


def _local_refinement(sig):
    """Local search: flip individual elements."""
    best_score = _exact_score(sig)
    improved = True
    start_time = time.time()
    deadline = start_time + 40.0
    
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            new_sig = sig.copy()
            new_sig[i] = -new_sig[i]
            s = _exact_score(new_sig)
            if s < best_score - 1e-12:
                best_score = s
                sig = new_sig
                improved = True
    
    return sig, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Phase 1: Continuous optimization
    best_sig, best_score = _construct_continuous()
    
    if best_sig is None:
        # Fallback: simple two-block
        best_sig = np.zeros(N)
        best_sig[:N//2] = 1.0
        best_sig[N//2:] = -1.0
        best_score = _exact_score(best_sig)
    
    if time.time() > deadline:
        return best_sig
    
    # Phase 2: Local refinement
    refined_sig, refined_score = _local_refinement(best_sig)
    
    if refined_score < best_score:
        best_sig = refined_sig
        best_score = refined_score
    
    # Phase 3: Try block-based refinements
    if time.time() < deadline - 5.0:
        # Convert to blocks
        blocks = []
        i = 0
        while i < N:
            h = best_sig[i]
            j = i
            while j < N and best_sig[j] == h:
                j += 1
            blocks.append((h, j - i))
            i = j
        
        # Try flipping individual blocks
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for bi in range(len(blocks)):
                if time.time() > deadline - 5.0:
                    break
                new_blocks = list(blocks)
                h, w = new_blocks[bi]
                new_blocks[bi] = (-h, w)
                
                new_sig = np.zeros(N)
                pos = 0
                for bh, bw in new_blocks:
                    new_sig[pos:pos+bw] = bh
                    pos += bw
                
                s = _exact_score(new_sig)
                if s < best_score - 1e-12:
                    best_score = s
                    best_sig = new_sig
                    blocks = new_blocks
                    improved = True
    
    return best_sig
# EVOLVE-BLOCK-END
