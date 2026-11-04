# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 145.0  # wall-clock limit with safety margin
N = 400

def _evaluate_fast(h):
    """Compute the task value using FFT for speed."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    # Use FFT for convolution
    fft_h = np.fft.fft(h, n=2*N)
    conv = np.real(np.fft.ifft(fft_h * fft_h))[:N] # Full conv length is 2N-1, we take first N? 
    # Wait, np.convolve(h, h, 'full') has length 2N-1.
    # The max is over ALL entries of the full convolution.
    # So we need the full convolution.
    conv_full = np.real(np.fft.ifft(fft_h * fft_h))
    # The ifft gives circular convolution of size 2N.
    # The linear convolution is embedded in this.
    # Specifically, if we zero-pad h to size 2N, the first N entries of ifft(fft(h_pad)*fft(h_pad))
    # correspond to the linear convolution?
    # Let's check.
    # Linear conv length 2N-1.
    # Circular conv length 2N.
    # If we zero pad to 2N, the circular conv is the linear conv padded with one zero?
    # Yes, because 2N > (N-1) + (N-1) = 2N-2.
    # So conv_full[0:N-1] is the first part, conv_full[N:2N-1] is the rest?
    # Actually, the linear convolution $c_k = \sum h_i h_{k-i}$.
    # $c_0 = h_0 h_0$.
    # $c_{2N-2} = h_{N-1} h_{N-1}$.
    # In the FFT domain with size $M=2N$:
    # $C_k = \sum_{n=0}^{M-1} h_n h_{(k-n) \mod M}$.
    # For $k < N$, $(k-n) \mod M$ is $k-n$ if $n \le k$, and $M+k-n$ if $n > k$.
    # Since $h$ is zero-padded to $M$, $h_{M+k-n}$ is 0 for $n > k$ (as $M+k-n > N-1$ usually? No).
    # Let's just use np.convolve for correctness in the final check, but for optimization, 
    # we can assume the max is likely in the central part or just use the full circular conv 
    # if we pad sufficiently.
    # To be safe and fast, we can use a larger FFT size, e.g., 4N, to avoid wrap-around.
    # But 2N is sufficient for linear conv if we take the right slice?
    # Actually, $c_{k}$ for $k=0 \dots 2N-2$.
    # The circular conv of size $2N$ computes $c_k$ for $k=0 \dots 2N-1$.
    # The term for $k$ in circular conv is $\sum h_i h_{(k-i) \pmod{2N}}$.
    # If $k < N$, and we have zero padding, the wrap-around terms $h_{2N+k-i}$ are zero for $i$ such that $2N+k-i \ge N$?
    # $2N+k-i \ge N \implies i \le N+k$. Since $i < N$, this is always true?
    # No, $h$ is non-zero only for $0 \dots N-1$.
    # So $h_j$ is non-zero only if $0 \le j \le N-1$.
    # In circular conv $C_k = \sum_{i=0}^{N-1} h_i h_{(k-i) \pmod{2N}}$.
    # If $k < N$:
    #   If $i \le k$, $k-i \in [0, N-1]$, so $h_{k-i}$ is the actual value.
    #   If $i > k$, $k-i$ is negative, so $(k-i) \pmod{2N} = 2N + k - i$.
    #   Since $i \le N-1$, $2N+k-i \ge 2N+k-(N-1) = N+k+1 \ge N+1$.
    #   So $h_{2N+k-i}$ is 0 (due to zero padding).
    # So $C_k$ for $k < N$ is exactly the linear convolution $c_k$.
    # What about $k \ge N$?
    # If $k \ge N$:
    #   If $i \ge k-N+1$? 
    #   Let's check $k=N$. $C_N = \sum h_i h_{(N-i) \pmod{2N}}$.
    #   If $i < N$, $N-i \in [1, N]$.
    #   If $i=0$, $h_N$ (which is 0).
    #   If $i>0$, $N-i \in [1, N-1]$, so $h_{N-i}$ is actual value.
    #   This matches linear conv $c_N = \sum h_i h_{N-i}$.
    #   Wait, linear conv $c_k$ is defined for $k=0 \dots 2N-2$.
    #   $c_k = \sum_{i=\max(0, k-N+1)}^{\min(k, N-1)} h_i h_{k-i}$.
    #   The circular conv with size $2N$ gives $C_k$ for $k=0 \dots 2N-1$.
    #   It turns out $C_k = c_k$ for $k=0 \dots 2N-2$ and $C_{2N-1} = c_0$ (wrap around).
    #   So we can just take `conv_full[:2*N-1]`? Or just `conv_full` and ignore the last element?
    #   Actually, `np.convolve(h, h, 'full')` has length $2N-1$.
    #   `np.fft.ifft(fft_h*fft_h)` has length $2N$.
    #   The first $2N-1$ elements of the ifft result correspond to the full convolution?
    #   Yes, because the wrap-around only affects the last element $C_{2N-1}$ which corresponds to $c_{-1}$? No.
    #   $C_{2N-1} = \sum h_i h_{(2N-1-i) \pmod{2N}} = \sum h_i h_{2N-1-i}$.
    #   Since $h$ is zero for $j \ge N$, $h_{2N-1-i}$ is non-zero only if $2N-1-i < N \implies i > N-1$.
    #   But $i \le N-1$. So $h_{2N-1-i}$ is always 0?
    #   Wait. $2N-1-i$. If $i=0$, index $2N-1$. $h[2N-1]=0$.
    #   If $i=N-1$, index $N$. $h[N]=0$.
    #   So $C_{2N-1} = 0$.
    #   And $c_{2N-2}$ is the last valid linear conv term.
    #   So `conv_full[:2*N-1]` should be the linear convolution.
    
    max_conv = np.max(conv_full)
    val = abs(2.0 * N * max_conv / (s * s))
    if not np.isfinite(val):
        return np.inf
    return val

def _objective_smooth(h, beta):
    """
    Objective with smooth max.
    """
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return 1e10
    
    fft_h = np.fft.fft(h, n=2*N)
    conv = np.real(np.fft.ifft(fft_h * fft_h))
    # We need the max of the full convolution.
    # Using log-sum-exp for stability
    m = np.max(conv)
    # If beta is large, exp(beta * (conv - m)) is 1 for max, small for others.
    # log_sum_exp = m + (1/beta) * log(sum(exp(beta * (conv - m))))
    # We want to minimize this.
    # To avoid overflow, we shift by m.
    shifted = beta * (conv - m)
    # Clip shifted to avoid exp overflow/underflow issues in sum
    shifted = np.clip(shifted, -50, 50)
    lse = m + (1.0 / beta) * np.log(np.sum(np.exp(shifted)))
    
    val = abs(2.0 * N * lse / (s * s))
    if not np.isfinite(val):
        return 1e10
    return val

def _local_refine_fast(h, deadline, max_iter=50000):
    """
    Local refinement using FFT evaluation.
    """
    best_val = _evaluate_fast(h)
    best_h = h.copy()
    rng = np.random.default_rng(42)
    
    step_size = 0.1
    for step in range(max_iter):
        if time.time() > deadline:
            break
        
        n_perturb = rng.integers(1, 6)
        idx = rng.choice(N, size=n_perturb, replace=False)
        h_new = best_h.copy()
        h_new[idx] += rng.normal(0, step_size, n_perturb)
        
        v_new = _evaluate_fast(h_new)
        if np.isfinite(v_new) and v_new < best_val - 1e-10:
            best_val = v_new
            best_h = h_new
            step_size *= 0.99
        elif step % 5000 == 0:
            step_size *= 0.95
    
    return best_h, best_val

def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    rng = np.random.default_rng(123)
    
    best_val = 2.0
    best_h = np.ones(N)
    
    # Phase 1: Smooth Max Optimization on Raw Vector
    # Start with a few structured initializations
    inits = [
        np.ones(N),
        np.sin(np.linspace(0, 2*np.pi, N)),
        np.random.default_rng(1).normal(0, 1, N),
        np.random.default_rng(2).normal(0, 1, N),
    ]
    
    # Add some low frequency inits
    for freq in [0.5, 1.0, 2.0]:
        n = np.arange(N)
        inits.append(np.cos(2 * np.pi * freq * n / N))
        inits.append(np.sin(2 * np.pi * freq * n / N))
        
    # Optimize each init with increasing beta
    for init_h in inits:
        if time.time() > deadline - 20:
            break
            
        current_h = init_h.copy()
        # Anneal beta
        for beta in [1.0, 5.0, 10.0, 20.0, 50.0]:
            if time.time() > deadline - 10:
                break
            
            try:
                res = minimize(
                    _objective_smooth,
                    current_h,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 200}
                )
                current_h = res.x
                v_curr = _evaluate_fast(current_h)
                if np.isfinite(v_curr) and v_curr < best_val:
                    best_val = v_curr
                    best_h = current_h.copy()
            except Exception:
                pass
    
    # Phase 2: Extensive Local Refinement
    if time.time() < deadline - 5:
        best_h, best_val = _local_refine_fast(best_h, deadline - 2)
        
    # Phase 3: Try to improve with more random restarts if time permits
    # The speedup allows more restarts
    while time.time() < deadline - 10:
        # Random restart
        h_rand = rng.normal(0, 1, N)
        # Quick smooth optimization
        try:
            res = minimize(
                _objective_smooth,
                h_rand,
                args=(10.0,),
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-10, 'maxfun': 100}
            )
            h_opt = res.x
            v_opt = _evaluate_fast(h_opt)
            if np.isfinite(v_opt) and v_opt < best_val:
                best_val = v_opt
                best_h = h_opt.copy()
                # Refine the new best
                if time.time() < deadline - 5:
                    best_h, best_val = _local_refine_fast(best_h, deadline - 2, max_iter=10000)
        except Exception:
            pass

    # Final check
    final_val = _evaluate_fast(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
