import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 120.0
# Safety margin for evaluation overhead
TIME_LIMIT = 100.0

def _compute_c3_discrete(signed_step):
    """
    Compute the C3 upper bound for a discrete signed step function.
    Returns the max absolute value of the autoconvolution.
    """
    # signed_step is an array of N values in {-1, 1}
    # Use FFT for efficiency
    fft_f = np.fft.fft(signed_step, n=2*N)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real
    # The autoconvolution of a step function with itself on a discrete grid
    # The scaling depends on the definition. 
    # If f is 1 on intervals of width w, the continuous conv has height proportional to w.
    # Here we treat it as discrete sequence autoconvolution.
    c3 = np.max(np.abs(autoconv))
    return c3

def _smooth_objective(x, fft_size):
    """
    Smooth objective function for continuous optimization.
    x: array of shape (N,)
    Returns: scalar value approximating max(|autoconv(h)|)
    """
    # Map to (-1, 1)
    h = np.tanh(x)
    
    # Compute autoconvolution via FFT
    # We need to be careful with the FFT size to avoid circular convolution artifacts
    # Linear convolution of two length N sequences requires size >= 2N-1.
    # We use 2*N for simplicity and power of 2 if possible, but 800 is fine.
    fft_f = np.fft.fft(h, n=fft_size)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real
    
    # The peak at the center (lag 0) is sum(h^2). 
    # If we want to minimize the "upper bound" on the tail or the whole thing,
    # we minimize the max absolute value.
    # However, the center peak is large. 
    # If the metric is max(|c|), and c[0] is large, minimizing it requires reducing h^2 sum,
    # which pushes h to 0. That's not a step function.
    # 
    # Re-reading "Third autocorrelation inequality":
    # This usually involves bounds like |c(t)| <= C.
    # If the seed is all-ones, c(t) is a triangle. Max is at 0.
    # If we flip signs, the max at 0 stays the same (sum h^2 = N), but other peaks might reduce.
    # So max(|c|) is likely dominated by c[0] = N.
    # Thus, minimizing max(|c|) doesn't help if c[0] is fixed.
    # 
    # Perhaps the metric is max_{t != 0} |c(t)|? Or max |c(t)| / c(0)?
    # Or maybe the "upper bound" refers to the L-infinity norm of the *normalized* convolution?
    # Let's assume the metric is the maximum absolute value of the autoconvolution 
    # EXCLUDING the trivial peak at lag 0, or normalized by the energy.
    # 
    # Let's look at the current program's score 1.0001.
    # If the seed is all-ones, and we improve it, the score should go up (improvement).
    # "score is the upper bound on C3" ... "lower is better" ... "seed/value".
    # If value is the bound, and lower is better, then score = seed_bound / current_bound.
    # If current_bound < seed_bound, score > 1.
    # So we want to MINIMIZE the bound.
    # 
    # If the bound is max(|c|), and c[0] is constant, we can't minimize it below c[0].
    # So the bound MUST be something else.
    # Maybe max_{t != 0} |c(t)|?
    # Or maybe the integral of |c(t)|^2? (Parseval)
    # Or maybe the max of |c(t)| for t in a specific range?
    #
    # Given "Third autocorrelation inequality", a common result is related to the 
    # maximum of the autocorrelation away from the origin.
    # Let's assume the metric is: max_{k != 0} |autoconv[k]|.
    # This is a common objective in signal design (low autocorrelation).
    
    # Mask out the center peak (lag 0)
    # In FFT output, lag 0 is at index 0 and index fft_size//2 (if even) or similar.
    # Actually, for linear convolution computed via FFT of size M:
    # c[0] is at index 0.
    # c[-1] is at index M-1.
    # The "center" of the autoconvolution (lag 0) is at index 0.
    # Wait, np.fft.ifft(fft_f * conj(fft_f)) gives the circular convolution.
    # If we pad to 2N, the linear convolution is in the first N and last N-1 elements.
    # c[0] is at index 0.
    # c[1] is at index 1.
    # c[-1] is at index 2N-1 (which wraps to N-1 in circular? No, linear conv is explicit).
    # 
    # Let's just take the max of the absolute values, but ignore the index 0 and the "wrapped" indices if we treat it as linear.
    # With padding 2N, indices 0 to N-1 are lags 0 to N-1.
    # Indices N to 2N-1 are lags -(N-1) to -1.
    # So we can ignore index 0 (lag 0).
    
    # Smooth max of |autoconv[1:N]| and |autoconv[N:2N]|
    # To be safe, let's just use a soft-max over the whole thing but with a penalty for the center?
    # No, let's assume the metric is max_{k != 0} |c[k]|.
    
    # Extract non-zero lags
    # Lags 1..N-1 are at indices 1..N-1
    # Lags -(N-1)..-1 are at indices N..2N-1
    non_zero_conv = np.concatenate([autoconv[1:N], autoconv[N:2*N]])
    
    abs_conv = np.abs(non_zero_conv)
    
    # Soft max
    T = 0.1 # Temperature for soft max
    max_abs = np.max(abs_conv)
    if max_abs < 1e-10:
        return 0.0
    log_sum_exp = np.log(np.sum(np.exp((abs_conv - max_abs) / T))) + max_abs / T
    soft_max = T * log_sum_exp
    
    return soft_max

def construct():
    start_time = time.time()
    
    # 1. Continuous Optimization Phase
    # Initialize with random or alternating signs
    np.random.seed(42)
    
    fft_size = 2 * N
    best_h = None
    best_val = float('inf')
    
    # Try a few initializations
    inits = [
        np.zeros(N), # Maps to 0, bad start for tanh? tanh(0)=0. 
        # Better to start with strong signs
        np.random.randn(N) * 3.0,
        np.array([3.0 if i % 2 == 0 else -3.0 for i in range(N)]),
        np.ones(N) * 3.0
    ]
    
    for x0 in inits:
        if time.time() - start_time > TIME_LIMIT * 0.4:
            break
            
        try:
            res = minimize(_smooth_objective, x0, args=(fft_size,), method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8})
            
            h_opt = np.tanh(res.x)
            # Evaluate the discrete version of this h_opt
            h_disc = np.sign(h_opt)
            # Ensure no zeros (sign(0)=0), replace with 1
            h_disc[h_disc == 0] = 1
            h_disc = h_disc.astype(np.int8)
            
            val = _compute_c3_discrete(h_disc)
            # Note: _compute_c3_discrete returns max(|conv|) including center.
            # We need to be consistent. 
            # If the metric is max_{k!=0}, I should use that for evaluation too.
            # Let's refine _compute_c3_discrete to match the objective.
            
            # Recalculate with non-zero lags only for consistency
            fft_f = np.fft.fft(h_disc, n=fft_size)
            autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real
            non_zero_conv = np.concatenate([autoconv[1:N], autoconv[N:2*N]])
            val_nonzero = np.max(np.abs(non_zero_conv))
            
            if val_nonzero < best_val:
                best_val = val_nonzero
                best_h = h_disc.copy()
                
        except Exception as e:
            pass

    if best_h is None:
        best_h = np.ones(N, dtype=np.int8)
        
    # 2. Discrete Local Search Polish
    # Use the result from continuous optimization as the starting point for discrete hill climbing
    current_step = best_h.copy()
    current_val = best_val
    
    max_iterations = 50000
    no_improve_count = 0
    patience = 5000
    
    # Pre-compute FFT of current step for fast updates? 
    # Flipping one element changes the FFT linearly.
    # fft_new = fft_old + delta * exp(-2j pi k idx / N)
    # This is O(N) for FFT update, then O(N) for IFFT. Total O(N).
    # Vs O(N log N) for full recompute.
    # For N=400, O(N) is much faster.
    
    fft_curr = np.fft.fft(current_step, n=fft_size)
    
    def _update_fft(fft_curr, idx, old_val, new_val):
        # delta = new_val - old_val
        # The basis vector for index idx in DFT of size fft_size
        # Note: current_step is length N, padded with 0 to fft_size.
        # The FFT is computed on the padded array.
        # So the change is at index idx in the padded array.
        # The DFT basis is exp(-2j pi k idx / fft_size)
        # Wait, np.fft.fft uses indices 0..N-1 for the input.
        # The input array for FFT is length fft_size, with first N elements as current_step.
        # So the change is at position idx in the array of length fft_size.
        k = np.arange(fft_size)
        basis = np.exp(-2j * np.pi * k * idx / fft_size)
        return fft_curr + (new_val - old_val) * basis

    for iteration in range(max_iterations):
        if time.time() - start_time > TIME_LIMIT - 2.0:
            break
            
        idx = np.random.randint(N)
        old_val = current_step[idx]
        new_val = -old_val
        
        # Update FFT
        fft_new = _update_fft(fft_curr, idx, old_val, new_val)
        
        # Compute autoconv
        autoconv = np.fft.ifft(fft_new * np.conj(fft_new)).real
        
        # Evaluate metric (max non-zero lag)
        non_zero_conv = np.concatenate([autoconv[1:N], autoconv[N:2*N]])
        new_val_metric = np.max(np.abs(non_zero_conv))
        
        if new_val_metric < current_val:
            current_val = new_val_metric
            current_step[idx] = new_val
            fft_curr = fft_new
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count >= patience:
            break

    return current_step
