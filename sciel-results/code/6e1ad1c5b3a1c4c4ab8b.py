"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft

def _eval_value(h):
    """Compute the objective value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _convolve_fft(h):
    """Compute full autoconvolution using FFT."""
    n_conv = 2 * len(h) - 1
    n_fft = 1
    while n_fft < n_conv:
        n_fft *= 2
    H = fft(h, n=n_fft)
    conv = ifft(H * H, n=n_fft).real[:n_conv]
    return conv

def _soft_max(x, beta):
    """Numerically stable soft maximum."""
    m = np.max(x)
    if not np.isfinite(m):
        return m
    exp_vals = np.exp(beta * (x - m))
    sm = m + np.log(np.sum(exp_vals)) / beta
    return sm

def _objective(h, beta):
    """
    Objective function for optimization.
    Minimizes the soft-max of the autoconvolution peak, normalized by sum^2.
    """
    if not np.all(np.isfinite(h)):
        return 1e10
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _convolve_fft(h)
    sm = _soft_max(conv, beta)
    return 2.0 * N * sm / (s * s)

def _objective_grad(h, beta):
    """
    Gradient of the soft-max objective with respect to h.
    
    The objective is: f(h) = 2N * soft_max(conv(h,h)) / (sum(h))^2
    
    We compute the gradient using the chain rule and FFT.
    soft_max(x) = max(x) + log(sum(exp(beta*(x - max(x)))))/beta
    
    d(soft_max)/dx = exp(beta*(x - max(x))) / sum(exp(beta*(x - max(x))))
    
    conv(h,h)_k = sum_i h_i * h_{i+k-N+1} (for full convolution)
    d(conv)_k / dh_j = h_{k-N+1-j} + h_{k-N+1-j} (with appropriate bounds)
    
    Actually, for full convolution c_k = sum_i h_i h_{k-i} where indices are valid.
    dc_k/dh_j = h_{k-j} + h_{k-j} if we consider both terms, but since it's h*h,
    dc_k/dh_j = 2 * h_{k-j} when 0 <= k-j < N, but we need to be careful.
    
    Let's use the FFT approach: conv = ifft(fft(h) * fft(h))
    The gradient can be computed via the chain rule through FFT.
    
    For simplicity and correctness, let's use a numerical gradient approach 
    but with a smarter method: we can compute the gradient of soft_max w.r.t. 
    conv, then use the fact that conv = ifft(H*H) to backpropagate.
    
    Actually, let's implement it properly:
    Let p_k = exp(beta*(conv_k - m)) / sum(exp(beta*(conv - m)))
    Then d(soft_max)/d(conv_k) = p_k
    
    Now, conv_k = sum_i h_i h_{k-i} (with proper indexing for full convolution)
    d(conv_k)/d(h_j) = h_{k-j} if 0 <= k-j < N else 0, but also we need to account
    for the fact that in the sum, when i=j, we get h_j * h_{k-j}, and when i=k-j, 
    we get h_{k-j} * h_j. So actually d(conv_k)/d(h_j) = h_{k-j} + h_{k-j} = 2*h_{k-j}
    when j != k-j, and h_j when j == k-j (i.e., k = 2j).
    
    Wait, that's not right either. Let me think more carefully.
    
    conv_k = sum_{i=0}^{N-1} h_i * h_{k-i} where h_m = 0 if m < 0 or m >= N.
    
    d(conv_k)/d(h_j) = h_{k-j} + h_{k-j} = 2*h_{k-j} when the term h_j appears twice,
    but actually h_j appears in conv_k when i=j (giving h_j * h_{k-j}) and when i=k-j 
    (giving h_{k-j} * h_j). These are the same term if j = k-j, i.e., k=2j.
    
    So d(conv_k)/d(h_j) = h_{k-j} + h_{k-j} if j != k-j, else h_j.
    Actually no: d/dh_j [h_i h_{k-i}] is h_{k-i} if i=j, plus h_i if k-i=j.
    So d(conv_k)/d(h_j) = h_{k-j} (from i=j) + h_{k-j} (from k-i=j, i.e., i=k-j) 
    but these are the same when j=k-j.
    
    So: d(conv_k)/d(h_j) = 2*h_{k-j} if j != k-j, and h_j if j == k-j.
    But h_{k-j} is only defined if 0 <= k-j < N.
    
    This is getting complex. Let's use a different approach: since we're using FFT,
    we can compute the gradient via the adjoint method.
    
    Given conv = ifft(H * H), where H = fft(h).
    Let g = d(soft_max)/d(conv) (a vector of length n_conv, but we only need the part
    that corresponds to the convolution).
    
    The gradient w.r.t. H is: d(conv)/dH = 2 * H * (conjugate of something)...
    
    Actually, let's just use numerical gradient with a step size, but be efficient.
    For N=400, we can do 400 evaluations of the objective per gradient step.
    With L-BFGS-B, this might be slow but let's try.
    
    Alternatively, we can use the fact that the problem is smooth enough and use
    a simpler approach: just call minimize with numerical gradient.
    """
    # For now, return None to let minimize use numerical gradient
    return None

def _simulated_annealing_binary(rng, start_time, deadline, max_steps_per_trial=5000):
    """
    Simulated annealing on binary sequences {-1, +1} to minimize max autoconvolution.
    Returns the best binary sequence found and its value.
    """
    best_seq = None
    best_val = np.inf
    
    # Try multiple restarts
    n_restarts = 3
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Initialize binary sequence
        if restart == 0:
            # Start with all +1
            seq = np.ones(N, dtype=np.int8)
        else:
            # Random initialization
            seq = rng.choice([-1, 1], size=N).astype(np.int8)
        
        # Ensure sum is not zero (for binary, sum must be even, so avoid sum=0)
        # If sum is 0, flip one element
        while np.sum(seq) == 0:
            idx = rng.randint(N)
            seq[idx] *= -1
        
        current_val = _eval_value(seq.astype(np.float64))
        current_seq = seq.copy()
        
        # Annealing schedule
        T_start = 1.0
        T_end = 0.01
        n_steps = max_steps_per_trial
        cooling = (T_end / T_start) ** (1.0 / n_steps)
        
        T = T_start
        for step in range(n_steps):
            if time.time() > deadline - 15:
                break
            
            # Pick a random index to flip
            idx = rng.randint(N)
            old_val = seq[idx]
            seq[idx] *= -1
            
            # Check if sum is still valid
            new_sum = np.sum(seq)
            if new_sum == 0:
                # Revert if sum becomes zero
                seq[idx] = old_val
                continue
            
            new_val = _eval_value(seq.astype(np.float64))
            
            # Acceptance criterion
            if new_val < current_val:
                # Always accept improvement
                current_val = new_val
                current_seq = seq.copy()
                if new_val < best_val:
                    best_val = new_val
                    best_seq = seq.copy()
            else:
                # Accept with probability exp(-delta/T)
                delta = new_val - current_val
                if delta > 0:
                    prob = np.exp(-delta / max(T, 1e-10))
                    if rng.random() < prob:
                        current_val = new_val
                        current_seq = seq.copy()
                    else:
                        # Revert
                        seq[idx] = old_val
            
            # Cool down
            T *= cooling
        
        # End of restart
        if current_val < best_val:
            best_val = current_val
            best_seq = current_seq.copy()
    
    if best_seq is None:
        return None, np.inf
    return best_seq, best_val

def _gradient_refine(h_init, start_time, deadline):
    """
    Gradient-based refinement using soft-max surrogate.
    Returns the best refined sequence and its value.
    """
    from scipy.optimize import minimize
    
    best_h = h_init.copy()
    best_val = _eval_value(h_init)
    
    betas = [5.0, 20.0, 50.0, 100.0, 200.0]
    current_h = h_init.copy()
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
        
        def obj(h):
            return _objective(h, beta)
        
        # Try a few restarts
        for trial in range(2):
            if time.time() > deadline - 5:
                break
            
            if trial == 0:
                h_try = current_h.copy()
            else:
                noise_scale = 0.1 / (beta + 1.0)
                h_try = current_h + noise_scale * np.random.randn(N)
                h_try -= np.mean(h_try)
                h_try += np.mean(current_h)
            
            try:
                result = minimize(
                    obj,
                    h_try,
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 30}
                )
                
                h_opt = result.x
                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    v_opt = _eval_value(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
                        current_h = h_opt.copy()
            except Exception:
                continue
    
    return best_h, best_val

def _polish_local(h_init, start_time, deadline):
    """
    Final high-precision polish pass using coordinate descent and local search.
    """
    best_h = h_init.copy()
    best_val = _eval_value(h_init)
    
    # Coordinate descent: try adjusting each coordinate slightly
    for iteration in range(3):
        if time.time() > deadline - 5:
            break
        
        improved = False
        for i in range(N):
            if time.time() > deadline - 5:
                break
            
            # Try small perturbations
            for delta in [-0.1, -0.01, 0.01, 0.1]:
                h_try = best_h.copy()
                h_try[i] += delta
                if not np.all(np.isfinite(h_try)) or abs(np.sum(h_try)) < 1e-10:
                    continue
                v_try = _eval_value(h_try)
                if v_try < best_val:
                    best_val = v_try
                    best_h = h_try.copy()
                    improved = True
        
        if not improved:
            break
    
    return best_h, best_val

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Uses simulated annealing on binary sequences followed by gradient refinement
    and a final polish pass.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Simulated annealing on binary sequences
    binary_seq, binary_val = _simulated_annealing_binary(rng, start_time, deadline)
    
    if binary_seq is not None:
        best_h = binary_seq.astype(np.float64)
        best_val = binary_val
    else:
        best_h = np.ones(N)
        best_val = 2.0
    
    # Phase 2: Gradient-based refinement
    if time.time() < deadline - 20:
        refined_h, refined_val = _gradient_refine(best_h, start_time, deadline)
        if refined_val < best_val:
            best_h = refined_h
            best_val = refined_val
    
    # Phase 3: Final polish
    if time.time() < deadline - 5:
        polished_h, polished_val = _polish_local(best_h, start_time, deadline)
        if polished_val < best_val:
            best_h = polished_h
            best_val = polished_val
    
    # Ensure finite and sum != 0
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
