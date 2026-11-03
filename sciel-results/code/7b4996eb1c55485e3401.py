"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Structural change: Uses circular autocorrelation (matching task spec/seed) 
with Nesterov accelerated gradient descent on a LogSumExp smooth surrogate.
Includes diverse initializations and a final exact-max polish phase."""

import numpy as np
import time

N = 100
_DEADLINE = time.time() + 145.0  # 5s safety margin before 150s kill

def _circular_overlap_values(h):
    """
    Compute O(s) = sum_i h[i] * (1 - h[(i+s)%N]) for all s.
    O(s) = N/2 - R_circ(s)
    R_circ(s) = sum_i h[i] h[(i+s)%N]
    """
    n = len(h)
    # Use FFT for circular autocorrelation
    H = np.fft.fft(h)
    R_circ = np.fft.ifft(H * np.conj(H)).real
    # Due to numerical errors, R_circ might have tiny imaginary parts or slight deviations
    # Ensure it's real
    R_circ = R_circ.real
    # O(s) = N/2 - R_circ(s)
    O = n / 2.0 - R_circ
    return O

def _score(h):
    """Compute the maximum overlap."""
    O = _circular_overlap_values(h)
    return float(np.max(O))

def _smooth_max_val_grad(h, beta):
    """
    Return (value, gradient) of LogSumExp(beta * O(h)).
    O(h) is the vector of overlaps.
    """
    n = len(h)
    O = _circular_overlap_values(h)
    # Numerical stability for LogSumExp
    max_O = np.max(O)
    exp_terms = np.exp(beta * (O - max_O))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights w_s
    
    val = max_O + np.log(sum_exp) / beta
    
    # Gradient of LogSumExp wrt h
    # d/dh LSE = sum_s w_s * (d/dh O(s))
    # O(s) = N/2 - R_circ(s)
    # d/dh O(s) = - d/dh R_circ(s)
    # d/dh_j R_circ(s) = h[(j-s)%n] + h[(j+s)%n]  (for s!=0)
    # d/dh_j R_circ(0) = 2*h[j]
    
    grad_O = np.zeros(n)
    # We can compute this efficiently using correlations
    # grad_O[j] = - sum_s w_s * (h[(j-s)%n] + h[(j+s)%n])
    # Note: for s=0, term is 2*h[j]. For s>0, it's h[j-s]+h[j+s].
    # Since R_circ(s) = R_circ(n-s), the weights w_s are symmetric if O is symmetric?
    # O(s) = N/2 - R_circ(s). R_circ is symmetric. So O is symmetric.
    # w_s = w_{n-s}.
    
    # Let's compute the sum directly for clarity and correctness, N=100 is small.
    # But we can vectorize the convolution part.
    # sum_s w_s * h[(j-s)%n] is the circular convolution of w and h.
    # sum_s w_s * h[(j+s)%n] is the circular convolution of w_rev and h?
    # Let conv1 = ifft(fft(w) * fft(h)) -> conv1[j] = sum_s w[s] h[(j-s)%n]
    # Let conv2 = ifft(fft(w) * fft(h)) -> wait.
    # h[(j+s)%n] = h[(j - (-s))%n].
    # So sum_s w_s h[j+s] = sum_s w_s h[j - (-s)].
    # This is conv(w, h) but with s negated?
    # Actually, circular convolution (w * h)[j] = sum_s w[s] h[(j-s)%n].
    # We need sum_s w[s] h[(j+s)%n].
    # Let k = -s. sum_k w[-k] h[(j-k)%n].
    # So it is conv(w_rev, h) where w_rev[k] = w[-k].
    
    W = np.fft.fft(w)
    H = np.fft.fft(h)
    
    # Term 1: sum_s w_s h[(j-s)%n]
    conv1 = np.fft.ifft(W * H).real
    # Term 2: sum_s w_s h[(j+s)%n]
    # This is equivalent to conv1 if w is symmetric?
    # w_s = exp(beta(O_s - max)) / sum. O_s is symmetric. So w_s is symmetric.
    # If w is symmetric (w_s = w_{n-s}), then conv1[j] = sum_s w_s h[j-s].
    # And sum_s w_s h[j+s] = sum_k w_k h[j-k] (by changing var k=-s, w_{-k}=w_k).
    # So if w is symmetric, Term 1 == Term 2.
    
    # Check symmetry of w
    # O is symmetric, so w is symmetric.
    # So grad_O = -2 * conv1 ?
    # Wait, for s=0, dR/dh = 2h.
    # In conv1, the s=0 term is w_0 h_j.
    # In the sum over s, the s=0 term contributes w_0 * 2 h_j.
    # The s>0 terms contribute w_s (h[j-s] + h[j+s]).
    # Since w is symmetric, sum_{s>0} w_s h[j-s] + sum_{s>0} w_s h[j+s] = 2 * sum_{s>0} w_s h[j-s] (if we map indices correctly).
    # Actually, just use the formula:
    # grad_O = - (2 * conv1 - w[0]*h) ? No.
    
    # Let's just do the O(N^2) loop for gradient to be safe and clear, N=100 is tiny.
    grad_O = np.zeros(n)
    for s in range(n):
        if w[s] < 1e-15:
            continue
        if s == 0:
            grad_O -= 2.0 * w[s] * h
        else:
            shifted1 = np.roll(h, -s) # h[(i+s)%n]
            shifted2 = np.roll(h, s)   # h[(i-s)%n]
            grad_O -= w[s] * (shifted1 + shifted2)
            
    return val, grad_O

def _fix_sum(h):
    """Project h to {0 <= h_i <= 1, sum(h) = N/2}."""
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    h = np.clip(h, 0.0, 1.0)
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        # Distribute difference to elements not at bounds
        mask = (h > 1e-10) & (h < 1.0 - 1e-10)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            # If still off, adjust one element
            diff2 = target - np.sum(h)
            if abs(diff2) > 1e-14:
                if diff2 > 0:
                    idx = np.argmin(h)
                else:
                    idx = np.argmax(h)
                h[idx] += diff2
                h = np.clip(h, 0.0, 1.0)
    return h

def _nesterov_optimize(h0, beta, max_iter=1000):
    """Nesterov accelerated projected gradient descent."""
    h = _fix_sum(h0)
    h_prev = h.copy()
    t_k = 1.0
    lr = 0.05 / beta  # Initial learning rate, scaled by beta
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5.0:
            break
        
        val, grad = _smooth_max_val_grad(h, beta)
        
        # Gradient step
        h_new = h - lr * grad
        
        # Projection
        h_new = _fix_sum(h_new)
        
        # Nesterov momentum
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        h_next = h_new + ((t_k - 1.0) / t_new) * (h_new - h_prev)
        h_next = _fix_sum(h_next)
        
        h_prev = h
        h = h_next
        t_k = t_new
        
        # Adaptive learning rate: decay slowly
        if it % 100 == 99:
            lr = max(1e-4, lr * 0.9)
            
    return h

def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
            
    i = np.arange(N)
    
    # Diverse initial guesses
    starts = []
    
    # 1. Constant
    starts.append(np.full(N, 0.5))
    
    # 2. Cosine
    starts.append(_fix_sum(0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)))
    
    # 3. Sine
    starts.append(_fix_sum(0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)))
    
    # 4. Two-frequency
    starts.append(_fix_sum(0.5 + 0.3 * np.cos(2.0 * np.pi * i / N) + 0.2 * np.cos(4.0 * np.pi * i / N)))
    
    # 5. Block pattern (like seed)
    h_block = np.zeros(N)
    for k in range(N):
        h_block[k] = 1.0 if (k % 8) < 4 else 0.0
    starts.append(_fix_sum(h_block))
    
    # 6. Random
    for _ in range(5):
        starts.append(_fix_sum(np.random.uniform(0, 1, N)))
    
    # Evaluate initial guesses
    for h0 in starts:
        _update_best(h0)
        
    # Iterative optimization with increasing beta
    betas = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    
    for beta in betas:
        if time.time() > _DEADLINE - 10.0:
            break
            
        # Optimize from current best
        h_opt = _nesterov_optimize(best_h, beta, max_iter=500)
        _update_best(h_opt)
        
        # Also try from a perturbed version of best
        h_pert = best_h + 0.1 * np.random.randn(N)
        h_pert = _fix_sum(h_pert)
        h_opt2 = _nesterov_optimize(h_pert, beta, max_iter=300)
        _update_best(h_opt2)
        
        # If time permits, try from random starts with high beta
        if time.time() < _DEADLINE - 30.0:
             for h0 in starts[:3]:
                 h_opt3 = _nesterov_optimize(h0, beta, max_iter=200)
                 _update_best(h_opt3)

    # Final Polish: Exact max reduction
    if best_h is not None and time.time() < _DEADLINE - 5.0:
        h = best_h.copy()
        for _ in range(100):
            if time.time() > _DEADLINE - 5.0:
                break
            O = _circular_overlap_values(h)
            max_idx = np.argmax(O)
            # Try to reduce O[max_idx]
            # O(s) = N/2 - R_circ(s)
            # To reduce O(s), increase R_circ(s)
            # R_circ(s) = sum_i h_i h_{i+s}
            # Gradient of R_circ(s) wrt h is h_shifted + h_shifted_back
            # We want to move h in the direction that increases R_circ(max_idx)
            # while keeping sum constant and bounds.
            
            # Simple heuristic: adjust pairs (i, i+max_idx) to be more correlated
            # If h[i] < h[i+max_idx], increase h[i] and decrease h[i+max_idx]?
            # No, we want h[i] and h[i+max_idx] to be similar (both high or both low) to increase product.
            # If h[i] < 0.5 and h[i+max_idx] < 0.5, increase both? No, sum constraint.
            # If h[i] < h[i+max_idx], we can increase h[i] and decrease h[i+max_idx] to make them closer?
            # Product x(1-x) is maximized at 0.5.
            # We want to maximize sum of products.
            # Let's use a small gradient step on the exact max objective.
            
            # Gradient of max(O) is 0 except at max_idx, where it is 1.
            # So we want to decrease O[max_idx].
            # dO/max_idx/dh = - (h_shifted + h_shifted_back)
            # So we want to increase R_circ(max_idx).
            # Direction d = h_shifted + h_shifted_back
            # h_new = h + alpha * d
            # But this increases sum(h). We need to project.
            
            s = max_idx
            if s == 0:
                d = 2.0 * h
            else:
                d = np.roll(h, -s) + np.roll(h, s)
                
            # Project d to have zero sum
            d = d - np.mean(d)
            
            # Line search for alpha
            best_alpha = 0.0
            best_val = _score(h)
            for alpha in [0.01, 0.005, 0.001, -0.01, -0.005, -0.001]:
                h_try = h + alpha * d
                h_try = _fix_sum(h_try)
                val_try = _score(h_try)
                if val_try < best_val - 1e-10:
                    best_val = val_try
                    best_alpha = alpha
                    h = h_try
            if best_alpha == 0.0:
                break
                
        _update_best(h)

    if best_h is None:
        best_h = np.full(N, 0.5)
        
    return best_h

# EVOLVE-BLOCK-END
