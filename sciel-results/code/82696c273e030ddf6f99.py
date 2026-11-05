"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np

def _compute_value(h):
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e6
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv) / (s * s))

def _smooth_max(conv, temperature=0.1):
    """Soft max using log-sum-exp for differentiability."""
    m = np.max(conv)
    if m < 0:
        m = 0
    if temperature <= 0:
        return m
    shifted = conv / temperature
    shifted = shifted - np.max(shifted)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return temperature * (log_sum + np.max(shifted))

def _objective_smooth(h, temperature=0.1):
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e6
    conv = np.convolve(h, h, mode='full')
    max_conv = _smooth_max(conv, temperature)
    return abs(2.0 * N * max_conv / (s * s))

def _gradient_step(h, temperature=0.1, lr=0.01):
    """One gradient descent step on the smooth objective."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return h
    
    conv = np.convolve(h, h, mode='full')
    
    # Compute gradient of smooth max
    m = np.max(conv)
    shifted = conv / temperature
    shifted = shifted - np.max(shifted)
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    
    # Gradient of conv[k] w.r.t. h[i]
    # conv[k] = sum_j h[j] * h[k-j] (with boundary handling)
    # d(conv[k])/d(h[i]) = h[k-i] + h[i-(k-N+1)] (terms that are in range)
    
    # For the weighted max, grad = sum_k weights[k] * d(conv[k])/d(h)
    grad_h = np.zeros(N)
    
    for k in range(len(conv)):
        if weights[k] < 1e-12:
            continue
        wk = weights[k]
        for i in range(N):
            # Term 1: h[i] * h[k-i], derivative w.r.t. h[i] is h[k-i] if k-i in [0,N-1]
            idx1 = k - i
            if 0 <= idx1 < N:
                grad_h[i] += wk * h[idx1]
            # Term 2: h[j] * h[k-j], when j=i, derivative w.r.t. h[i] is h[k-i]
            # But we also have the symmetric term: when k-j = i, j = k-i
            idx2 = k - (N - 1) - i  # This handles the other direction
            # Actually, let's be more careful
            # conv[k] = sum_{j=0}^{N-1} h[j] * h[k-j] where h[x]=0 if x not in [0,N-1]
            # d(conv[k])/d(h[i]) = h[k-i] (if 0<=k-i<N) + h[i-(k-N+1)] (if 0<=i-(k-N+1)<N)
            # The second term comes from when k-j = i, i.e., j = k-i, but that's the same as first
            # Wait, no. Let me redo.
            # conv[k] = sum_{j} h[j] h[k-j]
            # d/dh[i] = sum_j [delta(j,i) h[k-j] + h[j] delta(k-j, i)]
            # = h[k-i] (if 0<=k-i<N) + sum_j h[j] delta(k-j,i)
            # = h[k-i] (if valid) + h[k-i] (if 0<=k-i<N, since j=k-i)
            # Wait, that gives 2*h[k-i] if k-i is in range?
            # No: sum_j h[j] delta(k-j, i) = h[j] where j = k-i, so = h[k-i] if 0<=k-i<N
            # So total = h[k-i] + h[k-i] = 2*h[k-i] if 0<=k-i<N
            # Hmm, that doesn't seem right. Let me think again.
            # Actually, conv[k] = sum_{j=0}^{N-1} h[j] * h[k-j]
            # If we differentiate w.r.t. h[i]:
            # Only terms where j=i or k-j=i contribute
            # j=i: contributes h[k-i] (if 0<=k-i<N)
            # k-j=i => j=k-i: contributes h[k-i] (if 0<=k-i<N)
            # But if j=i and k-j=i, that's when i=k-i, i.e., k=2i, and we'd double count
            # Actually no, these are separate terms in the sum unless i = k-i
            # If i != k-i, we get two separate contributions: h[k-i] from j=i and h[k-i] from j=k-i
            # If i = k-i (k=2i), we get h[i] * h[i] and derivative is 2*h[i]
            # So in all cases: d(conv[k])/d(h[i]) = h[k-i] + h[k-i] = 2*h[k-i] if 0<=k-i<N
            # Wait, that's only if both j=i and j=k-i are valid indices
            # j=i is always valid (0<=i<N)
            # j=k-i is valid if 0<=k-i<N
            # So: if 0<=k-i<N: d(conv[k])/d(h[i]) = h[k-i] + h[k-i] = 2*h[k-i]
            # else: d(conv[k])/d(h[i]) = h[k-i] (only the j=i term)
            # Hmm wait, I think I'm overcomplicating. Let me just do it numerically or use FFT.
            pass
    
    # This is getting too complex. Let me use a simpler approach.
    # Just use finite differences for the gradient.
    eps = 1e-7
    grad = np.zeros(N)
    for i in range(N):
        h_plus = h.copy()
        h_minus = h.copy()
        h_plus[i] += eps
        h_minus[i] -= eps
        f_plus = _objective_smooth(h_plus, temperature)
        f_minus = _objective_smooth(h_minus, temperature)
        grad[i] = (f_plus - f_minus) / (2 * eps)
    
    h_new = h - lr * grad
    return h_new

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 135s deadline, 15s margin
    
    np.random.seed(42)
    
    best_h = np.ones(N)
    best_val = _compute_value(best_h)
    
    # Generate candidate signals from random walks
    candidates = []
    
    drift_noise_pairs = [
        (0.0, 1.0), (0.0, 0.5), (0.0, 2.0),
        (-0.1, 1.0), (0.1, 1.0), (-0.05, 0.5),
        (0.05, 0.5), (-0.2, 1.5), (0.2, 1.5),
        (0.0, 0.1), (0.0, 3.0),
    ]
    
    for drift, noise in drift_noise_pairs:
        if time.time() > deadline - 40:
            break
        for trial in range(2):
            if time.time() > deadline - 40:
                break
            steps = drift + noise * np.random.randn(N)
            h = np.cumsum(steps)
            h = h - np.mean(h)
            s = np.sum(h)
            if abs(s) < 1e-10:
                h[0] += 1e-5
            candidates.append(h.copy())
    
    # Evaluate all candidates
    for h in candidates:
        val = _compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Refine the best candidate with gradient descent
    if time.time() < deadline - 20:
        h_curr = best_h.copy()
        for step in range(50):
            if time.time() > deadline - 10:
                break
            h_new = _gradient_step(h_curr, temperature=0.1, lr=0.01)
            val_new = _compute_value(h_new)
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
                h_curr = h_new
            else:
                # Reduce learning rate
                h_curr = h_new
    
    return best_h
# EVOLVE-BLOCK-END
