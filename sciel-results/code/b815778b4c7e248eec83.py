"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _max_overlap(h):
    """
    Compute the maximum overlap of h with shifted (1-h).
    
    For step heights h[0..N-1], the overlap at shift k is:
    sum_i h[i] * (1 - h[(i+k) % N])
    
    We want to minimize the maximum over all shifts k.
    Note: shift 0 gives sum h[i]*(1-h[i]), which is fixed given sum h = N/2
    only if we also fix sum h^2. But actually shift 0 is just sum h[i] - sum h[i]^2.
    
    Wait, let's reconsider. The "overlap" of h with shifted 1-h at lag k:
    overlap(k) = sum_i h[i] * (1 - h[(i+k) mod N])
               = sum_i h[i] - sum_i h[i] * h[(i+k) mod N]
               = N/2 - autocorr(k)
    
    where autocorr(k) = sum_i h[i] * h[(i+k) mod N].
    
    So minimizing max_k overlap(k) is equivalent to maximizing min_k autocorr(k),
    since overlap(k) = N/2 - autocorr(k).
    
    Actually wait: we want to MINIMIZE the maximum overlap. 
    max_k overlap(k) = max_k (N/2 - autocorr(k)) = N/2 - min_k autocorr(k).
    
    So minimizing max overlap = maximizing min autocorr.
    """
    # Compute circular autocorrelation
    # autocorr[k] = sum_i h[i] * h[(i+k) % N]
    # Use FFT for efficiency
    h_fft = np.fft.fft(h)
    power = h_fft * np.conj(h_fft)
    autocorr = np.real(np.fft.ifft(power))
    # autocorr[0] = sum h[i]^2, autocorr[k] for k=1..N-1
    # The overlap at lag k is N/2 - autocorr[k]
    # We want max over k of (N/2 - autocorr[k])
    # = N/2 - min(autocorr[1:])  (excluding lag 0 since that's the "self" overlap)
    # 
    # Actually, should we include lag 0? The problem says "shifted 1-h".
    # A shift of 0 means no shift, so overlap(0) = sum h[i]*(1-h[i]).
    # Let's include all shifts 0 to N-1 to be safe, but typically "shifted" implies non-zero.
    # Let's compute max over all k including 0.
    
    overlaps = N/2.0 - autocorr
    return np.max(overlaps)


def _soft_max_overlap(h, beta):
    """
    Smooth surrogate: soft-max of overlaps using log-sum-exp.
    """
    autocorr = np.real(np.fft.ifft(np.fft.fft(h) * np.conj(np.fft.fft(h))))
    overlaps = N/2.0 - autocorr
    # log-sum-exp for numerical stability
    max_ov = np.max(overlaps)
    log_sum = np.log(np.sum(np.exp(beta * (overlaps - max_ov))))
    return max_ov + np.log(np.sum(np.exp(beta * (overlaps - max_ov)))) / beta


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # Strategy: chirp-based parameterization
    # h[i] = 0.5 + a * cos(2*pi*alpha*i + beta_phase)
    # But we need h in [0,1] and sum = N/2.
    # With sum h = N/2, the mean is 0.5.
    # Let h[i] = 0.5 + 0.5 * sin(2*pi*alpha*i + phi)
    # Then h is in [0,1] automatically.
    # Sum h = N*0.5 + 0.5*sum sin(...) = N/2 + 0.5*0 = N/2 (for integer alpha or full periods).
    # For non-integer alpha, the sum of sin may not be exactly 0.
    # Let's use: h[i] = 0.5 + 0.5 * sin(2*pi*alpha*i)
    # Sum = N/2 + 0.5 * sum_{i=0}^{N-1} sin(2*pi*alpha*i)
    # This sum is 0 if alpha is an integer. For non-integer, it's not exactly 0.
    # We can correct by adjusting the amplitude or using a projection.
    
    # Better: use a parameterization that guarantees sum = N/2.
    # Let v[i] = sin(2*pi*alpha*i + phi)
    # h[i] = 0.5 + c * (v[i] - mean(v))
    # Choose c so that h stays in [0,1].
    # max c = 1 / (2 * max|v[i] - mean(v)|) to ensure h in [0,1].
    
    # Actually, let's try a simpler approach:
    # h[i] = 0.5 + 0.5 * sin(2*pi*alpha*i)
    # Then project to [0,1] and renormalize to sum N/2.
    
    # Let's search over alpha and phi.
    # For a given alpha, phi, compute h, project, normalize.
    
    # Chirp: quadratic phase
    # h[i] = 0.5 + 0.5 * sin(2*pi*(alpha*i + 0.5*beta*i^2))
    # This is a chirp with linear frequency sweep.
    
    # Let's do a grid search over (alpha, beta, phi) first, then refine.
    
    # Actually, let's just do a simple optimization.
    # Parameters: alpha (sweep rate), phi (phase)
    # h[i] = 0.5 + 0.5 * sin(2*pi*alpha*i + phi)
    # Project to [0,1] and normalize.
    
    def make_h(alpha, phi):
        v = 0.5 + 0.5 * np.sin(2*np.pi*alpha*np.arange(N) + phi)
        # Project to [0,1]
        v = np.clip(v, 0, 1)
        # Normalize to sum N/2
        s = np.sum(v)
        if s > 0:
            v = v * (N/2.0) / s
            v = np.clip(v, 0, 1)
            # Re-normalize after clip
            s2 = np.sum(v)
            if s2 > 0:
                v = v * (N/2.0) / s2
        return v
    
    # Grid search over alpha and phi
    # alpha in [0, N/2], phi in [0, 2*pi)
    # For N=100, let's try alpha from 0 to 50 with step 0.01, phi from 0 to 2pi with step 0.1
    
    # But this might be slow. Let's be smarter.
    # Try a few good alpha values: around N/2 / k for small k.
    
    # Actually, let's just do a random search with refinement.
    
    np.random.seed(42)
    
    # Phase 1: Random search over (alpha, phi)
    n_random = 500
    for i in range(n_random):
        if time.time() > deadline - 30:
            break
        alpha = np.random.uniform(0, N/2)
        phi = np.random.uniform(0, 2*np.pi)
        h = make_h(alpha, phi)
        score = _max_overlap(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Phase 2: Local refinement using gradient-free optimization
    # Use Nelder-Mead on (alpha, phi)
    if time.time() < deadline - 20:
        from scipy.optimize import minimize
        
        def objective(params):
            alpha, phi = params
            h = make_h(alpha, phi)
            return _max_overlap(h)
        
        # Try refinement from best point
        # We need to recover alpha, phi from best_h... hard.
        # Instead, just do a few random starts with Nelder-Mead.
        
        for _ in range(10):
            if time.time() > deadline - 10:
                break
            x0 = [np.random.uniform(0, N/2), np.random.uniform(0, 2*np.pi)]
            try:
                res = minimize(objective, x0, method='Nelder-Mead', 
                             options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
                if res.fun < best_score:
                    best_score = res.fun
                    h = make_h(res.x[0], res.x[1])
                    best_h = h.copy()
            except:
                pass
    
    # Phase 3: Final polish with projected gradient descent
    if time.time() < deadline - 5:
        h = best_h.copy()
        beta = 50.0
        
        for it in range(1000):
            if time.time() > deadline - 1:
                break
            # Compute gradient of soft-max overlap
            # d/dh_k [soft_max(overlaps)] where overlaps = N/2 - autocorr
            # This is complex. Let's use a simple subgradient approach.
            # Or just do a few iterations of projected gradient on the max overlap.
            
            # Compute overlaps
            autocorr = np.real(np.fft.ifft(np.fft.fft(h) * np.conj(np.fft.fft(h))))
            overlaps = N/2.0 - autocorr
            max_ov = np.max(overlaps)
            
            # Subgradient: for each k where overlap is max, 
            # d overlap(k)/d h[i] = -h[(i+k)%N] for the h[i] term, and 1 for the (1-h) term
            # Actually overlap(k) = sum_i h[i]*(1-h[(i+k)%N])
            # d overlap(k)/d h[j] = (1-h[(j+k)%N]) + sum_{i: (i+k)%N = j} h[i]*(-1)
            # = (1-h[(j+k)%N]) - h[(j-k)%N]
            
            # For the max, we average over all k that achieve the max
            max_mask = (overlaps > max_ov - 1e-9)
            n_max = np.sum(max_mask)
            
            grad = np.zeros(N)
            for k in range(N):
                if max_mask[k]:
                    # d overlap(k)/d h[j] = (1-h[(j+k)%N]) - h[(j-k)%N]
                    j = np.arange(N)
                    grad += (1 - h[(j+k)%N]) - h[(j-k)%N]
            
            grad /= n_max
            
            # Projected gradient step
            lr = 0.01
            h_new = h - lr * grad
            h_new = np.clip(h_new, 0, 1)
            
            # Normalize to sum N/2
            s = np.sum(h_new)
            if s > 0:
                h_new = h_new * (N/2.0) / s
                h_new = np.clip(h_new, 0, 1)
                s2 = np.sum(h_new)
                if s2 > 0:
                    h_new = h_new * (N/2.0) / s2
            
            new_score = _max_overlap(h_new)
            if new_score < best_score:
                best_score = new_score
                best_h = h_new.copy()
                h = h_new
            else:
                # Reduce learning rate
                lr *= 0.9
    
    # Ensure exact sum
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h * (N/2.0) / s
        best_h = np.clip(best_h, 0, 1)
        s2 = np.sum(best_h)
        if s2 > 0:
            best_h = best_h * (N/2.0) / s2
    
    return best_h
# EVOLVE-BLOCK-END
