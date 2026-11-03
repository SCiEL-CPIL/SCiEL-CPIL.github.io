import numpy as np
import time

N = 400
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N:
    _FFT_SIZE *= 2

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1]"""
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=_FFT_SIZE)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=_FFT_SIZE)
    return np.real(acf[:N])

def _max_abs_acf_from_array(A):
    return float(np.max(np.abs(A[1:])))

def _peak_sidelobe(h):
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return float(np.max(np.abs(ac[1:])))

def _compute_acf(h):
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return ac

def _phase_gradient_descent(s_init, time_budget, start_time, deadline):
    pass

def _anneal_block_flip(h_init, time_budget, start_time, deadline, rng):
    h = h_init.copy()
    ac = _compute_acf(h)
    cur = float(np.max(np.abs(ac[1:])))
    best_h = h.copy()
    best = cur
    T = 1.0
    T_min = 1e-4
    cooling_rate = 0.999
    
    # Precompute indices for potential block flips to speed up random selection? 
    # No, just keep it simple but tune parameters.
    
    while time.time() - start_time < time_budget and time.time() < deadline:
        r = rng.random()
        if r < 0.5:
            L = int(rng.integers(1, 10))
        elif r < 0.9:
            L = int(rng.integers(10, 30))
        else:
            L = int(rng.integers(30, 60))
        L = min(L, N)
        
        i = int(rng.integers(0, N - L + 1))
        
        h[i:i+L] *= -1.0
        ac_new = _compute_acf(h)
        new_val = float(np.max(np.abs(ac_new[1:])))
        d = new_val - cur
        
        if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            cur = new_val
            ac = ac_new
            if cur < best:
                best = cur
                best_h = h.copy()
        else:
            h[i:i+L] *= -1.0
        
        T = max(T_min, T * cooling_rate)
        
    return best_h, best

def _greedy_polish(h, time_budget, start_time, deadline):
    h = h.copy()
    ac = _compute_acf(h)
    cur = float(np.max(np.abs(ac[1:])))
    best = cur
    best_h = h.copy()
    
    while time.time() - start_time < time_budget and time.time() < deadline:
        best_i = -1
        best_c = cur
        best_ac = None
        
        for i in range(N):
            if time.time() > deadline:
                return best_h, best
            h[i] *= -1.0
            ac_try = _compute_acf(h)
            c = float(np.max(np.abs(ac_try[1:])))
            if c < best_c:
                best_c = c
                best_i = i
                best_ac = ac_try.copy()
            h[i] *= -1.0
        
        if best_i < 0:
            break
        
        h[best_i] *= -1.0
        ac = best_ac
        cur = best_c
        if cur < best:
            best = cur
            best_h = h.copy()
            
    return best_h, best

def _mean_field_init(s_init, n_iters=100, beta=0.5):
    s = s_init.copy()
    kernel = np.zeros(N)
    kernel[0] = 1.0
    for k in range(1, N):
        kernel[k] = -beta / (k ** 2)
        kernel[N - k] = kernel[k]
    
    for _ in range(n_iters):
        F_s = np.fft.rfft(s.astype(np.float64))
        F_k = np.fft.rfft(kernel)
        h = np.fft.irfft(F_s * F_k, n=N)
        new_s = np.sign(h)
        new_s[new_s == 0] = s[new_s == 0]
        if np.all(new_s == s):
            break
        s = new_s
    return s

def _smooth_surrogate_optimize(h_init, time_budget, start_time, deadline, rng):
    """
    Optimize a smooth surrogate objective using gradient descent on continuous variables,
    then project to signs. The surrogate uses a soft-max of the squared autocorrelation
    sidelobes to approximate the peak sidelobe, making it differentiable.
    """
    # We optimize x in R^N, and project to signs at the end.
    # Objective: f(x) = sum_{k=1}^{N-1} exp(beta * acf_k(x)^2)
    # This approximates max(acf_k^2) for large beta.
    
    x = h_init.astype(np.float64)
    beta = 10.0
    
    # Precompute FFT size
    n_fft = _FFT_SIZE
    
    def compute_acf_and_grad(x):
        # acf = ifft(fft(x)^2)
        F = np.fft.rfft(x, n=n_fft)
        P = F * np.conj(F)
        acf = np.fft.irfft(P, n=n_fft)[:N]
        
        # Gradient of acf_k with respect to x
        # acf = ifft(|F|^2)
        # d(acf_k)/dx_i = 2 * Re( ifft( F * conj(F) * delta_{k,i} ... ) ) ?
        # Actually, gradient of sum_k exp(beta * acf_k^2) is complex.
        # Let's use finite differences for gradient? No, too slow.
        # Analytical gradient:
        # acf_k = sum_j x_j x_{j+k}
        # d(acf_k)/dx_i = x_{i-k} + x_{i+k} (with boundary handling)
        
        grad_acf = np.zeros((N, N)) # Too big? N=400, 400x400 = 160k floats. OK.
        # But we only need grad of f = sum exp(beta * acf_k^2)
        # grad_f_i = sum_k 2*beta*acf_k * exp(beta*acf_k^2) * d(acf_k)/dx_i
        
        # Let's compute weights w_k = 2*beta*acf_k*exp(beta*acf_k^2)
        # Then grad_f_i = sum_k w_k * (x_{i-k} + x_{i+k})
        
        # This can be done via convolution!
        # grad_f = w * x (circular correlation?)
        
        # acf_k = <x, shift_k(x)>
        # grad_f_i = sum_k w_k (x_{i-k} + x_{i+k})
        # This is 2 * (w * x) where * is circular correlation?
        
        # Let's verify:
        # sum_k w_k x_{i-k} is circular correlation of w and x.
        # sum_k w_k x_{i+k} is circular correlation of w and reverse(x)?
        
        # Actually, let's just use FFT for the gradient computation.
        # grad_f = 2 * ifft( w_fft * F ) ?
        
        # Let's stick to a simpler approach: use the fact that we can compute
        # the gradient via FFT efficiently.
        
        # For now, let's just do a simple projected gradient step with a small step size
        # and rely on the projection to signs to handle the non-convexity.
        
        # We need the gradient of f(x) = sum_{k=1}^{N-1} exp(beta * acf_k^2)
        
        # Compute acf
        F = np.fft.rfft(x, n=n_fft)
        P = F * np.conj(F)
        acf_full = np.fft.irfft(P, n=n_fft)
        acf = acf_full[:N]
        
        # Weights for the soft-max
        # We want to approximate max_{k>=1} acf_k^2
        # Let z_k = acf_k^2 for k>=1
        # f = sum exp(beta * z_k)
        # grad_f_i = sum_{k>=1} 2*beta*z_k*exp(beta*z_k) * d(acf_k)/dx_i
        
        # d(acf_k)/dx_i = x_{i-k} + x_{i+k} (circularly)
        
        # Let v_k = 2*beta*acf_k^2*exp(beta*acf_k^2) for k>=1, 0 for k=0
        # v is a vector of length N.
        # grad_f_i = sum_k v_k * (x_{i-k} + x_{i+k})
        # This is 2 * circular_correlation(v, x)
        
        # Circular correlation can be computed via FFT:
        # corr(v, x) = ifft( conj(V) * X )
        
        V = np.fft.rfft(v, n=n_fft)
        X = np.fft.rfft(x, n=n_fft)
        grad = 2.0 * np.fft.irfft(np.conj(V) * X, n=n_fft)[:N]
        
        return acf, grad

    v = np.zeros(N)
    
    step_size = 0.1
    
    while time.time() - start_time < time_budget and time.time() < deadline:
        acf, grad = compute_acf_and_grad(x)
        
        # Compute v
        v = np.zeros(N)
        for k in range(1, N):
            z = acf[k] * acf[k]
            v[k] = 2.0 * beta * z * np.exp(beta * z)
            
        # Recompute grad using v
        V = np.fft.rfft(v, n=n_fft)
        X = np.fft.rfft(x, n=n_fft)
        grad = 2.0 * np.fft.irfft(np.conj(V) * X, n=n_fft)[:N]
        
        # Gradient step
        x_new = x - step_size * grad
        
        # Project to signs
        x_proj = np.sign(x_new)
        x_proj[x_proj == 0] = 1.0
        
        # Check if projection changed anything
        if np.all(x_proj == np.sign(x)):
            # Converged or stuck
            break
            
        x = x_proj
        
    return x

def construct():
    start_time = time.time()
    deadline = 145.0
    
    best_val = float('inf')
    best_seq = np.ones(N)
    rng = np.random.default_rng(42)
    
    def _try_sequence(s):
        nonlocal best_val, best_seq
        A = _calc_acf(s)
        v = _max_abs_acf_from_array(A)
        if v < best_val:
            best_val = v
            best_seq = s.copy()
        return v
    
    # Phase 1: Mean-Field Initializations
    initializations = []
    n_inits = 6
    
    for i in range(n_inits):
        if time.time() > deadline - 40:
            break
        s_rand = rng.choice([-1.0, 1.0], size=N)
        s_mf = _mean_field_init(s_rand, n_iters=300, beta=0.5)
        initializations.append(s_mf)
        _try_sequence(s_mf)
        
    # Phase 2: Smooth Surrogate Optimization
    # Use the best MF result as start
    if time.time() < deadline - 20:
        time_for_smooth = min(15.0, deadline - 20 - time.time())
        h_smooth = _smooth_surrogate_optimize(best_seq, time_for_smooth, time.time(), deadline, rng)
        _try_sequence(h_smooth)
        
    # Phase 3: Simulated Annealing
    if time.time() < deadline - 10:
        time_for_anneal = deadline - 10 - time.time()
        # Use multiple starts for annealing if time permits
        n_anneals = 2
        time_per_anneal = time_for_anneal / n_anneals
        
        # Start from best_seq and best_smooth
        starts = [best_seq, h_smooth]
        
        for i in range(n_anneals):
            if time.time() > deadline - 10:
                break
            h_init = starts[i % len(starts)]
            h_best, v_best = _anneal_block_flip(h_init, time_per_anneal, time.time(), deadline, rng)
            _try_sequence(h_best)
            
    # Phase 4: Greedy Polish
    if time.time() < deadline - 5:
        time_for_polish = deadline - 5 - time.time()
        h_best, v_best = _greedy_polish(best_seq, time_for_polish, time.time(), deadline)
        _try_sequence(h_best)
        
    return best_seq

if __name__ == "__main__":
    construct()
