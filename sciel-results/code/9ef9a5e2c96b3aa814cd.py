import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _calc_score(h):
    """
    Calculate the score (upper bound on C3) for a given step function h.
    Score = max_{k != 0} |(h*h)_k| / ||h||_2^2
    For binary +/- 1, ||h||_2^2 = N.
    """
    corr = np.correlate(h, h, mode='full')
    mask = np.ones(len(corr), dtype=bool)
    mask[N-1] = False
    peak = np.max(np.abs(corr[mask]))
    norm = np.sum(h**2)
    if norm == 0:
        return 0.0
    return peak / norm

def _autocorr_fast(x):
    """
    Compute autocorrelation R_k for k=1..N-1 using FFT.
    R_k = sum_{j=0}^{N-1-k} x[j] * x[j+k]
    This is the real part of the autocorrelation.
    We compute the full circular autocorrelation, then extract the linear part.
    
    For linear autocorrelation:
    Let X = FFT(x, 2*N)
    Power = X * conj(X)
    AC = IFFT(Power).real
    The linear autocorrelation r[k] for k=0..N-1 is in AC[0:N] and AC[2*N-(N-1):] ?
    Actually, for zero-padded x to length 2N:
    The autocorrelation is circular with period 2N.
    r[k] = sum_{i=0}^{N-1-k} x[i] x[i+k] for k=0..N-1.
    r[-k] = r[k].
    
    Using FFT of length 2N:
    x_pad = [x, 0, ..., 0] (length 2N)
    X = FFT(x_pad)
    Power = X * conj(X)
    AC = IFFT(Power).real
    AC[k] = sum_{i=0}^{2N-1} x_pad[i] x_pad[(i+k) % 2N]
    For k=0..N-1:
    AC[k] = sum_{i=0}^{N-1-k} x[i] x[i+k] + sum_{i=N}^{2N-1} ... (zeros) 
    Wait, x_pad[i] is 0 for i>=N.
    So if i+k >= N, x_pad[i+k] might be 0? No, modulo 2N.
    If i < N and i+k >= N, then x_pad[i+k] = 0 (since k < N, i+k < 2N-1, so no wrap to start, just into zero region).
    Wait, if i is large, say i=N-1, k=1, i+k=N. x_pad[N]=0.
    So AC[k] = sum_{i=0}^{N-1-k} x[i] x[i+k]. This is exactly R_k.
    For k=N..2N-1, it wraps.
    So R_k for k=1..N-1 is AC[1:N].
    """
    size = 2 * N
    F = np.fft.fft(x, size)
    P = F * np.conj(F)
    AC = np.fft.ifft(P).real
    # R_k is AC[k] for k=1..N-1
    return AC[1:N]

def _objective_and_grad(x, beta):
    """
    Objective: smooth max of autocorrelation sidelobes.
    x is a continuous vector of length N.
    We want to minimize max_{k!=0} |R_k|.
    """
    R = _autocorr_fast(x)
    
    # Soft max
    abs_R = np.abs(R)
    m = np.max(abs_R)
    if m == 0:
        f = 0.0
        w = np.ones(N-1) / (N-1)
    else:
        shifted = beta * (abs_R - m)
        # Clip to avoid overflow in exp, though m is max so shifted <= 0
        # exp(shifted) <= 1. Sum <= N-1. Safe.
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        f = m + np.log(sum_exp) / beta
        w = exp_shifted / sum_exp
    
    # Gradient
    # grad = sum_k w_k * sign(R_k) * dR_k/dx
    # dR_k/dx_i = x_{i+k} + x_{i-k}
    # grad[i] = sum_{k=1}^{N-1} c_k * (x_{i+k} + x_{i-k})
    # where c_k = w_k * sign(R_k).
    # This is correlation of c (extended to negative lags) with x.
    
    c = np.zeros(2*N-1)
    # c is symmetric kernel.
    # c[N-1+k] = c_k for k=1..N-1
    # c[N-1-k] = c_k for k=1..N-1
    # c[N-1] = 0
    
    # Vectorize filling c
    lags = np.arange(1, N)
    vals = w * np.sign(R)
    c[N-1 + lags] = vals
    c[N-1 - lags] = vals
    
    # Now compute grad = correlate(x, c, 'valid')?
    # np.correlate(x, c) computes sum x[n-m] c[m].
    # We want sum_k c_k x[i+k].
    # Let's check np.correlate definition again.
    # np.correlate(a, v, mode)
    # a is first arg, v is second.
    # c[k] = sum n a[n+k] conj(v[k]) ? No.
    # NumPy doc: "Cross-correlation of two 1 dimensional sequences."
    # correlate(a, v) = sum_n a[n+k] conj(v[n]) ?
    # Actually, for mode='full':
    # out[k] = sum_n a[n] conj(v[n-k])?
    # Let's just use the convolution trick or the loop if it's fast enough.
    # With N=400, loop is 160k ops. Very fast in Python?
    # 160k * 1000 iterations = 160M ops. Might be slow in Python loop.
    # Let's use FFT for gradient too?
    # grad = IFFT(FFT(c_full) * conj(FFT(x_padded)))?
    # Let's try the loop first, it's simple. If too slow, switch to FFT.
    # But wait, I can construct the gradient via FFT easily.
    
    # Let's define a function to compute the gradient via FFT.
    # We want y[i] = sum_{k=-(N-1)}^{N-1} c[k] x[i+k].
    # This is cross-correlation of x and c.
    # Let C = FFT(c, 2*N).
    # Let X = FFT(x, 2*N).
    # The correlation is IFFT(conj(C) * X).
    # Let's verify.
    # Correlation y[i] = sum_k c[k] x[i+k].
    # Let z = convolve(x, c_rev).
    # Since c is symmetric, c_rev = c.
    # So y = convolve(x, c).
    # Convolution y[i] = sum_k x[k] c[i-k].
    # This is not sum_k c[k] x[i+k].
    # sum_k c[k] x[i+k] = sum_j x[j] c[j-i].
    # This is convolve(x, c)[i] if c is symmetric?
    # conv(x, c)[i] = sum_j x[j] c[i-j].
    # We have c[j-i]. Since c is symmetric, c[j-i] = c[i-j].
    # So yes, grad = convolve(x, c).
    
    # We need to extract the part corresponding to indices 0..N-1.
    # x is length N. c is length 2N-1.
    # conv is length N + 2N - 1 - 1 = 3N - 2.
    # The "center" of c is at N-1.
    # The valid correlation for x (0..N-1) should be where the center of c aligns with x.
    
    # Let's just use the loop. It's safer for indexing.
    # And N=400 is small.
    grad = np.zeros(N)
    # Loop over k
    # To speed up, we can do:
    # For each k, grad[:N-k] += vals[k-1] * x[k:]
    #            grad[k:]   += vals[k-1] * x[:N-k]
    # This is 2*(N-1) vector additions of length ~N/2.
    # Total ops ~ N^2.
    
    for k in range(1, N):
        val = w[k-1] * np.sign(R[k-1])
        if val == 0:
            continue
        grad[:N-k] += val * x[k:]
        grad[k:] += val * x[:N-k]
        
    return f, grad

def _local_search(h, time_limit):
    """
    Fast local search using incremental autocorrelation update.
    """
    current_score = _calc_score(h)
    best_score = current_score
    best_h = h.copy()
    
    # Precompute R for current h
    # R[k] for k=1..N-1
    R = _autocorr_fast(h)
    
    def update_R(R, idx, old_val, new_val):
        """
        Update R when h[idx] changes from old_val to new_val.
        delta = new_val - old_val.
        R_k changes by delta * (h[idx+k] + h[idx-k]).
        """
        delta = new_val - old_val
        # For k=1..N-1-idx: term h[idx+k]
        max_k1 = N - 1 - idx
        if max_k1 >= 0:
            R[:max_k1+1] += delta * h[idx+1:idx+1+max_k1]
        # For k=1..idx: term h[idx-k]
        max_k2 = idx
        if max_k2 >= 0:
            R[:max_k2+1] += delta * h[idx-1::-1][:max_k2] # h[idx-1], h[idx-2]...
            
    improved = True
    stuck_count = 0
    start = time.time()
    
    while improved and (time.time() - start) < time_limit:
        improved = False
        for j in range(N):
            if time.time() > start + time_limit:
                break
            old_val = h[j]
            new_val = -old_val
            h[j] = new_val
            
            update_R(R, j, old_val, new_val)
            new_score = np.max(np.abs(R)) / N # approx, ignoring norm change? Norm is constant N.
            
            if new_score < current_score:
                current_score = new_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_h = h.copy()
            else:
                h[j] = old_val
                update_R(R, j, new_val, old_val)
        
        if not improved:
            # Perturb
            stuck_count += 1
            if stuck_count > 5:
                break
            
            for _ in range(3):
                j = np.random.randint(N)
                old_val = h[j]
                new_val = -old_val
                h[j] = new_val
                update_R(R, j, old_val, new_val)
            
            current_score = np.max(np.abs(R)) / N
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
            else:
                # Maybe revert perturbation if it's too bad?
                # For now, keep it to explore.
                pass

    return best_h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin (limit 150s)
    
    best_h = None
    best_score = float('inf')
    
    # 1. Random Initialization
    num_initial = 50
    for _ in range(num_initial):
        if time.time() > deadline - 20:
            break
        h = np.random.choice([-1, 1], size=N)
        score = _calc_score(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    if best_h is None:
        return np.ones(N, dtype=int)
        
    # 2. Gradient Optimization (Continuous Relaxation)
    # Try a few different betas or restarts?
    # Let's do 1 main optimization run.
    x0 = best_h.astype(float)
    
    # We can try to optimize from a few different random starts or the best random.
    # Let's stick to best random for now to save time, or try 3 randoms.
    
    candidates = [x0]
    for _ in range(2):
        if time.time() > deadline - 30:
            break
        h_rand = np.random.choice([-1, 1], size=N)
        candidates.append(h_rand.astype(float))
    
    for x0 in candidates:
        if time.time() > deadline - 10:
            break
            
        beta = 10.0
        try:
            res = minimize(
                _objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 500}
            )
            x_opt = res.x
            h_opt = np.sign(x_opt)
            h_opt[h_opt == 0] = 1
            score_opt = _calc_score(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
        except Exception:
            pass
            
    # 3. Local Search Refinement
    # Give remaining time to local search
    remaining = deadline - 5 - time.time()
    if remaining > 2:
        h_refined = _local_search(best_h, remaining)
        score_refined = _calc_score(h_refined)
        if score_refined < best_score:
            best_score = score_refined
            best_h = h_refined
            
    return best_h
