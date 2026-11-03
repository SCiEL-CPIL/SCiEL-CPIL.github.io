import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _fft_overlap(x):
    """Compute all circular cross-correlations of x with (1-x) via FFT.
    Returns array of length N where result[k] = sum_i x[i]*(1-x[(i+k)%N])"""
    omx = 1.0 - x
    # Circular cross-correlation: result[k] = sum_i x[i] * omx[(i+k) % N]
    # This is the circular convolution of x with reversed omx
    # Using FFT: ifft(fft(x) * fft(roll(omx, 1))) gives circular cross-correlation
    X = np.fft.fft(x)
    O = np.fft.fft(omx)
    # Cross-correlation: result[k] = sum_i x[i] * omx[(i+k) mod N]
    # = ifft(X * conj(O)) shifted... let me think carefully.
    # sum_i x[i] * omx[(i+k) mod N]
    # Let y[j] = omx[j]. We want sum_i x[i] * y[(i+k) mod N]
    # This is the circular cross-correlation (x ⋆ y)[k] = sum_i x[i] * y[(i+k) mod N]
    # In FFT: ifft(fft(x) * conj(fft(y))) gives sum_i x[i] * conj(y[(i-k) mod N])
    # Since y is real: sum_i x[i] * y[(i-k) mod N]
    # So to get sum_i x[i] * y[(i+k) mod N], we use ifft(fft(y) * conj(fft(x)))
    # which gives sum_i y[i] * x[(i-k) mod N] = sum_i x[i] * y[(i+k) mod N] by reindexing
    corr = np.fft.ifft(np.fft.fft(omx) * np.conj(np.fft.fft(x))).real
    return corr

def _max_overlap(x):
    """Compute max over all circular shifts of sum_i x[i]*(1-x[(i+k)%N])"""
    corr = _fft_overlap(x)
    return float(np.max(corr))

def _normalize(x):
    """Scale and clip x so that sum is exactly N/2, values in [0,1]"""
    x = np.clip(x, 0.0, 1.0)
    s = x.sum()
    if s < 1e-15:
        return np.full(N, 0.5)
    x = x * (N / 2.0) / s
    x = np.clip(x, 0.0, 1.0)
    for _ in range(200):
        s = x.sum()
        target = N / 2.0
        if abs(s - target) < 1e-13:
            break
        need = target - s
        if need > 0:
            room = 1.0 - x
            room[x >= 1.0] = 0
            total_room = room.sum()
            if total_room < 1e-15:
                break
            x += room * (need / total_room)
            x = np.clip(x, 0.0, 1.0)
        else:
            avail = x.copy()
            avail[x <= 0.0] = 0
            total_avail = avail.sum()
            if total_avail < 1e-15:
                break
            x -= avail * (-need / total_avail)
            x = np.clip(x, 0.0, 1.0)
    return x

def _smooth_max_obj(x, beta):
    """Smooth approximation of max overlap using log-sum-exp.
    Returns (value, gradient)"""
    corr = _fft_overlap(x)
    # log-sum-exp for numerical stability
    cmax = np.max(corr)
    logsum = cmax + np.log(np.sum(np.exp((corr - cmax) * beta)) / beta)
    # Gradient: d/dx of logsum = beta * sum_j exp(beta*(corr_j - cmax)) / sum_j exp(beta*(corr_j - cmax)) * dcorr_j/dx
    weights = np.exp((corr - cmax) * beta)
    weights /= weights.sum()
    # dcorr_k/dx[i] = sum over the contribution... 
    # corr[k] = sum_i x[i] * (1-x[(i+k)%N])
    # dcorr_k/dx[i] = (1-x[(i+k)%N]) - x[i] * sum over j where (j+k)%N == i, i.e., j = (i-k)%N
    # Actually: corr[k] = sum_i x[i]*(1-x[(i+k)%N])
    # dcorr_k/dx[i] = (1-x[(i+k)%N]) + x[i] * (-1) * [contribution from x[i] appearing in omx term]
    # x[i] appears in omx[(i'+k)%N] when (i'+k)%N = i, i.e., i' = (i-k)%N
    # So dcorr_k/dx[i] = (1-x[(i+k)%N]) - x[(i-k)%N]
    # Gradient of logsum w.r.t. x[i] = sum_k weights[k] * dcorr_k/dx[i]
    # = sum_k weights[k] * ((1-x[(i+k)%N]) - x[(i-k)%N])
    # = sum_k weights[k] - sum_k weights[k]*x[(i+k)%N] - sum_k weights[k]*x[(i-k)%N]
    # = 1 - sum_k weights[k]*x[(i+k)%N] - sum_k weights[k]*x[(i-k)%N]
    # By symmetry of weights (if we consider all k), sum_k weights[k]*x[(i+k)%N] and sum_k weights[k]*x[(i-k)%N]
    # are both = (weights * x)_circular[i] where * is circular convolution
    # sum_k weights[k] * x[(i+k)%N] = (weights circularly convolved with x reversed)[i]... 
    # Let me just compute directly using FFT for efficiency
    # sum_k weights[k] * x[(i+k)%N]: let w_rev[k] = weights[N-k mod N], then this is (w_rev * x)[i] circularly
    # Actually sum_k w[k] * x[(i+k) mod N] = sum_j w[(j-i) mod N] * x[j] = (w_rev * x)[i] where w_rev[m] = w[-m mod N]
    w_rev = np.zeros(N)
    for k in range(N):
        w_rev[(-k) % N] = weights[k]
    # (w_rev * x) circular = ifft(fft(w_rev) * fft(x))
    term1 = np.fft.ifft(np.fft.fft(w_rev) * np.fft.fft(x)).real
    # sum_k weights[k] * x[(i-k)%N] = (weights * x)[i] circularly
    term2 = np.fft.ifft(np.fft.fft(weights) * np.fft.fft(x)).real
    grad = 1.0 - term1 - term2
    return logsum, grad

def _subgradient_step(x, lr, iters=50):
    """Projected subgradient descent on the max overlap"""
    x = x.copy()
    for it in range(iters):
        corr = _fft_overlap(x)
        kmax = np.argmax(corr)
        # Subgradient of max overlap at x: dcorr_kmax/dx[i] = (1-x[(i+kmax)%N]) - x[(i-kmax)%N]
        omx = 1.0 - x
        sg = omx.copy() - np.roll(x, kmax)
        # Projected gradient step
        x_new = x - lr * sg
        x_new = _normalize(x_new)
        if _max_overlap(x_new) < _max_overlap(x):
            x = x_new
    return x

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    best_h = np.full(N, 0.5)
    best_val = _max_overlap(best_h)
    
    # Multi-restart L-BFGS-B with smooth max objective
    np.random.seed(42)
    
    # Try different initializations and beta values
    inits = []
    # Start from uniform with small perturbation
    inits.append(np.full(N, 0.5) + np.random.randn(N) * 0.01)
    # Sinusoidal inits
    for freq in [1, 2, 3, 4, 5]:
        t = np.arange(N)
        inits.append(0.5 + 0.3 * np.sin(2*np.pi*freq*t/N))
        inits.append(0.5 + 0.3 * np.cos(2*np.pi*freq*t/N))
    # Two-level inits
    for frac in [0.3, 0.4, 0.5, 0.6]:
        init = np.full(N, 0.5)
        init[:int(N*frac)] = 0.8
        init[int(N*frac):] = 0.2
        inits.append(init)
    # Random inits
    for _ in range(5):
        inits.append(np.random.uniform(0.2, 0.8, N))
    
    betas = [5.0, 10.0, 20.0, 40.0]
    
    for init in inits:
        if time.time() > deadline - 30:
            break
        init = _normalize(init)
        for beta in betas:
            if time.time() > deadline - 10:
                break
            try:
                result = minimize(
                    _smooth_max_obj,
                    init.copy(),
                    args=(beta,),
                    jac=True,
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * N,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8,
                             'maxfun': 500, 'disp': False}
                )
                if result.success or result.fun < best_val:
                    h_opt = _normalize(result.x)
                    val = _max_overlap(h_opt)
                    if val < best_val - 1e-15:
                        best_val = val
                        best_h = h_opt.copy()
            except Exception:
                pass
    
    # Final refinement with subgradient steps
    if time.time() < deadline - 5:
        refined = _subgradient_step(best_h, lr=0.01, iters=30)
        val = _max_overlap(refined)
        if val < best_val - 1e-15:
            best_val = val
            best_h = refined.copy()
    
    # Ensure exact constraints
    best_h = _normalize(best_h)
    
    return best_h
