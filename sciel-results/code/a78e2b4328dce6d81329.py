"""Second autocorrelation inequality: choose 50 nonnegative heights to maximize
||h||_2^2 / (||h||_1 * ||h*h||_inf).

New strategy:
- The objective is scale-invariant: scaling h by c>0 leaves the ratio unchanged.
  So we can constrain ||h||_1 = 1 (probability simplex) and maximize
  f(h) = ||h||_2^2 / ||h*h||_inf  (with ||h||_1 = 1).
- We perform projected gradient ascent on the simplex using the analytical
  gradient of log f = log(||h||_2^2) - log(||h*h||_inf).
- The gradient of log(||h*h||_inf) w.r.t. h[i] is (2 * (h*h_active)[i]) / ||h*h||_inf
  where h*h_active is the auto-convolution value at the lag achieving the supremum.
  Since h is nonnegative, the supremum of h*h is at lag 0 (the DC component) 
  ONLY IF h is nonnegative? Actually for nonnegative h, (h*h)[0] = sum h[i]^2 = ||h||_2^2,
  and (h*h)[k] for k!=0 can be larger or smaller. But wait: for nonnegative h,
  (h*h)[0] = sum_i h[i]^2. Other lags: (h*h)[k] = sum_i h[i] h[i+k]. 
  By Cauchy-Schwarz, (h*h)[k] <= ||h||_2 * ||shifted h||_2 <= ||h||_2^2 = (h*h)[0].
  So for nonnegative h, the supremum of the auto-convolution is ALWAYS at lag 0!
  
  This is a critical insight: for nonnegative h, ||h*h||_inf = (h*h)[0] = ||h||_2^2.
  
  Therefore the objective becomes:
  f(h) = ||h||_2^2 / (||h||_1 * ||h||_2^2) = 1 / ||h||_1.
  
  To maximize 1/||h||_1, we want to MINIMIZE ||h||_1 subject to... but there's no 
  other constraint. We can make ||h||_1 arbitrarily small by scaling down, but the
  problem says "nonnegative heights" - presumably there's an implicit constraint that
  the heights are on [0,1] or that we're choosing the shape.
  
  Wait, re-reading: "Choose 50 nonnegative heights on equal intervals of [-1/4,1/4]".
  The heights are the step heights of a piecewise constant function. The L2 norm and L1
  norm are of this function. If the function is h(x) = sum_i a_i * 1_{interval_i}(x),
  then ||h||_1 = sum_i a_i * dx, ||h||_2^2 = sum_i a_i^2 * dx.
  The auto-convolution is (h*h)(t) = integral h(s)h(t-s) ds.
  
  For a step function with equal intervals of width dx = (1/2)/50 = 1/100,
  the discrete auto-convolution relates to the continuous one by a factor of dx.
  
  But the key insight remains: for nonnegative h, the peak of the continuous auto-convolution
  is at t=0, and equals ||h||_2^2. So:
  score = ||h||_2^2 / (||h||_1 * ||h||_2^2) = 1 / ||h||_1.
  
  This means the score is simply 1/||h||_1, and to maximize it we minimize ||h||_1.
  But we can make ||h||_1 -> 0 by scaling all heights to 0. That can't be right.
  
  There must be a normalization constraint. Perhaps the heights are in [0,1] and at least
  one is 1? Or perhaps the L1 norm is fixed? Let me re-read.
  
  "Choose 50 nonnegative heights on equal intervals of [-1/4,1/4] to maximize the ratio
   of squared L2 norm to L1 norm times supremum norm of the autoconvolution."
  
  If there's no constraint, the problem is unbounded (scale h -> 0 gives score -> inf).
  So there must be an implicit constraint. Common ones: max(h) = 1, or ||h||_1 = 1, or
  the heights are in [0,1].
  
  Given that the seed (uniform) scores 1.0 and a spike scores ~1.003, and the insight
  that for nonnegative h the auto-conv peak is at lag 0 = ||h||_2^2, the score is
  1/||h||_1 (if we use the continuous norms with the correct dx factors).
  
  Actually, let's be careful with the continuous vs discrete. Let dx = 1/100.
  h(x) = a_i for x in [(i-25)*dx, (i-24)*dx) for i=0..49.
  ||h||_1 = dx * sum a_i
  ||h||_2^2 = dx * sum a_i^2
  (h*h)(0) = integral h(s)^2 ds = dx * sum a_i^2 = ||h||_2^2
  
  So score = (dx * sum a_i^2) / (dx * sum a_i) * (dx * sum a_i^2) ... no wait.
  score = ||h||_2^2 / (||h||_1 * ||h*h||_inf) = (dx * sum a_i^2) / ((dx * sum a_i) * (dx * sum a_i^2))
        = (dx * sum a_i^2) / (dx^2 * sum a_i * sum a_i^2) = 1 / (dx * sum a_i)
  
  So score = 100 / sum(a_i). To maximize, minimize sum(a_i). Again unbounded below.
  
  So there MUST be a constraint. The most natural one given "heights" is that max(a_i) = 1
  or that the heights are in [0,1] with at least one equal to 1.
  
  If max(a_i) = 1, then to minimize sum(a_i) we set one a_i = 1 and all others = 0.
  Then sum = 1, score = 100. But the seed scores 1.0, so the actual metric must be normalized
  differently, or the "score" reported is relative to the seed.
  
  The task says "score is the lower bound on C2" and "the search score is value/seed".
  So the raw value for the spike (one height=1, rest=0) is some number, and the seed's
  raw value is another, and score = raw/seed_raw.
  
  If raw = 100/sum(a_i) with max(a_i)=1 constraint, then:
  - Spike: sum=1, raw=100
  - Uniform: sum=50, raw=100/50=2
  - Score for spike = 100/2 = 50. But observation says spike scores 1.003.
  
  This doesn't match. So my insight that ||h*h||_inf = ||h||_2^2 for nonnegative h must be
  wrong in the discrete/continuous setup, OR the auto-convolution is defined differently.
  
  Let me reconsider: maybe the "autoconvolution" is the discrete convolution of the 
  sequence [a_0, ..., a_49] with itself, not the continuous convolution of the step function.
  In that case, (h*h)[0] = sum a_i^2, and (h*h)[k] = sum a_i a_{i+k}. For nonnegative a,
  (h*h)[0] is NOT necessarily the maximum! Example: a = [1,1,0,...,0]. 
  (h*h)[0] = 1+1 = 2. (h*h)[1] = 1*1 = 1. (h*h)[-1] = 1. So max is 2 at lag 0.
  But a = [1, 0.5, 0, ..., 0]: (h*h)[0] = 1+0.25=1.25, (h*h)[1]=0.5. Max at 0.
  a = [1, 1, 1, 0,...]: (h*h)[0]=3, (h*h)[1]=2, (h*h)[2]=1. Max at 0.
  
  Actually, for any nonnegative sequence, (h*h)[0] = sum a_i^2, and (h*h)[k] = sum a_i a_{i+k}
  <= sum a_i^2 by rearrangement or just by noting that the sets {a_i} and {a_{i+k}} are 
  "more correlated" when k=0. In fact, (h*h)[k] <= (h*h)[0] for all k, with equality iff
  the sequence is periodic with period |k| or constant. So for nonnegative sequences,
  the supremum of the discrete auto-convolution IS at lag 0.
  
  So the score = sum(a_i^2) / (sum(a_i) * sum(a_i^2)) = 1/sum(a_i). Still unbounded.
  
  I'm stuck on the normalization. Let me just implement the objective as stated, assume
  there's an implicit max(h)=1 constraint (or that the evaluator normalizes), and use
  gradient ascent on the simplex with the constraint that at least one element is 1.
  
  Actually, the simplest interpretation: the evaluator computes the raw score as
  ||h||_2^2 / (||h||_1 * ||h*h||_inf) for the given h, and the "score" is that raw value
  divided by the seed's raw value. The seed is presumably the uniform vector [1,1,...,1].
  
  For uniform [1,...,1] (N=50): 
  ||h||_2^2 = 50, ||h||_1 = 50, (h*h)[0] = 50. Score = 50/(50*50) = 0.02.
  For spike [0,...,1,...,0]:
  ||h||_2^2 = 1, ||h||_1 = 1, (h*h)[0] = 1. Score = 1/(1*1) = 1.
  Relative score = 1/0.02 = 50.
  
  But observation says spike scores 1.003. So the metric must be different.
  
  Perhaps the auto-convolution is normalized by the L1 norm? Or perhaps it's the
  cross-correlation? Or perhaps the "supremum norm of the autoconvolution" means
  the L-infinity norm of the *normalized* auto-convolution (h*h)/(||h||_1^2)?
  
  If we define (h*h)_norm = (h*h) / ||h||_1^2, then ||(h*h)_norm||_inf = ||h*h||_inf / ||h||_1^2.
  Then score = ||h||_2^2 / (||h||_1 * ||h*h||_inf / ||h||_1^2) = ||h||_2^2 * ||h||_1 / ||h*h||_inf.
  For uniform: 50 * 50 / 50 = 50. For spike: 1 * 1 / 1 = 1. Relative = 1/50 = 0.02. No.
  
  Let me try: score = ||h||_2^2 / (||h||_1^2 * ||h*h||_inf / ||h||_1^2) = ||h||_2^2 / ||h*h||_inf.
  For uniform: 50/50 = 1. For spike: 1/1 = 1. Both give 1. Relative = 1. That matches "seed scores 1.0"!
  And for a slightly perturbed spike, it might score 1.003.
  
  So the metric is likely: score = ||h||_2^2 / ||h*h||_inf.
  For nonnegative h, ||h*h||_inf = (h*h)[0] = ||h||_2^2, so score = 1 always!
  
  That can't be right either, because then all nonnegative h give score 1.
  
  Unless... the auto-convolution is NOT the discrete convolution but the continuous one,
  and for the continuous case, the peak is NOT necessarily at lag 0 for step functions.
  
  For a step function h(x) = sum a_i 1_{I_i}(x), the auto-convolution is
  (h*h)(t) = sum_{i,j} a_i a_j * |I_i ∩ (I_j + t)|.
  At t=0: (h*h)(0) = sum a_i^2 * |I_i| = dx * sum a_i^2.
  At t = dx: (h*h)(dx) = sum_{i} a_i a_{i+1} * dx (assuming adjacent intervals overlap by dx).
  
  So the ratio (h*h)(dx) / (h*h)(0) = sum a_i a_{i+1} / sum a_i^2.
  This can be > 1 if the sequence is "smooth" (e.g., a_i = 1 for all i gives 
  sum a_i a_{i+1} = N-1, sum a_i^2 = N, ratio = (N-1)/N < 1).
  For a = [1, 1, 0, ..., 0]: sum a_i a_{i+1} = 1, sum a_i^2 = 2, ratio = 0.5 < 1.
  For a = [1, 0.5, 0.25, ...] geometric: sum a_i a_{i+1} = sum r^i r^{i+1} = r sum r^{2i} = r/(1-r^2),
  sum a_i^2 = sum r^{2i} = 1/(1-r^2). Ratio = r < 1.
  
  So for nonnegative step functions, the peak of the continuous auto-convolution is ALWAYS
  at t=0. So ||h*h||_inf = (h*h)(0) = dx * ||h||_2^2 (discrete) = ||h||_2^2 (continuous).
  
  I'm going in circles. Let me just implement the objective as ||h||_2^2 / (||h||_1 * ||h*h||_inf)
  with the discrete auto-convolution, and use gradient-based optimization. The key is to
  use a DIFFERENT method: projected gradient ascent on the simplex with analytical gradients.
"""

import numpy as np
import time

N = 50

def _autoconvolution(h):
    """Compute the full autoconvolution of h with itself (discrete)."""
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)
    return conv[:2 * N - 1]

def _score(h):
    """Compute the score: ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h, dtype=float)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h ** 2)
    conv = _autoconvolution(h)
    l_inf = np.max(conv)  # h is nonnegative, so conv is nonnegative
    if l_inf == 0:
        return 0.0
    return l2_sq / (l1 * l_inf)

def _score_and_grad(h):
    """Compute score and gradient of log(score) w.r.t. h for nonnegative h.
    score = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    log(score) = log(||h||_2^2) - log(||h||_1) - log(||h*h||_inf)
    
    grad log(||h||_2^2) = 2h / ||h||_2^2
    grad log(||h||_1) = 1 / ||h||_1
    grad log(||h*h||_inf): if the max is at lag k*, then (h*h)[k*] = sum_i h[i] h[i+k*]
    (with appropriate indexing). d/dh[j] (h*h)[k*] = h[j+k*] + h[j-k*] (with boundary handling).
    So grad log(||h*h||_inf)[j] = (h[j+k*] + h[j-k*]) / ||h*h||_inf, where h[out of bounds] = 0.
    """
    h = np.asarray(h, dtype=float)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0, np.zeros_like(h)
    l2_sq = np.sum(h ** 2)
    if l2_sq == 0:
        return 0.0, np.zeros_like(h)
    conv = _autoconvolution(h)
    l_inf = np.max(conv)
    if l_inf == 0:
        return 0.0, np.zeros_like(h)
    score = l2_sq / (l1 * l_inf)
    
    # Find the lag achieving the supremum
    k_star = np.argmax(conv)
    # conv is indexed from 0 to 2N-2, where conv[k] corresponds to lag k - (N-1)
    # Actually, for two arrays of length N, conv[k] = sum_i h[i] h[i+k-N+1] ... let's be careful.
    # np.convolve(h, h, 'full')[k] = sum_i h[i] h[k-i] for i where both indices are in [0,N-1].
    # So conv[k] corresponds to the sum over i of h[i]*h[k-i], which is the auto-conv at lag k-(N-1)
    # if we think of it as (N-1 - k) or something. Let's just use the fact that
    # d/dh[j] conv[k] = h[j] * h[k-j] + h[k-j] * h[j] ... no.
    # conv[k] = sum_i h[i] h[k-i]. d/dh[j] conv[k] = h[k-j] + h[j] * (d/dh[j] h[k-j]) ... 
    # Actually, conv[k] = sum_i h[i] h[k-i]. If we differentiate w.r.t. h[j]:
    # The term where i=j gives h[k-j]. The term where k-i=j i.e. i=k-j gives h[k-j].
    # So d/dh[j] conv[k] = 2 * h[k-j] if 0 <= k-j < N, else 0. But we need to be careful:
    # if i=j and i=k-j are the same (i.e., j = k/2), then it's h[j]^2 and the derivative is 2h[j].
    # In general, d/dh[j] conv[k] = h[k-j] + h[k-j] = 2 h[k-j] if j != k-j, else 2 h[j].
    # Wait, that's not right. Let me redo:
    # conv[k] = sum_{i=0}^{N-1} h[i] h[k-i]
    # d/dh[j] conv[k] = sum_i [delta(i,j) h[k-i] + h[i] delta(k-i,j)]
    # = h[k-j] + h[k-j] = 2 h[k-j], provided 0 <= k-j < N.
    # But if k-j < 0 or k-j >= N, then h[k-j] = 0, so the derivative is 0.
    # However, this double-counts when i=j and k-i=j simultaneously, i.e., j = k-j, i.e., k=2j.
    # In that case, the term is h[j]^2, and d/dh[j] h[j]^2 = 2 h[j]. And our formula gives 2 h[k-j] = 2 h[j]. Correct.
    
    # So grad_j log(conv[k*]) = 2 * h[k*-j] / conv[k*] if 0 <= k*-j < N, else 0.
    # But k* is the index in the full convolution (0 to 2N-2). The lag is k* - (N-1).
    # Let lag = k* - (N-1). Then conv[k*] = sum_i h[i] h[i+lag] (with h[out of bounds]=0).
    # d/dh[j] conv[k*] = h[j+lag] + h[j-lag] (with h[out of bounds]=0).
    
    lag = k_star - (N - 1)
    grad_conv = np.zeros(N)
    for j in range(N):
        val = 0.0
        if 0 <= j + lag < N:
            val += h[j + lag]
        if 0 <= j - lag < N:
            val += h[j - lag]
        grad_conv[j] = val
    
    grad_log_score = 2 * h / l2_sq - 1 / l1 - grad_conv / l_inf
    
    return score, grad_log_score

def _project_to_simplex(v):
    """Project v onto the probability simplex {x: x >= 0, sum x = 1}."""
    v = np.asarray(v, dtype=float)
    n = len(v)
    if np.sum(v) <= 0:
        return np.ones(n) / n
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    k = np.nonzero(cond)[0]
    if len(k) == 0:
        return np.ones(n) / n
    theta = cssv[k[-1]] / (k[-1] + 1)
    w = np.maximum(v - theta, 0)
    return w

def _project_to_max_one(v):
    """Project v onto {x: x >= 0, max(x) <= 1}. We want to stay in this set.
    Actually, for the simplex projection, we just need x >= 0 and sum = 1.
    The max <= 1 is automatically satisfied if sum = 1 and x >= 0 (since max <= sum = 1).
    So we just need the standard simplex projection.
    """
    return _project_to_simplex(v)

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin
    
    best_h = None
    best_score = -np.inf
    
    def update_best(h):
        nonlocal best_h, best_score
        s = _score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
        return s
    
    # Initializers: all should be on the simplex (sum=1) or scaled to have max=1
    # We'll work on the simplex (sum=1) since the objective is scale-invariant.
    initializers = []
    
    # 1. Single spike (one element = 1, rest = 0)
    for i in range(N):
        h = np.zeros(N)
        h[i] = 1.0
        initializers.append(h)
    
    # 2. Two adjacent spikes
    for i in range(N - 1):
        h = np.zeros(N)
        h[i] = 0.5
        h[i + 1] = 0.5
        initializers.append(h)
    
    # 3. Narrow Gaussian centered at various positions
    for center in range(0, N, 5):
        for sigma in [0.5, 1.0, 1.5, 2.0, 3.0]:
            x = np.arange(N)
            h = np.exp(-0.5 * ((x - center) / sigma) ** 2)
            if np.sum(h) > 0:
                h /= np.sum(h)
            initializers.append(h)
    
    # 4. Triangular profiles
    for width in [5, 10, 15, 20, 25, 30]:
        for center in range(0, N, 5):
            h = np.zeros(N)
            for i in range(N):
                d = abs(i - center)
                if d < width:
                    h[i] = 1 - d / width
            if np.sum(h) > 0:
                h /= np.sum(h)
            initializers.append(h)
    
    # 5. Geometric decay (centered)
    for r in [0.5, 0.7, 0.8, 0.9, 0.95]:
        for center in [N // 4, N // 2, 3 * N // 4]:
            h = np.zeros(N)
            for i in range(N):
                d = abs(i - center)
                h[i] = r ** d
            if np.sum(h) > 0:
                h /= np.sum(h)
            initializers.append(h)
    
    # Evaluate all initializers
    for h in initializers:
        if time.time() > deadline - 10:
            break
        update_best(h)
    
    # Gradient ascent on the simplex
    # We'll do multiple restarts from the best initializer and a few others
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() > deadline - 15:
            break
        
        # Choose starting point
        if restart == 0:
            h = best_h.copy()
        elif restart % 2 == 0:
            # Pick a random initializer
            h = initializers[np.random.randint(len(initializers))].copy()
        else:
            # Perturb the best
            h = best_h.copy()
            h += np.random.normal(0, 0.1, N)
            h = np.maximum(h, 0)
            if np.sum(h) > 0:
                h /= np.sum(h)
            else:
                h = best_h.copy()
        
        h = _project_to_simplex(h)
        current_score = _score(h)
        
        # Gradient ascent with adaptive step size
        step_size = 0.1
        patience = 0
        max_patience = 100
        
        for step in range(500):
            if time.time() > deadline - 10:
                break
            if patience > max_patience:
                break
            
            score, grad = _score_and_grad(h)
            if score > current_score:
                current_score = score
                patience = 0
            else:
                patience += 1
            
            # Projected gradient step
            h_new = h + step_size * grad
            h_new = _project_to_simplex(h_new)
            
            new_score = _score(h_new)
            if new_score > current_score:
                h = h_new
                current_score = new_score
                step_size = min(step_size * 1.05, 0.5)
                patience = 0
            else:
                step_size = max(step_size * 0.9, 0.001)
            
            if current_score > best_score:
                best_score = current_score
                best_h = h.copy()
        
        # After gradient ascent, try a few coordinate updates
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            idx = np.random.randint(N)
            for delta in [0.01, 0.05, 0.1, 0.2]:
                h_try = h.copy()
                h_try[idx] += delta
                if h_try[idx] < 0:
                    h_try[idx] = 0
                h_try = _project_to_simplex(h_try)
                s_try = _score(h_try)
                if s_try > current_score:
                    h = h_try
                    current_score = s_try
                    break
    
    # Final: also try scaling to max=1 (in case the evaluator expects that)
    # But since the objective is scale-invariant, it shouldn't matter.
    # Return the best found
    if best_h is None:
        # Fallback: single spike
        h = np.zeros(N)
        h[N // 2] = 1.0
        return list(h)
    
    return list(best_h)

if __name__ == "__main__":
    uniform = np.ones(N)
    print("Uniform score:", _score(uniform))
    
    spike = np.zeros(N)
    spike[N // 2] = 1.0
    print("Spike score:", _score(spike))
    
    # Test gradient
    h = np.random.exponential(1, N)
    h /= np.sum(h)
    score, grad = _score_and_grad(h)
    print("Test score:", score)
    print("Grad sum:", np.sum(grad))
    
    result = construct()
    print("Best score:", _score(np.array(result)))
    print("Result sum:", sum(result))
