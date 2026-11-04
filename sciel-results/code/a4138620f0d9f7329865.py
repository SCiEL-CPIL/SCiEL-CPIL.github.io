import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _autoconvolution(h):
    """Compute the full autoconvolution of h with itself."""
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)
    return conv[:2 * N - 1]

def _score(h):
    """Compute the score: ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h, dtype=float)
    if np.any(h < 0):
        return 0.0
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h ** 2)
    conv = _autoconvolution(h)
    l_inf = np.max(np.abs(conv))
    if l_inf == 0:
        return 0.0
    return l2_sq / (l1 * l_inf)

def _score_normalized(h):
    """Score assuming h is non-negative. Returns 0 if invalid."""
    if np.any(h < 0):
        return 0.0
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h ** 2)
    conv = _autoconvolution(h)
    l_inf = np.max(np.abs(conv))
    if l_inf == 0:
        return 0.0
    return l2_sq / (l1 * l_inf)

def _objective_log_inf(x):
    """
    Objective to minimize: log(||h*h||_inf) subject to ||h||_2 = 1.
    We enforce ||h||_2 = 1 by normalizing inside the objective if needed, 
    but L-BFGS-B works on unconstrained variables. 
    Actually, it's better to use a penalty or just normalize h in the objective.
    However, standard approach for scale invariant problems:
    Minimize f(h) = log(||h*h||_inf) with constraint ||h||_2 = 1.
    We can use the method of Lagrange multipliers or just normalize h in the objective.
    Let's normalize h in the objective to ensure scale invariance is handled.
    """
    h = np.abs(x)  # Ensure non-negativity
    l2 = np.linalg.norm(h)
    if l2 < 1e-12:
        return 1e10
    h_norm = h / l2
    conv = _autoconvolution(h_norm)
    l_inf = np.max(np.abs(conv))
    if l_inf < 1e-12:
        return 1e10
    return np.log(l_inf)

def _objective_log_inf_grad(x):
    """
    Gradient of log(||h*h||_inf) w.r.t h, assuming h is normalized to unit L2.
    Let g = h*h. Let m = max(g).
    d/dh log(m) = (1/m) * dm/dh.
    dm/dh = sum_{k: g_k = m} (2 * (h * delta_k)_i) where delta_k is the shift.
    Actually, (h*h)_k = sum_j h_j h_{k-j}.
    d(g_k)/dh_i = h_{k-i} + h_{k-N+i} (due to symmetry and full conv definition).
    Wait, standard convolution: (f*g)[n] = sum_k f[k] g[n-k].
    For autoconvolution of h (length N), the result has length 2N-1.
    (h*h)[k] = sum_{j=0}^{N-1} h[j] h[k-j] where h is zero-padded.
    d/dh_i (h*h)[k] = h[k-i] + h[k-i-N] ... no.
    Let's use the FFT definition or direct differentiation.
    (h*h)[k] = sum_j h[j] h[k-j].
    d/dh_i (h*h)[k] = h[k-i] + h[k-i] (if i and k-i are distinct? No, product rule).
    d/dh_i (sum_j h[j] h[k-j]) = (d/dh_i h[i]) * h[k-i] + sum_j h[j] * (d/dh_i h[k-j])
    = h[k-i] + h[k-i] (since d/dh_i h[k-j] is 1 if k-j=i => j=k-i).
    So d/dh_i (h*h)[k] = 2 * h[k-i] (with zero padding for out of bounds).
    Note: if k-i is out of [0, N-1], h[k-i] is 0.
    
    So grad_x log(m) = (1/m) * sum_{k in argmax} 2 * h[k-i].
    
    But we are optimizing x where h = |x|? No, let's just optimize h >= 0 directly.
    The constraint is ||h||_2 = 1.
    If we use L-BFGS-B on h >= 0, we need to handle the constraint.
    A common trick: optimize on the sphere. But L-BFGS-B doesn't do that directly.
    Alternative: Optimize on h >= 0 without explicit constraint, but normalize in objective?
    If we normalize in objective, the gradient is not straightforward.
    
    Better approach: Use the fact that the problem is scale invariant.
    We can fix ||h||_2 = 1.
    Let's use `minimize` with method 'SLSQP' or 'trust-constr' for equality constraints?
    SLSQP might be slow.
    
    Let's stick to the simple heuristic that worked in inspiration 3:
    Minimize ||h*h||_inf directly? No, log is smoother.
    
    Let's try a different perspective:
    The score is S = ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    Since ||h||_2^2 = 1 (fixed), we want to minimize ||h||_1 * ||h*h||_inf.
    Let P = ||h||_1 * ||h*h||_inf.
    This is still non-smooth.
    
    Let's go back to the insight: "fix ||h||2=1, minimize ||h*h||inf".
    Why is this equivalent?
    S = 1 / (||h||_1 * ||h*h||_inf).
    So maximizing S is equivalent to minimizing Q = ||h||_1 * ||h*h||_inf.
    Note that ||h||_1 is not constant if ||h||_2 = 1.
    However, for sparse vectors, ||h||_1 is close to 1. For dense vectors, ||h||_1 is larger.
    Minimizing ||h*h||_inf tends to concentrate mass (sparsity), which also minimizes ||h||_1 (for fixed L2=1, min ||h||_1 is 1, max is sqrt(N)).
    So minimizing ||h*h||_inf is a good proxy.
    
    Let's implement gradient descent on the sphere ||h||_2=1, h>=0.
    We can parameterize h using a softmax or just project.
    Simplest: Use L-BFGS-B on h >= 0, and add a penalty for ||h||_2 != 1?
    Or just normalize h in the objective and use the chain rule for gradient?
    
    Let's use the projection approach:
    Objective f(h) = log(||h*h||_inf) + lambda * (||h||_2^2 - 1)^2.
    
    Actually, the most robust and fast method for this specific problem (maximizing ratio of norms of autoconvolution) is often found by concentrating mass.
    Let's try a direct optimization of the score using `minimize` with a penalty for non-negativity and L2 norm.
    
    Revised Plan:
    1. Define objective to maximize: -log( ||h||_1 * ||h*h||_inf / ||h||_2^2 )
       Wait, S = ||h||_2^2 / (||h||_1 * ||h*h||_inf).
       Maximize S <=> Minimize -log(S) = log(||h||_1 * ||h*h||_inf) - 2*log(||h||_2).
       This is scale invariant, so we can fix ||h||_2 = 1.
       Then Minimize log(||h||_1 * ||h*h||_inf).
       Let g(h) = log(||h||_1) + log(||h*h||_inf).
       This is smooth enough for L-BFGS-B if we handle the max.
       
    2. Use L-BFGS-B with bounds [0, inf] and a penalty for ||h||_2 != 1?
       Or better: Use the fact that we can normalize.
       Let's just use the gradient of log(||h*h||_inf) and log(||h||_1).
       
    Gradient of log(||h||_1) = 1/||h||_1 * (1, 1, ..., 1).
    Gradient of log(||h*h||_inf) as derived above.
    
    Total Gradient G = grad_log_l1 + grad_log_linf.
    
    We need to enforce ||h||_2 = 1.
    We can do this by projecting the gradient onto the tangent space of the sphere?
    Or simply use a constrained optimizer.
    
    Let's use `scipy.optimize.minimize` with method='SLSQP' and constraint ||h||_2^2 - 1 = 0.
    SLSQP is slower but handles constraints.
    Alternatively, use 'trust-constr'.
    
    Given the 120s limit, SLSQP on 50 variables might be okay if we limit iterations.
    
    Let's try a different, simpler approach that is very effective for these "concentration" problems:
    Start with a spike. The optimum is likely a spike or a small cluster.
    We can do a grid search over the "width" of the cluster and the "shape" (Gaussian, Box, Triangle).
    But the seed program already did that and got 1.0.
    
    Wait, the seed program got 1.0. The inspiration 3 got 1.078.
    The inspiration 3 used L-BFGS-B on log-surrogate.
    
    Let's implement the L-BFGS-B on the log-surrogate with explicit gradient.
    To handle the constraint ||h||_2=1, we can re-normalize h after each step? No, that breaks the gradient.
    
    Better: Optimize in log-space? h_i = exp(y_i).
    Then ||h||_2 = 1 is sum exp(2y_i) = 1.
    This is still a constraint.
    
    Let's use the "penalty method" with L-BFGS-B.
    Objective: log(||h||_1 * ||h*h||_inf) + mu * (||h||_2^2 - 1)^2.
    Start with mu = 100, increase mu if constraint violation is high.
    
    Actually, the simplest robust method for this specific task (often solved by spikes) is to test specific structured candidates and then do a local Nelder-Mead polish.
    The seed program's local search was weak.
    
    Let's construct a set of very strong candidates:
    1. Single Spike at each index.
    2. Two adjacent spikes.
    3. Small Gaussian bumps of various widths.
    4. Then apply Nelder-Mead to the best few.
    
    Nelder-Mead is derivative-free and handles non-smoothness (max) well.
    It is slower per iteration but very robust.
    With 50 dimensions, it might be slow, but if we start close to the optimum (spike), it should converge quickly.
    
    Let's combine:
    1. Evaluate a grid of structured candidates (spikes, small bumps).
    2. Take the top 5.
    3. Run Nelder-Mead on each with tight tolerances.
    4. Return the best.
    
    This avoids the complexity of constrained gradient methods and exploits the likely sparsity of the optimum.
    """
    h = np.abs(x)
    l2 = np.linalg.norm(h)
    if l2 < 1e-12:
        return 1e10, np.zeros_like(x)
    
    # Normalize to unit L2 for the objective calculation
    h_norm = h / l2
    
    l1 = np.sum(h_norm)
    conv = _autoconvolution(h_norm)
    l_inf = np.max(np.abs(conv))
    
    if l1 < 1e-12 or l_inf < 1e-12:
        return 1e10, np.zeros_like(x)
    
    val = np.log(l1) + np.log(l_inf)
    
    # Gradient w.r.t h_norm
    # d/dh log(l1) = 1/l1
    grad_l1 = np.ones_like(h) / l1
    
    # d/dh log(l_inf)
    # max indices
    max_idx = np.where(np.abs(conv) == l_inf)[0]
    grad_linf = np.zeros_like(h)
    for k in max_idx:
        # d/dh_i (h*h)[k] = 2 * h[k-i] (with zero padding)
        # We need to add 2*h[k-i] to grad_linf[i] for all i.
        # This is a convolution of a delta at k with h, scaled by 2.
        # Specifically, grad_linf[i] += 2 * h[k-i] if k-i is in [0, N-1].
        # Let's construct this efficiently.
        # For a fixed k, the vector v_i = h[k-i] is a shifted/reversed h.
        # We can use np.roll or slicing.
        # Let's just loop i for simplicity since N=50 is small.
        for i in range(N):
            j = k - i
            if 0 <= j < N:
                grad_linf[i] += 2 * h_norm[j]
            # Note: autoconvolution is symmetric, so (h*h)[k] = (h*h)[2N-2-k].
            # The derivative formula 2*h[k-i] assumes standard convolution.
            # Let's verify with a small example.
            # h = [1, 0]. h*h = [1, 0, 0].
            # k=0: d/dh_0 = 2*h[0] = 2. d/dh_1 = 2*h[-1]=0.
            # k=1: d/dh_0 = 2*h[1] = 0. d/dh_1 = 2*h[0] = 2.
            # k=2: d/dh_0 = 2*h[2] = 0. d/dh_1 = 2*h[1] = 0.
            # Correct.
            
    grad_linf /= l_inf
    
    # Total gradient w.r.t h_norm
    grad_hnorm = grad_l1 + grad_linf
    
    # Chain rule: h_norm = h / l2.
    # We are optimizing x (which is h in this context, but we passed x).
    # Actually, we should optimize h directly, not x.
    # The function _objective_log_inf takes x, which we treat as h.
    # But we normalized inside.
    # This is problematic for gradient.
    # Let's change strategy: Optimize h directly, and add a penalty for ||h||_2 != 1.
    
    # Penalty term: mu * (||h||_2^2 - 1)^2
    mu = 1000.0
    penalty = mu * (np.sum(h**2) - 1.0)**2
    val += penalty
    
    # Gradient of penalty: 2 * mu * (||h||_2^2 - 1) * 2 * h
    grad_penalty = 4 * mu * (np.sum(h**2) - 1.0) * h
    
    # Gradient of log terms w.r.t h (unnormalized)
    # We calculated grad w.r.t h_norm.
    # h_norm = h / l2.
    # d h_norm_i / d h_j = (delta_ij * l2 - h_i * 2 * h_j / l2) / l2^2 = (delta_ij / l2) - (2 * h_i * h_j) / l2^3
    # This is getting messy.
    
    # Let's restart the gradient calculation for unnormalized h.
    # Objective: f(h) = log(sum(h)) + log(max(conv(h))) + mu * (sum(h^2)-1)^2
    
    # d/dh log(sum(h)) = 1/sum(h)
    g1 = np.ones_like(h) / np.sum(h)
    
    # d/dh log(max(conv(h)))
    # Let M = max(conv(h)).
    # d/dh_i M = sum_{k in argmax} d/dh_i conv_k
    # d/dh_i conv_k = 2 * h[k-i] (zero padded)
    g2 = np.zeros_like(h)
    max_idx = np.where(np.abs(conv) == l_inf)[0]
    for k in max_idx:
        for i in range(N):
            j = k - i
            if 0 <= j < N:
                g2[i] += 2 * h[j]
    g2 /= l_inf
    
    # Total gradient
    grad = g1 + g2 + grad_penalty
    
    return val, grad

def _objective_log_inf_simple(x):
    """Simpler objective for Nelder-Mead: just the score, negated."""
    h = np.abs(x)
    l2 = np.linalg.norm(h)
    if l2 < 1e-12:
        return 1e10
    h_norm = h / l2
    l1 = np.sum(h_norm)
    conv = _autoconvolution(h_norm)
    l_inf = np.max(np.abs(conv))
    if l1 < 1e-12 or l_inf < 1e-12:
        return 1e10
    score = 1.0 / (l1 * l_inf)
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # 1. Generate diverse initial candidates
    candidates = []
    
    # Spikes
    for i in range(N):
        h = np.zeros(N)
        h[i] = 1.0
        candidates.append(h)
        
    # Adjacent pairs
    for i in range(N - 1):
        h = np.zeros(N)
        h[i] = 1.0
        h[i+1] = 1.0
        candidates.append(h)
        
    # Small Gaussians
    for center in range(0, N, 2):
        for width in [1.0, 1.5, 2.0, 3.0]:
            x = np.arange(N)
            h = np.exp(-((x - center) ** 2) / (2 * width ** 2))
            if np.max(h) > 0:
                h = h / np.linalg.norm(h)
                candidates.append(h)
                
    # Small Box functions
    for center in range(0, N, 2):
        for half_width in [1, 2, 3]:
            h = np.zeros(N)
            start = max(0, center - half_width)
            end = min(N, center + half_width + 1)
            h[start:end] = 1.0
            if np.sum(h) > 0:
                h = h / np.linalg.norm(h)
                candidates.append(h)
                
    # Evaluate all candidates
    for h in candidates:
        if time.time() > deadline - 30:
            break
        s = _score_normalized(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
            
    # 2. Polish the top candidates with Nelder-Mead
    # We need to track the top few to polish them.
    # Let's just polish the best one found so far, and maybe a few others if time permits.
    # To be safe, let's collect the top 5 from the candidates.
    
    scored_candidates = []
    for h in candidates:
        s = _score_normalized(h)
        scored_candidates.append((s, h))
        
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    top_k = 5
    for s, h in scored_candidates[:top_k]:
        if time.time() > deadline - 10:
            break
            
        # Normalize to unit L2 for optimization
        h_opt = h.copy()
        l2 = np.linalg.norm(h_opt)
        if l2 > 0:
            h_opt = h_opt / l2
        else:
            continue
            
        # Nelder-Mead on the negative score
        # We use a simple objective that doesn't require gradient
        def nm_obj(x):
            return _objective_log_inf_simple(x)
            
        try:
            res = minimize(nm_obj, h_opt, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            h_res = np.abs(res.x)
            l2_res = np.linalg.norm(h_res)
            if l2_res > 0:
                h_res = h_res / l2_res
                s_res = _score_normalized(h_res)
                if s_res > best_score:
                    best_score = s_res
                    best_h = h_res.copy()
        except Exception:
            pass
            
    return list(best_h)

if __name__ == "__main__":
    # Quick test
    uniform = np.ones(N)
    print("Uniform score:", _score(uniform))
    
    spike = np.zeros(N)
    spike[N//2] = 1.0
    print("Spike score:", _score(spike))
    
    result = construct()
    print("Best score:", _score(np.array(result)))
    print("Result sum:", sum(result))
