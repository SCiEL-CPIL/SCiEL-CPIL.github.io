import numpy as np
from scipy.optimize import minimize
import time

N = 50

def compute_conv_max(h):
    """Compute the supremum norm of the autoconvolution of step heights h."""
    # h has N elements, autoconvolution has 2N-1 elements
    # conv[k] = sum_{i+j=k} h[i]*h[j] for k=0,...,2N-2
    # But for step functions on equal intervals, the max of the piecewise-linear
    # autoconvolution occurs at the midpoints of the sum intervals.
    # For simplicity and correctness with step functions, the peak of the
    # autoconvolution of two step functions with N bins each is at the
    # 2N-1 positions corresponding to sums of bin indices.
    
    # The autoconvolution of step functions: if bin i has height h[i] and width w,
    # then (h*h)(t) is the integral. For step functions, the max occurs at
    # t = (i+j) * w for i+j ranging, but actually it's piecewise linear with
    # peaks at the "center" of each sum.
    
    # For step functions on equal intervals, the autoconvolution is a piecewise
    # linear function. The maximum occurs at one of the 2N-1 points where
    # the "diagonal" i+j is constant. At position k (0-indexed), the value is:
    # sum_{i=max(0,k-N+1)}^{min(N-1,k)} h[i] * h[k-i]
    
    # This is the standard discrete convolution, and for step functions with
    # equal widths, the continuous autoconvolution's max equals the max of
    # these discrete conv values (since the function is linear between breakpoints
    # and the breakpoints are at half-integer positions, but the peaks are at
    # integer positions in the discrete sense).
    
    # Actually, let me be more careful. For step functions on [a,b] divided into N
    # equal bins, the autoconvolution (h*h)(t) = ∫ h(τ)h(t-τ)dτ. This is a
    # piecewise linear function. The breakpoints are at t = x_i + x_j where x_i
    # are bin edges. The maximum occurs at one of the "center" points.
    
    # For simplicity, the discrete convolution sum gives the correct values at
    # the peak locations for step functions.
    
    conv = np.convolve(h, h, mode='full')
    return np.max(conv)

def objective_and_grad(h, t):
    """Epigraph formulation: minimize t subject to t >= (sum(h)*conv_max)/sum(h^2), h>=0"""
    pass

def ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    l2_sq = np.sum(h**2)
    if l2_sq < 1e-30:
        return 0.0
    l1 = np.sum(h)
    conv_max = compute_conv_max(h)
    if l1 < 1e-30 or conv_max < 1e-30:
        return 0.0
    return l2_sq / (l1 * conv_max)

def objective(h):
    """Minimize the reciprocal of the ratio, i.e., ||h||_1 * ||h*h||_inf / ||h||_2^2"""
    l2_sq = np.sum(h**2)
    if l2_sq < 1e-30:
        return 1e10
    l1 = np.sum(h)
    conv_max = compute_conv_max(h)
    return l1 * conv_max / l2_sq

def objective_grad(h):
    """Compute gradient of objective = ||h||_1 * ||h*h||_inf / ||h||_2^2"""
    l2_sq = np.sum(h**2)
    if l2_sq < 1e-30:
        return np.ones(N) * 1e10
    l1 = np.sum(h)
    conv = np.convolve(h, h, mode='full')
    conv_max = np.max(conv)
    # Find which index achieves the max
    max_idx = np.argmax(conv)
    
    # conv[k] = sum_i h[i]*h[k-i]
    # d(conv[k])/d(h[j]) = h[k-j] + h[j-(k-j)] ... need to be careful
    # For k fixed, conv[k] = sum_{i=0}^{k} h[i]*h[k-i] where valid indices
    # d(conv[k])/d(h[j]) = h[k-j] if 0<=k-j<N, plus h[j] * d/d(h[j])h[k-j]
    # Actually: conv[k] = sum_{i: 0<=i<N, 0<=k-i<N} h[i]*h[k-i]
    # d(conv[k])/d(h[j]) = sum_{i: i=j or k-i=j} h[other]
    # = h[k-j] (if 0<=k-j<N) + h[j] * (if j == k-j, i.e., k=2j, then it's 2*h[j])
    # Wait, let me redo:
    # conv[k] = sum_{i=max(0,k-N+1)}^{min(N-1,k)} h[i] * h[k-i]
    # d(conv[k])/d(h[j]) = h[k-j] if j is in range and k-j is in range
    #                     + h[j] if (k-j) is in range and j is in range (from the other term)
    # But if j == k-j, i.e., k = 2j, then it's 2*h[j]
    
    # More carefully:
    # conv[k] = sum_i h[i] h[k-i]
    # d/dh[j] = h[k-j] + h[j] * delta_{j, k-j} ... no
    # d/dh[j] of h[i]h[k-i]:
    #   if i = j: h[k-j]
    #   if k-i = j, i.e., i = k-j: h[k-j]
    #   if j = k-j (i.e., k=2j): both terms give h[j], so 2*h[j]
    
    # So d(conv[k])/d(h[j]) = h[k-j] + h[k-j] if k-j == j, else h[k-j] if k-j valid
    # Let me just compute it directly:
    
    grad_conv = np.zeros(N)
    k = max_idx
    for j in range(N):
        # Contribution from h[j] in the first position: h[k-j] if 0<=k-j<N
        if 0 <= k - j < N:
            grad_conv[j] += h[k - j]
        # Contribution from h[j] in the second position: h[k-j] if 0<=k-j<N  
        # Wait, the second position is h[k-i], so when k-i=j, i=k-j
        # d/dh[j] of h[i]*h[k-i] where k-i=j means i=k-j
        # This term is h[k-j] * h[j], derivative w.r.t h[j] is h[k-j]
        if 0 <= k - j < N:
            # This is the same condition, and it's the same value
            # But we need to check if this is a double count
            # If j == k-j, i.e., k=2j, then both i=j and k-i=j refer to the same term
            # In that case, d/dh[j] of h[j]*h[j] = 2*h[j]
            # My formula gives h[k-j] + h[k-j] = 2*h[j] when k=2j, which is correct
            # When k != 2j, we have two distinct terms: h[j]*h[k-j] and h[k-j]*h[j]
            # Wait no, conv[k] = sum_i h[i]*h[k-i]. The term with i=j is h[j]*h[k-j].
            # The term with k-i=j, i.e., i=k-j, is h[k-j]*h[j].
            # These are the same term! So I'm double counting.
            # Let me reconsider.
            pass
    
    # Let me redo this more carefully
    grad_conv = np.zeros(N)
    k = max_idx
    for i in range(max(0, k - N + 1), min(N, k + 1)):
        j = k - i
        if 0 <= j < N:
            # term is h[i] * h[j]
            # d/dh[i] = h[j]
            # d/dh[j] = h[i]
            if i == j:
                grad_conv[i] += 2 * h[i]
            else:
                grad_conv[i] += h[j]
                grad_conv[j] += h[i]
    
    # Now objective = l1 * conv_max / l2_sq
    # d(obj)/d(h[j]) = (1 * conv_max + l1 * grad_conv_max[j]) / l2_sq 
    #                  - l1 * conv_max * 2*h[j] / l2_sq^2
    # where grad_conv_max[j] is d(conv_max)/d(h[j])
    
    # Note: conv_max is the max over k, so we only differentiate the active constraint
    # (the one achieving the max). If there are ties, we'd need subgradient, but
    # for SLSQP this should be fine.
    
    grad = (conv_max + l1 * grad_conv) / l2_sq - 2 * h * l1 * conv_max / l2_sq**2
    return grad

def solve_with_slsqp(h0, maxiter=1000, ftol=1e-12):
    """Solve using SLSQP with analytic gradients"""
    # Normalize to unit L2 norm to avoid scale issues
    norm = np.sqrt(np.sum(h0**2))
    if norm < 1e-10:
        return h0, 0.0
    h0 = h0 / norm
    
    bounds = [(0, None)] * N
    constraints = []
    
    res = minimize(
        objective,
        h0,
        jac=objective_grad,
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': maxiter, 'ftol': ftol}
    )
    
    h_opt = res.x
    # Ensure nonnegativity
    h_opt = np.maximum(h_opt, 0)
    # Renormalize
    norm = np.sqrt(np.sum(h_opt**2))
    if norm < 1e-10:
        return h0, 0.0
    h_opt = h_opt / norm
    
    score = ratio(h_opt)
    return h_opt, score

def initial_guesses():
    """Generate diverse initial guesses"""
    guesses = []
    
    # All ones
    guesses.append(np.ones(N))
    
    # Uniform
    guesses.append(np.ones(N))
    
    # Triangle
    t = np.linspace(0, 1, N)
    guesses.append(np.maximum(0, 1 - np.abs(2*t - 1)))
    
    # Parabolic
    t = np.linspace(-1, 1, N)
    guesses.append(np.maximum(0, 1 - t**2))
    
    # Flat top
    guesses.append(np.ones(N) * 0.5)
    guesses[N//4:3*N//4] = 1.0
    
    # Bump in center
    guesses.append(np.exp(-5 * (np.linspace(-1, 1, N))**2))
    
    # Wide bump
    guesses.append(np.exp(-2 * (np.linspace(-1, 1, N))**2))
    
    # Two bumps
    t = np.linspace(-1, 1, N)
    guesses.append(np.exp(-20*(t-0.3)**2) + np.exp(-20*(t+0.3)**2))
    
    # Random
    np.random.seed(42)
    guesses.append(np.random.rand(N) + 0.1)
    
    # Another random
    np.random.seed(123)
    guesses.append(np.random.rand(N) + 0.1)
    
    # Smooth random
    np.random.seed(456)
    r = np.random.rand(N)
    # Smooth it
    kernel = np.ones(5) / 5
    r = np.convolve(r, kernel, mode='same')
    guesses.append(r + 0.1)
    
    return guesses

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s
    
    best_h = np.ones(N)
    best_score = ratio(best_h)
    
    guesses = initial_guesses()
    
    for i, h0 in enumerate(guesses):
        if time.time() > deadline:
            break
        
        h_opt, score = solve_with_slsqp(h0, maxiter=500, ftol=1e-12)
        
        if score > best_score:
            best_score = score
            best_h = h_opt.copy()
        
        # Also try polishing from the result
        if time.time() < deadline - 10:
            h_polish, score_polish = solve_with_slsqp(h_opt, maxiter=1000, ftol=1e-14)
            if score_polish > best_score:
                best_score = score_polish
                best_h = h_polish.copy()
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        h_final, score_final = solve_with_slsqp(best_h, maxiter=2000, ftol=1e-15)
        if score_final > best_score:
            best_score = score_final
            best_h = h_final.copy()
    
    # Ensure nonnegativity and return as list
    best_h = np.maximum(best_h, 0)
    return list(best_h)

