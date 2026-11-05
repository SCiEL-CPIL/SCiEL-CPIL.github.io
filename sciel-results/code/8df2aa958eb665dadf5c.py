import numpy as np
import time
from scipy.optimize import linprog

N = 400

def construct():
    start_time = time.time()
    # Keep a safety margin of 20 seconds
    deadline = start_time + 100.0
    
    # Scale invariance: We can fix sum(h) = 1.
    # Then value = abs(2 * N * max(conv(h,h)))
    # We want to minimize V such that exists h with:
    # 1. sum(h) = 1
    # 2. h_i >= 0 (We can assume non-negative w.l.o.g for the LP relaxation? 
    #    Actually, h can be negative. But if h has negative parts, 
    #    sum(h)=1 is still possible. 
    #    However, the constraint is max(conv) <= V/(2N).
    #    conv(h,h) is quadratic, not linear. 
    #    Wait. The objective involves max(conv(h,h)). 
    #    conv(h,h)_k = sum_i h_i h_{i+k}. This is quadratic in h.
    #    So it's NOT a linear program. 
    #    
    #    Let's re-evaluate. 
    #    The "LP" idea fails because the constraint is quadratic.
    #    
    #    Alternative Strategy: 
    #    The current program uses random search. 
    #    The "inspiration" mentioned gradient-based optimization on soft-max.
    #    My "Materially Different" strategy: 
    #    Use **Subgradient Optimization** or **Projected Gradient Descent** 
    #    on the smooth approximation of the max function, 
    #    specifically targeting the **asymmetric** structures mentioned in the findings.
    #    
    #    Actually, let's look at the structure again.
    #    value = 2 * N * max_k (sum_i h_i h_{i+k}) / (sum h)^2
    #    
    #    If we assume h is non-negative, the max is at k=0 (sum h^2).
    #    Then value = 2 * N * sum(h^2) / (sum h)^2.
    #    Min of this is 2 (when h is constant).
    #    
    #    To get < 2, we MUST have negative entries in h.
    #    If h has negative entries, some convolution terms (for k != 0) might be negative,
    #    but the term at k=0 is always sum(h^2) > 0.
    #    So max(conv) is at least sum(h^2).
    #    Wait, is it? 
    #    conv(h,h)_0 = sum h_i^2.
    #    If h has negative values, sum(h^2) is still positive.
    #    So max(conv) >= sum(h^2).
    #    Therefore value >= 2 * N * sum(h^2) / (sum h)^2.
    #    By Cauchy-Schwarz, N * sum(h^2) >= (sum h)^2.
    #    So value >= 2.
    #    
    #    Is it possible to have value < 2?
    #    The inequality V >= 2 * (sum h^2) / (sum h)^2 >= 2 is a lower bound.
    #    The task says "lower is better" and seed is 2.0.
    #    The current program score is 1.1774, which means value = 2.0 / 1.1774? 
    #    No, score = seed_value / current_value? 
    #    "search score is seed/value". Seed value is 2.0.
    #    If score is 1.1774, then 2.0 / value = 1.1774 => value = 1.699.
    #    So it IS possible to get < 2.0.
    #    
    #    How? 
    #    The bound max(conv) >= sum(h^2) is only true if all other conv terms are smaller.
    #    But wait, max(conv) takes the MAX over ALL lags.
    #    If h has negative parts, conv(h,h)_0 = sum h^2 is positive.
    #    So max(conv) >= sum h^2.
    #    So value >= 2 * N * sum(h^2) / (sum h)^2.
    #    This quantity is always >= 2.
    #    
    #    Let me re-read the evaluator code carefully.
    #    value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)
    #    
    #    If my derivation is correct, value >= 2.
    #    But the score 1.1774 implies value < 2.
    #    There must be a flaw in my reasoning.
    #    
    #    Ah! np.convolve(h, h) computes the FULL convolution.
    #    The indices are 0 to 2N-2.
    #    h is length N.
    #    conv[0] = h[0]*h[0]
    #    conv[1] = h[0]*h[1] + h[1]*h[0] = 2 h[0] h[1]
    #    ...
    #    conv[N-1] = sum_{i=0}^{N-1} h[i] h[N-1-i] (if symmetric) ... no.
    #    Standard convolution: (a*b)[k] = sum_i a[i] b[k-i].
    #    For h*h, c[k] = sum_{i=max(0, k-N+1)}^{min(N-1, k)} h[i] h[k-i].
    #    
    #    My previous claim: c[0] = h[0]^2.
    #    c[1] = h[0]h[1] + h[1]h[0] = 2h[0]h[1].
    #    c[N-1] = sum_{i=0}^{N-1} h[i]h[N-1-i].
    #    c[2N-2] = h[N-1]^2.
    #    
    #    Where is sum(h^2)?
    #    sum(h^2) = c[0] + c[1]/2 + ... ? No.
    #    sum(h^2) is NOT a single term in the convolution unless N=1.
    #    The term c[k] for k=0 is just h[0]^2.
    #    The term c[N-1] (center) involves all pairs.
    #    
    #    So, if we make h[0] very small, c[0] is small.
    #    We can make the max of the convolution small by distributing the mass?
    #    
    #    Let's check the constant case h=1.
    #    c[k] = N - |k-N+1|? No.
    #    h=1. c[k] = number of overlapping ones.
    #    c[0]=1, c[1]=2, ..., c[N-1]=N, c[N]=N-1 ...
    #    Max is N.
    #    sum(h) = N.
    #    value = 2 * N * N / N^2 = 2.
    #    
    #    Can we get lower?
    #    We want to minimize max(c[k]) / (sum h)^2.
    #    Let sum h = 1. Minimize max(c[k]).
    #    For h=1/N (normalized), c[k] is triangular peak at N.
    #    Max c[N-1] = N * (1/N)^2 = 1/N.
    #    value = 2 * N * (1/N) = 2.
    #    
    #    If we use h with alternating signs?
    #    h = [1, -1, 1, -1 ...]
    #    sum h = 0. Invalid.
    #    h = [1, -1, 1, -1 ... 1, 0] (N even, last 0? or adjust)
    #    
    #    Let's try h = [1, -1] repeated.
    #    If N=2, h=[1, -1]. sum=0.
    #    If N=4, h=[1, -1, 1, -1]. sum=0.
    #    Add a small epsilon to one element?
    #    
    #    Strategy: 
    #    The "asymmetric structures" mentioned in findings suggest that 
    #    breaking symmetry allows the max convolution entry to be lower 
    #    relative to the square of the sum.
    #    
    #    I will implement a **Gradient Descent** approach on the 
    #    **Smoothed Max** of the convolution entries.
    #    f(h) = 2 * N * logsumexp(c / T) / (sum h)^2  (Softmax approximation of max)
    #    We minimize this.
    #    
    #    This is a different design family: First-order optimization on a 
    #    smooth surrogate, rather than random search or exact max evaluation.
    
    def compute_value_exact(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        c = np.convolve(h, h)
        return abs(2.0 * N * np.max(c) / (s * s))

    # Initialize with a non-trivial asymmetric start
    # Start with a ramp, which has lower max conv relative to sum^2?
    # Let's test a few analytic starts.
    
    candidates = []
    # 1. Ones
    candidates.append(np.ones(N))
    # 2. Ramp
    candidates.append(np.arange(1, N+1, dtype=np.float64))
    # 3. Sine wave (asymmetric phase?)
    x = np.linspace(0, 2*np.pi, N)
    candidates.append(np.sin(x) + 2) # Keep positive to avoid sum=0 issues initially
    # 4. Random with fixed seed
    rng = np.random.default_rng(42)
    candidates.append(rng.normal(0, 1, N))
    
    h_best = candidates[0]
    val_best = compute_value_exact(h_best)
    
    for h_cand in candidates[1:]:
        v = compute_value_exact(h_cand)
        if v < val_best:
            val_best = v
            h_best = h_cand
    
    # Gradient Descent on Smoothed Max
    # Objective: J(h) = 2 * N * logsumexp(c(h) / T) / (sum h)^2
    # We fix sum(h) = 1 to remove scale invariance and simplify gradient.
    # h = h_raw / sum(h_raw)
    # Actually, easier to just optimize h and normalize at the end?
    # No, gradient of normalized function is messy.
    # Let's optimize h directly, but project to sum=1 after each step?
    
    h = h_best.copy()
    # Normalize to sum=1
    h = h / np.sum(h)
    
    T = 0.1 # Temperature for softmax. Smaller T -> sharper max approximation.
    lr = 1e-3
    max_iters = 5000
    
    # Precompute indices for convolution? 
    # np.convolve is fast enough for N=400.
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        s = np.sum(h)
        if abs(s) < 1e-12:
            h = np.ones(N) / N
            s = 1.0
            continue
            
        c = np.convolve(h, h)
        
        # Smoothed max: logsumexp(c/T)
        # c_max = np.max(c)
        # logsumexp(c/T) = c_max/T + log(sum(exp((c-c_max)/T)))
        c_shifted = c - np.max(c)
        lse = np.max(c)/T + np.log(np.sum(np.exp(c_shifted/T)))
        
        # Objective value (approx)
        obj_val = 2.0 * N * lse / (s*s)
        
        # Gradient of lse w.r.t h
        # d/dh logsumexp(c/T) = (1/T) * softmax(c/T) * dc/dh
        # softmax(c/T) = exp(c_shifted/T) / sum(exp(c_shifted/T))
        p = np.exp(c_shifted/T)
        p = p / np.sum(p)
        
        # dc/dh is a Toeplitz-like matrix. 
        # c[k] = sum_i h[i] h[k-i]
        # dc[k]/dh[j] = h[k-j] + h[j-k] (with boundary checks)
        # Actually, gradient of sum_k p[k] c[k] w.r.t h[j]
        # = sum_k p[k] (h[k-j] + h[j-k]) (if indices valid)
        # This is equivalent to 2 * convolve(p, h)[j] ?
        # Let's check.
        # g[j] = sum_k p[k] h[k-j] + sum_k p[k] h[j-k]
        # Let m = k-j. sum_m p[m+j] h[m]. This is cross correlation.
        # np.convolve(p, h)[j] = sum_i p[i] h[j-i].
        # So sum_k p[k] h[k-j] = convolve(p, h)[j] ?
        # conv(p,h)[j] = sum_i p[i] h[j-i]. Let i=k. sum_k p[k] h[j-k].
        # So term 2 is conv(p,h)[j].
        # Term 1: sum_k p[k] h[k-j]. Let i=j-k. No.
        # h[k-j] is h at index k-j.
        # This is conv(h, p)[j]?
        # conv(h,p)[j] = sum_i h[i] p[j-i]. Let i=k-j. sum_{k-j} h[k-j] p[k]. Yes.
        # So grad_lse[j] = (1/T) * (conv(h,p)[j] + conv(p,h)[j])
        # Since conv(h,p) = conv(p,h) for real signals? No, convolution is commutative.
        # h*p = p*h. So grad_lse = (2/T) * conv(h, p)[j] ?
        # Wait. conv(h,p)[j] = sum_i h[i] p[j-i].
        # My term 1 was sum_k p[k] h[k-j].
        # Let i = k-j. k = i+j. sum_i p[i+j] h[i].
        # This is NOT conv(h,p)[j].
        # conv(h,p)[j] = sum_i h[i] p[j-i].
        # If p is symmetric, maybe?
        # Let's just compute the gradient numerically or use the explicit sum.
        # For N=400, explicit sum is O(N^2) = 160k ops. Fast.
        
        grad_lse = np.zeros(N)
        # p has length 2N-1.
        # c has length 2N-1.
        # We need sum_k p[k] * (h[k-j] + h[j-k])
        # Valid k for h[k-j]: 0 <= k-j < N => j <= k < j+N
        # Valid k for h[j-k]: 0 <= j-k < N => j-N < k <= j
        
        # Efficiently:
        # g[j] = sum_k p[k] h[k-j] + sum_k p[k] h[j-k]
        # Note that h[j-k] = h[(-1)*(k-j)]? No.
        # Let's use FFT? Or just loops?
        # N=400, 400 iterations of 400 sums is 160k. Python loop is slow.
        # Vectorize.
        
        # g1[j] = sum_k p[k] h[k-j]
        # This is cross-correlation of p and h.
        # np.correlate(p, h, mode='same')? 
        # np.correlate(p, h, 'full') gives sum_i p[i] h[i+j] ?
        # Definition: (a*b)[n] = sum_m a[m] b[n-m].
        # We want sum_k p[k] h[k-j]. Let n=k-j => k=n+j.
        # sum_{n+j} p[n+j] h[n].
        # This is conv(p, h) but with p reversed?
        # Let's stick to the fact that gradient of quadratic form x^T A x is (A+A^T)x.
        # Here c = vec(h) ^T M vec(h)? No, c is vector of quadratic forms.
        # c_k = h^T T_k h.
        # grad_h (p^T c) = sum_k p_k grad_h (h^T T_k h) = sum_k p_k (T_k + T_k^T) h.
        # T_k is the matrix for convolution lag k.
        # T_k is a Toeplitz matrix.
        # (T_k + T_k^T) is symmetric Toeplitz.
        # The sum is still a Toeplitz matrix operation.
        # Specifically, grad = 2 * (p * h) where * is convolution?
        # Let's test with small N.
        
        # Let's just use a simple finite difference check for the first few steps to be safe,
        # or use the analytic form: 
        # grad_obj[j] = (2/T) * sum_k p[k] * (h[k-j] + h[j-k]) (with 0 padding)
        # This is 2 * (p * h)[j] if p is symmetric? 
        # Actually, sum_k p[k] h[k-j] is (p * h)[j] if we define convolution as sum p[k] h[j-k]?
        # Standard conv: (p*h)[j] = sum_k p[k] h[j-k].
        # My term 2: sum_k p[k] h[j-k]. This IS (p*h)[j].
        # My term 1: sum_k p[k] h[k-j].
        # Let i = j-k. k = j-i.
        # sum_i p[j-i] h[i].
        # This is (p_reversed * h)[j] where p_rev[i] = p[-i]?
        # Actually, if we just compute conv(p, h), we get term 2.
        # Term 1 is conv(h, p)[j]?
        # conv(h,p)[j] = sum_i h[i] p[j-i].
        # Let i=k-j. sum_{k-j} h[k-j] p[k]. Yes.
        # So Term 1 = conv(h,p)[j].
        # Since convolution is commutative, conv(h,p) = conv(p,h).
        # So Term 1 = Term 2?
        # Wait.
        # Term 2: sum_k p[k] h[j-k].
        # Term 1: sum_k p[k] h[k-j].
        # Are these equal?
        # Only if p is symmetric or h is symmetric.
        # They are NOT generally equal.
        # However, the gradient of c_k = sum_i h_i h_{k-i} w.r.t h_j
        # is h_{k-j} + h_{j-k}.
        # So we need sum_k p_k (h_{k-j} + h_{j-k}).
        # This is (p*h)[j] + (p*h_rev)[j]?
        
        # Let's just compute it directly using FFT for speed.
        # grad = 2 * ifft(fft(p) * fft(h)) ?
        # No, that gives conv(p,h).
        # We need conv(p,h) + conv(p, h_reversed)?
        
        # Let's use the explicit vectorized sum for N=400. It's fast enough.
        # p has length 2N-1.
        # h has length N.
        
        # g[j] = sum_{k} p[k] h[k-j] + sum_{k} p[k] h[j-k]
        
        # Part 2: sum_k p[k] h[j-k] is conv(p, h)[j].
        # Part 1: sum_k p[k] h[k-j].
        # Let's compute Part 1.
        # h_idx = k-j.
        
        # We can use np.correlate?
        # np.correlate(p, h, 'full')
        
        # Let's just do it with matrix multiplication or loops if necessary, 
        # but N=400 is small.
        
        # Vectorized approach for Part 2:
        part2 = np.convolve(p, h, mode='same') # Length N? 
        # np.convolve(p, h) has length (2N-1) + N - 1 = 3N-2.
        # We want indices corresponding to h[j].
        # conv(p,h)[j] = sum_i p[i] h[j-i].
        # This matches Part 2 exactly.
        
        # Part 1: sum_k p[k] h[k-j].
        # This is np.convolve(h, p, mode='same')?
        # conv(h,p)[j] = sum_i h[i] p[j-i].
        # Let i = k-j. sum_{k-j} h[k-j] p[k].
        # This matches Part 1.
        
        # So grad_lse = (1/T) * (conv(h,p)[j] + conv(p,h)[j])
        
        # Note: conv(h,p) and conv(p,h) are the same array!
        # Convolution is commutative.
        # So grad_lse = (2/T) * conv(p,h)[j].
        
        # Let's verify this logic.
        # c_k = sum_i h_i h_{k-i}.
        # d c_k / d h_j = h_{k-j} + h_{j-k}.
        # Sum_k p_k d c_k / d h_j = sum_k p_k h_{k-j} + sum_k p_k h_{j-k}.
        # Term A: sum_k p_k h_{k-j}.
        # Term B: sum_k p_k h_{j-k}.
        # B is conv(p,h)[j].
        # A is conv(h,p)[j].
        # Since conv is commutative, A=B.
        # So grad = 2 * conv(p,h)[j].
        
        # Correct!
        
        grad_lse_vec = np.convolve(p, h, mode='full')
        # We need the part corresponding to h indices 0..N-1.
        # conv(p,h) length is len(p)+len(h)-1 = (2N-1)+N-1 = 3N-2.
        # The index j in h corresponds to which index in conv?
        # conv(p,h)[m] = sum_i p_i h_{m-i}.
        # We want the coefficient for h_j.
        # In the gradient formula, we summed over k (lag).
        # The result is a vector of length N.
        # The convolution result has length 3N-2.
        # The "center" of the convolution corresponds to the valid overlaps.
        # For h_j, the relevant terms are where p_k overlaps with h_j.
        # Actually, the vector `conv(p,h)` contains all cross-correlations.
        # We just need to slice it correctly.
        # The gradient with respect to h_j is the sum over all lags k of p_k * (contribution).
        # It turns out that the vector `2 * conv(p, h)` sliced appropriately gives the gradient.
        # Which slice?
        # If we treat h as a signal, the gradient is the cross-correlation of p with h.
        # The standard `np.convolve(p, h)` gives the convolution.
        # For the gradient of quadratic form, it is indeed related to convolution.
        # Let's assume the slice is [N-1 : 2*N-1] or similar.
        # Let's test with N=2.
        # h=[h0, h1]. p=[p0, p1, p2].
        # c0 = h0^2. c1 = 2h0h1. c2 = h1^2.
        # grad c0/dh0 = 2h0.
        # grad c1/dh0 = 2h1.
        # grad c2/dh0 = 0.
        # Total grad dh0 = 2 p0 h0 + 2 p1 h1.
        # conv(p,h) = [p0h0, p0h1+p1h0, p1h1+p2h0, p2h1].
        # We want 2(p0h0 + p1h1).
        # This is not directly a slice of conv(p,h).
        # Wait.
        # My derivation: grad_lse = (1/T) sum_k p_k (h_{k-j} + h_{j-k}).
        # For j=0:
        # k=0: p0 (h0 + h0) = 2 p0 h0.
        # k=1: p1 (h1 + h[-1]) = p1 h1. (h[-1]=0)
        # k=2: p2 (h2 + h[-2]) = 0.
        # Sum = 2 p0 h0 + p1 h1.
        # My previous formula "2 * conv" might be wrong if boundaries are handled differently.
        # Let's just compute the gradient explicitly using the sum formula for safety, 
        # since N=400 is small.
        
        grad_obj = np.zeros(N)
        # p is length 2N-1.
        # for j in range(N):
        #    term1 = 0
        #    term2 = 0
        #    for k in range(2*N-1):
        #       if 0 <= k-j < N:
        #           term1 += p[k] * h[k-j]
        #       if 0 <= j-k < N:
        #           term2 += p[k] * h[j-k]
        #    grad_obj[j] = (1/T) * (term1 + term2)
        
        # This is O(N^2) per iteration. 400^2 = 160,000.
        # 5000 iterations = 800M operations. Might be too slow in Python.
        # Let's use the FFT approximation or the "2*conv" with correct slicing.
        
        # Actually, the gradient of f(h) = sum_k p_k (h * h)_k
        # is 2 * (p * h) where * is convolution, BUT we must be careful with the "full" vs "valid" and the fact that p is the weight of the max.
        
        # Let's use the property that grad = 2 * ifft(fft(p) * fft(h)) ?
        # No.
        
        # Let's try a different smooth approximation that is easier to differentiate:
        # J = 2 * N * sum(c^2 / (2*sigma^2)) / (sum h)^2 ?
        # No, that minimizes the L2 norm of c, not the max.
        
        # Given the time, I will stick to the **Random Search** but with a **much better initialization** 
        # and **Adaptive Step Size** based on the gradient direction of the EXACT max.
        # The "Materially Different" strategy is: **Stochastic Gradient Descent on the Active Set**.
        # Identify the lag k* that achieves the max.
        # Compute gradient of c[k*] w.r.t h.
        # Update h in the negative gradient direction.
        # This is a subgradient method for the non-smooth max.
        
        k_max = np.argmax(c)
        # Gradient of c[k_max] w.r.t h[j]
        # c[k] = sum_i h_i h_{k-i}
        # grad h_j = h_{k-j} + h_{j-k}
        
        grad_c_max = np.zeros(N)
        k = k_max
        for j in range(N):
            val = 0.0
            if 0 <= k - j < N:
                val += h[k - j]
            if 0 <= j - k < N:
                val += h[j - k]
            grad_c_max[j] = val
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad_c_max)
        if grad_norm > 1e-9:
            grad_c_max /= grad_norm
            
        # Update
        h_new = h - lr * grad_c_max
        
        # Project to sum=1
        s_new = np.sum(h_new)
        if abs(s_new) < 1e-12:
            h_new = h
        else:
            h_new = h_new / s_new
            
        # Check improvement
        val_new = compute_value_exact(h_new)
        if val_new < val_best:
            val_best = val_new
            h_best = h_new.copy()
            h = h_new
            # Decay learning rate
            lr *= 0.99
        else:
            # If no improvement, try a smaller step or random jitter
            lr *= 0.9
            if lr < 1e-6:
                # Restart with random perturbation
                h = h_best + rng.normal(0, 0.01, N)
                h = h / np.sum(h)
                lr = 1e-3

    return h_best
