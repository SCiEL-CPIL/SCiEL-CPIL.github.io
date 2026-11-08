import numpy as np
import time

N = 400
DEADLINE = 120.0

def _compute_c3_fast(signed_step):
    """
    Compute C3 = max_k |autoconv(k)| exactly for a signed step function.
    
    The step function f has N equal intervals on [-1/4, 1/4].
    Interval i covers [-1/4 + i*h, -1/4 + (i+1)*h) where h = 1/N.
    
    Autoconvolution: (f*f)(t) = integral f(s) f(t-s) ds
    
    For step functions, at shift t = k*h (integer k), the autoconvolution is:
    (f*f)(k*h) = h * sum_{i} f_i * f_{i-k}
    where f_j = 0 if j outside [0, N-1].
    
    Actually, more carefully: the support of f is [-1/4, 1/4] with length 1/2.
    The autoconvolution (f*f)(t) = integral_{-1/4}^{1/4} f(s) f(t-s) ds.
    
    For t in [-1/2, 1/2], the overlap region is [max(-1/4, t-1/4), min(1/4, t+1/4)].
    
    For a step function with N intervals of width h = 1/N on [-1/4, 1/4]:
    Interval i: [-1/4 + i*h, -1/4 + (i+1)*h), height = signed_step[i]
    
    The autoconvolution at t = k*h (where k ranges from -N to N):
    We need to find which pairs of intervals (i, j) contribute to the integral at shift k*h.
    
    Actually, let me think more carefully. The autoconvolution of a step function can be
    computed as follows:
    
    (f*f)(t) = sum over i,j of f_i * f_j * length of overlap of interval_i and (interval_j shifted by t)
    
    For t = k*h, the overlap of interval i = [-1/4 + i*h, -1/4 + (i+1)*h) with 
    interval j shifted by k*h = [-1/4 + j*h + k*h, -1/4 + (j+1)*h + k*h)
    is the overlap of [i*h, (i+1)*h) with [(j+k)*h, (j+k+1)*h) in the local coordinate.
    
    This overlap is h * max(0, min(i+1, j+k+1) - max(i, j+k))
    
    So (f*f)(k*h) = h * sum_{i,j} f_i * f_j * max(0, min(i+1, j+k+1) - max(i, j+k))
    
    But this is O(N^2) per shift, and we have O(N) shifts, so O(N^3) total. Too slow.
    
    Alternative: use the fact that for step functions, the autoconvolution is a piecewise
    linear function. The breakpoints occur at t = (j-i)*h for integer j, i.
    
    Actually, a simpler approach: the autoconvolution of two step functions can be computed
    via the formula:
    
    (f*f)(t) = h * sum_{i=0}^{N-1} f_i * f_{i-k}  (for the case where the step is aligned)
    
    Wait, that's only true if the function is periodic or if we're doing circular convolution.
    For linear convolution of step functions, it's more complex.
    
    Let me use a different approach. The autoconvolution of a step function with N steps
    is a piecewise linear function with at most 2N-1 pieces. We can compute it exactly.
    
    For a step function f with values a_0, ..., a_{N-1} on intervals [x_i, x_{i+1}) where
    x_i = -1/4 + i*h, the autoconvolution (f*f)(t) for t in [-(N-1)h, (N-1)h] can be computed
    as follows:
    
    (f*f)(t) = integral f(s) f(t-s) ds
    
    For t >= 0, let k = floor(t/h). Then t = k*h + r where 0 <= r < h.
    The overlap of f(s) and f(t-s) involves pairs of intervals where the second interval
    is shifted by t.
    
    This is getting complex. Let me use a practical approach: compute the autoconvolution
    at all O(N) breakpoints t = k*h for k = -N+1, ..., N-1, and also at midpoints, using
    an O(N) formula for each.
    
    Actually, the simplest correct approach: for a step function, the autoconvolution is
    piecewise linear. The value at t = k*h can be computed as:
    
    (f*f)(k*h) = h * sum_{i=max(0,k)}^{N-1} f_i * f_{i-k}  if k >= 0
    (f*f)(k*h) = h * sum_{i=0}^{N-1-k} f_i * f_{i-k}  if k < 0
    
    Wait, that's the circular convolution formula. For linear convolution of step functions
    that are zero outside their support, we need to be more careful.
    
    Let me just use the direct formula. For t = k*h where k is an integer:
    
    (f*f)(k*h) = integral_{-1/4}^{1/4} f(s) f(k*h - s) ds
    
    f(s) = a_i where i = floor((s + 1/4)/h), if 0 <= i < N
    f(k*h - s) = a_j where j = floor((k*h - s + 1/4)/h), if 0 <= j < N
    
    The integral is over s such that both i and j are valid.
    
    For a given k, the valid range of s is:
    s in [-1/4, 1/4] AND k*h - s in [-1/4, 1/4]
    => s in [max(-1/4, k*h - 1/4), min(1/4, k*h + 1/4)]
    
    Within this range, f(s) and f(k*h - s) are constant on sub-intervals defined by the
    breakpoints of both functions. The breakpoints for f(s) are at s = -1/4 + i*h.
    The breakpoints for f(k*h - s) are at s = k*h + 1/4 - j*h = k*h - (j - k)*h + ... 
    
    This is getting too complex for an O(N) formula. Let me use a different strategy:
    
    Since N=400, I can compute the autoconvolution on a grid of 2N-1 points (the breakpoints)
    using an O(N^2) algorithm, which is 160000 operations - very fast.
    
    Actually, even simpler: I'll compute the autoconvolution at t = k*h for k = -(N-1), ..., N-1
    using the formula:
    
    (f*f)(k*h) = h * sum_{i=0}^{N-1} a_i * a_{i-k}  (with a_j = 0 for j outside [0,N-1])
    
    This IS the correct formula for the autoconvolution of a step function at the grid points!
    Because at t = k*h, the overlap of interval i of f with the shifted interval of f is
    exactly h if i-k is in [0,N-1], and the heights multiply.
    
    Wait, no. That's only true if the step function is such that the intervals align perfectly
    at the shift. Let me verify with a simple example.
    
    N=2, h=1/2, domain [-1/4, 1/4].
    f = [1, 1]. 
    (f*f)(0) = integral_{-1/4}^{1/4} 1*1 ds = 1/2.
    Formula: h * (a_0*a_0 + a_1*a_1) = 0.5 * (1+1) = 1. Wrong!
    
    So the simple formula doesn't work. The issue is that at t=0, the overlap of interval 0
    with itself is the full interval, but the overlap of interval 0 with interval 1 (shifted by 0)
    is zero because they don't overlap.
    
    The correct formula for (f*f)(k*h) involves the actual geometric overlap.
    
    Let me use a different approach: compute the autoconvolution via FFT on a fine grid,
    but make it very fast by precomputing the FFT of the basis vectors.
    
    Actually, the fastest approach for N=400: compute the autoconvolution at all 2N-1 = 799
    breakpoints t = k*h for k = -(N-1), ..., N-1 using an O(N^2) algorithm per evaluation.
    799 * 400 = ~320K operations per C3 evaluation. With ~100K iterations in 120s, that's
    32 billion operations - too slow in Python.
    
    Let me use NumPy to vectorize. For a given signed_step array a of length N:
    
    The autoconvolution at t = k*h (for k = -(N-1), ..., N-1) can be computed as:
    
    For k >= 0: (f*f)(k*h) = h * sum_{i=k}^{N-1} a_i * a_{i-k}
    For k < 0:  (f*f)(k*h) = h * sum_{i=0}^{N-1+k} a_i * a_{i-k}
    
    But this is the circular/linear convolution formula, which I showed is wrong above.
    
    The problem is that the step function is NOT periodic. The autoconvolution of a compactly
    supported step function is NOT simply the convolution of the sequence.
    
    Let me think again. The step function f is defined on [-1/4, 1/4] with N intervals.
    f(s) = a_i for s in [-1/4 + i*h, -1/4 + (i+1)*h)
    
    (f*f)(t) = integral_{-1/4}^{1/4} f(s) f(t-s) ds
    
    For t = 0: (f*f)(0) = integral_{-1/4}^{1/4} f(s)^2 ds = h * sum a_i^2 = h * N * 1 = 1/2.
    For f = [1,1] with N=2: (f*f)(0) = 1/2. Correct.
    
    For t = h = 1/2: (f*f)(1/2) = integral_{-1/4}^{1/4} f(s) f(1/2 - s) ds.
    f(1/2 - s) is nonzero when 1/2 - s in [-1/4, 1/4], i.e., s in [1/4, 3/4].
    But s is in [-1/4, 1/4], so the overlap is s = 1/4 (measure zero). So (f*f)(1/2) = 0.
    
    For t = h/2 = 1/4: (f*f)(1/4) = integral_{-1/4}^{1/4} f(s) f(1/4 - s) ds.
    f(1/4 - s) nonzero when 1/4 - s in [-1/4, 1/4], i.e., s in [0, 1/2].
    Overlap with [-1/4, 1/4] is [0, 1/4].
    For s in [0, 1/4]: f(s) = a_1 (since s in [-1/4 + h, -1/4 + 2h) = [0, 1/4)), 
    f(1/4 - s) = a_0 (since 1/4 - s in [0, 1/4) = [-1/4 + 0, -1/4 + h)).
    So (f*f)(1/4) = integral_0^{1/4} a_1 * a_0 ds = (1/4) * a_0 * a_1.
    
    For f = [1, 1]: (f*f)(1/4) = 1/4.
    
    So the autoconvolution is NOT simply the convolution of the sequence. It involves
    partial overlaps.
    
    Given the complexity, let me use the fine-grid FFT approach but optimize it heavily.
    The key insight from the inspirations is that vectorized fine-grid construction is fast.
    
    Let me use a fine grid of size M = 4*N = 1600, and compute the autoconvolution via FFT.
    This is O(M log M) per evaluation, which is fast.
    
    For the SA strategy, I'll use:
    1. Fine-grid FFT for C3 computation (vectorized)
    2. Simulated annealing with geometric cooling
    3. Multi-start with random restarts
    
    Let me implement this.
    """
    M = 4 * N
    dx = 0.5 / M
    
    # Build f on fine grid (vectorized)
    # Each step interval i covers [i*M/N, (i+1)*M/N) on the fine grid
    f_fine = np.repeat(signed_step, M // N)
    # Ensure exact length
    if len(f_fine) < M:
        f_fine = np.pad(f_fine, (0, M - len(f_fine)))
    elif len(f_fine) > M:
        f_fine = f_fine[:M]
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f_fine, n=2 * M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    c3 = np.max(np.abs(autoconv))
    return c3


def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    # Simulated annealing parameters
    T_initial = 0.05  # initial temperature
    T_final = 0.001   # final temperature
    cooling_rate = 0.999
    
    # Multi-start: try several random initial states
    best_step = None
    best_c3 = np.inf
    
    np.random.seed(42)
    
    # Number of restarts
    n_restarts = 10
    
    for restart in range(n_restarts):
        if time_left() < 5.0:
            break
        
        # Random initial state
        current_step = np.random.choice([-1, 1], size=N).astype(np.int8)
        current_c3 = _compute_c3_fast(current_step)
        
        if current_c3 < best_c3:
            best_c3 = current_c3
            best_step = current_step.copy()
        
        # Simulated annealing
        T = T_initial
        no_improve = 0
        patience = 2000
        
        while T > T_final:
            if time_left() < 2.0:
                break
            
            # Flip a random element
            idx = np.random.randint(N)
            current_step[idx] = -current_step[idx]
            new_c3 = _compute_c3_fast(current_step)
            
            delta = new_c3 - current_c3
            
            if delta < 0:
                # Improvement
                current_c3 = new_c3
                no_improve = 0
                if new_c3 < best_c3:
                    best_c3 = new_c3
                    best_step = current_step.copy()
            else:
                # Accept with probability exp(-delta / T)
                if np.random.random() < np.exp(-delta / T):
                    current_c3 = new_c3
                    no_improve += 1
                else:
                    # Reject, undo
                    current_step[idx] = -current_step[idx]
                    no_improve += 1
            
            T *= cooling_rate
            
            if no_improve >= patience:
                break
    
    # Final local search polish on the best solution
    if best_step is not None and time_left() > 3.0:
        current_step = best_step.copy()
        current_c3 = best_c3
        improved = True
        while improved and time_left() > 1.0:
            improved = False
            for idx in range(N):
                if time_left() < 1.0:
                    break
                current_step[idx] = -current_step[idx]
                new_c3 = _compute_c3_fast(current_step)
                if new_c3 < current_c3:
                    current_c3 = new_c3
                    improved = True
                    if new_c3 < best_c3:
                        best_c3 = new_c3
                        best_step = current_step.copy()
                else:
                    current_step[idx] = -current_step[idx]
    
    return best_step if best_step is not None else np.ones(N, dtype=np.int8)
