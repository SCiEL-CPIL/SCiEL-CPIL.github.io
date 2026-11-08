"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_c3(h):
    """
    Compute C3 = max |autoconvolution| for a signed step function.
    h: array of N values in {-1, 1}
    Domain: [-1/4, 1/4], N intervals, each of width h = 1/(2N)
    The autoconvolution is computed via FFT.
    """
    n = len(h)
    # Linear convolution size is 2n - 1. We need a power of 2 >= 2n-1 for FFT.
    # Since n=400, 2n-1 = 799. Next power of 2 is 1024.
    size = 1 << (2 * n - 1).bit_length()
    
    # FFT of the signal
    H = np.fft.rfft(h, size)
    
    # Autoconvolution in frequency domain is H * conj(H)
    # Inverse FFT gives the circular convolution. Since size >= 2n-1, it is linear.
    ac = np.fft.irfft(H * np.conj(H), size)
    
    # The linear convolution is the first 2n-1 elements
    ac_linear = ac[:2*n-1]
    
    # The width of each interval is 1/(2N)
    # The autoconvolution of step functions involves scaling by the interval width squared?
    # Let's check the definition.
    # f(t) = sum h_i * 1_{[x_i, x_{i+1})}(t)
    # (f*f)(t) = integral f(t-s) f(s) ds
    # If h_i are constants, the convolution is a piecewise constant function? No, it's a piecewise linear function (tent functions convolved).
    # Wait, the task says "signed step function".
    # Usually, C3 bound refers to the L-infinity norm of the autoconvolution.
    # Let's look at the inspirations.
    # Inspiration 1: _compute_c3.
    # The prompt says: "Paper and published result notebook place the absolute value differently; execution must explicitly select a metric."
    # "score is the upper bound on C3"
    # "evaluator measures the raw value (lower is better)"
    
    # Let's assume the standard definition:
    # f(t) = h_i on interval i.
    # (f*f)(t) = \int f(t-s)f(s) ds.
    # If we sample the autoconvolution at the centers of the intervals or just take the max of the continuous function?
    # For step functions, the autoconvolution is a continuous piecewise linear function.
    # Its maximum occurs at one of the breakpoints (interval boundaries) or centers?
    # Actually, for a convolution of two step functions, the result is a "trapezoidal" or "triangular" shape sum.
    # The maximum of the continuous autoconvolution is what we want to minimize.
    
    # How to compute the max of the continuous autoconvolution from the discrete signal?
    # The autoconvolution of a step function can be computed by summing the products of heights multiplied by the length of the overlap.
    # Let the intervals be I_k = [k/N, (k+1)/N] scaled to [-1/4, 1/4].
    # Actually, the domain is [-1/4, 1/4], length 1/2. N=400.
    # Interval width w = (1/2) / 400 = 1/800.
    
    # The autoconvolution at shift tau is:
    # (f*f)(tau) = \sum_{i,j} h_i h_j * Length(I_i \cap (I_j + tau))
    
    # This is equivalent to the convolution of the discrete sequence with itself, but scaled and interpolated.
    # Specifically, if we define the discrete autoconvolution c_k = \sum_{i=0}^{N-1} h_i h_{i+k} (for k>=0, and symmetric for k<0),
    # The continuous autoconvolution is a piecewise linear function that passes through points related to these sums.
    # Actually, it's simpler:
    # The autoconvolution of a step function with height h_i on interval of width w is:
    # F(tau) = \sum_{i,j} h_i h_j \int 1_{I_i}(t) 1_{I_j}(t-tau) dt
    # This is equal to w * (sum of h_i h_j for overlapping intervals) ... wait.
    # Let's use the property that the convolution of two step functions is a piecewise linear function whose slope changes at the boundaries of the intervals.
    # The values at the midpoints of the intervals?
    
    # Let's look at the "seed" program's `_autoconv_l2`. It computed L2 norm.
    # The task asks for "upper bound on C3". C3 is typically the L-infinity norm of the autoconvolution in the context of these inequalities (e.g., related to the C3 coefficient in the expansion of the autocorrelation or similar).
    
    # Let's assume C3 = max_t |(f*f)(t)|.
    
    # To compute max_t |(f*f)(t)| efficiently:
    # The function (f*f)(t) is continuous and piecewise linear.
    # The maxima must occur at the points where the slope changes, i.e., at the boundaries of the intervals of f or f shifted.
    # The boundaries are at k*w for k in Z.
    # So we just need to evaluate (f*f)(t) at t = k*w for k = -(N-1)...N-1.
    
    # How to evaluate (f*f)(k*w)?
    # (f*f)(k*w) = \sum_{i=0}^{N-1} h_i \int_{I_i} f(t-k*w) dt
    # Let j be such that t-k*w is in I_j.
    # This is effectively the discrete autoconvolution scaled by w?
    # Let's check:
    # If f is constant h_i on I_i = [i*w, (i+1)*w].
    # (f*f)(k*w) = \int f(t-k*w) f(t) dt.
    # Let u = t-k*w. dt = du.
    # = \int f(u) f(u+k*w) du.
    # The function f(u) is piecewise constant.
    # The integral is the sum over intervals of u where f(u) is constant, of h_i * \int_{I_i} f(u+k*w) du.
    # \int_{i*w}^{(i+1)*w} f(u+k*w) du.
    # Let v = u+k*w. dv=du. Limits: (i+k)w to (i+k+1)w.
    # = \int_{(i+k)w}^{(i+k+1)w} f(v) dv.
    # If the interval [(i+k)w, (i+k+1)w] falls entirely within an interval I_j of f, the integral is h_j * w.
    # If it straddles, it's a weighted sum.
    
    # Wait, for step functions, the autoconvolution at integer multiples of the step width w is actually:
    # c_k = w * \sum_{i} h_i h_{i-k} ?
    # Let's test with N=2, h=[1, 1], w=1.
    # f(t) = 1 on [0,1], 1 on [1,2]. f(t)=1 on [0,2].
    # (f*f)(t) is triangle from 0 to 2, peak 2 at t=1.
    # Discrete sum: \sum h_i h_{i-k}.
    # k=0: 1*1 + 1*1 = 2. w*2 = 2. Correct.
    # k=1: 1*1 (i=1, i-k=0) = 1. w*1 = 1.
    # Continuous (f*f)(1) = 2. (f*f)(2) = 0?
    # Wait, the domain is [-1/4, 1/4].
    # Let's re-read the prompt carefully.
    # "Choose a signed step function with 400 equal intervals on [-1/4,1/4]"
    # "score is the upper bound on C3"
    
    # If the function is supported on [-1/4, 1/4], the autoconvolution is supported on [-1/2, 1/2].
    # The maximum of the autoconvolution is the C3 value.
    
    # The values of the continuous autoconvolution at the points t_k = k * (1/(2N)) are NOT simply w * discrete_conv[k].
    # Because the step function is not a train of deltas.
    
    # However, there is a known relationship.
    # For a step function, the autoconvolution is a piecewise linear function.
    # The peaks occur at the centers of the symmetry?
    # Actually, the maximum of the autoconvolution of a step function is attained at one of the points t = m * w, where w is the step width.
    # Is this true?
    # Let's consider f(t) = 1 on [0,1], 0 elsewhere. w=1.
    # (f*f)(t) is 0 for t<0, t for 0<t<1, 2-t for 1<t<2, 0 for t>2.
    # Max is 1 at t=1.
    # Discrete conv of [1] with [1] is [1]. w*1 = 1.
    
    # Consider f(t) = 1 on [0,1], -1 on [1,2]. w=1.
    # (f*f)(t):
    # t in [0,1]: overlap of [0,1] and [t, t+1] shifted?
    # Let's use the formula:
    # (f*f)(t) = \int f(t-s)f(s) ds.
    # For t in [0,1]:
    # s in [0, t]: f(s)=1, f(t-s)=1 (since t-s in [0,t] subset [0,1]). Integral = t.
    # s in [t, 1]: f(s)=1, f(t-s) = f(negative) = 0? No, support is [0,2].
    # f(s) is 1 on [0,1], -1 on [1,2].
    # If t in [0,1]:
    # s in [0,t]: f(s)=1, f(t-s)=1. Int = t.
    # s in [t,1]: f(s)=1, f(t-s). t-s is in [0, t-t]=[0,0]? No.
    # t-s is in [t-1, 0]. Since support starts at 0, f(t-s) is 0 if t-s < 0.
    # So only s in [0,t] contributes?
    # Wait, f(t-s) is non-zero if t-s in [0,2].
    # If s in [0,1], t-s in [t-1, t].
    # If t<1, t-1 < 0. So part of the interval is outside support.
    # This is getting complicated to derive from scratch.
    
    # Let's use a robust numerical approach.
    # We can compute the autoconvolution at the critical points.
    # The critical points for a step function with N steps are the boundaries of the steps and their shifts.
    # There are O(N) such points.
    # We can compute the value at each point using the discrete convolution of the heights?
    
    # Actually, the value of the autoconvolution at the midpoint of the k-th interval of the domain of the convolution (which spans 2N-1 intervals) is:
    # C_k = w * \sum_{i} h_i h_{i+k} ?
    # No, that's for the case where the function is a sum of deltas.
    
    # Let's look at the "inspiration" code snippets if they had details.
    # They define `_compute_c3`.
    # A common efficient way to compute the max autoconvolution of a step function:
    # 1. Compute the discrete autoconvolution of the heights: `c = np.convolve(h, h, mode='full')`.
    # 2. The continuous autoconvolution is a piecewise linear function.
    # 3. The values at the "centers" of the convolution intervals are `w * c`?
    # 4. The values at the "edges" are `w * (c[k] + c[k-1])/2`?
    
    # Let's verify with the single step [1, 0].
    # h = [1, 0]. w=1.
    # c = [1, 0, 0].
    # Continuous f*f is triangle on [0,2].
    # Peaks at t=1. Value 1.
    # c[1] = 0. w*c[1] = 0. Wrong.
    # The peak is at the boundary between the 1st and 2nd interval of the convolution?
    # The convolution of two boxes of width 1 is a triangle of base 2, height 1.
    # The triangle peaks at t=1.
    # The discrete convolution of [1] and [1] is [1].
    # The mapping is: The discrete convolution index k corresponds to the shift k*w.
    # The value of the continuous autoconvolution at shift k*w is NOT simply w*c[k].
    
    # Correct Formula:
    # For a step function f(t) = h_i on [i*w, (i+1)*w],
    # The autoconvolution (f*f)(t) at t = k*w is:
    # (f*f)(k*w) = w * \sum_{i=0}^{N-1} h_i h_{i-k}  (if we define h_j=0 outside 0..N-1)
    # Wait, let's re-evaluate [1,0] with w=1.
    # h=[1,0].
    # k=0: \sum h_i h_i = 1*1 + 0*0 = 1. w*1 = 1.
    # (f*f)(0) = \int f(s)f(s) ds = 1. Correct.
    # k=1: \sum h_i h_{i-1} = h_1 h_0 = 0*1 = 0.
    # (f*f)(1) = 1 (from triangle peak).
    # So the formula `w * discrete_conv[k]` gives 0 at k=1, but the true value is 1.
    # Why? Because the peak of the triangle is at t=1, which is the boundary between the interval [0,1] and [1,2] of the convolution support.
    # The discrete convolution `c[k]` approximates the integral over the interval [k*w, (k+1)*w]?
    # No, `c[k] = \sum h_i h_{i-k}`.
    # The continuous autoconvolution is the convolution of the piecewise constant functions.
    # It turns out that the maximum of the continuous autoconvolution is attained at the midpoints of the intervals defined by the grid points k*w?
    # No, for [1,0], the peak is at t=1, which is a grid point.
    
    # Let's check the value at t = (k+0.5)*w.
    # For [1,0], t=0.5. (f*f)(0.5) = 0.5.
    # c[0]=1, c[1]=0.
    # Maybe the values are interpolated?
    
    # Actually, there is a simpler way.
    # The autoconvolution of a step function is a "spline" of order 1 (piecewise linear).
    # The coefficients of this spline are related to the discrete convolution.
    # Specifically, if we define the discrete autoconvolution `c_k = \sum_{i} h_i h_{i-k}`,
    # then the continuous autoconvolution `F(t)` satisfies:
    # `F(k*w) = w * c_k`? No, we saw that fails for k=1 in [1,0].
    # Let's re-calculate [1,0] carefully.
    # f(t) = 1 on [0,1].
    # (f*f)(t) = \int_0^1 f(t-s) ds.
    # If t=1: \int_0^1 f(1-s) ds. 1-s is in [0,1]. f(1-s)=1. Integral = 1.
    # So F(1) = 1.
    # Discrete conv c_1 = \sum h_i h_{i-1}. h_0=1, h_1=0.
    # i=1: h_1 h_0 = 0*1 = 0.
    # So c_1 = 0.
    # So F(1) != w * c_1.
    
    # What is the relationship?
    # F(t) is linear between k*w and (k+1)*w.
    # The values at the endpoints are needed.
    # It turns out that F(k*w) = w * (c_k + c_{k-1}) / 2 ?
    # Let's test.
    # k=1: w * (c_1 + c_0) / 2 = 1 * (0 + 1) / 2 = 0.5.
    # But F(1) = 1.
    
    # Let's try F(k*w) = w * c_k ? No.
    
    # Let's look at the integral definition again.
    # F(t) = \int f(t-s) f(s) ds.
    # This is the standard convolution.
    # If we sample f at the midpoints, we get a discrete convolution approximation.
    # But f is constant on intervals.
    
    # Correct approach:
    # The autoconvolution of a step function can be computed exactly.
    # The function F(t) is piecewise linear.
    # The break points are at t = m*w for all integers m.
    # The value of F(t) at t = m*w is:
    # F(m*w) = \sum_{i} h_i \int_{I_i} f(t-m*w) dt ?
    # No, F(t) = \int f(t-s) f(s) ds.
    # Let's use the property that the derivative of F(t) is f(t) * f'(t) ? No.
    # F'(t) = \int f'(t-s) f(s) ds.
    # Since f is a step function, f' is a sum of deltas.
    # F'(t) = \sum_{j} h_j \delta(t - (j+0.5)w) * \int ... ?
    # Actually, the slope of F(t) changes at t = (i+j+1)w ?
    
    # Let's use a different property.
    # The autoconvolution of a step function with heights h and width w is:
    # F(t) = w * (h * h)(t/w) where * is discrete convolution, BUT interpolated linearly?
    # No.
    
    # Let's just compute F(t) at the critical points directly using the definition, but efficiently.
    # Critical points are t_k = k * w for k = -(N-1) ... N-1.
    # F(t_k) = \int f(t_k - s) f(s) ds.
    # Since f(s) is constant on I_j = [j*w, (j+1)*w],
    # F(t_k) = \sum_{j=0}^{N-1} \int_{j*w}^{(j+1)*w} f(t_k - s) h_j ds.
    # Let u = t_k - s. du = -ds.
    # When s = j*w, u = t_k - j*w = (k-j)*w.
    # When s = (j+1)*w, u = t_k - (j+1)*w = (k-j-1)*w.
    # Integral = h_j \int_{(k-j-1)w}^{(k-j)*w} f(u) du.
    # The interval of integration is [(k-j-1)w, (k-j)w], which is exactly the interval I_{k-j-1} (if indices are valid).
    # So \int_{I_{k-j-1}} f(u) du = h_{k-j-1} * w.
    # Therefore, F(t_k) = \sum_{j=0}^{N-1} h_j * h_{k-j-1} * w.
    # Let m = k-j-1 => j = k-m-1.
    # F(t_k) = w * \sum_{m} h_{k-m-1} h_m.
    # This is w * (h * h)_{k-1} (discrete convolution index k-1).
    
    # Let's test with [1,0], w=1.
    # h=[1,0].
    # k=1 (t=1).
    # F(1) = 1 * \sum_{m} h_{1-m-1} h_m = \sum h_{-m} h_m.
    # m=0: h_0 h_0 = 1*1 = 1.
    # m=1: h_{-1} h_1 = 0.
    # Sum = 1. F(1) = 1. Correct!
    
    # So the value of the continuous autoconvolution at the grid points t_k = k*w is:
    # F(t_k) = w * c_{k-1}
    # where c is the discrete autoconvolution of h.
    
    # What about the values between the grid points?
    # F(t) is linear between t_k and t_{k+1}.
    # So the maximum of F(t) on [t_k, t_{k+1}] is max(F(t_k), F(t_{k+1})).
    # Therefore, the global maximum of |F(t)| is the maximum of |F(t_k)| over all k.
    
    # So, C3 = max_k |w * c_{k-1}| = w * max_k |c_{k-1}|.
    # Since w is constant, we just need to minimize max_k |c_k| where c is the discrete autoconvolution.
    
    n = len(h)
    # Compute discrete autoconvolution
    # c[k] = sum_i h[i] h[i-k]
    # Using FFT is efficient.
    size = 1 << (2 * n - 1).bit_length()
    H = np.fft.rfft(h, size)
    C = np.fft.irfft(H * np.conj(H), size)
    
    # C contains the circular convolution. Since size >= 2n-1, the first 2n-1 elements are the linear convolution.
    # The linear convolution c_lin has length 2n-1.
    # c_lin[k] corresponds to the sum for shift k.
    # Our formula derived F(t_k) = w * c_{k-1}.
    # The indices of c_lin are 0 to 2n-2.
    # The range of k for t_k is such that the convolution is non-zero.
    # The max will be over all c_lin.
    
    c_lin = C[:2*n-1]
    max_val = np.max(np.abs(c_lin))
    
    # Scale by w
    w = 1.0 / (2.0 * N)
    return float(max_val * w)

def _local_search_block(h, block_size, deadline):
    """
    Local search by flipping blocks of signs.
    """
    best_h = h.copy()
    best_val = _compute_c3(best_h)
    n = len(best_h)
    
    # Precompute block indices
    blocks = []
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        blocks.append((start, end))
    
    # Shuffle blocks for randomization
    np.random.shuffle(blocks)
    
    while time.time() < deadline:
        improved = False
        for start, end in blocks:
            if time.time() > deadline:
                break
            
            # Flip the block
            best_h[start:end] *= -1
            val = _compute_c3(best_h)
            
            if val < best_val:
                best_val = val
                improved = True
            else:
                # Revert
                best_h[start:end] *= -1
        
        if not improved:
            # Try smaller blocks or break?
            # Let's try to reduce block size if stuck
            if block_size > 1:
                block_size = max(1, block_size // 2)
                # Re-shuffle
                blocks = []
                for start in range(0, n, block_size):
                    end = min(start + block_size, n)
                    blocks.append((start, end))
                np.random.shuffle(blocks)
            else:
                break
                
    return best_h, best_val

def construct():
    deadline = time.time() + 140  # Safety margin
    best_h = None
    best_val = float('inf')
    rng = np.random.default_rng(42)
    
    # Start with a random sign pattern
    # Or maybe start with alternating signs?
    # Alternating signs often give low autoconvolution.
    h_init = np.ones(N)
    h_init[1::2] = -1
    
    # Try a few different initializations
    inits = [
        h_init,
        rng.choice([-1.0, 1.0], size=N),
        np.sin(np.linspace(0, 10*np.pi, N)).astype(float) * np.sign(np.sin(np.linspace(0, 10*np.pi, N))) # No, just signs
    ]
    
    # Better inits:
    # 1. Alternating
    # 2. Random
    # 3. Blocks of 1, -1, 1, -1 ...
    
    candidates = []
    candidates.append(h_init)
    candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # Block pattern: 1111 -1111 1111 ...
    h_block = np.ones(N)
    for i in range(0, N, 4):
        h_block[i:i+4] = 1 if (i//4) % 2 == 0 else -1
    candidates.append(h_block)
    
    for h in candidates:
        if time.time() > deadline:
            break
        h_opt, val = _local_search_block(h.copy(), block_size=16, deadline=deadline)
        if val < best_val:
            best_val = val
            best_h = h_opt.copy()
            
    if best_h is None:
        return np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
