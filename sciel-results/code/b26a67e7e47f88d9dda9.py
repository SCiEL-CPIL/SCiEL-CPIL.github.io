"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 600
# Interval length
dx = (1/4 - (-1/4)) / N  # = 0.5 / 600 = 1/1200

# Grid points for the autoconvolution.
# The step function is defined on [-1/4, 1/4].
# We approximate the integral by summing f[i] * f[j] * dx^2 where the midpoint distance corresponds to the shift.
# Actually, for a step function on intervals [x_i, x_{i+1}], the autoconvolution at shift y is:
# (f * f)(y) = ∫∫ f(x) f(x-y) dx dy ... no, (f*f)(y) = ∫ f(x) f(y-x) dx.
# Discretized:
# Let f_i be the height on interval i.
# (f*f)(y_k) ≈ Σ_i Σ_j f_i f_j dx^2 * δ(y_k - (x_j - x_i))
# Where x_i are the midpoints? Or boundaries?
# Usually, if we treat f as piecewise constant, the autoconvolution is a piecewise quadratic function.
# However, the problem asks for a "normalized maximum".
# The normalization is likely L1 or L2.
# "Choose a nonnegative step function... to minimize the normalized maximum of its autoconvolution".
# This usually refers to minimizing ||f*f||_∞ / ||f||_p^2 or similar.
# Given the context of "upper bound on C1", this is related to the first autocorrelation inequality.
# The standard inequality is ||f*f||_∞ ≤ C ||f||_1^2.
# So we want to minimize max(f*f) / (∫f)^2.
# Since we can scale f, the ratio is scale-invariant. We can fix ∫f = 1.
# Then we minimize max(f*f).

# Let's define the grid.
# Intervals: i=0..N-1.
# x_i = -1/4 + i*dx. Midpoint m_i = x_i + dx/2.
# The autoconvolution value at a shift y is approximately:
# (f*f)(y) ≈ dx^2 * Σ_i f_i * f_j where m_j - m_i ≈ y.
# Actually, the convolution of two step functions is a quadratic spline.
# But we can just evaluate it on a fine grid of shifts.
# The shift y ranges from -1/2 to 1/2.
# We can compute the convolution using FFT for speed, or direct summation.
# N=600 is small enough for direct O(N^2) or O(N log N) with FFT.
# Let's use FFT to compute the autoconvolution on a grid of shifts.

# To be precise about the "normalized maximum":
# If we fix sum(f_i * dx) = 1, then we minimize max_y (f*f)(y).
# Note: (f*f)(0) = ∫ f^2.
# For a box function of width W, (f*f)(0) = 1/W (if normalized to 1).
# Max is at 0.
# For a hollow structure, the peak at 0 might be reduced if the mass is spread out?
# No, (f*f)(0) = ∫ f^2. By Cauchy-Schwarz, ∫f^2 ≥ (∫f)^2 / Support = 1/W.
# So the minimum possible value of (f*f)(0) is 1/W_max = 1/(1/2) = 2.
# Wait, the support is [-1/4, 1/4], so max width is 1/2.
# So min ||f||_2^2 is 2.
# Thus max(f*f) is at least 2.
# The uniform function has f_i = 1/(N*dx) = 2.
# (f*f)(0) = Σ (2)^2 dx^2 = N * 4 * (1/1200)^2 = 600 * 4 / 1440000 = 2400 / 1440000 = 1/600?
# Let's re-calculate.
# ∫f = 1. f = 1 / (1/2) = 2.
# ∫f^2 = ∫ 4 dx = 4 * 0.5 = 2.
# So (f*f)(0) = 2.
# My discrete calculation:
# f_i = 2.
# Sum f_i^2 dx^2 = N * 4 * dx^2 = 600 * 4 * (1/1200)^2 = 2400 / 1440000 = 2/1200 = 1/600.
# This is wrong. The approximation (f*f)(y) ≈ dx^2 Σ f_i f_j is for the integral ∫∫ f(x)f(y-x) dx dy?
# No. (f*f)(y) = ∫ f(x) f(y-x) dx.
# Discretization: ∫ g(x) dx ≈ Σ g(x_i) dx.
# So (f*f)(y_k) ≈ Σ_i f(m_i) f(y_k - m_i) dx.
# If we interpolate f(y_k - m_i) as the value of the interval containing y_k - m_i, then:
# (f*f)(y_k) ≈ dx Σ_i f_i f_j where m_j ≈ y_k - m_i.
# So the factor is dx, not dx^2.
# Let's check units. f has units 1/length. (f*f) has units 1/length.
# dx Σ f_i f_j has units (1/L) * (1/L * 1/L) * L = 1/L^2? No.
# f is height. ∫f = 1 (dimensionless). So f has units 1/L.
# (f*f) = ∫ f f dx. Units (1/L)^2 * L = 1/L.
# Discrete: dx * f_i * f_j. Units L * (1/L)^2 = 1/L. Correct.
# So (f*f)(y_k) ≈ dx * Σ_{i,j: m_j - m_i ≈ y_k} f_i f_j.

# Let's compute the convolution matrix.
# Shifts y_k. Range [-0.5, 0.5].
# Let's use a grid for y of size 2N or 4N to be safe.
# Actually, the max will likely be at y=0 or near it.
# We can just compute the full convolution.

def compute_convolution(f):
    """
    Compute the autoconvolution of the step function f on the grid.
    Returns an array of values on a grid of shifts.
    """
    # f is an array of length N.
    # We want (f*f)(y) for y in [-0.5, 0.5].
    # Using FFT:
    # Zero-pad f to length 2N (or more) to avoid circular convolution.
    # Length of convolution result is 2N-1.
    # The shifts corresponding to the convolution indices k=0..2N-2 are:
    # y_k = (k - (N-1)) * dx? 
    # Let's map indices to shifts.
    # m_i = -0.25 + (i + 0.5)dx.
    # m_j - m_i = (j - i)dx.
    # So the shifts are integer multiples of dx.
    # The possible shifts are - (N-1)dx to (N-1)dx.
    # Total 2N-1 shifts.
    
    f_pad = np.zeros(2 * N)
    f_pad[:N] = f
    
    # FFT
    F = np.fft.fft(f_pad, n=4*N) # Pad more to be safe, or just 2N is enough for linear conv of size N+N-1
    conv = np.fft.ifft(F * F).real
    # The linear convolution is in conv[0:N] and conv[2N-1:N-1:-1]?
    # Standard: ifft(fft(a, n)*fft(b, n)) gives circular convolution.
    # If n >= len(a)+len(b)-1, it's linear.
    # Here a=b=f_pad (length 2N, but only first N non-zero).
    # Effective length N. Convolution length 2N-1.
    # So we need n >= 2N-1. 4N is safe.
    
    # The linear convolution of f with f:
    # c_k = sum_i f_i f_{k-i}
    # This corresponds to the sum over pairs (i,j) such that j-i = k?
    # Wait, convolution (f*f)(t) = sum_i f_i f_{t-i}.
    # If we index f by 0..N-1.
    # The shift t in "index units" is j - i.
    # The value at shift t is sum_{i,j: j-i=t} f_i f_j.
    # This is exactly the linear convolution coefficient for lag t.
    
    # Let's extract the lags.
    # The FFT result conv[k] corresponds to lag k (for k < N) and lag k-N (for k >= N)?
    # No, ifft gives c_k = sum_i a_i b_{k-i}.
    # For a=b=f (padded to 2N), c_k is the sum for lag k.
    # Lags range from -(N-1) to N-1.
    # In the FFT output of length 4N:
    # Indices 0 to N-1 correspond to lags 0 to N-1.
    # Indices 4N-(N-1) to 4N-1 correspond to lags -(N-1) to -1.
    
    lags = np.arange(-(N-1), N)
    # We need to map these lags to the correct FFT bins.
    # Bin 0 -> lag 0
    # Bin 1 -> lag 1
    # ...
    # Bin N-1 -> lag N-1
    # Bin N -> lag ? (Wrap around)
    # Actually, simpler:
    # Compute linear convolution directly using np.convolve?
    # np.convolve(f, f, 'full')
    # This is O(N^2) for small N, but N=600 is 360,000 ops. Very fast.
    
    c = np.convolve(f, f, 'full')
    # c has length 2N-1.
    # c[k] = sum_{i=0}^{k} f[i] f[k-i] ?
    # No, np.convolve definition:
    # c[k] = sum_i a[i] b[k-i]
    # For a=b=f.
    # c[0] = f[0]f[0] (lag 0? No, this is sum of products where indices sum to 0? No.)
    # Let's check numpy docs.
    # "convolve(a, v, mode='full') ... c_k = sum_i a_i v_{k-i}"
    # If a=v=f.
    # c_0 = a_0 v_0 = f_0 f_0.
    # c_1 = a_0 v_1 + a_1 v_0 = f_0 f_1 + f_1 f_0 = 2 f_0 f_1.
    # This corresponds to the shift where the overlap is...
    # Wait, the autoconvolution (f*f)(y) involves f(x) and f(y-x).
    # The discrete sum is sum_i f_i f_j where m_j - m_i = y.
    # Let k = j - i.
    # Then we sum f_i f_{i+k} for all valid i.
    # This is the cross-correlation, not convolution?
    # Correlation: R[k] = sum_i f_i f_{i+k}.
    # Convolution: C[k] = sum_i f_i f_{k-i}.
    # For real symmetric f, they are related.
    # Our f is not necessarily symmetric.
    # The autoconvolution (f*f)(y) = ∫ f(x)f(y-x)dx.
    # Let u = x, v = y-x. u+v=y.
    # Sum f_i f_j where m_i + m_j = y? No, m_j - m_i = y?
    # (f*f)(y) = ∫ f(x) f(y-x) dx.
    # Let x = m_i. y-x = y - m_i.
    # We need f(y-m_i). This is the value of f at position y-m_i.
    # If y-m_i = m_j, then j - i = y/dx.
    # So we sum f_i f_j where m_j = m_i + y.
    # So j = i + k, where k = y/dx.
    # So we sum f_i f_{i+k}.
    # This is the cross-correlation of f with itself, with lag k.
    # R[k] = sum_{i} f_i f_{i+k}.
    
    # How to get R[k] from np.convolve?
    # np.convolve(f, f, 'full') gives C[k] = sum_i f_i f_{k-i}.
    # If we reverse f, let g_i = f_{N-1-i}.
    # Then C[k] of f and g is sum_i f_i g_{k-i} = sum_i f_i f_{N-1-(k-i)} = sum_i f_i f_{N-1-k+i}.
    # Let j = N-1-k+i. Then i = j - (N-1-k).
    # Sum_j f_{j-(N-1-k)} f_j.
    # This looks like correlation with lag N-1-k.
    
    # Easier: just compute the correlation directly.
    # R[k] for k = -(N-1) ... (N-1).
    # R[0] = sum f_i^2.
    # R[1] = sum f_i f_{i+1}.
    # We can use FFT for this too.
    # FFT of f (padded).
    # IFFT(FFT(f) * conj(FFT(f))) gives the circular correlation.
    # Or just use the convolution result and flip it.
    # Convolution c = convolve(f, f, 'full').
    # c[k] = sum_i f_i f_{k-i}.
    # Correlation r[k] = sum_i f_i f_{i+k}.
    # r[k] = c[N-1-k] ?
    # Let's test N=2. f=[a,b].
    # conv: [a^2, 2ab, b^2].
    # corr[0] = a^2+b^2.
    # corr[1] = a*b.
    # corr[-1] = b*a.
    # c[0]=a^2, c[1]=2ab, c[2]=b^2.
    # r[0] = a^2+b^2. This is not directly in c.
    # Wait, the sum in convolution c[k] is over i such that 0<=i<N and 0<=k-i<N.
    # For k=0: i=0. f_0 f_0.
    # For k=1: i=0,1. f_0 f_1 + f_1 f_0.
    # For k=2: i=1. f_1 f_1.
    # The correlation sum r[k] = sum_i f_i f_{i+k}.
    # r[0] = f_0^2 + f_1^2.
    # r[1] = f_0 f_1.
    # r[-1] = f_1 f_0.
    
    # It seems c[k] is not directly r[k].
    # However, note that (f*f)(y) is symmetric if f is symmetric.
    # Generally, (f*f)(y) is not symmetric.
    # But we want the MAX over y.
    # The values are proportional to r[k].
    # Specifically, (f*f)(y_k) ≈ dx * r[k].
    
    # Let's compute r[k] using FFT.
    # r = ifft(fft(f) * conj(fft(f))).real
    # This gives circular correlation.
    # We need linear correlation.
    # Pad f to 2N.
    f_pad = np.zeros(2 * N)
    f_pad[:N] = f
    F = np.fft.fft(f_pad, n=4*N)
    corr = np.fft.ifft(F * np.conj(F)).real
    # The linear correlation is in corr[0:N] and corr[4N-(N-1):4N]?
    # Actually, for zero-padded signals, the linear correlation is just the first 2N-1 elements?
    # No, circular correlation of length 4N.
    # The non-zero lags are -N+1 to N-1.
    # In the FFT array of length 4N:
    # Index 0 -> lag 0
    # Index 1 -> lag 1
    # ...
    # Index N-1 -> lag N-1
    # Index 4N-1 -> lag -1
    # Index 4N-(N-1) -> lag -(N-1)
    
    # Extract lags -N+1 to N-1.
    # lags = np.arange(-(N-1), N)
    # indices = np.mod(lags, 4*N)
    # r = corr[indices]
    
    # Let's just use np.correlate?
    # np.correlate(f, f, 'full')
    # This computes sum_i f_i f_{i+k}.
    # Result length 2N-1.
    # r = np.correlate(f, f, 'full')
    # r[k] corresponds to lag k-(N-1).
    # r[0] -> lag -(N-1)
    # r[N-1] -> lag 0
    # r[2N-2] -> lag N-1
    
    r = np.correlate(f, f, 'full')
    # Scale by dx
    conv_vals = r * dx
    return conv_vals

def objective(params, time_start, deadline):
    """
    params: [width_inner, height_inner, width_outer, height_outer]
    Actually, let's define the structure.
    Double box:
    Outer box: width W_out, height H_out.
    Inner box (hole): width W_in, height 0.
    So f is H_out on the outer parts, 0 in the middle.
    Due to symmetry, we define half-widths.
    Total support [-1/4, 1/4].
    Let's parameterize by the inner hole half-width h_in and outer box height H.
    The outer box is fixed to the support?
    The problem says "Choose a nonnegative step function... on [-1/4, 1/4]".
    It doesn't say the support must be the whole interval.
    But scaling the support changes the normalization.
    If we shrink the support, ||f||_2^2 increases (for fixed L1 norm).
    So we should use the full support [-1/4, 1/4].
    
    So f is supported on [-1/4, 1/4].
    Structure:
    0 on [-h_in, h_in].
    H on [-1/4, -h_in] and [h_in, 1/4].
    
    Variables: h_in (0 to 1/4), H (height).
    Normalization: ∫f = 1.
    Length of support = 1/2 - 2*h_in.
    H * (1/2 - 2*h_in) = 1 => H = 1 / (0.5 - 2*h_in).
    
    We need to discretize this into N intervals.
    """
    if time.time() > deadline:
        return 1e10 # Timeout penalty
    
    h_in = params[0]
    # h_in must be < 0.25
    if h_in >= 0.25:
        return 1e10
    
    # Construct f
    f = np.zeros(N)
    # Grid points
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + dx/2
    # Mask for outer regions
    mask = (x < -h_in) | (x > h_in)
    H = 1.0 / (0.5 - 2.0 * h_in)
    f[mask] = H
    
    # Compute max autoconvolution
    conv_vals = compute_convolution(f)
    return np.max(conv_vals)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Initial guess: h_in = 0 (uniform box)
    # H = 1 / 0.5 = 2.
    # Max conv = 2.
    # Score = 1 / 2 = 0.5?
    # Wait, the seed score is 1.0.
    # Seed returns [1.0]*N.
    # ∫f = N * 1.0 * dx = 600 * 1/1200 = 0.5.
    # Normalized max = max(f*f) / (∫f)^2 ?
    # If we use the seed f=[1]*N:
    # ∫f = 0.5.
    # (f*f)(0) = ∫ f^2 = N * 1 * dx = 600 * 1/1200 = 0.5.
    # Ratio = 0.5 / 0.25 = 2.
    # So the "normalized maximum" is 2 for the seed.
    # If the score is 1.0, and score = seed_value / value, then seed_value = 2.
    # So we want to minimize the value V = max(f*f) / (∫f)^2.
    # For the double box with L1 norm 1:
    # V = max(f*f).
    # We found min V for uniform box (h_in=0) is 2.
    # Can we get lower than 2?
    # For a box of width W, V = 1/W.
    # If we use the full support W=0.5, V=2.
    # If we use a hollow box, the L2 norm increases, so (f*f)(0) increases.
    # (f*f)(0) = ∫f^2 = H^2 * (0.5 - 2 h_in) = (1/(0.5-2h_in))^2 * (0.5-2h_in) = 1 / (0.5 - 2 h_in).
    # As h_in increases, 0.5 - 2 h_in decreases, so 1/(...) increases.
    # So (f*f)(0) increases.
    # Since max(f*f) >= (f*f)(0), the maximum will increase.
    # So the uniform box is optimal among symmetric double boxes?
    
    # Wait, is it possible that the max is NOT at 0?
    # For a hollow box, the autoconvolution might have peaks away from 0?
    # The autoconvolution of a hollow box is the convolution of two hollow boxes.
    # It will be a trapezoid with a dip in the middle?
    # Let's check.
    # f = H (box - 1/4 to -h_in) + H (box h_in to 1/4).
    # f*f = H^2 [ (box1*box1) + (box1*box2) + (box2*box1) + (box2*box2) ].
    # box1*box1 is a trapezoid centered at 0 (shift 0).
    # box2*box2 is a trapezoid centered at 0.
    # box1*box2 is a trapezoid centered at 2*h_in.
    # box2*box1 is a trapezoid centered at -2*h_in.
    # So the sum has a peak at 0 and peaks at ±2*h_in.
    # The value at 0 is H^2 * (0.5 - 2 h_in).
    # The value at 2*h_in:
    # The overlap of box1 and box2 at shift 2*h_in is the full width of the smaller box?
    # Box1 is [-0.25, -h_in]. Box2 is [h_in, 0.25].
    # Shift 2*h_in: Box1 shifted by 2*h_in is [0.25-h_in, 0.5-h_in].
    # Overlap with Box2 [h_in, 0.25].
    # Overlap is [0.25-h_in, 0.25]. Width h_in.
    # So contribution is H^2 * h_in.
    # Total at 2*h_in: H^2 * (contribution from box1*box2 + box2*box1 + box1*box1 + box2*box2).
    # At shift 2*h_in, box1*box1 (centered 0) has value?
    # Distance 2*h_in from center. Width of box1 is 0.25-h_in.
    # If 2*h_in < 0.25-h_in, there is overlap.
    # This is getting complicated.
    
    # Let's just optimize numerically.
    
    best_val = 1e10
    best_params = [0.0]
    
    # Try a few random starts and local optimization
    # Parameters: h_in
    # Range [0, 0.24]
    
    # Since we suspect h_in=0 is best, let's check if any h_in > 0 is better.
    # If not, we should try asymmetric structures or other shapes.
    # But the directive says "KEEP this design... double-box".
    # So I will optimize h_in.
    
    # Use scipy minimize on a single variable h_in.
    def obj_1d(h):
        return objective([h], start_time, deadline)
    
    # Try multiple initial points
    for h0 in [0.0, 0.05, 0.1, 0.15, 0.2]:
        try:
            res = minimize(obj_1d, [h0], method='Nelder-Mead', bounds=[(0.0, 0.24)], options={'maxiter': 1000})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except:
            pass
        
        if time.time() > deadline - 10:
            break

    # Construct the final function from best_params
    h_in = best_params[0]
    f = np.zeros(N)
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + dx/2
    mask = (x < -h_in) | (x > h_in)
    H = 1.0 / (0.5 - 2.0 * h_in)
    f[mask] = H
    
    # Normalize to ensure sum(f)*dx = 1?
    # The objective function assumed L1=1.
    # But the score is max(f*f) / (∫f)^2.
    # If we return f, the evaluator computes this ratio.
    # So we can return any scaling.
    # Returning the L1=1 version is fine.
    
    return list(f)

# EVOLVE-BLOCK-END
