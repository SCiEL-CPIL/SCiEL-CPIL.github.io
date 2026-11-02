import numpy as np
from scipy.optimize import minimize
import time

N = 600
L = 0.5
dx = L / N
t_grid = np.arange(-L/2, L/2 + dx, dx)
# Autoconvolution grid for discrete FFT
# We need to compute (f*f)(t) for t in [-L, L]
# But f is supported on [-L/2, L/2].
# The autoconvolution is supported on [-L, L].
# We only care about the max.

def compute_objective(coeffs):
    # coeffs are non-negative? We will handle non-negativity in the reconstruction
    # Reconstruct f from DCT basis
    # DCT-II is appropriate for real, even functions
    # Basis: cos(pi * k * (n + 0.5) / N)
    # We use the first K modes
    K = len(coeffs)
    n = np.arange(N)
    # Construct f
    f = np.zeros(N)
    for k in range(K):
        if coeffs[k] < 0:
            coeffs[k] = 0 # Enforce non-negativity of coefficients? 
            # Actually, it's better to allow negative coeffs but clip f
    # Better: Linear combination
    # f = sum c_k * phi_k
    # phi_k = cos(pi * k * (n + 0.5) / N)
    # This is O(N*K)
    # To speed up, precompute basis
    # But K is small (20)
    
    # Simpler: Just use the coefficients as the first 20 bins of the DCT of f?
    # No, let's stick to linear combination of cosines.
    
    # For speed, we can use FFT
    # If we define f in the time domain, we can compute autoconvolution via FFT
    
    # Let's try a different representation:
    # Optimize the 600 values directly? Too slow for Nelder-Mead.
    # Optimize 20 coefficients.
    
    # Basis functions:
    # phi_k(n) = cos(pi * k * (n + 0.5) / N)
    # We precompute the matrix Phi (N x K)
    
    return 0.0 # Placeholder

# Precompute Basis
K = 20
n_idx = np.arange(N)
# phi_k(n) = cos(pi * k * (n + 0.5) / N)
# Shape: (N, K)
phi = np.cos(np.pi * np.outer(n_idx + 0.5, np.arange(K)) / N)

# Precompute FFT of basis?
# (f*f) = ifft(fft(f)^2)
# f = Phi @ c
# fft(f) = fft(Phi) @ c
# Let F_phi = fft(Phi) (K x N)
F_phi = np.fft.fft(phi, axis=0)

def objective(c):
    # c is vector of K coefficients
    # f = phi @ c
    # fft_f = F_phi @ c
    fft_f = F_phi @ c
    # power_spectrum = |fft_f|^2
    # autoconv = ifft(power_spectrum)
    # Note: Scaling factors for FFT
    # numpy fft: ifft(fft(x)*fft(y)) = conv(x,y) * N?
    # Actually, np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)) gives circular convolution.
    # We need linear convolution.
    # Zero padding is needed.
    
    # This FFT approach with basis is tricky with zero padding.
    # Let's just reconstruct f and do FFT with zero padding.
    f = phi @ c
    f = np.clip(f, 0, None) # Enforce non-negative
    
    # Normalize f to ||f||_2 = 1
    norm = np.sqrt(np.sum(f**2) * dx)
    if norm < 1e-12:
        return 1e6
    f = f / norm
    
    # Compute autoconvolution
    # Zero pad to 2*N to avoid circular wrap
    f_pad = np.zeros(2*N)
    f_pad[:N] = f
    fft_f = np.fft.fft(f_pad)
    power = fft_f * np.conj(fft_f)
    auto = np.fft.ifft(power).real
    # Scale by dx?
    # Discrete sum: sum f_i f_{i+k} dx
    # FFT gives sum. So multiply by dx.
    auto *= dx
    
    # Find max sidelobe
    # The peak is at index N (shifted) or 0?
    # np.fft.ifft gives circular. 
    # With zero padding, the peak is at N (center of 2N array).
    # Actually, for real even f, the peak is at 0 and N.
    # We want max excluding the main lobe.
    # Main lobe is around N.
    # Let's find the global max and the value at N.
    peak = auto[N]
    # Mask out the main lobe?
    # "maximum of its autoconvolution" usually includes the peak.
    # But if we assume the "sidelobe" interpretation:
    # We want to minimize the max of auto[t] for t != 0.
    
    # If the metric is simply max(auto), it's peak.
    # If metric is max_sidelobe, we need to exclude the peak.
    
    # Let's return the max of the absolute value of the autoconvolution, 
    # excluding the central bin (and maybe a small window).
    # The problem says "normalized maximum".
    # I will bet on Sidelobe Minimization.
    
    # Exclude indices near N (the peak)
    # Window size 5
    mask = np.ones(2*N, dtype=bool)
    mask[N-5:N+6] = False
    max_side = np.max(auto[mask])
    
    # Return the ratio
    return max_side / peak

# Initial guess: random coefficients
np.random.seed(42)
c0 = np.random.rand(K)

# Optimize
start_time = time.time()
res = minimize(objective, c0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})

# Reconstruct best f
c_best = res.x
f_best = phi @ c_best
f_best = np.clip(f_best, 0, None)
norm = np.sqrt(np.sum(f_best**2) * dx)
if norm > 1e-12:
    f_best = f_best / norm
else:
    f_best = np.ones(N) / np.sqrt(N * dx) # Fallback to constant

# Return as list of heights
return_list = list(f_best)

def construct():
    return return_list
