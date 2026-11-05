import numpy as np
from scipy.signal import fftconvolve

N = 400
w = 1.0 / 200.0

def construct():
       # Build Haar Wavelet Basis
       # We want an orthogonal basis.
       # Haar is easiest.
       basis = np.zeros((N, N))
       
       # DC component
       basis[0, :] = 1.0 / np.sqrt(N)
       
       # Other components
       # For each scale, generate the Haar functions
       # This is a standard construction.
       # Let's just use a random orthogonal matrix? No, "wavelet".
       # Let's implement a simple Haar basis generation.
       
       def generate_haar_basis(N):
           basis = np.zeros((N, N))
           # DC
           basis[0, :] = 1/np.sqrt(N)
           idx = 1
           # Scales
           for scale in range(int(np.log2(N))):
               length = 2 ** (scale + 1)
               # Number of such functions
               num_funcs = N // length
               for j in range(num_funcs):
                   # Detail function at this scale and position
                   # It is +1 for first half, -1 for second half of the block
                   start = j * length
                   end = start + length
                   half = length // 2
                   basis[idx, start:start+half] = 1/np.sqrt(N)
                   basis[idx, start+half:end] = -1/np.sqrt(N)
                   idx += 1
           return basis

       W = generate_haar_basis(N)
       
       # Initialize coefficients
       np.random.seed(42)
       theta = np.random.randn(N)
       
       def compute_autoconv_max(s):
           # s is real valued here
           n_fft = 2 * N
           S = np.fft.rfft(s, n_fft)
           r = np.fft.irfft(S * np.conj(S), n_fft)[:N]
           return np.max(np.abs(r)) * w, r

       best_theta = theta.copy()
       best_val = np.inf
       
       # Annealing schedule for T
       T_start = 1.0
       T_end = 0.1
       
       iterations = 2000
       
       for i in range(iterations):
           # Anneal T
           T = T_start * (T_end / T_start) ** (i / iterations)
           
           # Forward pass
           u = W @ theta
           s_soft = np.tanh(u / T)
           
           # Compute objective and gradient
           # r[k] = sum s[i] s[i+k]
           # We need grad of max |r[k]| wrt s
           
           n_fft = 2 * N
           S = np.fft.rfft(s_soft, n_fft)
           r = np.fft.irfft(S * np.conj(S), n_fft)[:N]
           
           k_max = np.argmax(np.abs(r))
           r_max = r[k_max]
           
           # Gradient of r[k_max] wrt s
           # r[k] = sum_{i=0}^{N-1-k} s[i] s[i+k]
           # d r[k] / d s[i] = s[i+k] (if i < N-k) + s[i-k] (if i >= k)
           grad_s = np.zeros(N)
           sign_r = np.sign(r_max)
           
           # Vectorized computation of grad_s
           # Term 1: s[i+k] for i < N-k
           if k_max > 0:
               grad_s[:N-k_max] += s_soft[k_max:]
               grad_s[k_max:] += s_soft[:N-k_max]
           else:
               # k=0, r[0] = sum s[i]^2
               # grad = 2 * s
               grad_s = 2 * s_soft
           
           grad_s *= sign_r
           
           # Backprop to theta
           # d s_soft / d u = sech^2(u/T) / T
           # d u / d theta = W
           # grad_theta = W^T (grad_s * ds_du)
           
           ds_du = np.cosh(u / T) ** (-2) / T
           grad_theta = W.T @ (grad_s * ds_du)
           
           # Update
           lr = 1.0 / (1 + i * 0.01)
           theta = theta - lr * grad_theta
           
           # Check hard sign performance occasionally
           if i % 10 == 0:
               s_hard = np.sign(s_soft)
               s_hard[s_hard == 0] = 1
               val, _ = compute_autoconv_max(s_hard)
               if val < best_val:
                   best_val = val
                   best_theta = theta.copy()
                   
       # Final hard sign
       u_final = W @ best_theta
       s_final = np.sign(u_final)
       s_final[s_final == 0] = 1
       
       return s_final
