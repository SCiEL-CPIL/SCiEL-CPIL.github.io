"""
First Autocorrelation Inequality: minimize the normalized maximum of the autoconvolution.

Approach:
- Represent a nonnegative even step function on [-1/4, 1/4] by a small set of
  low-frequency cosine coefficients (Fourier cosine series on the interval).
- Enforce nonnegativity by clipping after each update.
- Compute the true (non-periodic) autoconvolution via FFT.
- Normalize by L2 energy so the objective is max(f*f) / ||f||_2^2.
- Use gradient-based optimization with multiple restarts.
"""

import numpy as np

N = 600  # number of equal intervals

def construct():
    L = 0.5  # length of interval [-1/4, 1/4]
    n_harm = 12  # number of cosine harmonics (k=1..n_harm), plus DC term

    # Grid for the function values (centers of the 600 intervals)
    interval_width = L / N
    centers = -L / 2 + (np.arange(N) + 0.5) * interval_width

    # Precompute cosine basis matrix: shape (N, n_harm + 1)
    # Column 0 is DC (ones), columns 1..n_harm are cos(2*pi*k*x/L)
    basis = np.ones((N, n_harm + 1))
    for k in range(1, n_harm + 1):
        basis[:, k] = np.cos(2 * np.pi * k * centers / L)

    def f_from_coeffs(a):
        """Compute function values from cosine coefficients."""
        return basis @ a

    def compute_autoconv_max(f_vals):
        """
        Compute the normalized maximum of the autoconvolution.
        f_vals is the function on N equal intervals of width dx.
        We compute (f*f)(t) = integral f(tau) f(t-tau) dtau via FFT.
        The autoconvolution is supported on [-L, L] = [-0.5, 0.5].
        """
        dx = interval_width
        # Pad to next power of 2 for FFT efficiency
        n_fft = 2 * N
        # f is zero outside [-1/4, 1/4], so we can zero-pad
        f_pad = np.zeros(n_fft)
        f_pad[:N] = f_vals
        # FFT
        F = np.fft.rfft(f_pad)
        # Autoconvolution in frequency domain: |F|^2
        G = np.fft.irfft(F * F, n=n_fft)
        # The autoconvolution G(t) has the same grid spacing dx
        # G[0] corresponds to t=0, G[1] to t=dx, ..., G[N-1] to t=(N-1)*dx
        # G[N] to t=N*dx=L=0.5, etc.
        # We need max over the full support [-L, L], which is indices 0..N and N..2N
        # Actually, due to even symmetry, G is even, so max is at t=0 or somewhere positive.
        # Let's just take the max over the first N+1 points (t from 0 to L)
        # Actually the full autoconvolution on [-L, L] is 2N+1 points.
        # G has n_fft=2N points. The autoconvolution of two length-N signals has length 2N-1.
        # With zero-padding to 2N, the circular convolution equals linear convolution.
        # The result G has 2N points. G[0] is at t=0, G[1] at t=dx, ..., G[N-1] at t=(N-1)dx,
        # G[N] at t=Nd x = L, G[N+1] would be t=(N+1)dx which wraps around.
        # Actually, for linear convolution of two N-length sequences, the result is 2N-1 long.
        # With FFT of size 2N, we get circular convolution which equals linear conv for this case.
        # The output G[0..2N-1] corresponds to t=0, dx, ..., (2N-1)dx.
        # But the true autoconvolution is supported on [-L, L] = [-N*dx, N*dx].
        # Due to even symmetry, G[-k] = G[k]. In the FFT output, negative times are at the end.
        # G[0] = t=0, G[1..N-1] = t=dx..(N-1)dx, G[N] = t=Nd x=L
        # G[N+1..2N-1] = t=(N+1)dx..(2N-1)dx which corresponds to negative times wrapped.
        # Actually, the autoconvolution (f*f)(t) for t in [-L, L]:
        # For t >= 0: (f*f)(t) = integral_{max(-1/4, t-1/4)}^{min(1/4, t+1/4)} f(tau) f(t-tau) dtau
        # The FFT approach with zero-padding should give us the correct linear convolution.
        # Let's just take the max over all 2N points, but note that due to symmetry,
        # the max should be at t=0 or near it.
        # Actually, let's be more careful. The linear convolution of f (length N) with itself
        # has length 2N-1. With FFT size 2N, the result is the circular convolution,
        # which for zero-padded signals equals the linear convolution for the first 2N-1 samples.
        # G[0] = (f*f)(0), G[1] = (f*f)(dx), ..., G[2N-2] = (f*f)((2N-2)dx)
        # But (f*f)(t) for t > L is zero, so G[N..2N-1] should be ~0 (up to numerical error).
        # The max of the autoconvolution is max(G[0..N]) approximately.
        max_conv = np.max(G[:N + 1])
        
        # Energy
        energy = np.sum(f_vals ** 2) * dx
        if energy < 1e-12:
            return np.inf
        return max_conv / energy

    def objective_and_grad(a):
        """
        Compute the objective and its gradient with respect to a.
        Objective: max_t (f*f)(t) / ||f||_2^2
        where f = basis @ a.
        """
        f_vals = f_from_coeffs(a)
        dx = interval_width
        
        # Clip to ensure nonnegativity for the actual evaluation
        f_nonneg = np.maximum(0, f_vals)
        
        # Energy
        energy = np.sum(f_nonneg ** 2) * dx
        if energy < 1e-12:
            return np.inf, np.zeros_like(a)
        
        # Compute autoconvolution
        n_fft = 2 * N
        f_pad = np.zeros(n_fft)
        f_pad[:N] = f_nonneg
        F = np.fft.rfft(f_pad)
        G = np.fft.irfft(F * F, n=n_fft)
        
        # Find t_max
        idx_max = np.argmax(G[:N + 1])
        max_conv = G[idx_max]
        
        score = max_conv / energy
        
        # Gradient computation:
        # d(score)/d(a) = d(max_conv/energy)/d(a)
        # = (d(max_conv)/d(a) * energy - max_conv * d(energy)/d(a)) / energy^2
        # 
        # d(energy)/d(a) = 2 * dx * sum(f_nonneg * df_nonneg/da)
        # For nonnegativity clipping, if f_vals[j] > 0, df_nonneg[j]/da = basis[j,:]
        # If f_vals[j] <= 0, df_nonneg[j]/da = 0 (subgradient)
        
        # d(max_conv)/d(a): max_conv = G[idx_max] = integral f(tau) f(t_max - tau) dtau
        # = dx * sum_j f_nonneg[j] * f_nonneg_at_tmax_minus_j
        # where f_nonneg_at_tmax_minus_j is the value of f at position corresponding to t_max - j*dx
        # This is complex to compute exactly. Let's use a numerical gradient for simplicity
        # given the time constraints, or use the fact that the gradient of the max of a
        # differentiable function is the gradient at the point of maximum.
        
        # For a more efficient approach, let's use finite differences for the gradient
        # since n_harm is small (13 parameters).
        
        eps = 1e-6
        grad = np.zeros_like(a)
        for i in range(len(a)):
            a_plus = a.copy()
            a_plus[i] += eps
            f_plus = np.maximum(0, f_from_coeffs(a_plus))
            energy_plus = np.sum(f_plus ** 2) * dx
            if energy_plus < 1e-12:
                grad[i] = 0
                continue
            n_fft = 2 * N
            f_pad_plus = np.zeros(n_fft)
            f_pad_plus[:N] = f_plus
            F_plus = np.fft.rfft(f_pad_plus)
            G_plus = np.fft.irfft(F_plus * F_plus, n=n_fft)
            score_plus = np.max(G_plus[:N + 1]) / energy_plus
            
            a_minus = a.copy()
            a_minus[i] -= eps
            f_minus = np.maximum(0, f_from_coeffs(a_minus))
            energy_minus = np.sum(f_minus ** 2) * dx
            if energy_minus < 1e-12:
                grad[i] = 0
                continue
            f_pad_minus = np.zeros(n_fft)
            f_pad_minus[:N] = f_minus
            F_minus = np.fft.rfft(f_pad_minus)
            G_minus = np.fft.irfft(F_minus * F_minus, n=n_fft)
            score_minus = np.max(G_minus[:N + 1]) / energy_minus
            
            grad[i] = (score_plus - score_minus) / (2 * eps)
        
        return score, grad

    # Optimization with multiple restarts
    best_score = np.inf
    best_a = np.zeros(n_harm + 1)
    best_a[0] = 1.0  # Start with DC only (uniform function)

    # Initial score
    f_init = np.maximum(0, f_from_coeffs(best_a))
    best_score = compute_autoconv_max(f_init)

    n_restarts = 10
    n_iter = 50
    lr = 0.02

    for restart in range(n_restarts):
        if restart == 0:
            a = best_a.copy()
        else:
            # Random initialization biased towards positive DC
            a = np.random.randn(n_harm + 1) * 0.2
            a[0] += 1.0  # Ensure positive DC component
        
        for it in range(n_iter):
            score, grad = objective_and_grad(a)
            
            if score < best_score:
                best_score = score
                best_a = a.copy()
            
            # Gradient descent step
            a = a - lr * grad
            
            # Ensure DC component stays positive to maintain nonnegativity
            if a[0] < 0.1:
                a[0] = 0.1
            
            # Adaptive learning rate
            if it % 10 == 9:
                lr = max(0.001, lr * 0.8)
    
    # Final result
    heights = np.maximum(0, f_from_coeffs(best_a))
    
    return list(heights)
