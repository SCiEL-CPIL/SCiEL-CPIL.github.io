"""Optimized signed step function via Gumbel-Sigmoid (Concrete) relaxation."""
import numpy as np
from scipy.optimize import minimize
import time

N = 400
DT = 0.5 / N  # Interval width: 0.5 / 400


def _compute_autoconv(signed_heights):
    """Compute the autoconvolution via FFT and return as a 1D array of length N (lags 0..N-1)."""
    n = 1
    while n < 2 * N:
        n <<= 1
    F = np.fft.rfft(signed_heights, n)
    R = np.fft.irfft(F * np.conj(F), n)
    # Normalize and extract lags 0..N-1
    return R[:N] * DT


def _gumbel_sigmoid_loss(logits, tau=1.0):
    """
    Loss function using Concrete (Gumbel-Sigmoid) relaxation.
    logits: shape (N,), parameters for sign probabilities.
    tau: temperature for the relaxation.
    """
    # Soft signs in [-1, 1]
    # s = 2 * sigmoid(logits / tau) - 1
    # To avoid overflow in sigmoid, use stable version
    z = logits / tau
    # Stable sigmoid
    p = np.zeros_like(z)
    pos_mask = z >= 0
    p[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))
    neg_mask = ~pos_mask
    exp_z = np.exp(z[neg_mask])
    p[neg_mask] = exp_z / (1.0 + exp_z)
    
    s = 2.0 * p - 1.0

    R = _compute_autoconv(s)
    R_sidelobes = R[1:]
    squared = R_sidelobes ** 2
    
    # Soft max loss
    m = np.max(squared)
    if m < 1e-8:
        return 0.0
    loss = m + 1.0 * np.log(np.sum(np.exp((squared - m) / 1.0)))
    return loss


def _gumbel_sigmoid_loss_grad(logits, tau=1.0):
    """
    Gradient of the loss w.r.t. logits.
    """
    z = logits / tau
    # Stable sigmoid and its derivative
    p = np.zeros_like(z)
    pos_mask = z >= 0
    p[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))
    neg_mask = ~pos_mask
    exp_z = np.exp(z[neg_mask])
    p[neg_mask] = exp_z / (1.0 + exp_z)
    
    s = 2.0 * p - 1.0
    # ds/dlogits = 2 * p(1-p) * (1/tau) = (1 - s^2) / (2 * tau)
    ds_dlogits = (1.0 - s ** 2) / (2.0 * tau)

    R = _compute_autoconv(s)
    R_sidelobes = R[1:]
    squared = R_sidelobes ** 2

    m = np.max(squared)
    if m < 1e-8:
        return np.zeros(N)

    # Softmax weights (tau_loss = 1.0 for the loss soft-max)
    exp_s = np.exp((squared - m) / 1.0)
    sum_exp = np.sum(exp_s)
    weights = exp_s / sum_exp

    dR_sidelobes = 2.0 * weights * R_sidelobes

    A = np.zeros(N)
    A[1:] = dR_sidelobes

    V = np.zeros(N)
    for k in range(1, N):
        if A[k] == 0.0:
            continue
        V[:N-k] += A[k] * s[k:]
        V[k:] += A[k] * s[:N-k]

    V *= DT

    grad = V * ds_dlogits
    return grad


def construct():
    start_time = time.time()
    timeout = 110.0

    np.random.seed(42)
    best_signs = None
    best_loss = float('inf')

    # Initialize logits to 0 (p=0.5, s=0) or small random
    init_logits = np.random.randn(N) * 0.1

    # Try different temperatures (annealing)
    # Start with high temperature to explore, lower to refine
    temperatures = [10.0, 5.0, 2.0, 1.0, 0.5]
    
    current_logits = init_logits.copy()

    for tau in temperatures:
        if time.time() - start_time > timeout - 10:
            break
            
        # Run optimization for this temperature
        res = minimize(
            _gumbel_sigmoid_loss,
            current_logits,
            method='L-BFGS-B',
            jac=lambda x: _gumbel_sigmoid_loss_grad(x, tau),
            options={'maxiter': 20, 'ftol': 1e-8}
        )
        
        if res.fun < best_loss:
            best_loss = res.fun
            best_logits = res.x
            # Snap to signs to evaluate discrete performance?
            # The task asks for a sequence of signed step heights.
            # We should evaluate the discrete version to ensure we are improving the actual metric.
            signs = np.sign(best_logits)
            signs[signs == 0] = 1.0
            
            # Compute actual loss for discrete signs
            R_disc = _compute_autoconv(signs)
            disc_loss = np.max(R_disc[1:] ** 2)
            
            if disc_loss < best_loss:
                best_loss = disc_loss
                best_signs = signs

        current_logits = res.x

    # If no valid signs found, fallback
    if best_signs is None:
        best_signs = np.sign(current_logits)
        best_signs[best_signs == 0] = 1.0

    return best_signs
