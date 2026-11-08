import numpy as np
import time
from scipy.optimize import minimize

N = 600
dx = 0.5 / N
START_TIME = time.time()
TIME_LIMIT = 130  # Safety margin before 150s kill

def _compute_autoconv(v):
    """
    Computes the linear autoconvolution of v.
    v is a 1D array of length N.
    Returns array of length 2N-1.
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    v_pad = np.zeros(n_fft)
    v_pad[:N] = v
    conv = np.fft.ifft(np.fft.fft(v_pad) ** 2).real[:2 * N - 1]
    return conv

def _smooth_max_val(theta, beta=200.0):
    """
    Computes the smooth max of the autoconvolution given theta (log-weights).
    """
    # Softmax
    # Shift for numerical stability
    theta_max = np.max(theta)
    exp_theta = np.exp(theta - theta_max)
    sum_exp = np.sum(exp_theta)
    v = exp_theta / sum_exp
    
    C = _lag_based_autoconv(v)
    
    # Smooth max
    C_max = np.max(C)
    if C_max < 1e-12:
        return 0.0, C, v
    shifted = C - C_max
    exp_shifted = np.exp(beta * shifted)
    sum_exp_shifted = np.sum(exp_shifted)
    val = np.log(sum_exp_shifted) / beta + C_max
    
    return val, C, v, exp_shifted / sum_exp_shifted

def _smooth_max_grad(theta, beta=200.0):
    """
    Computes the value and gradient of the smooth max objective.
    """
    val, C, v, weights = _smooth_max_val(theta, beta)
    
    # grad_v_j = sum_lag weights[lag_idx] * (v_{j-lag} + v_{j+lag})
    # where lag_idx = lag + (N-1), so weights[lag_idx] corresponds to lag = lag_idx - (N-1)
    # Due to symmetry of weights (C is symmetric), grad_v = 2 * correlation(weights, v)
    # where correlation is computed appropriately.
    
    # Use FFT-based convolution as derived in _objective_and_grad
    n_fft = 1 << (3 * N).bit_length()
    w_arr = weights  # length 2N-1, index k corresponds to lag k-(N-1)
    v_arr = v  # length N
    
    w_fft = np.fft.fft(w_arr, n_fft)
    v_fft = np.fft.fft(v_arr, n_fft)
    conv_fft = w_fft * v_fft
    conv_lin = np.fft.ifft(conv_fft).real
    
    # Extract grad_v: j=0..N-1 corresponds to conv_lin[N-1..2N-2]
    grad_v = 2.0 * conv_lin[N-1 : 2*N-1]
    
    # Transform grad_v to grad_theta
    # dv_l/dtheta_j = v_l (delta_lj - v_j)
    # grad_theta_j = sum_l grad_v_l * dv_l/dtheta_j = v_j * (grad_v_j - sum_l grad_v_l * v_l)
    sum_grad_v_v = np.dot(grad_v, v)
    grad_theta = v * (grad_v - sum_grad_v_v)
    
    return val, grad_theta

def _lag_based_autoconv(v):
    """
    Computes linear autoconvolution indexed by lag.
    Output length 2N-1.
    Index 0 corresponds to lag -(N-1).
    Index N-1 corresponds to lag 0.
    Index 2N-2 corresponds to lag N-1.
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    v_pad = np.zeros(n_fft)
    v_pad[:N] = v
    V_fft = np.fft.fft(v_pad)
    P_fft = V_fft * np.conj(V_fft)
    r_circ = np.fft.ifft(P_fft).real
    
    # r_circ[0] is lag 0.
    # r_circ[1] is lag 1 (circular).
    # Since we zero-padded to n_fft >= 2N-1, lags 0..N-1 are in r_circ[0..N-1].
    # Lags -(N-1)..-1 are in r_circ[n_fft-N+1 .. n_fft-1].
    
    r_lin = np.zeros(2 * N - 1)
    r_lin[N-1 : 2*N-1] = r_circ[0 : N]
    r_lin[0 : N-1] = r_circ[n_fft-N+1 : n_fft]
    
    return r_lin

def _objective_and_grad(theta, beta=200.0):
    # Softmax
    theta_max = np.max(theta)
    exp_theta = np.exp(theta - theta_max)
    sum_exp = np.sum(exp_theta)
    v = exp_theta / sum_exp
    
    C = _lag_based_autoconv(v)
    
    # Smooth max
    C_max = np.max(C)
    if C_max < 1e-12:
        return 0.0, np.zeros(N)
    
    shifted = C - C_max
    exp_shifted = np.exp(beta * shifted)
    sum_exp_shifted = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp_shifted
    val = np.log(sum_exp_shifted) / beta + C_max
    
    # Gradient via FFT convolution
    n_fft = 1 << (3 * N).bit_length()
    w_arr = weights  # length 2N-1
    v_arr = v  # length N
    
    w_fft = np.fft.fft(w_arr, n_fft)
    v_fft = np.fft.fft(v_arr, n_fft)
    conv_fft = w_fft * v_fft
    conv_lin = np.fft.ifft(conv_fft).real
    
    grad_v = 2.0 * conv_lin[N-1 : 2*N-1]
    
    # Transform to grad_theta
    sum_grad_v_v = np.dot(grad_v, v)
    grad_theta = v * (grad_v - sum_grad_v_v)
    
    return val, grad_theta

def construct():
    global START_TIME
    TIME_LIMIT = 130
    
    def _time_left():
        return TIME_LIMIT - (time.time() - START_TIME)
    
    best_theta = np.zeros(N)
    best_val = 1e10
    
    # Try different beta values
    betas = [100.0, 200.0, 500.0]
    
    # Initializations
    x = np.linspace(-1, 1, N)
    inits = [
        np.zeros(N),
        -2 * x**2,
        -4 * x**2,
        -8 * x**2,
        -16 * x**2
    ]
    
    rng = np.random.default_rng(42)
    for _ in range(3):
        inits.append(rng.standard_normal(N) * 0.05)
    
    for beta in betas:
        if _time_left() < 15:
            break
        for x0 in inits:
            if _time_left() < 5:
                break
            try:
                res = minimize(
                    lambda th: _objective_and_grad(th, beta), 
                    x0.copy(), 
                    method='L-BFGS-B', 
                    jac=True,
                    options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-11, 'maxls': 50}
                )
                # Evaluate true max
                theta_max = np.max(res.x)
                exp_theta = np.exp(res.x - theta_max)
                v = exp_theta / np.sum(exp_theta)
                C = _lag_based_autoconv(v)
                true_val = np.max(C)
                
                if true_val < best_val:
                    best_val = true_val
                    best_theta = res.x.copy()
            except Exception:
                pass
    
    # Final polish with best beta
    if _time_left() > 20:
        try:
            res = minimize(
                lambda th: _objective_and_grad(th, 200.0), 
                best_theta.copy(), 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 100}
            )
            theta_max = np.max(res.x)
            exp_theta = np.exp(res.x - theta_max)
            v = exp_theta / np.sum(exp_theta)
            C = _lag_based_autoconv(v)
            true_val = np.max(C)
            
            if true_val < best_val:
                best_val = true_val
                best_theta = res.x.copy()
        except Exception:
            pass

    # Final conversion to v
    theta_max = np.max(best_theta)
    exp_theta = np.exp(best_theta - theta_max)
    v = exp_theta / np.sum(exp_theta)
    
    return v.tolist()

construct()
