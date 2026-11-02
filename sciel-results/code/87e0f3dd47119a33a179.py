import time
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline

N = 600
DEADLINE = 140.0  # Safety margin before 150s limit

def compute_score(h):
    """Compute normalized max of autoconvolution: max(auto_conv) / ((sum h)^2 / N)"""
    h = np.asarray(h, dtype=np.float64)
    total = np.sum(h)
    if total <= 0:
        return float('inf')
    H = np.fft.fft(h)
    auto_conv = np.fft.ifft(np.abs(H)**2).real
    return np.max(auto_conv) / (total * total / N)

def construct():
    """
    Returns a sequence of nonnegative step heights on equal intervals of [-1/4, 1/4].
    Uses a cubic spline parameterization with gradient-based optimization.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    def elapsed():
        return time.time() - start_time
    
    def time_left():
        return deadline - time.time()

    # Grid definition
    x_grid = np.linspace(-0.25, 0.25, N)
    
    # Spline control points
    K = 50  # Number of control points
    x_ctrl = np.linspace(-0.25, 0.25, K)
    
    # Initial guess: Uniform (log(1) = 0)
    w0 = np.zeros(K)
    
    def spline_to_h(w):
        """Convert log-control-points to heights on grid"""
        y_ctrl = np.exp(w)
        try:
            spline = CubicSpline(x_ctrl, y_ctrl, bc_type='natural')
            h = spline(x_grid)
            h = np.maximum(h, 0)
            return h
        except Exception:
            return np.ones(N)

    def objective_and_grad(w):
        h = spline_to_h(w)
        total = np.sum(h)
        if total <= 0:
            return 1e6, np.zeros_like(w)
        
        H = np.fft.fft(h)
        auto_conv = np.fft.ifft(np.abs(H)**2).real
        
        # LogSumExp approximation of max
        # tau controls the sharpness. Start moderate.
        tau = 50.0 
        max_ac = np.max(auto_conv)
        if max_ac <= 0:
            return 1e6, np.zeros_like(w)
        
        # Shift for numerical stability
        shifted = tau * auto_conv - tau * max_ac
        log_sum_exp = np.log(np.sum(np.exp(shifted))) + tau * max_ac
        L = log_sum_exp / tau
        
        # Score = L / (total^2 / N)
        S0 = (total * total) / N
        score = L / S0
        
        # Gradient calculation
        # dL/dh = 2 * convolve(p, h) where p is softmax(tau * auto_conv)
        p = np.exp(shifted) / np.sum(np.exp(shifted))
        # p is the probability distribution over the auto-convolution peaks
        # Gradient of L w.r.t h:
        # L = (1/tau) log sum exp(tau C_k)
        # dL/dC_k = p_k
        # C = h * h (circular)
        # dC_k/dh_j = h_{k-j} + h_{j-k}
        # dL/dh_j = sum_k p_k (h_{k-j} + h_{j-k})
        # This is 2 * (p * h)_j
        grad_L_h = 2 * np.fft.ifft(np.fft.fft(p) * np.fft.fft(h)).real
        
        # dS0/dh = 2 * h / N
        grad_S0_h = 2 * h / N
        
        # dScore/dh = (1/S0) * grad_L_h - (L/S0^2) * grad_S0_h
        grad_score_h = grad_L_h / S0 - (L / (S0 * S0)) * grad_S0_h
        
        # Backprop through spline
        # h = spline(x_grid)
        # We need dh/dw.
        # CubicSpline interpolation is linear in y_ctrl.
        # h = A @ y_ctrl, where A is the interpolation matrix.
        # We can compute A by evaluating the spline basis functions.
        # Or simply use finite differences for the backprop if analytic is too hard?
        # Let's try to get the basis functions.
        # scipy.interpolate.CubicSpline doesn't directly expose the matrix easily without digging.
        # Alternative: Use a simpler basis, e.g., Linear Interpolation (PCHIP) or just evaluate basis functions manually.
        
        # Let's use a simple approach: Finite difference for the backprop step?
        # No, that's slow.
        # Let's construct the interpolation matrix A explicitly for cubic splines?
        # Or use a different parameterization: Sum of Gaussians?
        
        # Let's switch to Gaussian Mixture for easier backprop.
        # h = sum A_j exp(-(x-c_j)^2/2s_j^2)
        # This is differentiable and easy to backprop.
        
        return score, None # Placeholder, will change strategy to Gaussian Mixture

    # Switching to Gaussian Mixture Model (GMM) for easier gradient computation
    # h(x) = A_0 + sum_{j=1}^M A_j exp(-(x-c_j)^2 / (2 s_j^2))
    # Params: A_0, A_1..A_M, c_1..c_M, s_1..s_M
    # M = 20
    M = 20
    dim = 1 + 3*M
    
    def gmm_to_h(params):
        A0 = params[0]
        A = params[1:1+M]
        c = params[1+M:1+2*M]
        s = params[1+2*M:1+3*M]
        
        # Ensure positive amplitudes and widths
        A = np.exp(A)
        s = np.exp(s)
        
        h = np.full(N, A0)
        for j in range(M):
            h += A[j] * np.exp(-(x_grid - c[j])**2 / (2 * s[j]**2))
        h = np.maximum(h, 0)
        return h

    def gmm_objective_and_grad(params):
        h = gmm_to_h(params)
        total = np.sum(h)
        if total <= 0:
            return 1e6, np.zeros(dim)
        
        H = np.fft.fft(h)
        auto_conv = np.fft.ifft(np.abs(H)**2).real
        
        tau = 50.0
        max_ac = np.max(auto_conv)
        if max_ac <= 0:
            return 1e6, np.zeros(dim)
            
        shifted = tau * auto_conv - tau * max_ac
        log_sum_exp = np.log(np.sum(np.exp(shifted))) + tau * max_ac
        L = log_sum_exp / tau
        
        S0 = (total * total) / N
        score = L / S0
        
        p = np.exp(shifted) / np.sum(np.exp(shifted))
        grad_L_h = 2 * np.fft.ifft(np.fft.fft(p) * np.fft.fft(h)).real
        grad_S0_h = 2 * h / N
        
        grad_score_h = grad_L_h / S0 - (L / (S0 * S0)) * grad_S0_h
        
        # Backprop to params
        # h = A0 + sum A_j exp(...)
        # dh/dA0 = 1
        # dh/dA_j = exp(...)
        # dh/dc_j = A_j * exp(...) * (x-c_j)/s_j^2
        # dh/ds_j = A_j * exp(...) * (x-c_j)^2 / s_j^3
        
        grad_params = np.zeros(dim)
        exp_terms = np.zeros((M, N))
        
        for j in range(M):
            exp_term = np.exp(-(x_grid - c[j])**2 / (2 * s[j]**2))
            exp_terms[j] = exp_term
            
            grad_params[0] += np.sum(grad_score_h) # dA0
            grad_params[1+j] += np.sum(grad_score_h * exp_term) # dA_j
            grad_params[1+M+j] += np.sum(grad_score_h * A[j] * exp_term * (x_grid - c[j]) / (s[j]**2)) # dc_j
            grad_params[1+2*M+j] += np.sum(grad_score_h * A[j] * exp_term * (x_grid - c[j])**2 / (s[j]**3)) # ds_j
            
        # Chain rule for exp transformations
        # A_j = exp(params[1+j]) -> dA_j/dparam = A_j
        # s_j = exp(params[1+2*M+j]) -> ds_j/dparam = s_j
        
        # The gradients above are dScore/dA_j, dScore/dc_j, dScore/ds_j
        # We need dScore/dparam
        # dScore/dparam_A_j = dScore/dA_j * A_j
        # dScore/dparam_s_j = dScore/ds_j * s_j
        
        for j in range(M):
            grad_params[1+j] *= A[j]
            grad_params[1+2*M+j] *= s[j]
            
        return score, grad_params

    # Initialize params
    # A0 = log(0.1)
    # A_j = log(1.0)
    # c_j = uniform random
    # s_j = log(0.05)
    rng = np.random.RandomState(42)
    w0 = np.zeros(dim)
    w0[0] = np.log(0.1)
    w0[1:1+M] = np.log(1.0)
    w0[1+M:1+2*M] = rng.uniform(-0.25, 0.25, M)
    w0[1+2*M:1+3*M] = np.log(0.05)
    
    best_h = gmm_to_h(w0)
    best_score = compute_score(best_h)
    
    # Optimize
    try:
        res = minimize(gmm_objective_and_grad, w0, method='L-BFGS-B', jac=True,
                       options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5})
        if res.fun < best_score:
            best_score = res.fun
            best_h = gmm_to_h(res.x)
    except Exception as e:
        pass

    # Cleanup
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) <= 0:
        best_h = np.ones(N)
        
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Best score: {compute_score(np.array(res))}")
