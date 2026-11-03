"""Rewritten program for signed autocorrelation maximum absolute-value variant."""
import numpy as np
import time
import sys

N = 400
# Deadline guard: keep a safety margin before the 150s wall-clock kill.
DEADLINE = 120.0  # seconds; leave 30s buffer for evaluator overhead/imports etc.

def _autoconv_max_abs(x):
    """Compute max absolute value of the circular autoconvolution of x."""
    # Use FFT for speed. Circular autoconvolution: IFFT(|FFT(x)|^2)
    X = np.fft.fft(x)
    ac = np.fft.ifft(X * X)
    ac = np.real(ac)
    return np.max(np.abs(ac))

def _squared_integral(x):
    """Squared integral (sum of squares) of x."""
    return np.sum(x * x)

def _score(x):
    """The objective to minimize: max_abs_autoconv / squared_integral."""
    si = _squared_integral(x)
    if si < 1e-12:
        return np.inf
    return _autoconv_max_abs(x) / si

def _soft_max_abs_ac(x, beta):
    """
    Smooth surrogate for max(|ac|) using log-sum-exp.
    ac = circular autoconvolution.
    soft_max(|ac|) = (1/beta) * log(sum(exp(beta * |ac|)))
    This is differentiable and approximates max(|ac|) as beta -> inf.
    """
    X = np.fft.fft(x)
    ac = np.fft.ifft(X * X)
    ac = np.real(ac)
    abs_ac = np.abs(ac)
    # log-sum-exp for numerical stability
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    lse = m + np.log(np.sum(np.exp(beta * (abs_ac - m))))
    return lse / beta

def _soft_score(x, beta):
    """Smooth surrogate for the score."""
    si = _squared_integral(x)
    if si < 1e-12:
        return np.inf
    return _soft_max_abs_ac(x, beta) / si

def _soft_score_grad(x, beta):
    """
    Gradient of the soft surrogate score with respect to x.
    score = soft_max_abs_ac(x, beta) / (x^T x)
    Let f = soft_max_abs_ac, g = x^T x.
    d(score)/dx = (f' * g - f * g') / g^2
    g' = 2x
    f' = gradient of log-sum-exp(|ac|)/beta w.r.t. x
    
    ac = IFFT(|FFT(x)|^2) -- but this is circular autoconvolution.
    Actually, ac_k = sum_j x_j x_{k-j mod N}
    d(ac_k)/dx_i = x_{k-i} + x_{i-k} (for i != k-i) or 2*x_{2k} if i = k-i mod N... 
    This is getting complex. Let's use the fact that:
    ac = IFFT(FFT(x) * FFT(x))
    d(ac)/dx in frequency domain: d(FFT(x)_k)/dx = FFT(e_i)_k where e_i is unit vector.
    This is expensive to compute analytically. Instead, use finite differences or 
    a simpler approach: since the problem is small (N=400), we can use scipy.optimize
    with numerical gradients, or we can derive the gradient.
    
    Actually, let's use a different approach: use scipy.optimize.minimize with 
    BFGS and let it compute numerical gradients, or use the fact that the 
    autoconvolution gradient can be computed via FFT.
    
    For ac = circular autoconvolution, the gradient of ac_k w.r.t. x_i is:
    d(ac_k)/dx_i = x_{(k-i) mod N} + x_{(i-k) mod N}
    (with the convention that if k-i == i-k mod N, i.e., 2i == k mod N, then it's 2*x_k... 
    actually no, the two terms are the same in that case, so it's still the sum)
    
    So grad_f_i = (1/beta) * sum_k [exp(beta*|ac_k|)/sum_j exp(beta*|ac_j|)] * sign(ac_k) * d(ac_k)/dx_i
    This is O(N^2) per evaluation which is 160,000 for N=400, acceptable.
    """
    X = np.fft.fft(x)
    ac = np.fft.ifft(X * X)
    ac = np.real(ac)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return np.zeros_like(x)
    weights = np.exp(beta * (abs_ac - m))
    w_sum = np.sum(weights)
    # p_k = exp(beta*|ac_k|) / sum = weights_k / w_sum
    p = weights / w_sum
    # d(f)/d(ac_k) = p_k * sign(ac_k)
    dac = p * np.sign(ac)
    
    # Now compute grad_f_i = sum_k dac_k * d(ac_k)/dx_i
    # d(ac_k)/dx_i = x_{(k-i) mod N} + x_{(i-k) mod N}
    # grad_f = 2 * circular_convolution(dac, x) ... let's verify:
    # grad_f_i = sum_k dac_k * (x_{(k-i)} + x_{(i-k)})
    # = sum_k dac_k * x_{(k-i)} + sum_k dac_k * x_{(i-k)}
    # Let j = k-i: first sum = sum_j dac_{j+i} * x_j
    # Let j = i-k, so k = i-j: second sum = sum_j dac_{i-j} * x_j
    # So grad_f_i = sum_j (dac_{(i+j) mod N} + dac_{(i-j) mod N}) * x_j
    # This is equivalent to 2 * real part of something, or we can compute via FFT.
    # Actually, grad_f = 2 * IFFT(FFT(dac) * FFT(x)) ... let's check.
    # The circular convolution (dac * x)_i = sum_j dac_j * x_{i-j}
    # We have sum_j dac_{(i+j)} x_j which is (dac * x_flipped)_i where x_flipped_j = x_{-j}
    # And sum_j dac_{(i-j)} x_j = (dac * x)_i
    # So grad_f_i = (dac * x_flipped)_i + (dac * x)_i
    # = 2 * Re( IFFT(FFT(dac) * FFT(x)) )_i ... need to be careful.
    
    # Let's just compute it directly with FFT for efficiency.
    # (dac * x)_i = IFFT(FFT(dac) * FFT(x))_i
    # (dac * x_flipped)_i where x_flipped_j = x_{-j mod N}
    # FFT(x_flipped)_k = conj(FFT(x)_k) (for real x)
    # So (dac * x_flipped)_i = IFFT(FFT(dac) * conj(FFT(x)))_i
    # grad_f = IFFT(FFT(dac) * FFT(x)) + IFFT(FFT(dac) * conj(FFT(x)))
    # = 2 * Re( IFFT(FFT(dac) * FFT(x)) )
    
    D = np.fft.fft(dac)
    grad_f = 2.0 * np.real(np.fft.ifft(D * X))
    
    # Now score = f / g, g = x^T x
    # d(score)/dx = (grad_f * g - f * 2x) / g^2
    f = _soft_max_abs_ac(x, beta)
    g = _squared_integral(x)
    if g < 1e-12:
        return np.zeros_like(x)
    grad_score = (grad_f * g - f * 2.0 * x) / (g * g)
    return grad_score

def construct():
    """
    Construct a sequence of signed step heights minimizing max_abs_autoconv / squared_integral.
    Returns a numpy array of length N.
    """
    start_time = time.time()
    
    # Initial guess: random signs
    np.random.seed(42)
    x = np.random.choice([-1.0, 1.0], size=N)
    
    best_x = x.copy()
    best_score = _score(x)
    
    # Schedule of beta values for the soft surrogate
    beta_schedule = [50.0, 100.0, 200.0, 400.0, 800.0]
    
    from scipy.optimize import minimize
    
    for beta in beta_schedule:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Try multiple initializations for robustness
        for init_idx in range(3):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            if init_idx == 0:
                x_curr = best_x.copy()
            elif init_idx == 1:
                x_curr = np.random.choice([-1.0, 1.0], size=N)
                # Scale to have unit L2 norm for better conditioning
                x_curr = x_curr / np.linalg.norm(x_curr)
            else:
                # Perturbed version of best
                x_curr = best_x.copy()
                noise = np.random.randn(N) * 0.1
                x_curr = x_curr + noise
                x_curr = x_curr / np.linalg.norm(x_curr)
            
            # Normalize x to have unit L2 norm (score is scale-invariant)
            x_curr = x_curr / np.linalg.norm(x_curr)
            
            def objective(x_flat):
                x_arr = x_flat.reshape(N)
                norm = np.linalg.norm(x_arr)
                if norm < 1e-10:
                    return np.inf
                x_arr = x_arr / norm
                return _soft_score(x_arr, beta)
            
            def gradient(x_flat):
                x_arr = x_flat.reshape(N)
                norm = np.linalg.norm(x_arr)
                if norm < 1e-10:
                    return np.zeros(N)
                x_arr = x_arr / norm
                grad = _soft_score_grad(x_arr, beta)
                # Project gradient to the unit sphere (tangent space)
                grad = grad - np.dot(grad, x_arr) * x_arr
                return grad.flatten()
            
            try:
                result = minimize(
                    objective,
                    x_curr.flatten(),
                    jac=gradient,
                    method='BFGS',
                    options={'maxiter': 200, 'gtol': 1e-6}
                )
                
                x_opt = result.x.reshape(N)
                x_opt = x_opt / np.linalg.norm(x_opt)
                
                # Evaluate with the true (non-soft) score
                true_score = _score(x_opt)
                
                if true_score < best_score:
                    best_score = true_score
                    best_x = x_opt.copy()
                    
            except Exception:
                pass
        
        # Peak suppression phase: directly target the argmax of autoconvolution
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Try to reduce the peak by adjusting x
        X = np.fft.fft(best_x)
        ac = np.fft.ifft(X * X)
        ac = np.real(ac)
        idx_max = np.argmax(np.abs(ac))
        
        # Try small perturbations around the peak location
        for _ in range(20):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            x_trial = best_x.copy()
            # Random perturbation
            x_trial += np.random.randn(N) * 0.01
            x_trial = x_trial / np.linalg.norm(x_trial)
            
            trial_score = _score(x_trial)
            if trial_score < best_score:
                best_score = trial_score
                best_x = x_trial.copy()
    
    # Final polish with high-precision BFGS on the true objective (numerical grad)
    if time.time() - start_time < DEADLINE - 5:
        from scipy.optimize import minimize
        
        def true_objective(x_flat):
            x_arr = x_flat.reshape(N)
            norm = np.linalg.norm(x_arr)
            if norm < 1e-10:
                return np.inf
            x_arr = x_arr / norm
            return _score(x_arr)
        
        try:
            result = minimize(
                true_objective,
                best_x.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            x_opt = result.x.reshape(N)
            x_opt = x_opt / np.linalg.norm(x_opt)
            opt_score = _score(x_opt)
            if opt_score < best_score:
                best_x = x_opt.copy()
                best_score = opt_score
        except Exception:
            pass
    
    return best_x

# Keep the original function signature
def construct():
    """
    Construct a sequence of signed step heights minimizing max_abs_autoconv / squared_integral.
    Returns a numpy array of length N.
    """
    start_time = time.time()
    
    np.random.seed(42)
    x = np.random.choice([-1.0, 1.0], size=N)
    
    best_x = x.copy()
    best_score = _score(x)
    
    beta_schedule = [50.0, 100.0, 200.0, 400.0, 800.0]
    
    from scipy.optimize import minimize
    
    for beta in beta_schedule:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        for init_idx in range(3):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            if init_idx == 0:
                x_curr = best_x.copy()
            elif init_idx == 1:
                x_curr = np.random.choice([-1.0, 1.0], size=N)
                x_curr = x_curr / np.linalg.norm(x_curr)
            else:
                x_curr = best_x.copy()
                noise = np.random.randn(N) * 0.1
                x_curr = x_curr + noise
                x_curr = x_curr / np.linalg.norm(x_curr)
            
            x_curr = x_curr / np.linalg.norm(x_curr)
            
            def objective(x_flat, beta=beta):
                x_arr = x_flat.reshape(N)
                norm = np.linalg.norm(x_arr)
                if norm < 1e-10:
                    return np.inf
                x_arr = x_arr / norm
                return _soft_score(x_arr, beta)
            
            def gradient(x_flat, beta=beta):
                x_arr = x_flat.reshape(N)
                norm = np.linalg.norm(x_arr)
                if norm < 1e-10:
                    return np.zeros(N)
                x_arr = x_arr / norm
                grad = _soft_score_grad(x_arr, beta)
                grad = grad - np.dot(grad, x_arr) * x_arr
                return grad.flatten()
            
            try:
                result = minimize(
                    objective,
                    x_curr.flatten(),
                    jac=gradient,
                    method='BFGS',
                    options={'maxiter': 200, 'gtol': 1e-6}
                )
                
                x_opt = result.x.reshape(N)
                x_opt = x_opt / np.linalg.norm(x_opt)
                
                true_score = _score(x_opt)
                
                if true_score < best_score:
                    best_score = true_score
                    best_x = x_opt.copy()
                    
            except Exception:
                pass
        
        if time.time() - start_time > DEADLINE - 15:
            break
        
        X = np.fft.fft(best_x)
        ac = np.fft.ifft(X * X)
        ac = np.real(ac)
        
        for _ in range(20):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            x_trial = best_x.copy()
            x_trial += np.random.randn(N) * 0.01
            x_trial = x_trial / np.linalg.norm(x_trial)
            
            trial_score = _score(x_trial)
            if trial_score < best_score:
                best_score = trial_score
                best_x = x_trial.copy()
    
    if time.time() - start_time < DEADLINE - 5:
        from scipy.optimize import minimize
        
        def true_objective(x_flat):
            x_arr = x_flat.reshape(N)
            norm = np.linalg.norm(x_arr)
            if norm < 1e-10:
                return np.inf
            x_arr = x_arr / norm
            return _score(x_arr)
        
        try:
            result = minimize(
                true_objective,
                best_x.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            x_opt = result.x.reshape(N)
            x_opt = x_opt / np.linalg.norm(x_opt)
            opt_score = _score(x_opt)
            if opt_score < best_score:
                best_x = x_opt.copy()
                best_score = opt_score
        except Exception:
            pass
    
    return best_x
