import numpy as np
from scipy.optimize import minimize
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # 1. Coarse Optimization
    N_coarse = 60
    beta_coarse = 10.0
    
    # Initialize g for coarse grid
    # Start with a boxcar in the middle
    g_coarse = np.zeros(N_coarse)
    # Maybe a slight random perturbation to break symmetry
    g_coarse += np.random.normal(0, 0.1, N_coarse)
    
    def objective_coarse(g):
        f = np.exp(g)
        f /= np.sum(f)
        # Compute autoconvolution
        c = np.fft.ifft(np.fft.fft(f) * np.fft.fft(f))
        c = np.real(c)
        # Soft-max
        return np.log(np.sum(np.exp(beta_coarse * c)))
        
    def gradient_coarse(g):
        f = np.exp(g)
        f /= np.sum(f)
        c = np.fft.ifft(np.fft.fft(f) * np.fft.fft(f))
        c = np.real(c)
        # Gradient of log-sum-exp
        probs = np.exp(beta_coarse * c) / np.sum(np.exp(beta_coarse * c))
        # Gradient of c w.r.t f
        # c = f * f (convolution)
        # dc/df = f (reversed) ? 
        # c_k = sum_i f_i f_{k-i}
        # dc_k/df_j = f_{k-j} + f_{j-k}
        # We need dL/df_j = sum_k probs_k (f_{k-j} + f_{j-k})
        # This is 2 * (probs * f)_j (convolution)
        grad_c = 2.0 * np.real(np.fft.ifft(np.fft.fft(probs) * np.fft.fft(f)))
        # Gradient w.r.t g
        # f = exp(g) / S
        # df/ dg = f (I - 1 1^T f^T) ? No.
        # d f_i / d g_j = f_i (delta_ij - f_j)
        # dL/dg_j = sum_i (dL/df_i) (d f_i / d g_j)
        # = sum_i grad_c_i f_i (delta_ij - f_j)
        # = grad_c_j f_j - f_j sum_i grad_c_i f_i
        # = f_j (grad_c_j - sum_i grad_c_i f_i)
        mean_grad = np.dot(grad_c, f)
        grad_g = f * (grad_c - mean_grad)
        return grad_g

    # Optimize coarse
    res_coarse = minimize(objective_coarse, g_coarse, jac=gradient_coarse, method='L-BFGS-B', 
                          options={'maxiter': 100, 'ftol': 1e-8})
    
    if time.time() > deadline - 30:
         # Fallback
         f = np.exp(res_coarse.x)
         f /= np.sum(f)
         # Upsample
         f_full = np.repeat(f, N // N_coarse)
         return f_full.tolist()

    # 2. Upsample
    f_coarse_opt = np.exp(res_coarse.x)
    f_coarse_opt /= np.sum(f_coarse_opt)
    
    # Linear interpolation for upsample
    x_coarse = np.linspace(0, 1, N_coarse, endpoint=False)
    x_full = np.linspace(0, 1, N, endpoint=False)
    f_full_init = np.interp(x_full, x_coarse, f_coarse_opt)
    f_full_init /= np.sum(f_full_init)
    
    # 3. Fine Optimization
    beta_fine = 50.0
    
    g_full = np.log(f_full_init + 1e-15)
    
    def objective_full(g):
        f = np.exp(g)
        f /= np.sum(f)
        c = np.fft.ifft(np.fft.fft(f) * np.fft.fft(f))
        c = np.real(c)
        return np.log(np.sum(np.exp(beta_fine * c)))
        
    def gradient_full(g):
        f = np.exp(g)
        f /= np.sum(f)
        c = np.fft.ifft(np.fft.fft(f) * np.fft.fft(f))
        c = np.real(c)
        probs = np.exp(beta_fine * c) / np.sum(np.exp(beta_fine * c))
        grad_c = 2.0 * np.real(np.fft.ifft(np.fft.fft(probs) * np.fft.fft(f)))
        mean_grad = np.dot(grad_c, f)
        grad_g = f * (grad_c - mean_grad)
        return grad_g

    res_full = minimize(objective_full, g_full, jac=gradient_full, method='L-BFGS-B', 
                        options={'maxiter': 200, 'ftol': 1e-10})
    
    if time.time() > deadline - 10:
         f = np.exp(res_full.x)
         f /= np.sum(f)
         return f.tolist()

    # 4. Polish with higher beta
    beta_polish = 100.0
    g_polish = res_full.x
    
    def objective_polish(g):
        f = np.exp(g)
        f /= np.sum(f)
        c = np.fft.ifft(np.fft.fft(f) * np.fft.fft(f))
        c = np.real(c)
        return np.log(np.sum(np.exp(beta_polish * c)))
        
    def gradient_polish(g):
        f = np.exp(g)
        f /= np.sum(f)
        c = np.fft.ifft(np.fft.fft(f) * np.fft.fft(f))
        c = np.real(c)
        probs = np.exp(beta_polish * c) / np.sum(np.exp(beta_polish * c))
        grad_c = 2.0 * np.real(np.fft.ifft(np.fft.fft(probs) * np.fft.fft(f)))
        mean_grad = np.dot(grad_c, f)
        grad_g = f * (grad_c - mean_grad)
        return grad_g

    res_polish = minimize(objective_polish, g_polish, jac=gradient_polish, method='L-BFGS-B', 
                          options={'maxiter': 100, 'ftol': 1e-12})
    
    f_final = np.exp(res_polish.x)
    f_final /= np.sum(f_final)
    
    return f_final.tolist()
