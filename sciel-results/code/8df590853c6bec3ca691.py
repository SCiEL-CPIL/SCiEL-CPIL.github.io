import numpy as np
import time
from scipy.optimize import minimize
from scipy.fft import fft, ifft, fftfreq

N = 600
TIME_LIMIT = 130  # Safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def spectral_flattening_objective(f):
    """
    Objective: Minimize the L4 norm of the Fourier transform (sum of squared power spectrum).
    This encourages a flat power spectrum, which corresponds to a flat autocorrelation
    (in the ideal case), thereby minimizing the peak of the autoconvolution.
    """
    # f is a flat array of size N
    f_vec = f.reshape(N)
    # Ensure non-negativity is handled by the solver bounds, but clip for safety in FFT
    f_vec = np.maximum(f_vec, 0.0)
    
    # Compute DFT
    F = fft(f_vec)
    P = np.abs(F) ** 2
    
    # Objective is sum of P^2
    return np.sum(P ** 2)

def spectral_flattening_gradient(f):
    """
    Gradient of the spectral flattening objective.
    J = sum |F_k|^4
    dJ/df_j = 4 * Re[ ifft( |F|^2 * F ) ]_j
    Derivation:
    J = sum_k F_k F_k* F_k F_k* (using real f, so F_{-k} = F_k*)
    Actually, simpler: J = sum P_k^2.
    dJ = 2 sum P_k dP_k
    dP_k = 2 Re(F_k* dF_k)
    dF_k = (1/sqrt(N)) sum_j f_j e^{-i 2 pi k j / N} (depending on FFT convention)
    numpy fft: F_k = sum_j f_j e^{-i 2 pi k j / N}
    ifft: f_j = (1/N) sum_k F_k e^{i 2 pi k j / N}
    
    dJ/df_j = sum_k 2 P_k * 2 Re(F_k* dF_k/d f_j)
    dF_k/d f_j = e^{-i 2 pi k j / N}
    dJ/df_j = 4 sum_k P_k Re(F_k* e^{-i 2 pi k j / N})
    Note: F_k* e^{-i 2 pi k j / N} is the term in the inverse FFT of (P * F*)?
    Let G_k = P_k * F_k*.
    ifft(G)_j = (1/N) sum_k G_k e^{i 2 pi k j / N} = (1/N) sum_k P_k F_k* e^{i 2 pi k j / N}
    This is not quite the same phase.
    Let's check the gradient of sum |F_k|^4 w.r.t f.
    Using chain rule:
    grad_f = 4 * ifft( np.abs(F)**2 * np.conj(F) ).real ?
    Let's test numerically or use complex calculus.
    J = sum_k F_k^2 F_k*^2 ? No, F_k is complex.
    J = sum_k (F_k F_k*)^2 = sum_k F_k^2 F_k*^2.
    dJ/df_j = sum_k [ 2 F_k F_k*^2 dF_k/d f_j + 2 F_k* F_k^2 dF_k*/d f_j ]
    dF_k/d f_j = e^{-i 2 pi k j / N}
    dF_k*/d f_j = e^{i 2 pi k j / N}
    Term 1: sum_k 2 P_k F_k* e^{-i 2 pi k j / N}
    Term 2: sum_k 2 P_k F_k e^{i 2 pi k j / N}
    Sum = 2 sum_k P_k ( F_k* e^{-i ...} + F_k e^{i ...} )
    Note that ifft( P * F ) gives sum P F e^{i ...} / N.
    ifft( P * F* ) gives sum P F* e^{i ...} / N.
    So Term 1 is related to ifft(P * F*) but with e^{-i}.
    Actually, let's just use the property:
    grad = 4 * ifft( np.abs(F)**2 * F ).real
    Let's verify with a small example in code if possible, but for now assume this standard form.
    Wait, the standard gradient of ||F(f)||_4^4 is 4 * ifft( |F|^2 * F ).
    """
    f_vec = f.reshape(N)
    f_vec = np.maximum(f_vec, 0.0)
    F = fft(f_vec)
    P = np.abs(F) ** 2
    # Gradient in frequency domain
    grad_F = 4.0 * P * F
    # Transform back to time domain
    grad_f = ifft(grad_F).real
    return grad_f

def construct():
    start_time = time.time()
    
    # Initial guess: Constant function
    f0 = np.ones(N)
    
    # We want to minimize the spectral flatness objective subject to sum(f)=1 and f>=0.
    # We will use SLSQP to handle the equality constraint.
    
    def objective(f):
        return spectral_flattening_objective(f)
        
    def gradient(f):
        return spectral_flattening_gradient(f)
        
    # Constraint: sum(f) - 1 = 0
    constraints = [
        {'type': 'eq', 'fun': lambda f: np.sum(f) - 1.0, 'jac': lambda f: np.ones(N)}
    ]
    
    # Bounds: f >= 0
    bounds = [(0.0, None)] * N
    
    best_f = f0.copy()
    best_score = compute_score(best_f)
    
    try:
        # Run optimization
        res = minimize(
            objective, 
            f0, 
            method='SLSQP', 
            jac=gradient, 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-9, 'disp': False}
        )
        
        if res.success:
            f_opt = res.x
            f_opt = np.maximum(f_opt, 0.0)
            # Normalize to ensure sum is exactly 1 (SLSQP might have small errors)
            s = np.sum(f_opt)
            if s > 1e-15:
                f_opt = f_opt / s
            
            score_opt = compute_score(f_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_f = f_opt
        else:
            # If SLSQP fails, try L-BFGS-B with a penalty method or just use the result if valid
            if hasattr(res, 'x'):
                f_opt = res.x
                f_opt = np.maximum(f_opt, 0.0)
                s = np.sum(f_opt)
                if s > 1e-15:
                    f_opt = f_opt / s
                score_opt = compute_score(f_opt)
                if score_opt < best_score:
                    best_score = score_opt
                    best_f = f_opt

    except Exception:
        pass
    
    # If time remains, we can try to polish the result by directly minimizing the max autoconvolution
    # using the spectral result as a good starting point.
    if time.time() - start_time < TIME_LIMIT - 10:
        try:
            # Use a smooth max approximation on the time domain for polishing
            beta = 50.0
            def smooth_max_obj(f):
                f = np.maximum(f, 0.0)
                s = np.sum(f)
                if s < 1e-15: return 1e10
                conv = np.convolve(f, f, mode='full')
                z = conv / (s * s)
                z_max = np.max(z)
                if z_max < 1e-15: return 0.0
                exp_z = np.exp(beta * (z - z_max))
                return z_max + np.log(np.sum(exp_z)) / beta

            def smooth_max_grad(f):
                f = np.maximum(f, 0.0)
                s = np.sum(f)
                if s < 1e-15: return np.zeros(N)
                conv = np.convolve(f, f, mode='full')
                z = conv / (s * s)
                z_max = np.max(z)
                if z_max < 1e-15: return np.zeros(N)
                exp_z = np.exp(beta * (z - z_max))
                p = exp_z / np.sum(exp_z)
                
                # Gradient of LSE w.r.t f
                # Reuse logic from seed but simplified
                c1 = np.sum(p * conv)
                S = np.zeros(N)
                for j in range(N):
                    i_start = max(0, j)
                    i_end = min(2*N-1, N+j)
                    if i_end > i_start:
                        S[j] = np.dot(p[i_start:i_end], f[i_start-j:i_end-j])
                grad = (2.0 / (s**3)) * (s * S - c1)
                return grad

            # Run a few steps of gradient descent with projection
            f_curr = best_f.copy()
            lr = 0.01
            for _ in range(50):
                if time.time() - start_time > TIME_LIMIT - 5:
                    break
                g = smooth_max_grad(f_curr)
                f_curr = f_curr - lr * g
                f_curr = np.maximum(f_curr, 0.0)
                s = np.sum(f_curr)
                if s < 1e-15: break
                f_curr = f_curr / s
                score_curr = compute_score(f_curr)
                if score_curr < best_score:
                    best_score = score_curr
                    best_f = f_curr.copy()
        except Exception:
            pass

    # Ensure nonnegative and return
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
