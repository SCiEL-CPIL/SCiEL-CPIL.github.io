"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400
import time
import numpy as np

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 140.0  # Kill at 150s, use 140s as safety margin
    
    def autoconv_energy(heights):
        """Compute the L2 norm of the autoconvolution (as a proxy for C3)."""
        n_fft = 2 * N
        f_ext = np.zeros(n_fft)
        f_ext[:N] = heights
        F = np.fft.rfft(f_ext)
        power = F * np.conj(F)
        autocorr = np.fft.irfft(power, n=n_fft)[:N]
        # Minimize sum of squares of non-zero lags
        return np.sum(autocorr[1:N] ** 2)
    
    def autoconv_energy_grad(heights):
        """Compute energy and gradient using FFT."""
        n_fft = 2 * N
        f_ext = np.zeros(n_fft)
        f_ext[:N] = heights
        F = np.fft.rfft(f_ext)
        power = F * np.conj(F)
        autocorr = np.fft.irfft(power, n=n_fft)[:N]
        energy = np.sum(autocorr[1:N] ** 2)
        
        # Gradient: d/dh_i sum_j autocorr[j]^2 for j>=1
        # autocorr[j] = sum_i f[i] f[i-j]
        # d(autocorr[j])/dh_i = f[i-j] + f[i+j] (for valid indices)
        # So gradient[i] = 2 * sum_{j=1}^{N-1} autocorr[j] * (f[i-j] + f[i+j])
        
        # Use circular convolution properties
        # autocorr[j] for j=1..N-1
        # gradient[i] = 2 * sum_{j=1}^{N-1} autocorr[j] * (h[i-j mod N] + h[i+j mod N])
        
        # This is equivalent to 2 * (autocorr_circ * h)[i] where autocorr_circ is circular
        # but we need to be careful about the range
        
        # Simpler: just compute directly for the gradient
        grad = np.zeros(N)
        for j in range(1, N):
            ac = autocorr[j]
            if ac == 0:
                continue
            # d(autocorr[j])/dh[i] = h[i-j] if i>=j else h[i+N-j] (circular)
            # Actually for linear autocorr with zero padding:
            # autocorr[j] = sum_{i=0}^{N-1-j} h[i] * h[i+j]
            # d/dh[i] = h[i-j] (if i>=j) + h[i+j] (if i+j < N)
            # So grad[i] += 2 * autocorr[j] * (h[i-j] if i>=j else 0) + 2 * autocorr[j] * (h[i+j] if i+j<N else 0)
            
            # Vectorized:
            # For i from j to N-1: grad[i] += 2*ac*h[i-j]
            # For i from 0 to N-1-j: grad[i] += 2*ac*h[i+j]
            
            grad[j:] += 2 * ac * h[:N-j]
            grad[:N-j] += 2 * ac * h[j:]
        
        return energy, grad
    
    # Better gradient using FFT-based circular correlation
    def autoconv_energy_grad_fast(heights):
        """Compute energy and gradient using FFT."""
        n_fft = 2 * N
        f_ext = np.zeros(n_fft)
        f_ext[:N] = heights
        F = np.fft.rfft(f_ext)
        power = F * np.conj(F)
        autocorr = np.fft.irfft(power, n=n_fft)[:N]
        energy = np.sum(autocorr[1:N] ** 2)
        
        # Gradient via circular correlation
        # For linear autocorrelation, the gradient at position i is:
        # 2 * sum_{j=1}^{N-1} autocorr[j] * (h[i-j] + h[i+j]) with appropriate bounds
        # 
        # Using FFT: create autocorr_ext of length n_fft with autocorr[1:N] at positions 1:N
        # and autocorr[N:2N-1] = autocorr[N-1:0:-1] (for negative lags)
        # Then grad = 2 * ifft(fft(autocorr_ext) * conj(fft(h_ext)))[:N]
        
        autocorr_ext = np.zeros(n_fft)
        autocorr_ext[1:N] = autocorr[1:N]
        autocorr_ext[N+1:2*N] = autocorr[N-1:0:-1]  # negative lags
        
        A = np.fft.rfft(autocorr_ext)
        H = np.fft.rfft(f_ext)
        grad_full = np.fft.irfft(A * np.conj(H), n=n_fft)[:N]
        grad = 2 * grad_full
        
        return energy, grad
    
    best_result = None
    best_score = float('inf')
    
    rng = np.random.RandomState(42)
    
    def time_check():
        return time.time() > deadline
    
    # Strategy: Multiple restarts with gradient-based optimization + simulated annealing
    num_restarts = 50
    max_sa_steps = 1000
    max_lbfgs_iter = 100
    
    for r in range(num_restarts):
        if time_check():
            break
            
        # Initialize with random sequence
        h = rng.choice([-1.0, 1.0], size=N)
        current_energy = autoconv_energy(h)
        
        if current_energy < best_score:
            best_score = current_energy
            best_result = h.copy()
        
        # Phase 1: Simulated Annealing
        T = 500.0
        T_min = 0.1
        cooling_rate = 0.999
        
        for step in range(max_sa_steps):
            if time_check():
                break
                
            i = rng.randint(N)
            h[i] *= -1
            new_energy = autoconv_energy(h)
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or (T > T_min and rng.rand() < np.exp(-delta_e / T)):
                current_energy = new_energy
                if current_energy < best_score:
                    best_score = current_energy
                    best_result = h.copy()
            else:
                h[i] *= -1
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
        # Phase 2: Gradient-based polish using L-BFGS
        if not time_check():
            # Use continuous relaxation, then round
            h_cont = h.copy()
            
            def obj_grad(x):
                x_signed = np.where(x > 0, 1.0, -1.0)
                e, g = autoconv_energy_grad_fast(x_signed)
                # Gradient with respect to x is 0 (since we're using sign)
                # But for L-BFGS on continuous variables, we need smooth objective
                # Use soft sign: x / (1 + |x|) approximation
                # Actually, let's just use the gradient of the energy w.r.t. signed values
                # and let L-BFGS handle it
                return e, g
            
            try:
                result = minimize(obj_grad, h_cont, method='L-BFGS-B', 
                                 jac=True, options={'maxiter': max_lbfgs_iter, 'gtol': 1e-6})
                if result.success and result.fun < best_score:
                    h_opt = np.where(result.x > 0, 1.0, -1.0)
                    e_opt = autoconv_energy(h_opt)
                    if e_opt < best_score:
                        best_score = e_opt
                        best_result = h_opt.copy()
            except Exception:
                pass
        
        # Phase 3: Greedy hill climbing with exact energy
        if not time_check():
            improved = True
            hc_steps = 0
            max_hc = 200
            while improved and not time_check() and hc_steps < max_hc:
                improved = False
                hc_steps += 1
                # Try flipping a random position
                i = rng.randint(N)
                h[i] *= -1
                new_energy = autoconv_energy(h)
                if new_energy < current_energy:
                    current_energy = new_energy
                    improved = True
                    if current_energy < best_score:
                        best_score = current_energy
                        best_result = h.copy()
                else:
                    h[i] *= -1
    
    if best_result is None:
        best_result = np.ones(N)
        
    return best_result
# EVOLVE-BLOCK-END
