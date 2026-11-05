import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.fft import fft, ifft, fftshift, ifftshift

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    def compute_value(h):
        """Exact evaluation of the objective function."""
        s = np.sum(h)
        if abs(s) < 1e-12:
            return float('inf')
        if not np.all(np.isfinite(h)):
            return float('inf')
        z = np.convolve(h, h)
        return abs(2 * N * np.max(z) / (s * s))

    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Spectral Representation:
    # h is real-valued. We represent it by its DFT coefficients H_k.
    # H_0 is real (controls mean/sum). H_k = conj(H_{N-k}) for k=1..N/2-1.
    # H_{N/2} is real.
    # We parameterize H_k (k=1..N/2) as complex numbers with parameters (r_k, theta_k).
    # H_k = r_k * exp(i * theta_k).
    # To ensure h is real, we set H_{N-k} = conj(H_k).
    
    # Parameters:
    # c[0] = H_0 (real)
    # c[1:N//2] = r_k (magnitudes, k=1..N/2-1)
    # c[N//2] = H_{N/2} (real)
    # c[N//2+1 : N//2+1+N//2-1] = theta_k (phases, k=1..N/2-1)
    
    dim = 1 + (N//2 - 1) + 1 + (N//2 - 1)
    
    def coeffs_to_h(c):
        h = np.zeros(N)
        H = np.zeros(N, dtype=complex)
        H[0] = c[0]
        H[N//2] = c[N//2]
        
        # r_k and theta_k for k=1..N/2-1
        # Indices in c:
        # r: 1 to N//2-1
        # theta: N//2+1 to N//2+1+N//2-2
        
        for k in range(1, N//2):
            r = c[k]
            theta = c[N//2 + 1 + k - 1]
            H[k] = r * np.exp(1j * theta)
            H[N-k] = np.conj(H[k])
        
        h = ifft(H).real
        return h

    def objective(c):
        if not np.all(np.isfinite(c)):
            return float('inf')
        h = coeffs_to_h(c)
        s = np.sum(h)
        if abs(s) < 1e-9:
            return float('inf')
        if not np.all(np.isfinite(h)):
            return float('inf')
        z = np.convolve(h, h)
        return abs(2 * N * np.max(z) / (s * s))

    # Global Search: Differential Evolution in Spectral Coefficient Space
    # Bounds:
    # H_0: [0.1, 10.0] (ensure non-zero sum)
    # r_k: [0.0, 5.0]
    # H_{N/2}: [-5.0, 5.0]
    # theta_k: [-pi, pi]
    
    bounds = [(0.1, 10.0)] + [(0.0, 5.0)] * (N//2 - 1) + [(-5.0, 5.0)] + [(-np.pi, np.pi)] * (N//2 - 1)
    
    try:
        res_de = differential_evolution(
            objective, 
            bounds, 
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-4,
            polish=False
        )
        
        if res_de.fun < best_val:
            h_de = coeffs_to_h(res_de.x)
            if np.all(np.isfinite(h_de)) and abs(np.sum(h_de)) > 1e-9:
                best_val = res_de.fun
                best_h = h_de
    except Exception:
        pass

    # Local Polish: Multi-start L-BFGS-B
    # We need a smooth objective for gradient-based methods.
    # max(z) is non-smooth. We use a smooth approximation: soft-max.
    # soft_max(z) = tau * log(sum(exp(z/tau)))
    # This is smooth and close to max for small tau.
    
    def smooth_objective(c):
        if not np.all(np.isfinite(c)):
            return float('inf')
        h = coeffs_to_h(c)
        s = np.sum(h)
        if abs(s) < 1e-9:
            return float('inf')
        if not np.all(np.isfinite(h)):
            return float('inf')
        z = np.convolve(h, h)
        
        # Soft max
        tau = 1.0
        z_max = np.max(z)
        # Avoid overflow
        if z_max > 50:
             z_shifted = z - z_max
             log_sum_exp = z_max + tau * np.log(np.sum(np.exp(z_shifted / tau)))
        else:
             log_sum_exp = tau * np.log(np.sum(np.exp(z / tau)))
        
        val = abs(2 * N * log_sum_exp / (s * s))
        return val

    # Starting points for local polish
    starts = []
    
    # 1. Result from DE
    if 'res_de' in locals() and hasattr(res_de, 'x'):
        starts.append(res_de.x)
    
    # 2. Random spectral starts
    rng = np.random.default_rng(42)
    for i in range(10):
        if time.time() > deadline - 15:
            break
        c_rand = np.zeros(dim)
        c_rand[0] = rng.uniform(1.0, 5.0)
        c_rand[1:N//2] = rng.uniform(0.1, 2.0, size=N//2-1)
        c_rand[N//2] = rng.uniform(-2.0, 2.0)
        c_rand[N//2+1:] = rng.uniform(-np.pi, np.pi, size=N//2-1)
        starts.append(c_rand)

    for c_start in starts:
        if time.time() > deadline - 10:
            break
        try:
            res_loc = minimize(
                smooth_objective,
                c_start,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if res_loc.fun < best_val:
                h_loc = coeffs_to_h(res_loc.x)
                if np.all(np.isfinite(h_loc)) and abs(np.sum(h_loc)) > 1e-9:
                    val_loc = compute_value(h_loc)
                    if val_loc < best_val:
                        best_val = val_loc
                        best_h = h_loc
        except Exception:
            pass

    # Final check
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
        
    return best_h

if __name__ == "__main__":
    h = construct()
    print(len(h))
    print(np.sum(h))
    z = np.convolve(h, h)
    print(abs(2 * N * np.max(z) / np.sum(h)**2))
