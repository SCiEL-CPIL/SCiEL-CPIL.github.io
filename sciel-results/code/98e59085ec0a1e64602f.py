import numpy as np
import scipy.special
import time
from scipy.optimize import minimize

def construct():
    # Deadline guard: 630s limit, use 600s as safety margin
    start_time = time.time()
    deadline = start_time + 580.0  # 20s safety margin

    def time_remaining():
        return deadline - time.time()

    try:
        m, n = 12, 1
        
        # Initial guess: Seed [4, 8, ..., 48]
        # This is known to score 1.0.
        # We will try to improve it by minimizing the uncertainty product.
        
        def compute_bound(roots):
            """
            Computes a proxy for the uncertainty bound C4.
            Uses the function f(x) = exp(-x^2/4) * prod(x^2 - r_i).
            Minimizes sigma_x * sigma_k.
            """
            # Ensure roots are positive and sorted
            roots = np.sort(np.asarray(roots, dtype=float))
            if np.any(roots <= 0):
                return 1e10
            
            # Construct the polynomial Q(y) = prod(y - r_i)
            # We need to integrate |f(x)|^2 and x^2 |f(x)|^2
            # f(x) = exp(-x^2/4) * Q(x^2)
            # |f(x)|^2 = exp(-x^2/2) * Q(x^2)^2
            
            # Use Gauss-Laguerre quadrature?
            # The weight is exp(-x^2/2). This is not standard Laguerre weight exp(-x).
            # We can change variable u = x^2/2. x = sqrt(2u). dx = du / sqrt(2u).
            # exp(-x^2/2) dx = exp(-u) du / sqrt(2u).
            # This is a generalized Laguerre weight with alpha = -0.5.
            # scipy.special.roots_genlaguerre supports alpha > -1.
            
            # Let's use numerical integration for robustness, it's safer for arbitrary roots.
            # Range: Roots go up to ~50. x^2 goes up to 2500.
            # exp(-x^2/4) decays fast. x > 10 is negligible.
            # Let's integrate x from 0 to 20.
            
            x = np.linspace(0, 20, 2000)
            dx = x[1] - x[0]
            
            # Compute Q(x^2)
            y = x**2
            # Q(y) = prod(y - r_i)
            # To avoid overflow/underflow, compute log or scale
            # log|Q| = sum(log|y - r_i|)
            # But we need the sign and magnitude for |f|^2.
            # |f|^2 = exp(-x^2/2) * Q(x^2)^2
            
            # Compute Q(x^2)
            # Use numpy polynomial evaluation for stability?
            # coeffs = np.poly(roots) -> P(x) = x^12 - ...
            # Q(y) is the same polynomial in y.
            try:
                coeffs = np.poly(roots)
                # Evaluate Q at y
                Q_vals = np.polyval(coeffs, y)
            except Exception:
                return 1e10
            
            f_sq = np.exp(-x**2 / 2.0) * Q_vals**2
            
            # Normalize
            norm_sq = np.trapz(f_sq, x)
            if norm_sq < 1e-15:
                return 1e10
            
            # sigma_x^2
            x2_f_sq = x**2 * f_sq
            mean_x2 = np.trapz(x2_f_sq, x) / norm_sq
            
            # For momentum sigma_k, we need the Fourier transform.
            # Approximation: Parseval's theorem or direct FFT.
            # Direct FFT on the grid.
            # f(x) = exp(-x^2/4) * Q(x^2)
            # Note: f(x) is not necessarily real positive, Q can be negative.
            # But |f|^2 is what we used. For sigma_k, we need |hat{f}(k)|^2.
            # Let's compute hat{f} via FFT.
            
            # Use a finer grid for FFT to get good k resolution
            N = 4096
            x_fft = np.linspace(0, 40, N)
            dx_fft = x_fft[1] - x_fft[0]
            y_fft = x_fft**2
            Q_fft = np.polyval(coeffs, y_fft)
            f_fft = np.exp(-x_fft**2 / 4.0) * Q_fft
            
            # FFT
            # Shift to center? The function is defined on x>0.
            # We can treat it as a function on R by extending it to x<0?
            # The uncertainty principle usually applies to functions on R.
            # If f is defined on R+ and extended to R-, it must be even or odd?
            # Cohn-Gonçalves constructions often use even functions.
            # Let's assume f(x) is even: f(x) = f(-x).
            # Then hat{f} is real (if f is real and even).
            
            # Construct even extension on [-20, 20]
            x_even = np.linspace(-20, 20, N)
            x_abs = np.abs(x_even)
            y_even = x_abs**2
            Q_even = np.polyval(coeffs, y_even)
            f_even = np.exp(-x_even**2 / 4.0) * Q_even
            
            F = np.fft.fftshift(np.fft.fft(f_even))
            k = np.fft.fftshift(np.fft.fftfreq(N, dx=x_even[1]-x_even[0]))
            
            F_sq = np.abs(F)**2
            norm_k = np.trapz(F_sq, k)
            if norm_k < 1e-15:
                return 1e10
            
            k2_F_sq = k**2 * F_sq
            mean_k2 = np.trapz(k2_F_sq, k) / norm_k
            
            # Uncertainty product
            # sigma_x = sqrt(mean_x2)
            # sigma_k = sqrt(mean_k2)
            # Note: The exact constants depend on the definition of Fourier transform.
            # We just need a monotonic proxy.
            bound = np.sqrt(mean_x2) * np.sqrt(mean_k2)
            
            return bound

        # Optimization
        # Start from seed
        roots_init = np.array([4.0 * (i + 1) for i in range(12)])
        
        # Use Nelder-Mead or L-BFGS-B
        # L-BFGS-B requires bounds. Roots must be > 0.
        # Bounds: [0.1, 100.0] for each root.
        
        res = minimize(
            compute_bound,
            roots_init,
            method='L-BFGS-B',
            bounds=[(0.1, 100.0)] * 12,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if res.success and res.fun < compute_bound(roots_init):
            best_roots = res.x
        else:
            # Try Nelder-Mead if L-BFGS-B failed or didn't improve
            res_nm = minimize(
                compute_bound,
                roots_init,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res_nm.fun < compute_bound(roots_init):
                best_roots = res_nm.x
            else:
                best_roots = roots_init

        # Ensure roots are positive
        best_roots = np.maximum(best_roots, 1e-6)
        
        # Sort roots
        best_roots = np.sort(best_roots)
        
        # Convert to list
        root_list = [float(r) for r in best_roots]
        
        # Verify 12 roots
        if len(root_list) != 12:
            return [4.0 * (i + 1) for i in range(12)]
            
        return root_list

    except Exception as e:
        # Fallback to seed
        return [4.0 * (i + 1) for i in range(12)]
