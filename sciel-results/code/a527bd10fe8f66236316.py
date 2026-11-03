"""Optimized program combining analytic exactness with global search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize
import math

def _compute_bound(c0, c4, c8):
    """
    Compute the uncertainty product bound for the given coefficients.
    c12 is determined by c0, c4, c8 to satisfy g(0)=0.
    Returns the product sqrt(<x^2> * <xi^2>).
    """
    # Hermite values at 0: H0(0)=1, H4(0)=12, H8(0)=168, H12(0)=1330
    # Wait, let's verify H4(0). H4(x) = 16x^4 - 48x^2 + 12. H4(0) = 12. Correct.
    # H8(x) = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168. H8(0) = 168. Correct.
    # H12(x) = 4096x^12 - ... + 1330. H12(0) = 1330. Correct.
    
    c12 = -(c0 + 12*c4 + 168*c8) / 1330.0
    
    # Coefficients of the polynomial P(x) = c0 H0 + c4 H4 + c8 H8 + c12 H12
    # P(x) = a12 x^12 + a10 x^10 + a8 x^8 + a6 x^6 + a4 x^4 + a2 x^2 + a0
    # a0 = c0 + 12c4 + 168c8 + 1330c12 = 0
    a0 = 0.0
    a2 = -48*c4 - 1344*c8 - 32256*c12
    a4 = 16*c4 + 2688*c8 + 133056*c12
    a6 = -1344*c8 - 180224*c12
    a8 = 256*c8 + 118272*c12
    a10 = -36864*c12
    a12 = 4096*c12
    
    p = np.array([a0, a2, a4, a6, a8, a10, a12])
    
    # Convolution to get coefficients of P(x)^2
    # q[k] is coeff of x^{2k} in P(x)^2
    q = np.convolve(p, p)
    # q has length 13, indices 0 to 12 corresponding to x^0 to x^24
    
    # Integrate x^{2k} e^{-x^2}
    # \int_{-inf}^{inf} x^{2k} e^{-x^2} dx = \sqrt{\pi} (2k-1)!! = \sqrt{\pi} 2^k k!
    factors = np.zeros(13)
    for k in range(13):
        factors[k] = math.sqrt(math.pi) * (2**k) * math.factorial(k)
        
    I = q * factors
    # I[k] = \int x^{2k} P(x)^2 e^{-x^2} dx
    
    norm_sq = I[0]
    if norm_sq <= 1e-15:
        return 1e10 # Invalid
    
    # Variance <x^2> = I[1] / I[0]
    x2 = I[1] / I[0]
    
    # Calculate g''(x) to get <xi^2>
    # g(x) = P(x) e^{-x^2/2}
    # g''(x) = (P'' - 2xP' + (x^2 - 1)P) e^{-x^2/2}
    # Let R(x) = P'' - 2xP' + (x^2 - 1)P
    # R(x) is even. We need \int R(x)^2 e^{-x^2} dx.
    
    # Construct R(x) coefficients r_j for x^{2j}
    # P(x) = \sum_{i=0}^6 p_i x^{2i}
    # P'(x) = \sum_{i=1}^6 2i p_i x^{2i-1}
    # P''(x) = \sum_{i=1}^6 2i(2i-1) p_i x^{2i-2}
    # -2xP'(x) = \sum_{i=1}^6 -4i p_i x^{2i}
    # (x^2-1)P(x) = \sum_{i=0}^6 p_i x^{2i+2} - \sum_{i=0}^6 p_i x^{2i}
    
    r = np.zeros(8) # indices 0..7
    for j in range(8):
        val = 0.0
        # From P'': term is 2(j+1)(2j+1) p_{j+1} (for j=0..5)
        if j <= 5:
            val += 2*(j+1)*(2*j+1)*p[j+1]
        # From -2xP': term is -4j p_j (for j=1..6)
        if 1 <= j <= 6:
            val += -4*j*p[j]
        # From x^2 P: term is p_{j-1} (for j=1..7)
        if 1 <= j <= 7:
            val += p[j-1]
        # From -P: term is -p_j (for j=0..6)
        if j <= 6:
            val += -p[j]
        r[j] = val
        
    # Compute \int R(x)^2 e^{-x^2} dx
    s = np.convolve(r, r)
    # s has length 15, indices 0 to 14
    
    factors_R = np.zeros(15)
    for k in range(15):
        factors_R[k] = math.sqrt(math.pi) * (2**k) * math.factorial(k)
        
    I_R = s * factors_R
    norm_R_sq = I_R[0]
    
    # <xi^2> = (1/4\pi^2) * \int |g''|^2 dx / \int |g|^2 dx
    xi2 = norm_R_sq / (4 * math.pi**2 * norm_sq)
    
    # Product
    product = math.sqrt(x2 * xi2)
    
    return product

def construct():
    start_time = time.time()
    deadline = start_time + 180.0  # 180s limit per task contract
    
    # Initial guess: Seed values
    seed_coeffs = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    best_val = _compute_bound(seed_coeffs[0], seed_coeffs[1], seed_coeffs[2])
    best_coeffs = seed_coeffs.copy()
    
    def _objective(x):
        return _compute_bound(x[0], x[1], x[2])
    
    # 1. Local Optimization from Seed
    try:
        res_local = scipy.optimize.minimize(
            _objective,
            seed_coeffs,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if res_local.success and res_local.fun < best_val:
            best_val = res_local.fun
            best_coeffs = res_local.x.copy()
    except Exception:
        pass
    
    # 2. Global Search using Differential Evolution
    # Bounds are set wide enough to explore the space but constrained by the problem nature
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    if time.time() < deadline - 30.0:
        try:
            # DE is stochastic, so we use a fixed seed for reproducibility if needed, 
            # but generally it explores well.
            de_res = scipy.optimize.differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=100,
                tol=1e-8,
                popsize=15,
                polish=False # We will polish manually
            )
            
            if de_res.success or (de_res.fun < best_val):
                candidate = de_res.x
                val_candidate = de_res.fun
                if val_candidate < best_val:
                    best_val = val_candidate
                    best_coeffs = candidate.copy()
        except Exception:
            pass
    
    # 3. Multi-start Local Optimization
    # Try a few random restarts to ensure we haven't missed a better local minimum
    if time.time() < deadline - 10.0:
        np.random.seed(123)
        for _ in range(10):
            if time.time() > deadline - 5.0:
                break
            # Start near the current best with some noise
            noise = np.random.normal(0, 0.5, 3)
            start_point = best_coeffs + noise
            try:
                res_restart = scipy.optimize.minimize(
                    _objective,
                    start_point,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12}
                )
                if res_restart.success and res_restart.fun < best_val:
                    best_val = res_restart.fun
                    best_coeffs = res_restart.x.copy()
            except Exception:
                pass

    # 4. Final High-Precision Polish
    if time.time() < deadline - 2.0:
        try:
            res_final = scipy.optimize.minimize(
                _objective,
                best_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            if res_final.success and res_final.fun < best_val:
                best_val = res_final.fun
                best_coeffs = res_final.x.copy()
        except Exception:
            pass

    return best_coeffs
# EVOLVE-BLOCK-END
