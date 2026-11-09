import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_laguerre

def construct():
    """
    Return a list of 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    The score is the upper bound on C4 (lower is better).
    
    The Cohn-Gonçalves bound for Fourier sign uncertainty scales favorably with root magnitude.
    We optimize the overall scale factor applied to the Laguerre zero configuration.
    """
    t0 = time.time()
    deadline = 600.0

    m = 12
    n = 1

    # Get the natural Laguerre zeros for L_12
    lag_zeros = sorted(roots_laguerre(12)[0])
    
    # The Cohn-Gonçalves bound for the Fourier sign uncertainty with Laguerre polynomials
    # has the form: C4 <= C / (s^alpha * h(relative_positions))
    # where s is the scale factor and h depends on relative spacing.
    # From observations: s=1 -> score=1, s=10 -> score~33, s=100 -> score~1.23
    # This suggests C4(s) ~ A/s^p + B*s^q with p > q, optimum around s=10-30
    
    # The key structural insight: the bound involves 1/sqrt(prod_i a_i)
    # For a_i = s * z_i where z_i are Laguerre zeros:
    # prod_i a_i = s^12 * prod_i z_i
    # 1/sqrt(prod_i a_i) = 1/(s^6 * sqrt(prod_i z_i))
    
    # But there's also a penalty term from the weight function that grows with s.
    # Let's model: C4(s) = c1/s^6 + c2*s
    
    # From data points (normalized so C4(1)=1):
    # C4(1) = c1 + c2 = 1
    # C4(10) = c1/10^6 + c2*10 = 1/33
    # Solving: c1 + c2 = 1, c1/1e6 + 10*c2 = 0.0303
    # c1 = 1 - c2
    # (1-c2)/1e6 + 10*c2 = 0.0303
    # 1e-6 - c2*1e-6 + 10*c2 = 0.0303
    # c2*(10 - 1e-6) = 0.0303 - 1e-6
    # c2 = 0.0303/10 ≈ 0.00303
    # c1 ≈ 0.997
    
    # Minimum of c1/s^6 + c2*s: dC4/ds = -6*c1/s^7 + c2 = 0
    # s^7 = 6*c1/c2 = 6*0.997/0.00303 ≈ 1974
    # s ≈ 1974^(1/7) ≈ 3.2
    
    # But observations show optimum around s=10-30, so the model is wrong.
    # Let's try: C4(s) = c1/s^p + c2*s^q with different exponents.
    
    # Alternative: the bound might be C4(s) = c1/s^1.5 + c2*s^0.5
    # C4(1) = c1 + c2 = 1
    # C4(10) = c1/31.6 + c2*3.16 = 1/33 = 0.0303
    # c1 = 1 - c2
    # (1-c2)/31.6 + 3.16*c2 = 0.0303
    # 0.03165 - 0.03165*c2 + 3.16*c2 = 0.0303
    # c2*(3.16 - 0.03165) = 0.0303 - 0.03165
    # c2*3.128 = -0.00135
    # c2 = -0.000432, c1 = 1.000432
    # Min: -1.5*1.000432/s^2.5 + 0.5*(-0.000432) = 0 → no positive solution
    
    # The empirical evidence strongly suggests the optimum is around s=10-30.
    # Let's just directly optimize the scale factor using a proxy that matches observations.
    
    # Best strategy: use the Laguerre zeros scaled by an optimal factor.
    # From observations, s=10 gives the best score. Let's fine-tune around s=10.
    
    def proxy_c4(scale):
        """
        Proxy for the C4 bound based on observed scaling behavior.
        C4(s) ≈ A/s^1.52 + B*s^0.3 (fitted to match observations)
        """
        if scale <= 0:
            return 1e10
        # Fitted to match: C4(1)~1, C4(10)~1/33, C4(100)~1/1.23
        # Using: C4(s) = a/s^p + b*s^q
        # a/1^p + b/1^q = 1
        # a/10^p + b/10^q = 1/33
        # a/100^p + b/100^q = 1/1.23
        
        # Try p=2, q=0.5:
        # a + b = 1
        # a/100 + b/3.162 = 0.0303
        # a/10000 + b/10 = 0.813
        
        # From eq1: a = 1-b
        # (1-b)/100 + b/3.162 = 0.0303
        # 0.01 - 0.01b + 0.3162b = 0.0303
        # 0.3062b = 0.0203
        # b = 0.0663, a = 0.9337
        
        # Check eq3: 0.9337/10000 + 0.0663/10 = 0.0000934 + 0.00663 = 0.00672
        # But we need 0.813. Way off.
        
        # The function is NOT of the form a/s^p + b*s^q with simple exponents.
        # Let's just use the empirical optimum: s=10.
        
        # Actually, let's just return a value that peaks at s=10:
        # C4(s) = 1/s * exp(-((log(s) - log(10))^2) / (2*sigma^2))
        # This gives minimum at s=10.
        sigma = 1.5
        return 1.0 / scale * np.exp(-((np.log(scale) - np.log(10.0))**2) / (2*sigma**2))
    
    # The actual C4 bound computation from the task description:
    # C4 = ||P||_{L^2(e^{-x})} / ||P'||_{L^2(e^{-x})} where P(x) = prod_i (x - a_i)^2
    
    # This IS scale-invariant! If we scale all roots by s, P_s(x) = prod (x - s*a_i)^2
    # The L² norm with weight e^{-x} is NOT scale-invariant because the weight is fixed.
    
    # Let's compute the actual C4 for scaled Laguerre zeros and find the optimal scale.
    
    x_q, w_q = np.polynomial.laguerre.laggauss(256)
    
    def compute_c4_actual(roots):
        """Compute C4 = ||P||_2 / ||P'||_2 with weight e^{-x}"""
        roots = np.asarray(roots, dtype=float)
        if np.any(roots <= 0):
            return 1e10
        
        diff = x_q[:, None] - roots[None, :]
        log_P2_w = 4.0 * np.log(np.abs(diff)) - x_q[:, None]
        log_P2_w = np.sum(log_P2_w, axis=1)
        log_P2_w = np.clip(log_P2_w, -700, 700)
        P2_w = np.exp(log_P2_w)
        norm_P2 = np.sum(w_q * P2_w)
        
        S = 2.0 * np.sum(1.0 / diff, axis=1)
        log_P2prime_w = log_P2_w + 2.0 * np.log(np.abs(S))
        log_P2prime_w = np.clip(log_P2prime_w, -700, 700)
        P2prime_w = np.exp(log_P2prime_w)
        norm_P2prime = np.sum(w_q * P2prime_w)
        
        if norm_P2prime <= 0 or norm_P2 <= 0:
            return 1e10
        
        return np.sqrt(norm_P2 / norm_P2prime)
    
    # Scan over scale factors to find the optimal one
    best_scale = 1.0
    best_val = 1e10
    
    for s in np.geomspace(0.5, 100, 200):
        if time.time() - t0 > deadline - 60:
            break
        roots = lag_zeros * s
        val = compute_c4_actual(roots)
        if val < best_val:
            best_val = val
            best_scale = s
    
    print(f"Best scale from scan: {best_scale:.4f}, C4: {best_val:.6e}")
    
    # Fine-tune the scale around the best found
    def scale_objective(log_s):
        if time.time() - t0 > deadline - 30:
            return 1e10
        s = np.exp(log_s)
        if s <= 0 or s > 1000:
            return 1e10
        roots = lag_zeros * s
        return compute_c4_actual(roots)
    
    res = minimize(scale_objective, np.log(best_scale), method='Nelder-Mead',
                  options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
    
    if res.fun < best_val:
        best_scale = np.exp(res.x[0])
        best_val = res.fun
        print(f"Fine-tuned scale: {best_scale:.6f}, C4: {best_val:.6e}")
    
    # Now optimize the relative positions with the optimal scale fixed
    def objective(theta):
        if time.time() - t0 > deadline - 10:
            return 1e10
        roots = np.sort(np.exp(theta))
        # Scale to match the optimal scale (normalize so geometric mean matches)
        geo_mean = np.exp(np.mean(theta))
        target_geo_mean = np.exp(np.mean(np.log(lag_zeros))) * best_scale
        roots = roots * (target_geo_mean / geo_mean)
        return compute_c4_actual(roots)
    
    # Multi-start optimization of relative positions
    starts = [
        np.log(lag_zeros),
        np.log(lag_zeros * (1 + 0.1 * np.random.randn(12))),
        np.log(lag_zeros * (1 + 0.05 * np.random.randn(12))),
    ]
    
    for i, th in enumerate(starts):
        if time.time() - t0 > deadline - 60:
            break
        try:
            res = minimize(objective, th, method='Nelder-Mead',
                          options={'maxiter': 20000, 'xatol': 1e-10,
                                   'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_val:
                roots = np.sort(np.exp(res.x))
                geo_mean = np.exp(np.mean(res.x))
                target_geo_mean = np.exp(np.mean(np.log(lag_zeros))) * best_scale
                roots = roots * (target_geo_mean / geo_mean)
                best_val = res.fun
                print(f"NM start {i}: {best_val:.6e}")
        except Exception as e:
            print(f"NM {i} failed: {e}")
    
    # Final result: scaled Laguerre zeros
    final_roots = lag_zeros * best_scale
    
    # Verify all positive
    if np.any(final_roots <= 0):
        final_roots = lag_zeros * 10.0  # Fallback to known good
    
    return [float(r) for r in final_roots]

