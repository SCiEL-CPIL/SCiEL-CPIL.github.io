"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_max_abs(x):
    """Max absolute value of the discrete autoconvolution of x."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))

def _integral(x):
    """Integral of the equal-width step function = sum of the heights."""
    return float(np.sum(x))

def _metric(x):
    """max |autoconv| / (integral)^2.  Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _construct():
    deadline = time.time() + 145.0
    
    # Initial best: constant sequence
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    rng = np.random.default_rng(42)
    
    # Strategy: Direct optimization in time domain using smooth max (log-sum-exp)
    # to handle the non-differentiability of max(|c|).
    # We minimize f(x) = log-sum-exp(|c_k|) / (sum x)^2
    # This is scale invariant, so we can fix sum(x) = N (or 1) to remove that degree of freedom,
    # but L-BFGS-B handles it fine if we just minimize the ratio.
    # Actually, fixing the scale is better for numerical stability.
    # Let's fix sum(x) = 1. Then we minimize max |c|.
    # But the metric is max |c| / (sum x)^2. If we scale x by alpha, c scales by alpha^2, sum by alpha.
    # Metric scales by (alpha^2 * max|c|) / (alpha^2 * (sum x)^2) = invariant.
    # So we can fix sum(x) = 1.
    
    # We will optimize x directly.
    # Objective: J(x) = log(sum(exp(|c_k|/tau)) * tau) / (sum x)^2
    # where c = ifft(|fft(x)|^2)
    
    def _smooth_max_abs_conv(x, tau=1.0):
        """
        Computes a smooth approximation of max |c_k| / (sum x)^2.
        c is the linear autoconvolution.
        """
        fx = np.fft.rfft(x, _FFT_SIZE)
        power = np.abs(fx)**2
        # Reconstruct full power spectrum for ifft
        # rfft returns M = N//2 + 1 bins.
        # ifft expects full spectrum or uses rfft/irfft.
        # irfft(power, _FFT_SIZE) gives the autoconvolution.
        c = np.fft.irfft(power, _FFT_SIZE)
        c_lin = c[:2*N-1]
        
        abs_c = np.abs(c_lin)
        s = np.sum(x)
        
        if abs(s) < 1e-10:
            return 1e10, np.zeros(N)
            
        # Smooth max approximation: tau * log(sum(exp(|c|/tau)))
        # To avoid overflow, subtract max
        max_c = np.max(abs_c)
        if max_c > 100: # Scaling heuristic for stability
             # If values are huge, just return max
             return max_c / (s*s), np.zeros(N)
             
        # We want to minimize this value.
        # The gradient of log-sum-exp is softmax.
        # d/dx log(sum(exp(y_i/tau))) = (1/tau) * sum( (exp(y_i/tau)/sum(exp(y_j/tau))) * dy_i/dx )
        # y_i = |c_i|
        # dy_i/dx = sign(c_i) * dc_i/dx
        
        # For the objective J = S_max / s^2, where S_max is the smooth max.
        # This is complicated. 
        # Alternative: Fix s=1. Then J = S_max.
        # We can add a penalty term to keep s close to 1, or project.
        # L-BFGS-B supports bounds, but not equality constraints easily.
        # We can use a penalty method or just rely on the scale invariance and 
        # initialize with sum=1. The optimizer might drift, but the metric is invariant.
        # Actually, if sum(x) changes, the metric stays the same if c scales correctly?
        # No, c(x) = x * x. If x -> alpha x, c -> alpha^2 c.
        # max |alpha^2 c| / (alpha sum x)^2 = alpha^2 max|c| / (alpha^2 (sum x)^2) = max|c|/(sum x)^2.
        # So the metric is truly scale invariant.
        # We can just minimize the smooth max of |c|, and the scale will be determined by the optimizer?
        # No, if we minimize max|c|, the optimizer will shrink x to 0 to make c=0.
        # So we MUST fix the scale.
        
        # Let's fix sum(x) = 1 by normalizing x inside the objective?
        # No, that breaks differentiability if done abruptly.
        # Better: Minimize J(x) = max|c(x)| / (sum x)^2.
        # This is homogeneous of degree 0.
        # We can use the fact that the gradient of J is 0 along the radial direction.
        # L-BFGS-B might struggle with flat directions.
        
        # Let's try a different parameterization: x = y / sum(y) * N.
        # This forces sum(x) = N.
        # Then we minimize max |c(x)|.
        # c(x) is quadratic in y.
        
        # Let's stick to x, but add a strong penalty if sum(x) deviates from N.
        # Or simply: minimize log-sum-exp(|c|) + lambda * (sum(x) - N)^2
        
        # Let's implement the gradient of J(x) = S(x) / s(x)^2
        # S(x) = tau * log(sum(exp(|c_k|/tau)))
        # s(x) = sum(x)
        
        # This is getting complex to compute gradient manually for L-BFGS-B.
        # Let's use numerical gradient (finite differences) with L-BFGS-B?
        # N=400. 400 evaluations per gradient step. 100 steps = 40,000 evals.
        # Each eval is FFT(400). Very fast. 40,000 * 10us = 0.4s. Feasible.
        
        # But finite difference might be noisy.
        # Let's try to compute the analytic gradient of the smooth max part.
        
        # Let u_k = |c_k|.
        # S = tau * log(sum_k exp(u_k/tau))
        # dS/du_k = exp(u_k/tau) / sum_j exp(u_j/tau) = p_k (softmax weights)
        # du_k/dx = sign(c_k) * dc_k/dx
        # c = ifft(|fft(x)|^2)
        # Let X = fft(x). |X|^2 = P.
        # c = ifft(P).
        # P_k = X_k \bar{X}_k.
        # dP_k/dx_m = (dX_k/dx_m) \bar{X}_k + X_k (d\bar{X}_k/dx_m)
        # X_k = sum_n x_n exp(-i 2 pi k n / L)
        # dX_k/dx_m = exp(-i 2 pi k m / L)
        # d\bar{X}_k/dx_m = exp(i 2 pi k m / L)
        # So dP_k/dx_m = exp(-i 2 pi k m / L) \bar{X}_k + X_k exp(i 2 pi k m / L)
        #                = \bar{X}_k exp(-i 2 pi k m / L) + X_k exp(i 2 pi k m / L)
        
        # dc_l/dx_m = (1/L) sum_k (dP_k/dx_m) exp(i 2 pi k l / L)
        # dc_l/dx_m = (1/L) sum_k [ \bar{X}_k exp(-i 2 pi k m / L) + X_k exp(i 2 pi k m / L) ] exp(i 2 pi k l / L)
        #            = (1/L) [ \bar{X}_k sum_k exp(i 2 pi k (l-m) / L) + X_k sum_k exp(i 2 pi k (l+m) / L) ]
        #            = \bar{X}_k delta_{l,m} + X_k delta_{l, -m mod L}
        # Wait, the sum over k is the inverse DFT of the constant sequence?
        # sum_k exp(i 2 pi k (l-m) / L) = L * delta_{l, m mod L}
        # sum_k exp(i 2 pi k (l+m) / L) = L * delta_{l, -m mod L}
        
        # So dc_l/dx_m = \bar{X}_0 delta_{l,m} + X_0 delta_{l, -m mod L} ?
        # No, the sums are over k. The terms \bar{X}_k and X_k depend on k.
        # The formula is:
        # dc_l/dx_m = (1/L) sum_k \bar{X}_k exp(i 2 pi k (l-m) / L) + (1/L) sum_k X_k exp(i 2 pi k (l+m) / L)
        
        # Let A = ifft(conj(X)) = conj(fft(x))_time? 
        # ifft(conj(X)) = conj(ifft(X)) = conj(x).
        # So sum_k \bar{X}_k exp(i 2 pi k (l-m) / L) is the circular convolution of conj(x) with delta?
        # Actually, ifft(conj(X)) is a sequence y_l.
        # y_l = (1/L) sum_k \bar{X}_k exp(i 2 pi k l / L).
        # Then the first term is y_{l-m} (circular shift).
        # Similarly, let z_l = (1/L) sum_k X_k exp(i 2 pi k l / L) = x_l.
        # The second term is z_{l+m} = x_{l+m} (circular shift).
        
        # So dc_l/dx_m = conj(x)_{(l-m) mod L} + x_{(l+m) mod L}
        
        # This is a sparse operation!
        # dc_l/dx_m is non-zero only for specific m for a given l?
        # No, for a fixed l, m varies.
        # We need dJ/dx_m = sum_l (dJ/dc_l) * (dc_l/dx_m)
        # dJ/dc_l = (dS/dc_l) / s^2 - 2 S / s^3 * (dc_l/ds)? No, s is sum x, independent of c in the derivative chain?
        # J = S / s^2.
        # dJ/dx_m = (1/s^2) dS/dx_m - (2S/s^3) (ds/dx_m)
        # ds/dx_m = 1.
        # dS/dx_m = sum_l (dS/dc_l) (dc_l/dx_m)
        # dS/dc_l = p_l * sign(c_l) / tau ? No.
        # S = tau log sum exp(|c|/tau).
        # dS/d|c_l| = exp(|c_l|/tau) / sum exp(|c_j|/tau) = p_l.
        # d|c_l|/dc_l = sign(c_l).
        # So dS/dc_l = p_l * sign(c_l).
        
        # So dJ/dx_m = (1/s^2) sum_l [ p_l * sign(c_l) * (conj(x)_{(l-m) mod L} + x_{(l+m) mod L}) ] - 2S/s^3
        
        # This can be computed efficiently.
        
        # Let's implement this.
        
        # 1. Compute c, s, S, p_l, sign_c_l
        # 2. Compute vector G_c_l = p_l * sign(c_l)
        # 3. We need sum_l G_c_l * conj(x)_{(l-m) mod L}
        #    Let v_l = G_c_l.
        #    We want y_m = sum_l v_l conj(x)_{(l-m) mod L}.
        #    Let k = l-m => l = k+m.
        #    y_m = sum_k v_{k+m} conj(x)_k.
        #    This is the cross-correlation of v and conj(x).
        #    corr(v, conj(x))_m = sum_k v_{k+m} conj(x)_k.
        #    Using FFT: fft(v) * conj(fft(conj(x))) = fft(v) * conj(conj(fft(x))) = fft(v) * fft(x).
        #    Wait, corr(a, b) = ifft(fft(a) * conj(fft(b))).
        #    Here a=v, b=conj(x).
        #    fft(b) = fft(conj(x)) = conj(fft(x)).
        #    conj(fft(b)) = conj(conj(fft(x))) = fft(x).
        #    So y = ifft(fft(v) * fft(x)).
        
        # 4. Similarly for x_{(l+m) mod L}.
        #    z_m = sum_l G_c_l x_{(l+m) mod L}.
        #    Let k = l+m => l = k-m.
        #    z_m = sum_k G_c_{k-m} x_k.
        #    This is cross-correlation of G_c and x, but reversed?
        #    corr(a, b)_m = sum_k a_{k+m} b_k.
        #    Here we have sum_k a_{k-m} b_k.
        #    Let j = k-m => k = j+m.
        #    sum_j a_j b_{j+m}.
        #    This is the correlation of b and a?
        #    corr(b, a)_m = sum_j b_{j+m} a_j. Yes.
        #    So z = corr(x, G_c).
        #    fft(corr(x, G_c)) = fft(x) * conj(fft(G_c)).
        #    So z = ifft(fft(x) * conj(fft(G_c))).
        
        # Total dS/dx_m = y_m + z_m.
        
        # 5. dJ/dx_m = (y_m + z_m) / s^2 - 2S/s^3.
        
        # Let's implement this gradient.
        
        L = _FFT_SIZE
        fx = np.fft.rfft(x, L)
        power = np.abs(fx)**2
        c = np.fft.irfft(power, L)
        c_lin = c[:2*N-1]
        # Note: c is circular autoconvolution of padded x.
        # The derivative formula dc_l/dx_m = conj(x)_{(l-m) mod L} + x_{(l+m) mod L}
        # holds for the circular convolution.
        # We are taking max over c_lin (first 2N-1).
        # The gradient should only include terms for l in 0..2N-2.
        # So we mask G_c_l to be zero for l >= 2N-1.
        
        abs_c = np.abs(c_lin)
        s = np.sum(x)
        
        if abs(s) < 1e-10:
            return 1e10, np.zeros(N)
            
        # Smooth max parameters
        # tau should be small enough to approximate max, but large enough for smoothness.
        # Let's use tau = 1.0 / 1000? Or adaptive?
        # Let's try tau = 0.1 * max(abs_c) or fixed 1.0.
        # If max(abs_c) is small, 1.0 is fine.
        tau = 1.0
        
        # Compute p_l = softmax(abs_c / tau)
        # Only for l in 0..2N-2
        max_ac = np.max(abs_c)
        exp_c = np.exp((abs_c - max_ac) / tau)
        sum_exp = np.sum(exp_c)
        p_lin = exp_c / sum_exp
        
        # S = tau * log(sum_exp) + max_ac
        S = tau * np.log(sum_exp) + max_ac
        
        # G_c_l = p_l * sign(c_l) for l in 0..2N-2, 0 otherwise.
        G_c = np.zeros(L)
        sign_c = np.sign(c_lin)
        G_c[:2*N-1] = p_lin * sign_c
        
        # Compute y = ifft(fft(G_c) * fft(x_padded))
        # x_padded is x with zeros.
        fx_full = np.fft.fft(x, L) # x is length N, padded to L
        fg = np.fft.fft(G_c, L)
        
        y = np.fft.ifft(fg * fx_full)
        
        # Compute z = ifft(fft(x_padded) * conj(fft(G_c)))
        # fft(x_padded) is fx_full.
        # conj(fft(G_c)) is conj(fg).
        z = np.fft.ifft(fx_full * np.conj(fg))
        
        # dS/dx = y + z (real part, since result should be real)
        dS_dx = np.real(y + z)
        
        # dJ/dx = dS_dx / s^2 - 2S/s^3
        grad = dS_dx / (s*s) - 2*S/(s**3)
        
        # grad is length L. We only care about first N.
        grad_x = grad[:N]
        
        val = S / (s*s)
        
        return val, grad_x

    # Optimization loop
    # We use L-BFGS-B with analytic gradient.
    
    best_x_local = np.ones(N)
    # Normalize to sum 1?
    # best_x_local /= np.sum(best_x_local)
    
    # Try a few random starts
    starts = [np.ones(N)]
    for _ in range(5):
        starts.append(rng.standard_normal(N))
        
    for x_init in starts:
        if time.time() > deadline - 10:
            break
            
        x_init = x_init.copy()
        # Normalize initial guess to sum 1 to help optimizer
        s_init = np.sum(x_init)
        if abs(s_init) > 1e-6:
            x_init = x_init / s_init * 1.0 # Scale to sum 1? 
            # Actually, let's keep scale free but start near sum=1.
            x_init = x_init / np.sum(x_init) * 1.0
            
        try:
            res = opt.minimize(
                _smooth_max_abs_conv,
                x_init,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
                
        except Exception:
            pass
            
    # Polish with local search if time permits
    if time.time() < deadline - 5:
        def _local_search(x, deadline, rng, max_rounds=5000):
            best_x_local = x.copy()
            best_val_local = _metric(best_x_local)
            no_improve = 0
            step_scale = 0.1
            
            for round_idx in range(max_rounds):
                if time.time() > deadline:
                    break
                
                if no_improve > 50:
                    step_scale *= 0.8
                    no_improve = 0
                    if step_scale < 0.001:
                        break
                
                idx = rng.integers(0, N)
                
                for sign in [1, -1]:
                    for scale in [step_scale, step_scale * 0.5]:
                        x_test = best_x_local.copy()
                        x_test[idx] += sign * scale
                        
                        v = _metric(x_test)
                        if v < best_val_local:
                            best_val_local = v
                            best_x_local = x_test
                            no_improve = 0
                            break
                    else:
                        continue
                    break
                else:
                    no_improve += 1
            
            return best_x_local, best_val_local
        
        x_ref, val_ref = _local_search(best_x, deadline - 2, rng, max_rounds=5000)
        if val_ref < best_val:
            best_val = val_ref
            best_x = x_ref

    return best_x

def construct():
    return _construct()
# EVOLVE-BLOCK-END
