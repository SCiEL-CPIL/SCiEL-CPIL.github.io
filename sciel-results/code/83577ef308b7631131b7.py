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
    
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    rng = np.random.default_rng(42)
    
    # Strategy: Direct optimization of the magnitude spectrum P_k = |X_k|^2.
    # The autoconvolution is c = IFFT(P). We want to minimize max |c_k| / P[0].
    # Note that sum(x) = X[0] = sqrt(P[0]). So the metric is max |c_k| / P[0].
    # This is a scale-invariant problem in P. We can fix P[0] = 1 (or any constant)
    # and minimize max |c_k|.
    # Wait, if we fix P[0]=1, then sum(x) = 1.
    # The metric is max |c_k| / 1.
    # We want to minimize max |c_k| subject to P >= 0, P symmetric, P[0]=1.
    # c = IFFT(P).
    # c_k = (1/_FFT_SIZE) * sum_j P_j exp(2j pi j k / _FFT_SIZE).
    # We want c_k to be small for k != 0.
    # c_0 = (1/_FFT_SIZE) * sum_j P_j.
    # If P is flat, P_j = 1 for all j, then c_0 = 1, c_k = 0 for k != 0.
    # Then max |c_k| = 1. Metric = 1.
    # But for x = ones, P[0] = N^2, P_j = 0 for j != 0.
    # Then c_0 = N, c_k = 0. Metric = N / N^2 = 1/N.
    # So we want P to be concentrated at DC?
    # If P is concentrated at DC, x is constant.
    # Let's check the metric for x = ones again.
    # P[0] = N^2. c = IFFT(P).
    # c_0 = (1/_FFT_SIZE) * N^2.
    # Wait, my FFT normalization might be off.
    # np.fft.fft(x) gives X_k = sum x_n exp(-2j pi n k / N).
    # np.fft.ifft(X) gives x_n = (1/N) sum X_k exp(2j pi n k / N).
    # So if X = P (real, symmetric), then x = ifft(P).
    # And autoconv = ifft(|X|^2) = ifft(P).
    # So c = ifft(P).
    # c_k = (1/_FFT_SIZE) sum P_j exp(2j pi j k / _FFT_SIZE).
    # For x = ones, X[0] = N, X[k] = 0 for k != 0.
    # So P[0] = N^2, P[k] = 0.
    # c = ifft(P). c[0] = (1/_FFT_SIZE) N^2.
    # But we know autoconv of ones is a triangle.
    # Ah, the FFT size is _FFT_SIZE, which is > N.
    # So the zero-padding is important.
    # If x = ones(N), padded to _FFT_SIZE, then X[0] = N, X[k] = 0.
    # P[0] = N^2.
    # c = ifft(P). c[0] = N^2 / _FFT_SIZE.
    # c[k] = 0 for k != 0.
    # So max |c| = N^2 / _FFT_SIZE.
    # Metric = (N^2 / _FFT_SIZE) / (N^2) = 1 / _FFT_SIZE.
    # This is wrong. The autoconvolution of ones is not a delta.
    # The issue is that ifft(P) is the circular autoconvolution of the padded signal.
    # But we want the linear autoconvolution.
    # The linear autoconvolution is the first 2N-1 samples of the circular autoconvolution
    # IF _FFT_SIZE >= 2N-1.
    # So c = ifft(P) gives the linear autoconvolution in the first 2N-1 samples.
    # For x = ones(N), padded to _FFT_SIZE:
    # x_pad = [1, 1, ..., 1, 0, 0, ..., 0].
    # X = fft(x_pad). X[0] = N. X[k] = sum_{n=0}^{N-1} exp(-2j pi k n / _FFT_SIZE).
    # This is not zero for k != 0. It's a Dirichlet kernel.
    # So P[k] = |X[k]|^2 is not zero.
    # My previous assumption was wrong.
    
    # Let's stick to optimizing P directly.
    # Variables: P[0], P[1], ..., P[M-1] where M = _FFT_SIZE//2 + 1.
    # Constraints: P >= 0. P[0] > 0.
    # Objective: max |ifft(P)[:2N-1]| / P[0].
    # We can fix P[0] = 1 to remove scale invariance.
    # Then minimize max |ifft(P)[:2N-1]|.
    
    M = _FFT_SIZE // 2 + 1
    P_init = np.zeros(M)
    P_init[0] = 1.0
    # Add some small noise to other frequencies to break symmetry?
    # Or start with a flat spectrum?
    # If P is flat, P[k] = 1/M.
    # Then ifft(P) is a delta at 0.
    # c[0] = 1, c[k] = 0.
    # Metric = 1 / 1 = 1.
    # But we know 1/N is achievable.
    # For x = ones, P is not flat.
    
    # Let's try to optimize P using scipy.optimize.minimize.
    # Objective function.
    def _objective(P):
        # P is the magnitude spectrum.
        # Ensure non-negativity? The optimizer might go negative.
        # We can use a penalty or project back.
        # Let's use a soft constraint: P_eff = np.maximum(P, 0).
        P_eff = np.maximum(P, 0)
        if P_eff[0] < 1e-12:
            return 1e10
        # Normalize P[0] to 1?
        # No, the metric is scale invariant.
        # But the objective max |c| / P[0] is not convex if we don't fix scale.
        # Let's fix P[0] = 1 in the variable space.
        # So we optimize P[1:] and set P[0] = 1.
        # Wait, if we fix P[0]=1, we restrict the solution space?
        # No, because the metric is scale invariant.
        # So any optimal solution can be scaled to have P[0]=1.
        
        # Construct full spectrum.
        X = np.zeros(_FFT_SIZE, dtype=complex)
        X[:M] = np.sqrt(P_eff)
        # Hermitian symmetry.
        X[M:] = np.conj(X[_FFT_SIZE-M+1:0:-1])
        # Note: X[0] is real.
        
        # Compute autoconvolution.
        # c = ifft(|X|^2) = ifft(P_eff extended).
        # Actually, |X|^2 is just P_eff extended.
        P_full = np.zeros(_FFT_SIZE)
        P_full[:M] = P_eff
        P_full[M:] = P_eff[_FFT_SIZE-M+1:0:-1]
        
        c = np.fft.ifft(P_full)
        c = np.real(c)
        
        # Max absolute value of the linear part.
        max_abs = np.max(np.abs(c[:2*N-1]))
        
        # Metric is max_abs / P[0].
        # Since we fixed P[0]=1, metric = max_abs.
        # But wait, if we fix P[0]=1, then sum(x) = 1.
        # The metric is max |c| / (sum x)^2 = max |c| / 1.
        # So we minimize max |c|.
        
        return max_abs

    # Define bounds. P >= 0.
    bounds = [(0, None)] * M
    
    # Initial guess.
    # Try a few different initializations.
    inits = []
    # 1. Delta at DC (x = constant)
    P_delta = np.zeros(M)
    P_delta[0] = 1.0
    inits.append(P_delta)
    
    # 2. Flat spectrum
    P_flat = np.ones(M) / M
    inits.append(P_flat)
    
    # 3. Random
    for _ in range(10):
        P_rand = np.abs(rng.normal(0, 1, M))
        P_rand /= np.sum(P_rand)
        inits.append(P_rand)
    
    best_val_local = float('inf')
    best_P_local = None
    
    for P_init in inits:
        if time.time() > deadline - 5:
            break
        
        try:
            res = opt.minimize(
                _objective,
                P_init,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            if res.success or res.fun < best_val_local:
                best_val_local = res.fun
                best_P_local = res.x
        except Exception:
            pass
    
    if best_P_local is not None:
        # Reconstruct x.
        # We need X such that |X|^2 = P.
        # We can choose phase to be 0.
        # X = sqrt(P) (real, non-negative).
        # Then x = ifft(X).
        P_best = np.maximum(best_P_local, 0)
        # Ensure P[0] is consistent.
        # The metric was computed with P[0] fixed to 1 in the optimization?
        # No, I didn't fix it in the code above. I just returned max_abs.
        # But the objective function I defined uses P_eff[0] in the denominator?
        # No, I said "Since we fixed P[0]=1...". But I didn't implement that fix.
        # I just returned max_abs.
        # So the optimizer minimized max |c|, not max |c| / P[0].
        # This is wrong. It will try to make P very small to make c small.
        # We MUST fix the scale.
        
        # Let's restart the optimization with the scale fixed.
        
    # Corrected Optimization
    
    def _objective_fixed(P):
        # P is the magnitude spectrum, with P[0] fixed to 1.
        # The input P has length M, but we ignore P[0] and set it to 1.
        P_eff = np.maximum(P, 0)
        P_eff[0] = 1.0
        
        P_full = np.zeros(_FFT_SIZE)
        P_full[:M] = P_eff
        P_full[M:] = P_eff[_FFT_SIZE-M+1:0:-1]
        
        c = np.fft.ifft(P_full)
        c = np.real(c)
        
        max_abs = np.max(np.abs(c[:2*N-1]))
        
        return max_abs

    # Variables: P[1:] (length M-1).
    def _objective_wrapper(P_vars):
        P_full = np.zeros(M)
        P_full[0] = 1.0
        P_full[1:] = P_vars
        return _objective_fixed(P_full)

    bounds_vars = [(0, None)] * (M - 1)
    
    best_val_fixed = float('inf')
    best_P_vars = None
    
    # Initializations for P_vars
    inits_vars = []
    # 1. All zeros (Delta at DC)
    inits_vars.append(np.zeros(M-1))
    # 2. Flat (excluding DC)
    inits_vars.append(np.ones(M-1) * (1.0 / M))
    # 3. Random
    for _ in range(10):
        inits_vars.append(np.abs(rng.normal(0, 1, M-1)))
    
    for P_init_vars in inits_vars:
        if time.time() > deadline - 5:
            break
        
        try:
            res = opt.minimize(
                _objective_wrapper,
                P_init_vars,
                method='SLSQP',
                bounds=bounds_vars,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            if res.fun < best_val_fixed:
                best_val_fixed = res.fun
                best_P_vars = res.x
        except Exception:
            pass
    
    if best_P_vars is not None:
        # Reconstruct x.
        P_best = np.zeros(M)
        P_best[0] = 1.0
        P_best[1:] = np.maximum(best_P_vars, 0)
        
        # X = sqrt(P_best). Phase = 0.
        X = np.zeros(_FFT_SIZE, dtype=complex)
        X[:M] = np.sqrt(P_best)
        X[M:] = np.conj(X[_FFT_SIZE-M+1:0:-1])
        
        x = np.fft.ifft(X)
        x = np.real(x)
        
        # Check metric.
        val = _metric(x)
        if val < best_val:
            best_val = val
            best_x = x
    
    # Local search on x to polish.
    def _local_search(x, deadline, rng, max_rounds=5000):
        best_x_local = x.copy()
        best_val_local = _metric(best_x_local)
        no_improve = 0
        step_scale = 0.5
        
        for round_idx in range(max_rounds):
            if time.time() > deadline:
                break
            
            if no_improve > 50:
                step_scale *= 0.8
                no_improve = 0
                if step_scale < 0.01:
                    break
            
            idx = rng.integers(0, N)
            
            for sign in [1, -1]:
                for scale in [step_scale, step_scale * 0.5, step_scale * 0.1]:
                    x_test = best_x_local.copy()
                    x_test[idx] += sign * scale
                    
                    S_new = np.sum(x_test)
                    if abs(S_new) > 1e-6:
                        x_test = x_test * (N / S_new)
                    
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
    
    # Run local search on the best solution found so far.
    if time.time() < deadline - 5:
        x_ref, val_ref = _local_search(best_x, deadline - 2, rng, max_rounds=10000)
        if val_ref < best_val:
            best_val = val_ref
            best_x = x_ref

    return best_x

def construct():
    return _construct()
# EVOLVE-BLOCK-END
