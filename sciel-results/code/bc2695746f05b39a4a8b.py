import numpy as np
import time
from scipy.optimize import minimize
import scipy.fft as fft

N = 400
TIME_LIMIT = 150.0
SAFETY_MARGIN = 15.0

def compute_value(h):
    """Exact evaluation of the objective function."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return float('inf')
    if not np.all(np.isfinite(h)):
        return float('inf')
    z = np.convolve(h, h)
    return abs(2 * N * np.max(z) / (s * s))

def spectral_to_h(P_half):
    """
    Reconstruct real h from half power spectrum.
    P_half is the power at frequencies 0..N/2.
    We construct the full spectrum, take sqrt, and ifft.
    To ensure h is real, we enforce conjugate symmetry.
    Note: This assumes we want to minimize the peak of autocorrelation.
    Autocorrelation r[k] = ifft(P)[k] for circular.
    For linear, it's more complex, but for large N, circular is a good proxy 
    if we are careful with the boundary. However, the task uses np.convolve (linear).
    
    Actually, minimizing the peak of the *linear* autocorrelation is the goal.
    The peak of the linear autocorrelation is related to the time-domain signal.
    
    Let's stick to a simpler structural change: 
    We know that for a fixed sum S, the value is proportional to max(z).
    We want to minimize max(z).
    
    A different approach: 
    The problem is equivalent to finding a vector h such that its autocorrelation 
    is as flat as possible (low peak). This is related to "low peak factor" sequences.
    
    Let's try a different optimization variable: 
    Optimize the phase of the Fourier transform? 
    Or better: Optimize the time-domain h directly but use a 
    "soft" constraint that pushes h to be non-negative? 
    Inspiration 2 mentions: "exploit scale invariance + nonnegativity to drive the".
    
    If we assume h >= 0, then sum(h) > 0 and max(z) is at z[0] = sum(h^2).
    Then value = 2*N*sum(h^2)/sum(h)^2.
    To minimize this, we want h to be as "spread out" as possible, i.e., constant.
    But constant h gives value 2.0.
    If we allow negative values, we can make sum(h) large while keeping sum(h^2) small?
    No, z[0] = sum(h^2). If we have negative values, sum(h) can be small, 
    making the denominator small, which increases the value. 
    So we want sum(h) to be LARGE in magnitude.
    
    Wait, the formula is: abs(2 * N * max(z) / sum(h)^2).
    If we scale h by c, sum(h) becomes c*sum(h), max(z) becomes c^2*max(z).
    Value becomes |2*N*c^2*max(z) / (c^2*sum(h)^2)| = constant.
    So the value is scale invariant.
    
    Key insight from Inspiration 2: "nonnegativity".
    If we restrict h to be non-negative, then max(z) is likely z[0] = sum(h^2).
    Then value = 2*N * sum(h^2) / sum(h)^2.
    By Cauchy-Schwarz, sum(h^2) * N >= sum(h)^2, so sum(h^2)/sum(h)^2 >= 1/N.
    Min value = 2*N*(1/N) = 2.
    So for non-negative h, the minimum is 2.0 (achieved by constant h).
    To get < 2.0, we MUST have negative values.
    
    If we have negative values, sum(h) can be small, but max(z) might be large.
    Or, sum(h) can be large (if signs are mixed but sum is large? No, if signs mixed, sum is smaller than sum of abs).
    
    Let's look at the structure:
    We want to minimize max(z) / sum(h)^2.
    This is equivalent to minimizing the peak of the autocorrelation relative to the DC component squared.
    
    Let's try a strategy based on **Barker-like sequences** or **Golay-like** structures but in continuous domain.
    Specifically, we can use a **parametric family** of functions that are known to have low autocorrelation peaks.
    Example: h[n] = cos(2*pi*k*n/N + phi) ?
    Or h[n] = sign(cos(2*pi*k*n/N))?
    
    Let's try optimizing over a small set of parameters for a structured function.
    h[n] = A * cos(2*pi*f1*n/N + p1) + B * cos(2*pi*f2*n/N + p2) ...
    
    Alternatively, let's use **Gradient Descent on the Time Domain** but with a **Sharp** max approximation (low temperature) 
    and a **Strong** initialization from a known low-autocorrelation structure.
    
    The current program uses tau=1.0. The log-sum-exp with tau=1.0 is a very smooth approximation.
    If we use a smaller tau (e.g., 0.1 or 0.01), the surrogate becomes closer to the true max, 
    but the gradient becomes noisier and optimization harder.
    
    However, the DIRECTIVE asks for a STRUCTURAL change.
    
    Let's try **Simulated Annealing** or **Genetic Algorithm** on the sign pattern?
    No, N=400 is too large for binary search.
    
    Let's try **Optimizing the Fourier Coefficients** directly, but with a **Constraint** that the time-domain signal is real and has a specific property.
    
    Actually, let's look at the "Inspiration 2" more closely: "exploit scale invariance + nonnegativity".
    Wait, if non-negativity leads to min 2.0, why would they suggest it?
    Maybe they mean: Start with non-negative, then allow small negative dips?
    
    Let's try a different structural approach:
    **Minimize the L-infinity norm of the autocorrelation subject to sum(h)=1.**
    This is a convex optimization problem if we ignore the fact that autocorrelation is quadratic.
    Actually, minimizing max(z) subject to sum(h)=1 is NOT convex because z depends quadratically on h.
    
    However, we can use **Alternating Minimization**:
    1. Fix the shape of h, scale it so sum(h)=1.
    2. Minimize max(z).
    
    Let's try a **Random Restart Gradient Descent** with a **Sharper** surrogate (smaller tau) 
    and **Better Initializations** based on **Chirp Signals** or **M-sequences**.
    
    A chirp signal h[n] = exp(i * pi * n^2 / N) has ideal autocorrelation (impulse).
    But we need real h.
    h[n] = cos(pi * n^2 / N) ?
    
    Let's test a few structured initializations:
    1. h[n] = cos(2*pi*n/N)
    2. h[n] = cos(pi*n^2/N)
    3. h[n] = (-1)^n
    4. h[n] = 1 if n < N/2 else -1
    
    And then refine with L-BFGS-B using a **lower temperature** tau for the log-sum-exp.
    
    Why this is different:
    Current program uses tau=1.0 and random/Chebyshev starts.
    New program uses structured starts (Chirp, Square, etc.) and a **lower tau** (sharper max approximation) 
    to drive the optimization towards the true max constraint.
    
    Also, we can use **Multiple Restarts** with different taus.
    
    Let's implement this.
    """
    pass

def smooth_objective_sharp(h, tau):
    """
    Differentiable surrogate with tunable temperature tau.
    """
    if not np.all(np.isfinite(h)):
        return float('inf')
    
    s = np.sum(h)
    if abs(s) < 1e-9:
        return float('inf')
        
    fft_len = 1024
    H_lin = fft.fft(h, n=fft_len)
    Z_lin = fft.ifft(H_lin * H_lin)
    z_lin = Z_lin.real[:2*N-1]
    
    z_max = np.max(z_lin)
    z_scaled = z_lin / tau
    z_max_scaled = np.max(z_scaled)
    
    exp_terms = np.exp(z_scaled - z_max_scaled)
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        log_sum_exp = z_max_scaled
    else:
        log_sum_exp = z_max_scaled + np.log(sum_exp)
        
    max_z_smooth = log_sum_exp * tau
    
    val = abs(2 * N * max_z_smooth / (s * s))
    return val

def construct():
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - SAFETY_MARGIN)
    
    best_h = None
    best_val = float('inf')
    
    rng = np.random.default_rng(123)
    
    # Structured Initializations
    candidates = []
    
    # 1. Constant (Seed)
    candidates.append(np.ones(N))
    
    # 2. Square Wave
    h_sq = np.ones(N)
    h_sq[N//2:] = -1.0
    candidates.append(h_sq)
    
    # 3. Chirp (Real part of complex chirp)
    n = np.arange(N)
    h_chirp = np.cos(np.pi * n**2 / N)
    candidates.append(h_chirp)
    
    # 4. Linear Chirp
    h_lin_chirp = np.cos(2 * np.pi * n * 5 / N)
    candidates.append(h_lin_chirp)
    
    # 5. Random Gaussian
    candidates.append(rng.normal(0, 1, N))
    
    # 6. Block structure
    h_block = np.zeros(N)
    h_block[:N//4] = 1.0
    h_block[N//4:N//2] = -1.0
    h_block[N//2:3*N//4] = 1.0
    h_block[3*N//4:] = -1.0
    candidates.append(h_block)
    
    # Evaluate initial candidates
    for h_cand in candidates:
        v = compute_value(h_cand)
        if v < best_val:
            best_val = v
            best_h = h_cand.copy()
    
    # Optimization Phase
    # We will try different temperatures tau.
    # Smaller tau -> sharper max -> closer to true objective, but harder to optimize.
    # Larger tau -> smoother -> easier to optimize, but might converge to a "flat" max which isn't optimal for the sharp max.
    
    taus = [0.5, 0.2, 0.1, 0.05]
    
    # Start from the best candidate found so far
    current_h = best_h.copy()
    
    for tau in taus:
        if time.time() > deadline - 10:
            break
            
        def obj_func(h):
            return smooth_objective_sharp(h, tau)
        
        try:
            res = minimize(
                obj_func, 
                current_h, 
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            h_new = res.x
            if np.all(np.isfinite(h_new)) and abs(np.sum(h_new)) > 1e-9:
                val_new = compute_value(h_new)
                if val_new < best_val:
                    best_val = val_new
                    best_h = h_new.copy()
                    current_h = h_new.copy() # Chain the optimization
        except Exception:
            pass
    
    # Additional Random Restarts with Sharp Tau
    if time.time() < deadline - 10:
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            
            # Random start
            h_rand = rng.normal(0, 1, N)
            
            # Try a few taus for this random start
            for tau in [0.1, 0.05]:
                if time.time() > deadline - 5:
                    break
                def obj_func(h):
                    return smooth_objective_sharp(h, tau)
                
                try:
                    res = minimize(
                        obj_func, 
                        h_rand, 
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-7}
                    )
                    
                    h_new = res.x
                    if np.all(np.isfinite(h_new)) and abs(np.sum(h_new)) > 1e-9:
                        val_new = compute_value(h_new)
                        if val_new < best_val:
                            best_val = val_new
                            best_h = h_new.copy()
                except Exception:
                    pass

    # Final Polish with very sharp tau
    if best_h is not None and time.time() < deadline - 5:
        tau_final = 0.01
        def obj_func_final(h):
            return smooth_objective_sharp(h, tau_final)
        
        try:
            res_final = minimize(
                obj_func_final, 
                best_h, 
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-9}
            )
            
            h_final = res_final.x
            if np.all(np.isfinite(h_final)) and abs(np.sum(h_final)) > 1e-9:
                val_final = compute_value(h_final)
                if val_final < best_val:
                    best_val = val_final
                    best_h = h_final.copy()
        except Exception:
            pass

    # Ensure validity
    if best_h is None or not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
        
    return best_h

if __name__ == "__main__":
    h = construct()
    print(len(h))
    print(np.sum(h))
    z = np.convolve(h, h)
    print(abs(2 * N * np.max(z) / np.sum(h)**2))
