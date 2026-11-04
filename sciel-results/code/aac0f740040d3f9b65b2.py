"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2


def _autoconv_max_abs(x):
    """Max absolute value of the discrete autoconvolution of x."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))


def _autoconv_values(x):
    """Full autoconvolution array for gradient computation."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return conv[:2 * N - 1]


def _integral(x):
    """Integral of the equal-width step function = sum of the heights."""
    return float(np.sum(x))


def _metric(x):
    """max |autoconv| / (integral)^2.  Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)


def _soft_max_abs(conv, beta=1000.0):
    """Smooth approximation of max |conv| using log-sum-exp."""
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    # log-sum-exp trick for numerical stability
    shifted = abs_conv - max_val
    return max_val + np.log(np.sum(np.exp(beta * shifted))) / beta


def _smooth_metric(x, beta=1000.0):
    """Smooth surrogate: soft_max |autoconv| / (integral)^2."""
    conv = _autoconv_values(x)
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _soft_max_abs(conv, beta) / (s * s)


def _phase_only_optimize(deadline, rng, max_iters=5000):
    """
    Optimize phase vector with fixed magnitude spectrum.
    This ensures spectral flatness (|X(k)| constant) which minimizes autoconv peaks in frequency domain sense.
    """
    # Initial phase: random or structured
    # We fix magnitude to be flat: |X(k)| = C for all k
    # X(k) = C * exp(i * theta_k)
    # x = IFFT(X)
    
    # We need to handle real-valued x, so X must be conjugate symmetric.
    # rfft returns N//2 + 1 bins.
    # Let M = N//2 + 1
    # theta[0] = 0 (DC, real)
    # theta[M-1] = 0 (Nyquist, real) if N even
    # For k in 1..M-2: X[k] and X[N-k] are conjugates.
    
    M = N // 2 + 1
    # theta is the phase vector for the rfft bins
    # We optimize theta[1:M-1] (excluding DC and Nyquist for simplicity, or include with constraints)
    # Actually, let's just optimize all phases and reconstruct real x via irfft.
    # To ensure x is real, we define X_full and force conjugate symmetry.
    
    # Strategy:
    # 1. Fix magnitude spectrum A[k] = 1 for all k (or some constant).
    # 2. Optimize phases phi[k].
    # 3. X[k] = A[k] * exp(i * phi[k]).
    # 4. x = IFFT(X).
    # 5. Ensure x is real by taking real part or enforcing symmetry.
    # Since we use rfft/irfft, we can work with the half spectrum.
    # Let phi be the phase for the rfft bins.
    # X_rfft[k] = exp(i * phi[k])
    # x = irfft(X_rfft * scale, N)
    
    # Wait, irfft expects the spectrum to be consistent with a real signal.
    # If we just pick arbitrary phases for the rfft bins, irfft will produce a real signal.
    # The magnitude of the full spectrum will be |exp(i*phi)| = 1 for the positive frequencies,
    # and by symmetry, 1 for the negative frequencies.
    # So the power spectrum is flat!
    
    # Initialize phases
    phi = rng.uniform(0, 2 * np.pi, size=M)
    # Fix DC phase to 0 to avoid sign ambiguity / simplify
    phi[0] = 0.0
    if N % 2 == 0:
        phi[M-1] = 0.0 # Nyquist
        
    # Scale factor to control amplitude
    # If X[k]=1, x will have mean 0 and variance N.
    # We want to maximize integral (sum) to minimize metric?
    # Metric = max|conv| / (sum x)^2
    # If we scale x by c, conv scales by c^2, sum scales by c.
    # Metric scales by c^2 / c^2 = 1. Scale invariant!
    # So we can fix scale. Let's fix sum(x) = 1 or just work with unit magnitude spectrum.
    # But if sum(x) is close to 0, metric explodes.
    # We should ensure sum(x) is not small.
    # The DC component X[0] determines the mean.
    # If we fix phi[0]=0, X[0]=1. Mean = 1/N. Sum = 1.
    # This is good.
    
    best_x = None
    best_val = float('inf')
    
    # Compute initial x
    X = np.exp(1j * phi)
    x = np.fft.irfft(X, n=N)
    # Note: irfft with n=N will scale appropriately?
    # numpy irfft is inverse of rfft.
    # rfft(x) -> X. irfft(X, N) -> x.
    # If we set X = exp(i*phi), then x = irfft(exp(i*phi), N).
    
    v = _metric(x)
    if v < best_val:
        best_val = v
        best_x = x.copy()
    
    # Gradient descent on phi
    # Objective: f(phi) = max|conv(x(phi))| / (sum(x(phi)))^2
    # Since scale is fixed by DC=1, sum(x) is determined by phi[0]=0 -> sum=1?
    # Let's check: x[0..N-1] = 1/N sum_k exp(i*phi[k]) exp(-i*2*pi*k*n/N) ...
    # Sum_n x[n] = x[0] * N? No.
    # Sum_n x[n] = N * X[0] / N = X[0] = 1.
    # So sum(x) = 1 always (if phi[0]=0).
    # Then metric = max|conv|.
    # We just need to minimize max|conv|.
    
    # Gradient of max|conv| w.r.t. phi is tricky.
    # Let's use a smooth surrogate: S(phi) = soft_max_abs(conv(x(phi)))
    # dS/dphi_k = sum_j w_j * d|conv_j|/dphi_k
    # This requires backprop through FFT.
    
    # Alternative: Use finite differences or simple random search in phase space.
    # Given time constraints, random search / hill climbing in phase space might be robust.
    
    # Let's try a simple local search on phases.
    current_phi = phi.copy()
    current_x = x.copy()
    current_val = v
    
    T = 0.1
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        # Pick a random phase index to perturb
        k = rng.integers(1, M - (1 if N % 2 == 0 else 0)) # Avoid DC and Nyquist
        # Or perturb all slightly?
        # Let's perturb a few random phases
        num_perturb = rng.integers(1, 4)
        indices = rng.choice(M, size=num_perturb, replace=False)
        # Exclude DC and Nyquist
        indices = [i for i in indices if i != 0 and (N % 2 != 0 or i != M-1)]
        if not indices:
            continue
            
        new_phi = current_phi.copy()
        for idx in indices:
            new_phi[idx] += rng.normal(0, T)
            
        new_X = np.exp(1j * new_phi)
        new_x = np.fft.irfft(new_X, n=N)
        
        new_val = _metric(new_x)
        
        if new_val < current_val:
            current_phi = new_phi
            current_x = new_x
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_x = new_x.copy()
        else:
            # Simulated annealing step
            if rng.random() < np.exp(-(new_val - current_val) / T):
                current_phi = new_phi
                current_x = new_x
                current_val = new_val
                
        T *= 0.999
        
    return best_x, best_val


def _hybrid_optimize(deadline, rng):
    """
    Combine phase-only optimization with time-domain polishing.
    """
    best_x = None
    best_val = float('inf')
    
    # Try multiple random phase initializations
    num_restarts = 5
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for r in range(num_restarts):
        t_start = time.time()
        x, v = _phase_only_optimize(t_start + time_per_restart, rng, max_iters=2000)
        if x is not None and v < best_val:
            best_val = v
            best_x = x
    
    if best_x is None:
        return np.ones(N), _metric(np.ones(N))
        
    # Polish in time domain
    # Use a fast local search
    step_scale = 0.1
    no_improve = 0
    for round_idx in range(10000):
        if time.time() > deadline:
            break
        
        if no_improve > 50:
            step_scale *= 0.9
            no_improve = 0
            if step_scale < 0.001:
                break
                
        k = int(rng.integers(1, 5))
        indices = rng.choice(N, size=k, replace=False)
        
        best_delta = 0
        best_idx = None
        for trial in range(2):
            x_test = best_x.copy()
            for idx in indices:
                x_test[idx] += rng.normal(0, step_scale)
            v = _metric(x_test)
            if v < best_val + best_delta:
                best_delta = v - best_val
                best_idx = x_test
                
        if best_idx is not None and best_delta < 0:
            best_val += best_delta
            best_x = best_idx
            no_improve = 0
        else:
            no_improve += 1
            
    return best_x, best_val


def construct():
    deadline = time.time() + 145.0
    
    rng = np.random.default_rng(42)
    
    # Start with the phase-only hybrid approach
    best_x, best_val = _hybrid_optimize(deadline, rng)
    
    # If we have time, try a few more random restarts or different strategies
    if time.time() < deadline - 10:
        # Try a simple random search in time domain for comparison
        x_rand = rng.normal(0, 1, N)
        v_rand = _metric(x_rand)
        if v_rand < best_val:
            best_x = x_rand
            best_val = v_rand
            
    return best_x
# EVOLVE-BLOCK-END
