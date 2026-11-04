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
    
    # Pre-compute the matrix A where A @ p = autoconv
    # p is the power spectrum |X|^2 (length _FFT_SIZE/2 + 1)
    # autoconv is length 2N-1
    # conv[l] = sum_k P_k * e^{i 2 pi k l / _FFT_SIZE}
    # Since P_k is real and non-negative, and conv is real (Hermitian symmetry in X implies P is symmetric? No, P is just |X|^2, which is symmetric for real x: P[k] = P[-k] = P[_FFT_SIZE-k].
    # rfft gives P[0...M] where M=_FFT_SIZE//2.
    # The autoconvolution is real.
    # conv[l] = P[0] + 2 * sum_{k=1}^{M-1} P[k] cos(2 pi k l / _FFT_SIZE) + P[M] cos(2 pi M l / _FFT_SIZE)
    # Note: for even _FFT_SIZE, the Nyquist term k=M is real.
    
    M = _FFT_SIZE // 2
    L = 2 * N - 1
    
    # Construct A matrix (L x (M+1))
    k_vals = np.arange(M + 1)
    l_vals = np.arange(L)
    
    # Outer product for cos terms
    # angles = 2 * pi * np.outer(l_vals, k_vals) / _FFT_SIZE
    # A = np.cos(angles)
    
    # Optimization: We only need to evaluate the objective. 
    # To speed up scipy, we can use a pre-computed matrix if L and M are manageable.
    # L = 799, M+1 = 257 (if _FFT_SIZE=1024). 799*257 ~ 200k floats. Very manageable.
    
    angles = 2 * np.pi * np.outer(l_vals, k_vals) / _FFT_SIZE
    A = np.cos(angles)
    
    # Handle the Nyquist term (k=M) which has cos(pi*l) = (-1)^l. 
    # The standard FFT reconstruction for irfft uses:
    # x[n] = (1/N_fft) * (X[0] + 2 * sum_{k=1}^{N/2-1} X[k] cos(...) + X[N/2] cos(pi*n))
    # Wait, the autoconvolution is the IFFT of |X|^2.
    # Let P = |X|^2.
    # conv[l] = IFFT(P)[l]
    # The IFFT formula includes a 1/N_fft factor.
    # However, the metric is scale invariant. 
    # max |conv| / (sum x)^2.
    # sum x = X[0] = P[0] (since X[0] is real for real x, and P[0]=|X[0]|^2? No. X[0] = sum x. P[0] = (sum x)^2.
    # So (sum x)^2 = P[0].
    # The metric is max_l |IFFT(P)[l]| / P[0].
    # IFFT(P)[l] = (1/_FFT_SIZE) * sum_{k=0}^{M} c_k P[k] cos(2 pi k l / _FFT_SIZE)
    # where c_0 = 1, c_M = 1 (if M>0), c_k = 2 for 0<k<M.
    
    coeffs = np.ones(M + 1)
    if M > 0:
        coeffs[1:-1] = 2.0
    
    A = A * coeffs[None, :] / _FFT_SIZE
    
    # Now the problem is:
    # Minimize t
    # Subject to:
    #   -t <= A @ p <= t
    #   p >= 0
    #   p[0] = 1  (Normalization: set P[0]=1, so integral^2=1)
    #   p is the power spectrum of some signal? 
    #   Any non-negative sequence p that is symmetric (which it is by definition in rfft) 
    #   corresponds to a valid power spectrum IF it is the PSD of a valid signal.
    #   Actually, ANY non-negative sequence P[0..M] with P[M] real (Nyquist) 
    #   defines a valid autocorrelation function (Bochner's theorem / positive definiteness).
    #   Wait, P[k] = |X[k]|^2. X is the FFT of x.
    #   Is every non-negative P a valid PSD? 
    #   Yes, if we define x = IFFT(sqrt(P) * phase). 
    #   But we need x to be real. 
    #   If P is symmetric (P[k]=P[-k]), then we can choose phases to make x real?
    #   Actually, the constraint is that the sequence r[l] = IFFT(P)[l] must be positive definite.
    #   Since P >= 0, r[l] is the autocorrelation of the signal whose PSD is P.
    #   Is it guaranteed that r[l] is positive definite? 
    #   Yes, by definition, the IFFT of a non-negative spectrum is a positive definite sequence 
    #   IF we interpret it correctly. 
    #   However, there is a subtlety: the "signal" x must be real.
    #   If we just pick P >= 0, can we always find a real x such that |FFT(x)|^2 = P?
    #   Yes. Let x = IFFT(sqrt(P) * exp(i*phi)). 
    #   For x to be real, we need X[k] = conj(X[-k]).
    #   This implies |X[k]| = |X[-k]|, which is satisfied by P[k]=P[-k].
    #   We can always choose phases phi[k] such that X is Hermitian symmetric.
    #   So the only constraints on P are P >= 0 and P[0] = 1 (normalization).
    
    # Variables: p[0..M-1] (since p[M] is determined? No, p[M] is a variable too).
    # But p[0] is fixed to 1.
    # So variables are p[1..M].
    # Number of variables: M.
    
    x0 = np.ones(M)  # Initial guess: flat spectrum (except DC)
    
    def objective(v):
        # v is p[1..M]
        p = np.empty(M + 1)
        p[0] = 1.0
        p[1:] = v
        conv = A @ p
        return np.max(np.abs(conv))
        
    def jac(v):
        # Gradient of max(|A@p|) w.r.t v
        p = np.empty(M + 1)
        p[0] = 1.0
        p[1:] = v
        conv = A @ p
        idx_max = np.argmax(np.abs(conv))
        sign = np.sign(conv[idx_max])
        # grad w.r.t p is sign * A[idx_max, :]
        # grad w.r.t v is sign * A[idx_max, 1:]
        return sign * A[idx_max, 1:]
        
    # Constraints: p >= 0
    # Since v are p[1..M], we need v >= 0.
    # We can use bounds.
    bounds = [(0, None)] * M
    
    # Run L-BFGS-B
    # L-BFGS-B is good for large scale.
    # However, objective is non-smooth (max). 
    # L-BFGS-B might struggle with the kinks.
    # Alternative: Use SLSQP or trust-constr.
    # Given M ~ 512 (if FFT_SIZE 1024), SLSQP might be slow but okay for one run.
    # Let's try L-BFGS-B first as it's fast. If it stalls, we can try others.
    
    best_val = float('inf')
    best_p = np.ones(M + 1)
    best_p[0] = 1.0
    
    # Multi-start to avoid local minima of the non-smooth max function?
    # The feasible region is convex (p>=0, p0=1). The objective is convex (max of linear functions).
    # So the global minimum is unique (or a face of it). 
    # Gradient descent methods should converge to the global min if we handle the non-smoothness.
    # L-BFGS-B uses gradient. At the kink, the gradient is not unique. 
    # The subgradient is the convex hull of the active gradients.
    # L-BFGS-B might oscillate.
    
    # Let's use a simpler approach: 
    # Since it's a convex problem, we can use `scipy.optimize.minimize` with method='SLSQP' 
    # or 'trust-constr'.
    # But SLSQP is O(n^3). n=512. 512^3 = 134M. Might be too slow per iteration.
    
    # Better: Use `scipy.optimize.linprog`? No, it's nonlinear (max of linear).
    # It IS a linear program if we introduce t!
    # Minimize t
    # Subject to:
    #   A @ p <= t * 1
    #   -A @ p <= t * 1
    #   p >= 0
    #   p[0] = 1
    
    # This is a Linear Program!
    # Variables: p[0..M], t.
    # Total vars: M + 2.
    # Constraints: 2*L inequalities + 1 equality + M inequalities (p>=0).
    # L = 799.
    # Total constraints ~ 2000.
    # Total vars ~ 514.
    # `scipy.optimize.linprog` can handle this easily and FAST.
    
    # Let's reformulate as LP.
    # c = [0, 0, ..., 0, 1] (coefficients for p[0..M], t)
    # We want to minimize t.
    
    c = np.zeros(M + 2)
    c[-1] = 1.0  # t is the last variable
    
    # Inequality constraints: A_ub @ x <= b_ub
    # 1. A @ p <= t  => A @ p - t <= 0
    # 2. -A @ p <= t => -A @ p - t <= 0
    # 3. -p <= 0 (p >= 0)
    # 4. p[0] = 1 => handled by bounds or A_eq.
    
    # Variables x = [p[0], p[1], ..., p[M], t]
    
    # A_ub block 1: [A, -1]
    # A_ub block 2: [-A, -1]
    # A_ub block 3: [-I, 0]
    
    # Dimensions:
    # A is L x (M+1)
    # A_ub will be (2L + M+1) x (M+2)
    
    A_ub = np.zeros((2 * L + M + 1, M + 2))
    b_ub = np.zeros(2 * L + M + 1)
    
    # Block 1: A @ p - t <= 0
    A_ub[:L, :M+1] = A
    A_ub[:L, M+1] = -1.0
    
    # Block 2: -A @ p - t <= 0
    A_ub[L:2*L, :M+1] = -A
    A_ub[L:2*L, M+1] = -1.0
    
    # Block 3: -p <= 0  => p >= 0
    # -1 on diagonal for p vars
    A_ub[2*L:, :M+1] = -np.eye(M + 1)
    # t is free (can be negative? No, t >= max|conv| >= 0. But LP will find t>=0 if A has non-zero entries).
    # Actually, we should constrain t >= 0? 
    # If p=0, conv=0, t can be 0. 
    # But p[0]=1, so conv is not 0.
    
    # Equality constraint: p[0] = 1
    A_eq = np.zeros((1, M + 2))
    A_eq[0, 0] = 1.0
    b_eq = np.array([1.0])
    
    # Bounds
    # p[i] >= 0 (already in A_ub)
    # t >= 0? 
    # If we don't bound t, it can be negative. 
    # But objective is minimize t. 
    # If we set t very negative, constraints A@p - t <= 0 => A@p <= t. 
    # Since A@p can be positive, t must be at least max(A@p).
    # So t will be pushed up.
    # Let's add lower bound 0 for t to be safe? 
    # Actually, max|conv| is always >= 0.
    # So t >= 0 is implied if A is not all zeros.
    
    # Solve LP
    # Use 'highs' method for speed.
    
    res = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='highs')
    
    if res.success:
        p_opt = res.x[:M+1]
        t_opt = res.x[-1]
        
        # Reconstruct x
        # We need a real signal x such that |FFT(x)|^2 = p_opt.
        # We can choose x to be real by setting the phase of X to be 0?
        # If X is real and non-negative, x is the IFFT of X, which is real and symmetric?
        # Not necessarily symmetric, but real.
        # Let X[k] = sqrt(p_opt[k]) for k=0..M.
        # Extend to full FFT with Hermitian symmetry.
        
        X_full = np.zeros(_FFT_SIZE, dtype=complex)
        # rfft indices: 0..M
        # X[0] = sqrt(p[0])
        # X[M] = sqrt(p[M]) (if M < _FFT_SIZE/2? No, M = _FFT_SIZE//2)
        # For k in 1..M-1:
        #   X[k] = sqrt(p[k])
        #   X[_FFT_SIZE-k] = sqrt(p[k]) (conjugate, so same since real)
        
        X_full[:M+1] = np.sqrt(p_opt)
        # Fill the rest
        for k in range(1, M):
            X_full[_FFT_SIZE - k] = X_full[k]
            
        # Note: X_full[M] is the Nyquist term. It is its own conjugate.
        # So X_full[M] is already set.
        
        x_opt = np.fft.irfft(X_full, _FFT_SIZE)[:N]
        
        # Check metric
        val_opt = _metric(x_opt)
        
        if val_opt < best_val:
            best_val = val_opt
            best_x = x_opt
    else:
        # Fallback
        best_x = np.ones(N)
        best_val = _metric(best_x)

    # Local refinement?
    # The LP solution is the global minimum for the relaxed problem (P >= 0).
    # Since any P >= 0 is a valid PSD for some real x (with appropriate phase), 
    # this IS the global minimum for the original problem too?
    # Wait. The original problem minimizes over x. 
    # The set of achievable P = |FFT(x)|^2 for real x is the set of non-negative, symmetric sequences.
    # The LP minimizes over ALL non-negative, symmetric sequences.
    # So the LP solution is a lower bound.
    # Is it achievable?
    # Yes, by choosing the phase of X to be 0 (or constant), x is real.
    # So the LP solution IS the global optimum.
    
    # Why did the previous program score 1.07?
    # Maybe numerical issues or the LP wasn't used?
    # The previous program used gradient descent on x, which is non-convex in x (quadratic).
    # But in P, it's linear.
    
    return best_x

def construct():
    return _construct()
# EVOLVE-BLOCK-END
