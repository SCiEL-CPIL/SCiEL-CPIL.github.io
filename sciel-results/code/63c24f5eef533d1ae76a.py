import scipy
import numpy as np

N = 400
M = 2 * N  # Zero-padding size for linear autocorrelation

def construct():
    # Initialize X in frequency domain
    # X[0] = 1 (Sum of x is 1)
    X = np.zeros(M, dtype=complex)
    X[0] = 1.0
    
    # Initialize R (autocorrelation)
    # Start with x = 1/N * ones(N)
    # X = fft(x_pad)
    # x_pad = [1/N, ..., 1/N, 0, ..., 0]
    # X[0] = 1, X[k] = 0 for k != 0
    # P = |X|^2 = [1, 0, ..., 0]
    # R = ifft(P) = 1/M * ones(M)
    R = np.ones(M) / M
    
    # Dual variable lambda
    lam = np.zeros(M, dtype=float)
    
    # ADMM parameter
    rho = 1.0
    
    best_x = None
    best_score = np.inf
    
    # Indices for linear autocorrelation sidelobes
    # Positive lags: 1 to N-1
    # Negative lags: M-N+1 to M-1
    pos_lags = np.arange(1, N)
    neg_lags = np.arange(M - N + 1, M)
    side_lags = np.concatenate([pos_lags, neg_lags])
    
    for iteration in range(200):
        # 1. X-update
        # Target P = fft(R) - lam/rho
        P_target = np.fft.fft(R) - lam / rho
        P_target = np.real(P_target) # Should be real
        P_target = np.maximum(P_target, 0) # Magnitude squared must be non-negative
        
        # Update X
        X_new = X.copy()
        X_new[0] = 1.0 # Constraint sum(x) = 1
        
        # For k != 0, set magnitude
        # We need to handle Hermitian symmetry carefully
        # P_target is real and non-negative, and symmetric P_target[k] = P_target[M-k]
        # So |X_new[k]| = sqrt(P_target[k])
        
        mags = np.sqrt(P_target)
        
        # Update magnitudes for all k != 0
        # But we must maintain Hermitian symmetry: X[k] = conj(X[M-k])
        # This implies |X[k]| = |X[M-k]| and phase[k] = -phase[M-k]
        # Since P_target is symmetric, mags are symmetric.
        # We keep the phase from the previous X.
        
        # To ensure symmetry, we can update the first half and mirror.
        # Or simply update all and then enforce symmetry.
        # Let's update all magnitudes.
        phases = np.angle(X)
        
        X_new = mags * np.exp(1j * phases)
        X_new[0] = 1.0 + 0j
        
        # Enforce Hermitian symmetry
        # X_new[k] = conj(X_new[M-k])
        # Average them to reduce error?
        # Or just set X_new[M-k] = conj(X_new[k]) for k=1..M/2
        for k in range(1, M // 2 + 1):
            if k == M - k: # k = M/2
                X_new[k] = np.real(X_new[k])
            else:
                # Average magnitude? No, just enforce one side
                # Let's keep the phase of k and set M-k
                X_new[M - k] = np.conj(X_new[k])
        
        # Compute x_new
        x_new = np.fft.ifft(X_new).real
        x_new = x_new[:N] # Extract the signal part
        
        # 2. R-update
        # Minimize t + rho/2 * || R - Z ||^2 s.t. |R[k]| <= t for k in side_lags
        # Z = ifft(|X_new|^2 - lam/rho)
        P_current = np.abs(X_new)**2
        Z = np.fft.ifft(P_current - lam / rho).real
        
        # Solve for t
        # Objective: f(t) = t + rho/2 * sum_{k in side_lags} (clamp(Z[k], -t, t) - Z[k])^2
        # We can use a simple search or Newton.
        # Let's use a simple bisection or minimize_scalar.
        # Since t is likely small, and Z values are bounded.
        
        # Derivative approach:
        # f'(t) = 1 + rho * sum_{k in side_lags} (clamp(Z[k], -t, t) - Z[k]) * d(clamp)/dt
        # d(clamp)/dt is 1 if Z[k] > t, -1 if Z[k] < -t, 0 otherwise.
        
        # Let's just use scipy.optimize.minimize_scalar for robustness, or a simple loop.
        # Given the time limit, a simple 1D optimization is fine.
        
        def objective(t):
            if t < 0:
                return np.inf
            # Clamp Z for side lags
            R_clamped = Z.copy()
            R_clamped[side_lags] = np.clip(Z[side_lags], -t, t)
            error = np.sum((R_clamped[side_lags] - Z[side_lags])**2)
            # Note: Terms for k not in side_lags are constant (R=Z)
            return t + 0.5 * rho * error

        # Use minimize_scalar
        # Bounds: t in [0, max(|Z[side_lags]|)]
        t_max = np.max(np.abs(Z[side_lags]))
        res = scipy.optimize.minimize_scalar(objective, bounds=(0, t_max), method='bounded')
        t_opt = res.x
        
        # Compute R_new
        R_new = Z.copy()
        R_new[side_lags] = np.clip(Z[side_lags], -t_opt, t_opt)
        
        # 3. Dual update
        # lam = lam + rho * (|X_new|^2 - fft(R_new))
        P_fft = np.fft.fft(R_new).real
        lam = lam + rho * (P_current - P_fft)
        
        # Update X and R
        X = X_new
        R = R_new
        
        # Calculate score
        sum_x = np.sum(x_new)
        if sum_x == 0:
            score = np.inf
        else:
            R_sidelobes = np.concatenate([R_new[1:N], R_new[M-N+1:M]])
            max_side = np.max(np.abs(R_sidelobes))
            score = max_side / (sum_x**2)
            
        if score < best_score:
            best_score = score
            best_x = x_new.copy()

    return best_x
