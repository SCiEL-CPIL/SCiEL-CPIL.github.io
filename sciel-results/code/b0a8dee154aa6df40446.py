"""Strategy: Nadaraya-Watson denoising + degree-5 polynomial fit with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _nadaraya_watson(x, y, x_eval, bandwidth, chunk_size=5000):
    """Compute Nadaraya-Watson kernel regression estimate at x_eval points using chunked computation."""
    n_eval = len(x_eval)
    y_est = np.empty(n_eval)
    for start in range(0, n_eval, chunk_size):
        end = min(start + chunk_size, n_eval)
        xe = x_eval[start:end]
        diff = xe[:, None] - x[None, :]
        weights = np.exp(-0.5 * (diff / bandwidth) ** 2)
        wsum = np.sum(weights, axis=1)
        wsum[wsum < 1e-30] = 1e-30
        y_est[start:end] = np.sum(weights * y[None, :], axis=1) / wsum
    return y_est


def _legendre_basis_scaled(x_scaled, degree=5):
    """Compute Legendre polynomial basis for x_scaled (already centered/scaled).
    Maps x_scaled to [-1,1] range using its min/max, then evaluates P_0..P_degree.
    Returns (n, degree+1) array and the mapping parameters (x_min, x_max)."""
    x_min = np.min(x_scaled)
    x_max = np.max(x_scaled)
    if x_max - x_min < 1e-10:
        x_max = x_min + 1.0
    t = 2.0 * (x_scaled - x_min) / (x_max - x_min) - 1.0
    n = len(x_scaled)
    P = np.zeros((n, degree + 1))
    P[:, 0] = 1.0
    if degree >= 1:
        P[:, 1] = t
    for k in range(2, degree + 1):
        P[:, k] = ((2 * k - 1) * t * P[:, k - 1] - (k - 1) * P[:, k - 2]) / k
    return P, (x_min, x_max)


def _legendre_to_monomial_coefficients(a, x_min, x_max, degree=5):
    """Convert Legendre coefficients a (for P_0..P_degree on [-1,1]) to monomial
    coefficients in the original x_scaled space.
    
    P_k(t) where t = 2*(x - x_min)/(x_max - x_min) - 1
    We need: sum_k a_k P_k(t(x)) = sum_i c_i x^i
    """
    # Evaluate on enough points to determine the monomial coefficients
    n_pts = degree + 1
    # Use points spread across the range
    nodes = np.linspace(x_min, x_max, n_pts)
    # Compute t values
    if x_max - x_min < 1e-10:
        t_vals = np.zeros(n_pts)
    else:
        t_vals = 2.0 * (nodes - x_min) / (x_max - x_min) - 1.0
    # Evaluate Legendre polynomials at t_vals
    P_vals = np.zeros((n_pts, degree + 1))
    P_vals[:, 0] = 1.0
    if degree >= 1:
        P_vals[:, 1] = t_vals
    for k in range(2, degree + 1):
        P_vals[:, k] = ((2 * k - 1) * t_vals * P_vals[:, k - 1] - (k - 1) * P_vals[:, k - 2]) / k
    # Evaluate monomial basis at nodes
    V = np.vander(nodes, degree + 1, increasing=True).T  # (n_pts, degree+1)
    # Solve V @ c = P_vals @ a
    c = np.linalg.solve(V, P_vals @ a)
    return c


def _legendre_basis(x, degree=5):
    """Compute Legendre polynomial basis values (shifted to [-1,1]) for x in [0,1].
    Returns array of shape (n, degree+1) with columns P_0..P_degree."""
    n = len(x)
    # Shift x to [-1, 1]
    t = 2.0 * x - 1.0
    P = np.zeros((n, degree + 1))
    P[:, 0] = 1.0
    if degree >= 1:
        P[:, 1] = t
    for k in range(2, degree + 1):
        P[:, k] = ((2 * k - 1) * t * P[:, k - 1] - (k - 1) * P[:, k - 2]) / k
    return P


def _legendre_to_monomial(P_basis, x_train, degree=5):
    """Convert Legendre-basis coefficients to monomial coefficients.
    Fits a degree-5 monomial polynomial to the Legendre basis evaluated at x_train."""
    # P_basis: (n, degree+1)
    # We want monomial coeffs c such that sum(c_i * x^i) ≈ sum(a_k * P_k(x))
    # Build Vandermonde for monomials
    V_mono = np.vander(x_train, degree + 1, increasing=True).T  # (n, degree+1)
    # Solve least squares: V_mono @ c ≈ P_basis @ a
    # But we don't have 'a' here; this function is for conversion after we have Legendre coeffs.
    # Actually, we need: given Legendre coeffs a, find monomial coeffs c.
    # P_k(x) = sum_{i=0}^{k} m_{k,i} x^i
    # So sum_k a_k P_k(x) = sum_i (sum_{k>=i} a_k m_{k,i}) x^i
    # We can compute the monomial expansion of each Legendre poly.
    pass
    return None


def _legendre_coeffs_to_monomial(a, degree=5):
    """Convert Legendre polynomial coefficients a (for P_0..P_degree) to monomial coefficients.
    Uses the known expansion of Legendre polynomials in powers of x (on [-1,1])."""
    # Build the transformation matrix: P_k(x) = sum_{i=0}^{k} T[k,i] x^i
    # We'll compute T by evaluating on enough points and solving, or use recurrence.
    # Simpler: evaluate both bases at enough distinct points and solve.
    n_pts = degree + 1
    # Use Chebyshev nodes for stability
    nodes = np.cos(np.pi * np.arange(n_pts) / (n_pts - 1)) if n_pts > 1 else np.array([0.0])
    # Evaluate Legendre at nodes
    P_nodes = _legendre_basis((nodes + 1.0) / 2.0, degree)  # shape (n_pts, degree+1)
    # Evaluate monomial basis at nodes
    V_nodes = np.vander(nodes, degree + 1, increasing=True).T  # shape (n_pts, degree+1)
    # We want: P_nodes @ a = V_nodes @ c  =>  c = V_nodes^{-1} P_nodes a
    # But V_nodes is square (n_pts x n_pts), so:
    T = np.linalg.solve(V_nodes, P_nodes)  # T @ a = c, shape (degree+1, degree+1)
    c = T @ a
    return c


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = len(x)
    if n < 20:
        return np.zeros(6)

    # Remove any NaNs or Infs
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)

    if n < 20:
        return np.zeros(6)

    # Center and scale x for better conditioning
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-10:
        x_std = 1.0
    x_scaled = (x - x_mean) / x_std

    # Stratified split: sort by x, then alternate to ensure both sets cover the full range
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    sorted_indices = perm[np.argsort(x_scaled[perm])]
    val_frac = 0.2
    val_size = max(100, int(n * val_frac))
    # Take every 1/val_frac-th point in sorted order for validation
    stride = max(1, int(1.0 / val_frac))
    val_idx = sorted_indices[::stride][:val_size]
    val_set = set(val_idx.tolist())
    train_idx = np.array([i for i in sorted_indices if i not in val_set])

    x_train = x_scaled[train_idx]
    y_train = y[train_idx]
    x_val = x_scaled[val_idx]
    y_val = y[val_idx]

    # Precompute Legendre basis for training and validation
    P_train, (x_min, x_max) = _legendre_basis_scaled(x_train, degree=5)
    P_val, _ = _legendre_basis_scaled(x_val, degree=5)

    def _score_candidate_coeff_scaled(coeff_scaled):
        """Compute normalized MSE on validation set using monomial coefficients in scaled space."""
        pred_val = np.polynomial.polynomial.polyval(x_val, coeff_scaled)
        mse = np.mean((pred_val - y_val) ** 2)
        var_y = np.var(y_val)
        if var_y < 1e-10:
            return 1e10
        return mse / var_y

    def _legendre_to_scaled_monomial(a):
        """Convert Legendre coefficients (on [-1,1] mapped from x_train range) to
        monomial coefficients in x_scaled space."""
        return _legendre_to_monomial_coefficients(a, x_min, x_max, degree=5)

    def _convert_to_original(coeff_scaled, x_mean, x_std):
        """Convert coefficients from scaled to original x scale."""
        a = 1.0 / x_std
        b = -x_mean / x_std
        coeff = np.zeros(6)
        for i in range(6):
            for j in range(i + 1):
                binom = 1.0
                for k in range(j):
                    binom *= (i - k) / (k + 1)
                coeff[j] += coeff_scaled[i] * binom * (a ** j) * (b ** (i - j))
        return np.real(coeff)

    best_coeff = None
    best_score = np.inf

    # Strategy 1: Nadaraya-Watson denoising + Legendre-based fit
    # Use a broader and finer grid of bandwidths
    bandwidths = [0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
    for bw in bandwidths:
        if time.time() > deadline - 30:
            break
        try:
            y_den = _nadaraya_watson(x_train, y_train, x_train, bw)
            # Fit in Legendre basis: solve (P^T P) a = P^T y_den
            A_leg = P_train.T @ P_train + 1e-10 * np.eye(6)
            B_leg = P_train.T @ y_den
            a_leg = np.linalg.solve(A_leg, B_leg)
            if np.all(np.isfinite(a_leg)):
                coeff_scaled = _legendre_to_scaled_monomial(a_leg)
                if np.all(np.isfinite(coeff_scaled)):
                    score = _score_candidate_coeff_scaled(coeff_scaled)
                    if score < best_score:
                        best_score = score
                        best_coeff = coeff_scaled.copy()
        except:
            pass

    # Strategy 1b: Two-stage NW denoising (coarse then fine)
    for bw_coarse in [0.5, 0.8, 1.0]:
        for bw_fine in [0.15, 0.2, 0.25, 0.3, 0.4]:
            if time.time() > deadline - 25:
                break
            try:
                y_den1 = _nadaraya_watson(x_train, y_train, x_train, bw_coarse)
                y_den2 = _nadaraya_watson(x_train, y_den1, x_train, bw_fine)
                A_leg = P_train.T @ P_train + 1e-10 * np.eye(6)
                B_leg = P_train.T @ y_den2
                a_leg = np.linalg.solve(A_leg, B_leg)
                if np.all(np.isfinite(a_leg)):
                    coeff_scaled = _legendre_to_scaled_monomial(a_leg)
                    if np.all(np.isfinite(coeff_scaled)):
                        score = _score_candidate_coeff_scaled(coeff_scaled)
                        if score < best_score:
                            best_score = score
                            best_coeff = coeff_scaled.copy()
            except:
                pass

    # Strategy 2: Ridge regression in Legendre basis with various lambdas
    PTP = P_train.T @ P_train
    lambdas = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    for lam in lambdas:
        if time.time() > deadline - 20:
            break
        try:
            A_leg = PTP + lam * np.eye(6)
            B_leg = P_train.T @ y_train
            a_leg = np.linalg.solve(A_leg, B_leg)
            if np.all(np.isfinite(a_leg)):
                coeff_scaled = _legendre_to_scaled_monomial(a_leg)
                if np.all(np.isfinite(coeff_scaled)):
                    score = _score_candidate_coeff_scaled(coeff_scaled)
                    if score < best_score:
                        best_score = score
                        best_coeff = coeff_scaled.copy()
        except:
            pass

    # Strategy 2b: Ridge on NW-denoised data
    for bw in [0.2, 0.3, 0.5]:
        for lam in [1e-6, 1e-4, 1e-3, 1e-2]:
            if time.time() > deadline - 15:
                break
            try:
                y_den = _nadaraya_watson(x_train, y_train, x_train, bw)
                A_leg = PTP + lam * np.eye(6)
                B_leg = P_train.T @ y_den
                a_leg = np.linalg.solve(A_leg, B_leg)
                if np.all(np.isfinite(a_leg)):
                    coeff_scaled = _legendre_to_scaled_monomial(a_leg)
                    if np.all(np.isfinite(coeff_scaled)):
                        score = _score_candidate_coeff_scaled(coeff_scaled)
                        if score < best_score:
                            best_score = score
                            best_coeff = coeff_scaled.copy()
            except:
                pass

    # Strategy 3: Standard polynomial fit on raw (non-denoised) data
    try:
        coeff_scaled = np.polynomial.polynomial.polyfit(x_train, y_train, 5)
        if np.all(np.isfinite(coeff_scaled)):
            score = _score_candidate_coeff_scaled(coeff_scaled)
            if score < best_score:
                best_score = score
                best_coeff = coeff_scaled.copy()
    except:
        pass

    # Strategy 4: Weighted polynomial fit - weight points by density
    for bw in [0.2, 0.3, 0.5]:
        if time.time() > deadline - 10:
            break
        try:
            diff = x_train[:, None] - x_train[None, :]
            dens = np.exp(-0.5 * (diff / bw) ** 2).sum(axis=1)
            weights = dens / (dens.sum() + 1e-10)
            coeff_scaled = np.polynomial.polynomial.polyfit(x_train, y_train, 5, w=weights)
            if np.all(np.isfinite(coeff_scaled)):
                score = _score_candidate_coeff_scaled(coeff_scaled)
                if score < best_score:
                    best_score = score
                    best_coeff = coeff_scaled.copy()
        except:
            pass

    # Strategy 5: NW denoising + standard polyfit
    for bw in [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.8, 1.0]:
        if time.time() > deadline - 8:
            break
        try:
            y_den = _nadaraya_watson(x_train, y_train, x_train, bw)
            coeff_scaled = np.polynomial.polynomial.polyfit(x_train, y_den, 5)
            if np.all(np.isfinite(coeff_scaled)):
                score = _score_candidate_coeff_scaled(coeff_scaled)
                if score < best_score:
                    best_score = score
                    best_coeff = coeff_scaled.copy()
        except:
            pass

    # Final safety check
    if best_coeff is None:
        # Fallback: simple polyfit on all data
        try:
            best_coeff = np.polynomial.polynomial.polyfit(x_scaled, y, 5)
        except:
            return np.zeros(6)
    else:
        # Polish: refit using all data with the best bandwidth strategy
        # Try fitting on the full dataset and compare
        try:
            coeff_full = np.polynomial.polynomial.polyfit(x_scaled, y, 5)
            if np.all(np.isfinite(coeff_full)):
                pred_full = np.polynomial.polynomial.polyval(x_val, coeff_full)
                mse_full = np.mean((pred_full - y_val) ** 2)
                var_y = np.var(y_val)
                score_full = mse_full / (var_y + 1e-10)
                if score_full < best_score:
                    best_coeff = coeff_full.copy()
                    best_score = score_full
        except:
            pass

    # Ensure all coefficients are finite
    if not np.all(np.isfinite(best_coeff)):
        return np.zeros(6)

    # Convert to original scale
    coeff_original = _convert_to_original(best_coeff, x_mean, x_std)

    # Ensure all coefficients are finite
    if not np.all(np.isfinite(coeff_original)):
        return np.zeros(6)

    # Clip extremely large coefficients that may cause numerical issues
    max_coeff = np.max(np.abs(coeff_original))
    if max_coeff > 1e6:
        coeff_original = np.sign(coeff_original) * np.minimum(np.abs(coeff_original), 1e6)

    return coeff_original

# EVOLVE-BLOCK-END
