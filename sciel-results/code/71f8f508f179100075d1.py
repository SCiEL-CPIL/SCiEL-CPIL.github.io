"""Legendre-basis regularized degree-5 polynomial fit with IRLS refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin within 120s limit

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    n = x.size

    if n < 6:
        return np.zeros(6)

    # Map x to [-1, 1] for Legendre polynomials
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max - x_min < 1e-12:
        x_max = x_min + 1.0
    x_leg = 2.0 * (x - x_min) / (x_max - x_min) - 1.0

    # Build Legendre basis matrix
    # P_0(t) = 1, P_1(t) = t, P_2(t) = (3t^2-1)/2, ...
    L = np.zeros((n, 6))
    L[:, 0] = 1.0
    L[:, 1] = x_leg
    L[:, 2] = (3.0 * x_leg**2 - 1.0) / 2.0
    L[:, 3] = (5.0 * x_leg**3 - 3.0 * x_leg) / 2.0
    L[:, 4] = (35.0 * x_leg**4 - 30.0 * x_leg**2 + 3.0) / 8.0
    L[:, 5] = (63.0 * x_leg**5 - 70.0 * x_leg**3 + 15.0 * x_leg) / 8.0

    # Regularized least squares in Legendre basis
    # Try several lambda values and pick best by validation-like criterion
    # Use simple OLS first, then try small regularization
    best_coeff_leg = None
    best_mse = np.inf

    # OLS in Legendre basis
    try:
        coeff_leg, _, _, _ = np.linalg.lstsq(L, y, rcond=None)
        pred = L @ coeff_leg
        mse = np.mean((y - pred)**2)
        if mse < best_mse:
            best_mse = mse
            best_coeff_leg = coeff_leg.copy()
    except np.linalg.LinAlgError:
        pass

    # Try small regularization (ridge) to reduce overfitting
    # Regularization in orthogonal basis is well-conditioned
    # Use a wider range of lambda values including very small ones
    for lam in [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        if time.time() > deadline - 15.0:
            break
        try:
            if lam == 0.0:
                coeff_reg, _, _, _ = np.linalg.lstsq(L, y, rcond=None)
            else:
                A = L.T @ L + lam * np.eye(6)
                b = L.T @ y
                coeff_reg = np.linalg.solve(A, b)
            pred = L @ coeff_reg
            mse = np.mean((y - pred)**2)
            if mse < best_mse:
                best_mse = mse
                best_coeff_leg = coeff_reg.copy()
        except np.linalg.LinAlgError:
            continue

    if best_coeff_leg is None:
        # Fallback: simple OLS
        coeff_leg = np.zeros(6)
        best_coeff_leg = coeff_leg

    # IRLS refinement in Legendre basis
    pred = L @ best_coeff_leg
    resid = y - pred
    scale = np.median(np.abs(resid)) / 0.6745
    if scale < 1e-12:
        scale = np.std(resid) + 1e-12

    def huber_weights(r, c=1.345):
        """Huber-style weights: linear downweighting for large residuals."""
        ar = np.abs(r) / scale
        w = np.ones_like(ar)
        mask = ar > c
        w[mask] = c / ar[mask]
        return w

    def tukey_weights(r, c=4.685):
        ar = np.abs(r) / scale
        w = np.zeros_like(ar)
        mask = ar <= c
        w[mask] = (1.0 - (ar[mask] / c)**2)**2
        return w

    # Use Huber first (more robust, less aggressive), then Tukey for final polish
    for it in range(20):
        if time.time() > deadline - 15.0:
            break
        # Alternate between Huber and Tukey based on iteration
        if it < 10:
            w = huber_weights(resid)
        else:
            w = tukey_weights(resid)
            if w.sum() < n * 0.05:
                w = np.ones(n)
        try:
            # Weighted least squares in Legendre basis
            A = L.T @ (L * w[:, None])
            b = L.T @ (y * w)
            # Add small regularization for stability
            A += 1e-10 * np.eye(6)
            coeff_irls = np.linalg.solve(A, b)
            pred = L @ coeff_irls
            resid = y - pred
            mse = np.mean(resid**2)
            new_scale = np.median(np.abs(resid)) / 0.6745
            if new_scale < 1e-12:
                new_scale = np.std(resid) + 1e-12
            scale = new_scale
            if mse < best_mse:
                best_mse = mse
                best_coeff_leg = coeff_irls.copy()
            # Check convergence
            if np.max(np.abs(coeff_irls - best_coeff_leg)) < 1e-12:
                break
        except np.linalg.LinAlgError:
            break

    # Final polish: unregularized OLS using the robustly-weighted solution as initialization
    # This removes any bias introduced by regularization in the IRLS step
    if time.time() < deadline - 5.0:
        try:
            # Use the IRLS solution to identify inliers, then do clean OLS
            pred = L @ best_coeff_leg
            resid = y - pred
            mad = np.median(np.abs(resid))
            inlier_mask = np.abs(resid) < 3.0 * mad
            if np.sum(inlier_mask) > 10:
                L_in = L[inlier_mask, :]
                y_in = y[inlier_mask]
                coeff_polish, _, _, _ = np.linalg.lstsq(L_in, y_in, rcond=None)
                pred_polish = L @ coeff_polish
                mse_polish = np.mean((y - pred_polish)**2)
                if mse_polish < best_mse:
                    best_mse = mse_polish
                    best_coeff_leg = coeff_polish.copy()
        except np.linalg.LinAlgError:
            pass

    # Convert Legendre coefficients to monomial coefficients in original x basis
    # p(x) = sum_{k=0}^{5} c_k * P_k(t) where t = 2(x-x_min)/(x_max-x_min) - 1
    # We need to express this as sum_{j=0}^{5} d_j * x^j
    # First, express P_k(t) in powers of t, then substitute t = (2/(x_max-x_min)) * x - (2*x_min/(x_max-x_min)) - 1
    # Let a = 2/(x_max-x_min), b = -(2*x_min/(x_max-x_min)) - 1
    # t = a*x + b
    # P_k(t) = sum_{m=0}^{k} p_{k,m} * t^m
    # Then P_k(a*x + b) = sum_{m=0}^{k} p_{k,m} * (a*x + b)^m
    # Expand (a*x + b)^m using binomial theorem

    a = 2.0 / (x_max - x_min)
    b = -(2.0 * x_min / (x_max - x_min)) - 1.0

    # Precompute binomial coefficients
    from math import comb

    # Legendre polynomial coefficients in t basis
    # P_0(t) = 1
    # P_1(t) = t
    # P_2(t) = 0.5*t^2 - 0.5
    # P_3(t) = 0.5*t^3 - 1.5*t
    # P_4(t) = 0.375*t^4 - 0.75*t^2 + 0.375
    # P_5(t) = 0.3125*t^5 - 0.875*t^3 + 0.9375*t

    legendre_coeff_t = np.zeros((6, 6))
    legendre_coeff_t[0, 0] = 1.0
    legendre_coeff_t[1, 1] = 1.0
    legendre_coeff_t[2, 2] = 0.5
    legendre_coeff_t[2, 0] = -0.5
    legendre_coeff_t[3, 3] = 0.5
    legendre_coeff_t[3, 1] = -1.5
    legendre_coeff_t[4, 4] = 0.375
    legendre_coeff_t[4, 2] = -0.75
    legendre_coeff_t[4, 0] = 0.375
    legendre_coeff_t[5, 5] = 0.3125
    legendre_coeff_t[5, 3] = -0.875
    legendre_coeff_t[5, 1] = 0.9375

    # Convert to monomial coefficients in x
    d = np.zeros(6)
    for k in range(6):
        ck = best_coeff_leg[k]
        if abs(ck) < 1e-15:
            continue
        for m in range(6):
            pkm = legendre_coeff_t[k, m]
            if abs(pkm) < 1e-15:
                continue
            # Expand (a*x + b)^m = sum_{j=0}^{m} C(m,j) * a^j * x^j * b^{m-j}
            for j in range(m + 1):
                d[j] += ck * pkm * comb(m, j) * (a**j) * (b**(m-j))

    return d

# EVOLVE-BLOCK-END
