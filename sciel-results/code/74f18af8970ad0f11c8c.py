"""Combined closure: standardized degree-5 polynomial with adaptive ridge penalty."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import comb


def construct(training_x, training_subgrid_tendency):
    t0 = time.time()
    deadline = 110.0  # Safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = x.shape[0]
    if n == 0:
        return np.zeros(6, dtype=np.float64)

    if n != y.shape[0]:
        m = min(n, y.shape[0])
        x = x[:m]
        y = y[:m]
        n = m
        if n == 0:
            return np.zeros(6, dtype=np.float64)

    # Standardize x for numerical stability
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    if sigma <= 1e-14:
        # Degenerate: constant x -> fit constant
        c = np.zeros(6, dtype=np.float64)
        c[0] = float(np.mean(y))
        return c

    u = (x - mu) / sigma

    # Standardize y for better conditioning
    y_mean = float(np.mean(y))
    y_std = float(np.std(y))
    if y_std <= 1e-14:
        y_std = 1.0
    y_hat = (y - y_mean) / y_std

    # Build Vandermonde matrix for u (degree 5, increasing powers)
    V = np.vander(u, N=6, increasing=True)

    # Adaptive ridge penalty selection
    # Try several lambda values and pick the one with best validation error
    # Use a simple train/val split if we have enough samples
    use_holdout = n >= 200
    if use_holdout:
        idx = np.random.permutation(n)
        split = int(0.8 * n)
        tr = idx[:split]
        va = idx[split:]
    else:
        tr = np.arange(n)
        va = None

    # Grid of lambda values to try
    lambdas = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    best_w = None
    best_err = np.inf

    for lam in lambdas:
        # Check time
        if time.time() - t0 > deadline:
            break

        if lam == 0.0:
            # Unregularized least squares
            try:
                w, *_ = np.linalg.lstsq(V[tr], y_hat[tr], rcond=None)
            except np.linalg.LinAlgError:
                continue
        else:
            # Ridge regression: (V^T V + lam * diag) w = V^T y
            # Keep intercept (first column) unregularized
            ATA = V[tr].T @ V[tr]
            ATy = V[tr].T @ y_hat[tr]
            lam_diag = np.full(6, lam)
            lam_diag[0] = 0.0  # Don't regularize intercept
            M = ATA + np.diag(lam_diag)
            try:
                w = np.linalg.solve(M, ATy)
            except np.linalg.LinAlgError:
                # Fallback to SVD-based solve
                try:
                    U, s, Vt = np.linalg.svd(V[tr], full_matrices=False)
                    # Apply ridge in SVD space
                    s_reg = np.where(s < 1e-12, 1e-12, s)
                    w = Vt.T @ (U.T @ y_hat[tr] * (s_reg / (s_reg**2 + lam)))
                except np.linalg.LinAlgError:
                    continue

        # Evaluate error on validation set (or training if no holdout)
        if va is not None:
            pred = V[va] @ w
            err = float(np.mean((pred - y_hat[va]) ** 2))
        else:
            pred = V[tr] @ w
            err = float(np.mean((pred - y_hat[tr]) ** 2))

        if err < best_err:
            best_err = err
            best_w = w

    # Fallback if no valid solution found
    if best_w is None:
        try:
            best_w, *_ = np.linalg.lstsq(V, y_hat, rcond=None)
        except np.linalg.LinAlgError:
            best_w = np.zeros(6, dtype=np.float64)

    # Convert from standardized basis (u) to raw basis (x)
    # We have: P(x) = sum_j w_j * u^j, where u = (x - mu)/sigma
    # We want: P(x) = sum_k c_k * x^k
    # So: sum_j w_j * ((x - mu)/sigma)^j = sum_k c_k * x^k
    # Expand: sum_j w_j * sigma^(-j) * (x - mu)^j
    # Let z = x - mu, then P(x) = sum_j (w_j / sigma^j) * z^j
    # Now expand z^j = (x - mu)^j = sum_{k=0}^{j} C(j,k) * x^k * (-mu)^(j-k)
    # So: P(x) = sum_j (w_j / sigma^j) * sum_{k=0}^{j} C(j,k) * x^k * (-mu)^(j-k)
    #         = sum_k x^k * [ sum_{j=k}^{5} (w_j / sigma^j) * C(j,k) * (-mu)^(j-k) ]
    # Therefore: c_k = sum_{j=k}^{5} (w_j / sigma^j) * C(j,k) * (-mu)^(j-k)

    c = np.zeros(6, dtype=np.float64)
    for k in range(6):
        c_k = 0.0
        for j in range(k, 6):
            term = (best_w[j] / (sigma ** j)) * comb(j, k) * ((-mu) ** (j - k))
            c_k += term
        c[k] = c_k

    # Un-standardize: we fitted y_hat = (y - y_mean)/y_std
    # So the fitted polynomial gives (y - y_mean)/y_std
    # We want y, so multiply by y_std and add y_mean
    # But wait: the polynomial is in terms of x, not y_hat
    # Actually, we need to be careful:
    # We fitted: y_hat = P_u(u) = sum_j w_j * u^j
    # And y_hat = (y - y_mean)/y_std
    # So y = y_mean + y_std * P_u(u)
    # And P_u(u) = P_x(x) (same polynomial, just different basis)
    # So y = y_mean + y_std * P_x(x)
    # Therefore, the coefficients of P_x need to be scaled:
    # P_x(x) = sum_k c_k * x^k
    # y = y_mean + y_std * sum_k c_k * x^k
    #     = y_mean + sum_k (y_std * c_k) * x^k
    # So the final coefficients are: c_final[k] = y_std * c[k] for k>=1, and c_final[0] = y_mean + y_std * c[0]

    # Actually, let me re-derive this more carefully.
    # We have: y_hat = (y - y_mean) / y_std
    # And: y_hat = sum_j w_j * u^j
    # So: (y - y_mean) / y_std = sum_j w_j * ((x - mu)/sigma)^j
    # Therefore: y = y_mean + y_std * sum_j w_j * ((x - mu)/sigma)^j
    #             = y_mean + sum_j (w_j * y_std / sigma^j) * (x - mu)^j
    # Now expand (x - mu)^j = sum_{k=0}^{j} C(j,k) * x^k * (-mu)^(j-k)
    # So: y = y_mean + sum_j (w_j * y_std / sigma^j) * sum_{k=0}^{j} C(j,k) * x^k * (-mu)^(j-k)
    #      = y_mean + sum_k x^k * [ sum_{j=k}^{5} (w_j * y_std / sigma^j) * C(j,k) * (-mu)^(j-k) ]
    # Therefore: c_final[k] = sum_{j=k}^{5} (w_j * y_std / sigma^j) * C(j,k) * (-mu)^(j-k)
    # And c_final[0] includes the y_mean term.

    # Let me redo the coefficient calculation with the y_std scaling included.
    c_final = np.zeros(6, dtype=np.float64)
    for k in range(6):
        c_k = 0.0
        for j in range(k, 6):
            term = (best_w[j] * y_std / (sigma ** j)) * comb(j, k) * ((-mu) ** (j - k))
            c_k += term
        c_final[k] = c_k

    # Add the constant offset y_mean to the constant term
    c_final[0] += y_mean

    return c_final
# EVOLVE-BLOCK-END
