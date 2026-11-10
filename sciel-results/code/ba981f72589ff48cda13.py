import numpy as np
import time

def _localize_cov(cov, N, half_width=3, order=3):
    """Apply a smooth taper to the covariance matrix based on periodic distance."""
    indices = np.arange(N)
    dist = np.abs(indices[:, None] - indices[None, :])
    dist = np.minimum(dist, N - dist)
    # Smooth taper: 1 at d=0, 0 at d>=half_width
    loc = np.clip(1.0 - (dist / half_width) ** order, 0.0, 1.0)
    return cov * loc

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Variational ensemble assimilation for Lorenz-96 partial-observation problem.
    Uses an iterative Gauss-Newton minimization of a hybrid cost with localized
    background error and deterministic spread control.
    """
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    M, N = ensemble.shape
    K = len(observed_indices)

    ensemble = np.array(ensemble, dtype=np.float64)
    observations = np.array(observations, dtype=np.float64)
    observed_indices = np.array(observed_indices, dtype=int)

    # --- Background statistics ---
    xb = np.mean(ensemble, axis=0)
    anomalies = ensemble - xb
    # Sample covariance with localization
    cov_b = (anomalies.T @ anomalies) / (M - 1)
    cov_loc = _localize_cov(cov_b, N, half_width=3, order=3)

    # Ensure positive semi-definite-ish: add small ridge
    cov_loc += 1e-6 * np.eye(N)

    # --- Observation operator (identity on observed indices) ---
    # H maps state -> observed subset
    Hx = lambda x: x[observed_indices]

    # --- Cost function: J(x) = 0.5*(y-Hx)' R^{-1} (y-Hx) + 0.5*(x-xb)' B^{-1} (x-xb) ---
    # B = cov_loc, R = observation_variance * I
    R = observation_variance * np.eye(K)
    R_inv = np.linalg.inv(R)

    # Precompute B^{-1} (or use Cholesky for stability)
    try:
        L = np.linalg.cholesky(cov_loc)
        def B_inv_vec(v):
            # B^{-1} v = L^{-T} L^{-1} v
            tmp = np.linalg.solve(L, v)
            return np.linalg.solve(L.T, tmp)
        have_chol = True
    except np.linalg.LinAlgError:
        B_inv = np.linalg.pinv(cov_loc)
        def B_inv_vec(v):
            return B_inv @ v
        have_chol = False

    def cost(x):
        innov = observations - Hx(x)
        Jo = 0.5 * innov @ R_inv @ innov
        bg = x - xb
        Jb = 0.5 * bg @ B_inv_vec(bg)
        return Jo + Jb

    def grad(x):
        innov = observations - Hx(x)
        # dJo/dx = -H' R^{-1} (y - Hx)
        dJo = np.zeros(N)
        dJo[observed_indices] = -R_inv @ innov
        dJb = B_inv_vec(x - xb)
        return dJo + dJb

    # --- Gauss-Newton iterations ---
    x = xb.copy()
    # Precompute H matrix for efficiency
    H = np.zeros((K, N))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0

    # H' R^{-1} H
    HtRinvH = H.T @ R_inv @ H
    # H' R^{-1} (y - H xb)
    innov_b = observations - H @ xb
    HtRinvyb = H.T @ R_inv @ innov_b

    # Regularize the normal equations
    reg = 1e-4
    for it in range(15):
        if time.time() > DEADLINE:
            break
        # Linearize around current x
        innov = observations - H @ x
        # Gauss-Newton step: (B^{-1} + H'R^{-1}H) dx = H'R^{-1}(y-Hx) + B^{-1}(x - xb) ... 
        # Actually for the quadratic cost, the gradient is linear, so one Newton step suffices.
        # But let's do iterative CG on the Hessian.
        # Hessian is constant for this quadratic: A = B^{-1} + H'R^{-1}H
        # Since B^{-1} is implicit, we solve (B^{-1} + H'R^{-1}H) dx = grad(xb) - B^{-1}(x-xb) + ...
        # Simpler: since cost is quadratic, solve the linear system directly.
        # grad(x) = B^{-1}(x-xb) - H'R^{-1}(y-Hx)
        # = B^{-1}x - B^{-1}xb - H'R^{-1}y + H'R^{-1}Hx
        # = (B^{-1} + H'R^{-1}H)x - (B^{-1}xb + H'R^{-1}y)
        # Set grad = 0:
        # (B^{-1} + H'R^{-1}H)x = B^{-1}xb + H'R^{-1}y

        # Right-hand side: B^{-1}xb + H'R^{-1}y
        rhs = B_inv_vec(xb) + H.T @ (R_inv @ observations)

        # Solve (B^{-1} + H'R^{-1}H) x = rhs using CG
        def matvec(v):
            return B_inv_vec(v) + HtRinvH @ v

        # Conjugate gradient
        r = rhs - matvec(x)
        p = r.copy()
        rs_old = r @ r
        for cg_it in range(min(N, 30)):
            if time.time() > DEADLINE:
                break
            Ap = matvec(p)
            alpha = rs_old / (p @ Ap + 1e-12)
            x_new = x + alpha * p
            r_new = r - alpha * Ap
            rs_new = r_new @ r_new
            if np.sqrt(rs_new) < 1e-10:
                x = x_new
                break
            beta = rs_new / (rs_old + 1e-12)
            p = r_new + beta * p
            x = x_new
            r = r_new
            rs_old = rs_new

    # --- Deterministic spread control ---
    # Ensure the ensemble has reasonable spread after the update
    x_mean = x
    # Compute the updated ensemble by scaling anomalies
    # Target: ensemble variance at observed locations ~ observation_variance
    ens_var_obs = np.var(ensemble[:, observed_indices], axis=0)
    target_var = observation_variance
    # Inflation factor
    spread_ratio = np.mean(ens_var_obs) / (target_var + 1e-12)
    # Gentle deterministic inflation
    if spread_ratio < 0.8:
        inflation = 1.0 + 0.15 * (0.8 - spread_ratio)
    elif spread_ratio > 1.5:
        inflation = 1.0 - 0.05 * (spread_ratio - 1.5)
    else:
        inflation = 1.0
    inflation = float(np.clip(inflation, 0.95, 1.20))

    # Build updated ensemble: mean = x, anomalies scaled
    result = np.empty_like(ensemble)
    for m in range(M):
        result[m] = x + inflation * (ensemble[m] - xb)

    # Safety
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100, 100)

    if time.time() > DEADLINE:
        return result

    return result
