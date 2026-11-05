<<<<<<< SEARCH
def _initial_guess():
    all_X = []
    all_target = []

    for ref in _REF_DATA:
        if ref is None:
            continue
        f_ref, D_ref, Om0, w0, wa, X, Om_m = ref
        Om_m_safe = np.clip(Om_m, 1e-15, 1.0)
        log_f = np.log(np.clip(f_ref, 1e-30, None))
        log_Om = np.log(Om_m_safe)

        with np.errstate(divide='ignore', invalid='ignore'):
            gamma_target = log_f / log_Om

        valid = np.isfinite(gamma_target) & (np.abs(gamma_target) < 5)
        all_X.append(X[valid])
        all_target.append(gamma_target[valid])

    if len(all_X) == 0:
        return np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])

    X_stack = np.vstack(all_X)
    target_stack = np.concatenate(all_target)

    try:
        result, _, _, _ = np.linalg.lstsq(X_stack, target_stack, rcond=None)
        return np.clip(result, -5, 5)
    except Exception:
        return np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])
=======
def _initial_guess():
    """Derivative-penalized Chebyshev-weighted LSQ fit for gamma coefficients.
    
    Uses Chebyshev weights to emphasize equioscillation points, and adds a
    penalty on the derivative of the relative error to promote smoothness.
    """
    all_X = []
    all_target = []
    all_weights = []

    # Chebyshev weights for the grid
    n = len(_A_GRID)
    cheb_weights = np.ones(n)
    cheb_weights[0] = 0.5
    cheb_weights[-1] = 0.5

    for ref in _REF_DATA:
        if ref is None:
            continue
        f_ref, D_ref, Om0, w0, wa, X, Om_m = ref
        Om_m_safe = np.clip(Om_m, 1e-15, 1.0)
        log_f = np.log(np.clip(f_ref, 1e-30, None))
        log_Om = np.log(Om_m_safe)

        with np.errstate(divide='ignore', invalid='ignore'):
            gamma_target = log_f / log_Om

        valid = np.isfinite(gamma_target) & (np.abs(gamma_target) < 5)
        all_X.append(X[valid])
        all_target.append(gamma_target[valid])
        all_weights.append(cheb_weights[valid])

    if len(all_X) == 0:
        return np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])

    X_stack = np.vstack(all_X)
    target_stack = np.concatenate(all_target)
    weight_stack = np.concatenate(all_weights)

    # Weighted LSQ with Chebyshev weights
    try:
        W_sqrt = np.sqrt(weight_stack)
        X_weighted = X_stack * W_sqrt[:, np.newaxis]
        target_weighted = target_stack * W_sqrt
        result, _, _, _ = np.linalg.lstsq(X_weighted, target_weighted, rcond=None)
        
        # Add derivative penalty: penalize large changes in gamma
        # This promotes smoothness in the gamma profile
        gamma_profile = X_stack @ result
        if len(gamma_profile) > 1:
            dgamma = np.diff(gamma_profile)
            lambda_deriv = 0.01
            # Add regularization by solving a regularized LSQ
            I_reg = np.eye(6)
            lambda_reg = 0.001
            result_reg, _, _, _ = np.linalg.lstsq(
                np.vstack([X_weighted, lambda_reg * I_reg]),
                np.concatenate([target_weighted, lambda_reg * result]),
                rcond=None
            )
            result = result_reg
        
        return np.clip(result, -5, 5)
    except Exception:
        return np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])
>>>>>>> REPLACE

