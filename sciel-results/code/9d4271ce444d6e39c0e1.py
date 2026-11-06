"""Model-and-optimize: parameterize the 3-layer Ry/CZ circuit and directly
minimize the exact mean energy gap per spin against the exact ground state."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import scipy.optimize

# Explicit global wall-clock deadline guard (evaluator kills at 150 s).
_GLOBAL_DEADLINE = time.time() + 140.0
_CACHE = {}


def _precompute_masks(n):
    """Precompute index arrays for fast Ry and CZ operations."""
    dim = 2 ** n
    idx = np.arange(dim)
    ry_masks = {}
    for q in range(n):
        mask = 1 << q
        zero_idx = idx[(idx & mask) == 0]
        one_idx = zero_idx | mask
        ry_masks[q] = (zero_idx, one_idx)
    cz_masks = {}
    for a in range(n - 1):
        for b in range(a + 1, n):
            if b == a + 1:  # only adjacent pairs needed
                mask = (1 << a) | (1 << b)
                both_one = idx[(idx & mask) == mask]
                cz_masks[(a, b)] = both_one
    return ry_masks, cz_masks


def _apply_ry_fast(psi, zero_idx, one_idx, theta):
    """Apply single-qubit Ry(theta) using precomputed indices."""
    c = np.cos(theta * 0.5)
    s = np.sin(theta * 0.5)
    out = psi.copy()
    a0 = psi[zero_idx]
    a1 = psi[one_idx]
    out[zero_idx] = c * a0 - s * a1
    out[one_idx] = s * a0 + c * a1
    return out


def _apply_cz_fast(psi, both_one_idx):
    """Apply controlled-Z using precomputed indices."""
    out = psi.copy()
    out[both_one_idx] = -out[both_one_idx]
    return out


def _circuit_state(theta, n, layers, ry_masks, cz_masks):
    dim = 2 ** n
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    for l in range(layers):
        for i in range(n):
            zero_idx, one_idx = ry_masks[i]
            psi = _apply_ry_fast(psi, zero_idx, one_idx, theta[l, i])
        if l < layers - 1:
            for i in range(n - 1):
                both_one = cz_masks[(i, i + 1)]
                psi = _apply_cz_fast(psi, both_one)
    return psi


def _build_H(n, coupling, field):
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    # ZZ (Ising) terms: -coupling * sum_i Z_i Z_{i+1}
    for i in range(n - 1):
        ma = 1 << i
        mb = 1 << (i + 1)
        for state in range(dim):
            za = 1.0 if ((state & ma) == 0) else -1.0
            zb = 1.0 if ((state & mb) == 0) else -1.0
            H[state, state] += -coupling * za * zb
    # Transverse-field terms: -field * sum_i X_i
    for i in range(n):
        m = 1 << i
        for state in range(dim):
            H[state, state ^ m] += -field
    return H


def _ground_state(n, coupling, field):
    """Compute exact ground state and energy."""
    H = _build_H(n, coupling, field)
    evals, evecs = np.linalg.eigh(H)
    E0 = float(evals[0])
    psi0 = evecs[:, 0]
    return E0, psi0, H


def _mean_field_init(n, coupling, field, layers):
    """
    Variational mean-field initialization.
    For each site, estimate the optimal Ry angle from local field balance.
    In transverse-field Ising model with open boundary, the effective transverse
    field on site i depends on its neighbors' magnetization.
    """
    theta = np.zeros((layers, n))
    
    # Simple mean-field: estimate effective field per site
    # For open chain, edge sites have 1 neighbor, bulk sites have 2
    # The optimal angle for Ry in a single-qubit Hamiltonian -h_x X - h_z Z
    # satisfies tan(theta/2) = h_x / h_z (approximately)
    
    # Estimate neighbor magnetization ~ 0 for initial guess
    # So effective transverse field ~ field, longitudinal ~ coupling * avg_neighbor_Z
    # For symmetric initialization, assume <Z> ~ 0, so theta ~ arctan(field/coupling)
    
    # More refined: use the fact that ground state has <X> > 0 and <Z> depends on field
    # For weak field, state is mostly |00...0> or |11...1>, theta small
    # For strong field, state is mostly |+>^n, theta ~ pi/2
    
    # Interpolate based on field strength
    # At field=0, ground state is |00...0> or |11...1>, theta ~ 0
    # At field>>coupling, ground state is |+>^n, theta ~ pi/2
    
    # Use a smooth interpolation
    ratio = field / max(coupling, 1e-10)
    # theta_last ~ arctan(ratio) capped at pi/2
    theta_last = np.arctan(ratio)
    theta_last = np.clip(theta_last, 0, np.pi / 2)
    
    # Earlier layers can be slightly different to allow entanglement building
    # Use a simple schedule: layer 0 slightly smaller, layer 1 medium, layer 2 = theta_last
    theta[0] = 0.7 * theta_last
    theta[1] = 0.85 * theta_last
    theta[2] = theta_last
    
    return theta


def _optimize_case(n, coupling, field, layers, time_budget):
    E0, psi0, H = _ground_state(n, coupling, field)
    dim = 2 ** n
    ry_masks, cz_masks = _precompute_masks(n)
    
    def circuit_state_vec(theta_flat):
        theta = theta_flat.reshape(layers, n)
        psi = _circuit_state(theta, n, layers, ry_masks, cz_masks)
        return psi
    
    def energy_obj(theta_flat):
        psi = circuit_state_vec(theta_flat)
        return float(np.real(np.vdot(psi, H @ psi))) - E0
    
    def state_mismatch(theta_flat):
        """Least-squares objective: match circuit state to ground state."""
        psi = circuit_state_vec(theta_flat)
        # Complex least squares: separate real and imaginary parts
        diff = psi - psi0
        return np.concatenate([diff.real, diff.imag])
    
    bounds_lower = -np.pi * np.ones(layers * n)
    bounds_upper = np.pi * np.ones(layers * n)
    
    rng = np.random.default_rng(12345)
    
    # Generate multiple starting points
    starts = []
    # Mean-field init
    starts.append(_mean_field_init(n, coupling, field, layers).reshape(-1))
    # Zero init
    starts.append(np.zeros(layers * n))
    # Random inits
    n_random = 15
    for _ in range(n_random):
        starts.append(rng.uniform(-np.pi, np.pi, size=layers * n))
    
    best_theta = np.zeros(layers * n)
    best_energy = np.inf
    t0 = time.time()
    
    for s in starts:
        if time.time() - t0 > time_budget - 3.0:
            break
        try:
            # Use least_squares for state matching (more robust landscape)
            res = scipy.optimize.least_squares(
                state_mismatch, s,
                bounds=(bounds_lower, bounds_upper),
                method='trf',
                max_nfev=200,
                xtol=1e-12,
                ftol=1e-12,
            )
            e_val = energy_obj(res.x)
            if e_val < best_energy:
                best_energy = e_val
                best_theta = res.x.copy()
        except Exception:
            pass
    
    # Polish with L-BFGS-B on energy for final precision
    if time.time() - t0 < time_budget - 1.0 and best_energy < np.inf:
        try:
            res_polish = scipy.optimize.minimize(
                energy_obj, best_theta, method="L-BFGS-B",
                bounds=list(zip(bounds_lower, bounds_upper)),
                options={"maxiter": 500, "ftol": 1e-16, "gtol": 1e-12},
            )
            if res_polish.fun < best_energy:
                best_energy = res_polish.fun
                best_theta = res_polish.x.copy()
        except Exception:
            pass
    
    # Guard against non-finite results.
    if not np.all(np.isfinite(best_theta)):
        best_theta = np.zeros(layers * n)
    
    return best_theta.reshape(layers, n)


def angles(n, coupling, field, layers):
    key = (int(n), float(coupling), float(field), int(layers))
    if key in _CACHE:
        return _CACHE[key].copy()

    remaining = _GLOBAL_DEADLINE - time.time()
    if remaining <= 3.0:
        # Not enough time: return a sane product-state default (|+>^n on last layer).
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        _CACHE[key] = theta
        return theta.copy()

    # Per-call budget: keep total under the deadline with a safety margin.
    budget = min(remaining * 0.85, 18.0)

    try:
        theta = _optimize_case(n, coupling, field, layers, budget)
    except Exception:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2

    if not np.all(np.isfinite(theta)):
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2

    _CACHE[key] = theta
    return theta.copy()
# EVOLVE-BLOCK-END
