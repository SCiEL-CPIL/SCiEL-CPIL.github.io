"""Combined program: sparse spike+background structure with soft-max L-BFGS-B refinement."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _eval_value(h):
    """Compute the exact evaluation metric."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _soft_max(x, beta):
    """Numerically stable soft maximum of x."""
    m = np.max(x)
    if not np.isfinite(m):
        return m
    exp_vals = np.exp(beta * (x - m))
    return m + np.log(np.sum(exp_vals)) / beta

def _objective_soft(h, beta):
    """
    Soft-max surrogate objective for autoconvolution peak.
    Internally normalizes so sum(h) = N, then returns the value formula.
    """
    if not np.all(np.isfinite(h)):
        return 1e10
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    h_norm = h * (N / s)
    conv = np.convolve(h_norm, h_norm)
    sm = _soft_max(conv, beta)
    return abs(2.0 * N * sm / (N * N))

def _spike_background_h(spike_positions, spike_amps, background):
    """
    Construct h from a uniform background plus spikes.
    spike_positions: array of integer positions (0..N-1)
    spike_amps: array of amplitudes at those positions
    background: scalar base level
    Returns h of length N.
    """
    h = np.full(N, background)
    for pos, amp in zip(spike_positions, spike_amps):
        pos = int(pos) % N
        h[pos] += amp
    return h

def _spike_params_to_h(params, n_spikes, bg_scale=1.0):
    """
    Convert a flat parameter vector to h.
    params layout: [bg_level, spike_1_amp, spike_1_pos_frac, spike_2_amp, spike_2_pos_frac, ...]
    n_spikes: number of spikes
    bg_scale: scales the background contribution
    """
    bg_level = params[0]
    spike_amps = np.zeros(n_spikes)
    spike_pos_fracs = np.zeros(n_spikes)
    for i in range(n_spikes):
        base = 1 + 2 * i
        spike_amps[i] = params[base]
        spike_pos_fracs[i] = params[base + 1]
    spike_positions = np.round(spike_pos_fracs * (N - 1)).astype(int)
    spike_positions = np.clip(spike_positions, 0, N - 1)
    h = _spike_background_h(spike_positions, spike_amps, bg_level * bg_scale)
    return h

def _objective_from_params(params, beta, n_spikes, bg_scale=1.0):
    """Objective function for L-BFGS-B in parameter space."""
    h = _spike_params_to_h(params, n_spikes, bg_scale)
    return _objective_soft(h, beta)

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Structural model: h = uniform background + K high-amplitude spikes.
    Refinement: soft-max surrogate + L-BFGS-B on spike parameters,
    followed by full time-domain L-BFGS-B polish.
    """
    deadline = time.time() + 143.0  # safety margin before 150s kill

    # Start with all-ones
    best_h = np.ones(N)
    best_val = 2.0

    # ---- Phase 1: Grid search over spike configurations ----
    # Try various numbers of spikes and configurations
    for n_spikes in [1, 2, 3, 4, 5, 8, 10, 15, 20]:
        if time.time() > deadline - 60:
            break

        # For each n_spikes, try a few random configurations
        rng = np.random.RandomState(100 + n_spikes)
        n_trials = 30 if n_spikes <= 5 else 15

        for trial in range(n_trials):
            if time.time() > deadline - 60:
                break

            # Random spike positions (well-separated to avoid clumping)
            if n_spikes == 1:
                spike_positions = np.array([rng.randint(0, N)])
            else:
                # Use random permutation to get well-separated positions
                perm = rng.permutation(N)
                # Take every (N // n_spikes)-th element for even spacing
                step = max(1, N // n_spikes)
                spike_positions = perm[::step][:n_spikes]
                spike_positions = np.sort(spike_positions)

            # Random spike amplitudes (can be negative)
            spike_amps = rng.uniform(-2.0, 2.0, size=n_spikes)

            # Background level
            bg_level = rng.uniform(-0.5, 0.5)

            h_candidate = _spike_background_h(spike_positions, spike_amps, bg_level)

            s = np.sum(h_candidate)
            if abs(s) < 1e-10:
                continue

            # Normalize to sum = N
            h_candidate = h_candidate * (N / s)

            v = _eval_value(h_candidate)
            if v < best_val:
                best_val = v
                best_h = h_candidate.copy()

    # ---- Phase 2: L-BFGS-B optimization in spike-parameter space ----
    # For several spike counts, optimize the parameters
    for n_spikes in [2, 3, 4, 5, 6, 8, 10, 15, 20, 30]:
        if time.time() > deadline - 25:
            break

        # Initialize from a random configuration
        rng = np.random.RandomState(500 + n_spikes)

        for trial in range(3):
            if time.time() > deadline - 25:
                break

            if n_spikes == 1:
                spike_positions = np.array([rng.randint(0, N)])
            else:
                perm = rng.permutation(N)
                step = max(1, N // n_spikes)
                spike_positions = perm[::step][:n_spikes]
                spike_positions = np.sort(spike_positions)

            spike_amps = rng.uniform(-1.5, 1.5, size=n_spikes)
            bg_level = rng.uniform(-0.3, 0.3)

            # Build initial parameter vector
            params_init = np.zeros(1 + 2 * n_spikes)
            params_init[0] = bg_level
            for i in range(n_spikes):
                params_init[1 + 2*i] = spike_amps[i]
                params_init[2 + 2*i] = spike_positions[i] / (N - 1)  # fraction in [0,1]

            # Bounds: bg in [-2, 2], amps in [-3, 3], pos_fracs in [0, 1]
            bounds = [(-2.0, 2.0)]
            for i in range(n_spikes):
                bounds.append((-3.0, 3.0))  # amp
                bounds.append((0.0, 1.0))   # pos_frac

            # Soft-max beta: start low, then refine
            for beta in [5.0, 20.0, 50.0]:
                if time.time() > deadline - 25:
                    break

                def obj(params):
                    return _objective_from_params(params, beta, n_spikes)

                try:
                    result = minimize(
                        obj,
                        params_init,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-10}
                    )

                    if result.fun < best_val:
                        h_opt = _spike_params_to_h(result.x, n_spikes)
                        s_opt = np.sum(h_opt)
                        if abs(s_opt) > 1e-10 and np.all(np.isfinite(h_opt)):
                            h_opt = h_opt * (N / s_opt)
                            v_opt = _eval_value(h_opt)
                            if v_opt < best_val:
                                best_val = v_opt
                                best_h = h_opt.copy()
                except Exception:
                    continue

    # ---- Phase 3: Full time-domain L-BFGS-B polish ----
    # Use the best_h from spike optimization as starting point
    for beta in [10.0, 30.0, 80.0]:
        if time.time() > deadline - 15:
            break

        def obj_time(h):
            return _objective_soft(h, beta)

        # Try a few perturbations around best_h
        rng = np.random.RandomState(9000 + int(beta))
        for trial in range(4):
            if time.time() > deadline - 15:
                break

            if trial == 0:
                h_init = best_h.copy()
            else:
                h_init = best_h + 0.05 * rng.randn(N)
                s_init = np.sum(h_init)
                if abs(s_init) < 1e-10:
                    continue
                h_init = h_init * (N / s_init)

            try:
                result = minimize(
                    obj_time,
                    h_init,
                    method='L-BFGS-B',
                    options={'maxiter': 120, 'ftol': 1e-14, 'gtol': 1e-12}
                )

                if result.fun < best_val:
                    h_opt = result.x
                    s_opt = np.sum(h_opt)
                    if abs(s_opt) > 1e-10 and np.all(np.isfinite(h_opt)):
                        h_opt = h_opt * (N / s_opt)
                        v_opt = _eval_value(h_opt)
                        if v_opt < best_val:
                            best_val = v_opt
                            best_h = h_opt.copy()
            except Exception:
                continue

    # ---- Phase 4: Coordinate descent polish on individual elements ----
    if time.time() < deadline - 8:
        h_work = best_h.copy()
        step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
        for step in step_sizes:
            if time.time() > deadline - 8:
                break
            improved_any = False
            for i in range(N):
                if time.time() > deadline - 8:
                    break
                old_val = h_work[i]
                for delta in [step, -step]:
                    h_work[i] = old_val + delta
                    s = np.sum(h_work)
                    if abs(s) < 1e-10:
                        h_work[i] = old_val
                        continue
                    v = _eval_value(h_work)
                    if v < best_val:
                        best_val = v
                        best_h = h_work.copy()
                        improved_any = True
                    else:
                        h_work[i] = old_val
            if not improved_any:
                break

    # Ensure validity
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)

    return best_h
# EVOLVE-BLOCK-END
