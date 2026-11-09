"""Robust X-pi gate control pulse using unitary evolution and multi-stage optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import time

def _precompute_operators():
    """Precompute all constant operators for the 3-level system."""
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2.0)],
                  [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n = a_dag @ a
    I3 = np.eye(3, dtype=complex)
    n_nm1 = n @ (n - I3)
    a_plus_adag = a + a_dag
    a_minus_adag = a - a_dag
    return a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag

def _simulate_unitary(pulse_IQ, detuning, scale, anharmonicity, dt, n_steps,
                      n, I3, n_nm1, a_plus_adag, a_minus_adag):
    """Simulate unitary evolution for a given pulse, detuning, and scale."""
    U_total = I3.copy()
    H_static = detuning * n + 0.5 * anharmonicity * n_nm1
    for i in range(n_steps):
        I_val = pulse_IQ[i, 0]
        Q_val = pulse_IQ[i, 1]
        H = H_static + 0.5 * scale * (I_val * a_plus_adag + 1j * Q_val * a_minus_adag)
        U_step = expm(-1j * H * dt)
        U_total = U_step @ U_total
    return U_total

def _compute_fidelities(U, X):
    """Compute infidelity for all 6 initial states given unitary U and target X."""
    psi_0 = np.array([1, 0, 0], dtype=complex)
    psi_1 = np.array([0, 1, 0], dtype=complex)
    psi_01 = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_01n = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_0i1 = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    psi_0mi1 = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    initial_states = [psi_0, psi_1, psi_01, psi_01n, psi_0i1, psi_0mi1]
    infidelities = []
    for psi in initial_states:
        phi = X @ psi
        psi_final = U @ psi
        F = np.abs(np.vdot(phi, psi_final)) ** 2
        infid = 1.0 - F
        infidelities.append(infid)
    mean_infid = np.mean(infidelities)
    max_infid = np.max(infidelities)
    return mean_infid + max_infid

def _check_validity(pulse_IQ, amp_bound, slew_bound, dt, n_steps):
    """Check amplitude and slew rate constraints."""
    amplitudes = np.sqrt(pulse_IQ[:, 0] ** 2 + pulse_IQ[:, 1] ** 2)
    if np.any(amplitudes > amp_bound + 1e-9):
        return False
    prev = np.array([0.0, 0.0])
    for i in range(n_steps):
        diff = np.linalg.norm(pulse_IQ[i] - prev)
        if diff / dt > slew_bound + 1e-9:
            return False
        prev = pulse_IQ[i]
    diff = np.linalg.norm(np.array([0.0, 0.0]) - prev)
    if diff / dt > slew_bound + 1e-9:
        return False
    return True

def _penalty(pulse_IQ, amp_bound, slew_bound, dt, n_steps):
    """Compute constraint violation penalty (0 if valid)."""
    p = pulse_IQ
    amplitudes = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)
    amp_pen = np.sum(np.maximum(0, amplitudes - amp_bound) ** 2)
    slew_pen = 0.0
    prev = np.array([0.0, 0.0])
    max_slew = slew_bound * dt
    for i in range(n_steps):
        diff = np.linalg.norm(p[i] - prev)
        if diff > max_slew:
            slew_pen += (diff - max_slew) ** 2
        prev = p[i]
    diff = np.linalg.norm(np.array([0.0, 0.0]) - prev)
    if diff > max_slew:
        slew_pen += (diff - max_slew) ** 2
    return amp_pen + slew_pen

def _seed_pulse():
    """Generate DRAG-style Gaussian seed pulse with proper area and constraints."""
    dt = 0.5
    n_steps = 80
    T_total = 40.0
    t_samples = (np.arange(n_steps) + 0.5) * dt
    t_center = T_total / 2.0
    t = t_samples - t_center

    sigma = T_total / 6.0
    A = 0.35
    beta = 0.08

    I_base = A * np.exp(-t ** 2 / (2 * sigma ** 2))
    dI_dt = -A * t / sigma ** 2 * np.exp(-t ** 2 / (2 * sigma ** 2))
    Q_drag = -beta * dI_dt

    pulse = np.column_stack([I_base, Q_drag])

    # Normalize area to pi
    area = np.sum(I_base) * dt
    if abs(area) > 1e-10:
        pulse *= (np.pi / area)

    # Enforce amplitude constraint
    amps = np.sqrt(pulse[:, 0] ** 2 + pulse[:, 1] ** 2)
    max_amp = np.max(amps)
    if max_amp > 0.4:
        pulse *= (0.4 / max_amp)

    # Enforce slew constraint
    max_slew = 0.0
    prev = np.array([0.0, 0.0])
    for i in range(n_steps):
        diff = np.linalg.norm(pulse[i] - prev)
        slew = diff / dt
        if slew > max_slew:
            max_slew = slew
        prev = pulse[i]
    diff = np.linalg.norm(np.array([0.0, 0.0]) - prev)
    slew = diff / dt
    if slew > max_slew:
        max_slew = slew
    if max_slew > 0.12:
        pulse *= (0.12 / max_slew)

    return pulse

def pulse(target_gate, hardware_model):
    """Return an (80, 2) array of I and Q amplitudes for a 40 ns X-pi gate."""
    start_time = time.time()
    deadline = start_time + 320.0  # 10s safety margin before 330s kill

    dt = 0.5
    n_steps = 80
    T_total = 40.0
    anharmonicity = -2 * np.pi * 0.25  # rad/ns
    amp_bound = 0.4
    slew_bound = 0.12
    detuning_bound = 2 * np.pi * 0.002  # rad/ns

    a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag = _precompute_operators()
    X = np.array([[0, 1, 0],
                  [1, 0, 0],
                  [0, 0, 0]], dtype=complex)

    detunings = [-detuning_bound, 0.0, detuning_bound]
    scales = [0.97, 1.0, 1.03]

    def objective_direct(p_flat):
        p = p_flat.reshape(80, 2)
        pen = _penalty(p, amp_bound, slew_bound, dt, n_steps)
        if pen > 1e-6:
            return 1000.0 * pen + 10.0

        scores = []
        for det in detunings:
            for sc in scales:
                U = _simulate_unitary(p, det, sc, anharmonicity, dt, n_steps,
                                      n, I3, n_nm1, a_plus_adag, a_minus_adag)
                score = _compute_fidelities(U, X)
                scores.append(score)

        raw_score = np.mean(scores) + np.max(scores)
        return raw_score + 100.0 * pen

    # Generate seed
    pulse_seed = _seed_pulse()
    x0 = pulse_seed.flatten()
    best_pulse = pulse_seed.copy()
    best_score = objective_direct(x0)

    bounds = [(-0.4, 0.4)] * 160

    # Multi-stage optimization
    stages = [
        ('L-BFGS-B', {'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-7, 'maxfun': 5000}, 120),
        ('Powell', {'maxiter': 300, 'xtol': 1e-8, 'ftol': 1e-10}, 80),
        ('L-BFGS-B', {'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 8000}, 80),
        ('Nelder-Mead', {'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}, 40),
    ]

    for method_name, opts, budget in stages:
        if time.time() > deadline - 15.0:
            break

        stage_start = time.time()
        stage_deadline = stage_start + budget - 5.0

        try:
            if method_name == 'L-BFGS-B':
                res = minimize(objective_direct, x0, method='L-BFGS-B',
                               bounds=bounds, options=opts)
            elif method_name == 'Powell':
                res = minimize(objective_direct, x0, method='Powell',
                               bounds=bounds, options=opts)
            elif method_name == 'Nelder-Mead':
                res = minimize(objective_direct, x0, method='Nelder-Mead',
                               options=opts)
            else:
                continue

            if res.fun < best_score:
                best_pulse = res.x.reshape(80, 2).copy()
                best_score = res.fun
                x0 = res.x.copy()
            else:
                x0 = best_pulse.flatten()

        except Exception:
            pass

        if time.time() > deadline - 10.0:
            break

    # Final validation
    if not _check_validity(best_pulse, amp_bound, slew_bound, dt, n_steps):
        # Fall back to seed if optimization broke constraints
        best_pulse = _seed_pulse()

    return best_pulse
# EVOLVE-BLOCK-END
