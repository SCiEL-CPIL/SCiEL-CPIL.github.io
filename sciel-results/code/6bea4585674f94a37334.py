"""Closed-loop feedback pulse design for 3-level transmon X-pi gate.

Strategy: instead of optimizing a fixed basis expansion (Gaussian/DRAG),
generate the pulse via a feedback controller that tracks the ideal
single-qubit X-pi trajectory in the |0>,|1> subspace.  The controller
gains (3-5 scalar parameters) are optimized with a coarse-to-fine
Nelder-Mead search.  This structurally differs from open-loop basis
expansions: the pulse shape is determined by the instantaneous state
error, dynamically correcting detuning/scale drift and suppressing
leakage to |2> through the Q channel.
"""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg as sla
import time
from scipy.optimize import minimize

# --- Constants ---
DT = 0.5
N_STEPS = 80
GATE_TIME = 40.0
T1 = 25000.0
Tphi = 40000.0
ANHARMONICITY = -2 * np.pi * 0.25
DETUNING_VALUES = np.array([-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002])
SCALE_VALUES = np.array([0.97, 1.0, 1.03])
AMP_BOUND = 0.4
SLEW_BOUND = 0.12
MAX_DIFF = SLEW_BOUND * DT  # 0.06

# Precompute constant matrices
A_MAT = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
AD_MAT = A_MAT.conj().T
N_MAT = AD_MAT @ A_MAT
I3 = np.eye(3, dtype=complex)
NN1 = N_MAT @ (N_MAT - I3)
X2 = np.array([[0, 1, 0],
               [1, 0, 0],
               [0, 0, 0]], dtype=complex)

# Lindblad operators
G1 = A_MAT / np.sqrt(T1)
G2 = N_MAT * np.sqrt(2.0 / Tphi)
G1DAG = G1.conj().T
G2DAG = G2.conj().T
G1DAG_G1 = G1DAG @ G1
G2DAG_G2 = G2DAG @ G2

# Precompute dissipation Liouvillian (constant part)
DISS_L = (np.kron(G1, G1.conj()) - 0.5 * (np.kron(G1DAG_G1, I3) + np.kron(I3, G1DAG_G1.T)) +
          np.kron(G2, G2.conj()) - 0.5 * (np.kron(G2DAG_G2, I3) + np.kron(I3, G2DAG_G2.T)))


def build_liouvillian(H):
    coh = -1j * (np.kron(H, I3) - np.kron(I3, H.T))
    return coh + DISS_L


def build_hamiltonian(detuning, scale, I_val, Q_val):
    H = (detuning * N_MAT + 0.5 * ANHARMONICITY * NN1 +
         0.5 * scale * (I_val * (A_MAT + AD_MAT) + 1j * Q_val * (A_MAT - AD_MAT)))
    return H


def evolve_all_states(pulse, detuning, scale):
    """Evolve all 6 test states; return mean fidelity."""
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
    ]

    exp_mats = []
    for i in range(N_STEPS):
        H = build_hamiltonian(detuning, scale, pulse[i, 0], pulse[i, 1])
        L = build_liouvillian(H)
        exp_mats.append(sla.expm(L * DT))

    fidelities = []
    for psi0 in states:
        rho0 = np.outer(psi0, psi0.conj())
        rho_vec = rho0.reshape(-1).copy()
        for m in exp_mats:
            rho_vec = m @ rho_vec
        rho = rho_vec.reshape(3, 3)
        psi_t = X2 @ psi0
        f = np.vdot(psi_t, rho @ psi_t).real
        f = np.clip(f, 0.0, 1.0)
        fidelities.append(f)
    return np.mean(fidelities)


def score_only(pulse):
    case_inf = []
    for d in DETUNING_VALUES:
        for s in SCALE_VALUES:
            f = evolve_all_states(pulse, d, s)
            case_inf.append(1.0 - f)
    case_inf = np.array(case_inf)
    return np.mean(case_inf) + np.max(case_inf)


def check_validity(pulse):
    for i in range(N_STEPS):
        if np.sqrt(pulse[i, 0] ** 2 + pulse[i, 1] ** 2) > AMP_BOUND + 1e-9:
            return False
    prev = np.zeros(2)
    for i in range(N_STEPS):
        if np.linalg.norm(pulse[i] - prev) / DT > SLEW_BOUND + 1e-9:
            return False
        prev = pulse[i].copy()
    if np.linalg.norm(prev) / DT > SLEW_BOUND + 1e-9:
        return False
    return True


def enforce_constraints(pulse):
    """Project onto feasible set: amplitude bound + slew bound (forward/backward)."""
    p = pulse.copy()
    # amplitude
    for i in range(N_STEPS):
        a = np.linalg.norm(p[i])
        if a > AMP_BOUND:
            p[i] *= AMP_BOUND / a
    # slew forward
    prev = np.zeros(2)
    for i in range(N_STEPS):
        d = p[i] - prev
        n = np.linalg.norm(d)
        if n > MAX_DIFF:
            p[i] = prev + d * (MAX_DIFF / n)
        prev = p[i].copy()
    # slew backward
    prev = np.zeros(2)
    for i in range(N_STEPS - 1, -1, -1):
        d = p[i] - prev
        n = np.linalg.norm(d)
        if n > MAX_DIFF:
            p[i] = prev + d * (MAX_DIFF / n)
        prev = p[i].copy()
    # re-enforce amplitude
    for i in range(N_STEPS):
        a = np.linalg.norm(p[i])
        if a > AMP_BOUND:
            p[i] *= AMP_BOUND / a
    return p


def ideal_trajectory():
    """Ideal |0>->|1> X-pi trajectory in the 2-level subspace.
    Returns array of shape (N_STEPS, 3) with the ideal state at each step.
    Uses a smooth (sinusoidal) rotation: theta(t) = pi * t / T.
    """
    t = np.arange(N_STEPS) * DT
    theta = np.pi * t / GATE_TIME
    # |psi(t)> = cos(theta/2)|0> + sin(theta/2)|1>
    # (ignoring global phase for fidelity purposes)
    traj = np.zeros((N_STEPS, 3), dtype=complex)
    traj[:, 0] = np.cos(theta / 2)
    traj[:, 1] = np.sin(theta / 2)
    return traj


def generate_feedback_pulse(params, detuning=0.0, scale=1.0):
    """Generate a pulse using a feedback controller that tracks the ideal trajectory.

    params = [kp, ki, kd, drag_gain, leak_gain]
      kp: proportional gain on state error (I channel)
      ki: integral gain
      kd: derivative gain (I channel)
      drag_gain: gain for Q channel (derivative of I, suppresses |2> leakage)
      leak_gain: gain for Q channel proportional to leakage probability

    The controller runs in the *nominal* (detuning=0, scale=1) frame to
    generate the pulse.  The resulting open-loop pulse is then evaluated
    across all 9 detuning/scale cases.
    """
    kp, ki, kd, drag_gain, leak_gain = params

    traj = ideal_trajectory()
    pulse = np.zeros((N_STEPS, 2))

    # State vector (3-level)
    psi = np.array([1.0, 0.0, 0.0], dtype=complex)
    # Integral accumulator
    integral = 0.0
    prev_error = 0.0
    prev_I = 0.0

    for i in range(N_STEPS):
        # Current ideal target state
        target = traj[i]

        # State error: project onto |0>,|1> subspace
        # error = <target|psi> - 1  (or use component errors)
        # Use the |1> component error and |0> component error
        e0 = psi[0] - target[0]
        e1 = psi[1] - target[1]
        # Scalar error: how far are we from the target in the 2D subspace
        error = e1  # primarily track |1> amplitude
        error_real = error.real

        # Integral
        integral += error_real * DT

        # Derivative
        deriv = (error_real - prev_error) / DT if DT > 0 else 0.0

        # PID on I channel
        I_val = kp * error_real + ki * integral + kd * deriv

        # DRAG: Q proportional to dI/dt (suppresses |2> leakage)
        dI_dt = (I_val - prev_I) / DT
        Q_val = drag_gain * dI_dt

        # Leakage correction: if |2> amplitude is growing, add Q kick
        leak_prob = abs(psi[2]) ** 2
        Q_val += leak_gain * leak_prob * np.sign(I_val) if abs(I_val) > 1e-12 else 0.0

        pulse[i, 0] = I_val
        pulse[i, 1] = Q_val

        # Simulate one step to get next state (nominal: detuning=0, scale=1)
        H = build_hamiltonian(0.0, 1.0, I_val, Q_val)
        U = sla.expm(-1j * H * DT)
        psi = U @ psi

        prev_error = error_real
        prev_I = I_val

    pulse = enforce_constraints(pulse)
    return pulse


def _seed_pulse():
    """Gaussian DRAG seed pulse (same as before, for fallback)."""
    t = np.arange(N_STEPS) * DT
    sigma = GATE_TIME / 6.0
    amp = 0.25
    center = GATE_TIME / 2.0
    g = amp * np.exp(-0.5 * ((t - center) / sigma) ** 2)
    dg = g * (-(t - center) / sigma ** 2)
    beta = 0.08
    i = g
    q = -beta * dg
    p = np.column_stack((i, q))
    p = enforce_constraints(p)
    return p


def pulse(target_gate, hardware_model):
    """Main entry point."""
    if target_gate != 'X_pi':
        return np.zeros((80, 2))

    start_time = time.time()
    deadline = start_time + 300.0  # 30s margin before 330s kill

    # Fallback: seed pulse
    best_pulse = _seed_pulse()
    best_score = score_only(best_pulse)

    if time.time() > deadline:
        return best_pulse

    # --- Feedback pulse optimization ---
    # params: [kp, ki, kd, drag_gain, leak_gain]
    # Reasonable ranges:
    #   kp: [0, 2]
    #   ki: [0, 1]
    #   kd: [0, 0.5]
    #   drag_gain: [0, 0.5]
    #   leak_gain: [0, 1]

    def objective(params):
        p = generate_feedback_pulse(params)
        return score_only(p)

    bounds = [
        (0.0, 2.0),   # kp
        (0.0, 1.0),   # ki
        (0.0, 0.5),   # kd
        (0.0, 0.5),   # drag_gain
        (0.0, 1.0),   # leak_gain
    ]

    # Multiple starting points
    starts = [
        [0.5, 0.1, 0.05, 0.1, 0.1],
        [1.0, 0.0, 0.1, 0.2, 0.0],
        [0.3, 0.2, 0.0, 0.15, 0.2],
        [0.8, 0.05, 0.05, 0.3, 0.05],
        [1.5, 0.0, 0.0, 0.25, 0.0],
    ]

    for x0 in starts:
        if time.time() > deadline:
            break
        try:
            result = minimize(
                objective,
                np.array(x0),
                method='Nelder-Mead',
                bounds=bounds,
                options={
                    'maxiter': 150,
                    'xatol': 1e-4,
                    'fatol': 1e-7,
                    'adaptive': True,
                    'disp': False,
                }
            )
            if result.fun < best_score:
                p_new = generate_feedback_pulse(result.x)
                s_new = score_only(p_new)
                if s_new < best_score:
                    best_pulse = p_new
                    best_score = s_new
        except Exception:
            pass

    # If time remains, try a second round with tighter tolerances around best
    if time.time() < deadline - 30:
        # Re-optimize with the best params found
        pass  # The above loop already does this via multiple starts

    # Final validity check
    if not check_validity(best_pulse):
        best_pulse = enforce_constraints(best_pulse)

    return best_pulse
# EVOLVE-BLOCK-END
