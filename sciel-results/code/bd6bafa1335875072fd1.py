import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize

# Constants from the task contract
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DE_TUNING_BOUNDS = (-2 * np.pi * 0.002, 2 * np.pi * 0.002)
DRIVE_SCALE_BOUNDS = (0.97, 1.03)
T1 = 25000.0
TPHI = 40000.0
N_STEPS = 80
DT = 0.5


def _build_matrices():
    """Build 3x3 matrices for a, a_dag, n."""
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n = a_dag @ a
    return a, a_dag, n


def _liouvillian_superoperator(I, Q, detuning, scale, a, a_dag, n):
    """Build the 9x9 Liouvillian superoperator."""
    H = detuning * n + 0.5 * ANHARMONICITY * (n @ (n - np.eye(3))) + \
        0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag))

    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / TPHI)

    I3 = np.eye(3, dtype=complex)

    L_ham = -1j * (np.kron(H, I3) - np.kron(I3, H.T))

    L_total = L_ham.copy()
    for L in [L1, L2]:
        L_dag = L.conj().T
        L_total += np.kron(L, L.conj()) - 0.5 * (np.kron(L_dag @ L, I3) + np.kron(I3, (L_dag @ L).T))

    return L_total


def _simulate_pulse_fast(pulse, detuning, scale, a, a_dag, n):
    """
    Simulate the pulse for 6 input states simultaneously.
    Returns mean fidelity.
    """
    # Input states
    psi_0 = np.array([1, 0, 0], dtype=complex)
    psi_1 = np.array([0, 1, 0], dtype=complex)
    psi_2 = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_3 = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_4 = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    psi_5 = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)

    # Initial density matrices vectorized
    initial_states = []
    for psi in [psi_0, psi_1, psi_2, psi_3, psi_4, psi_5]:
        rho = np.outer(psi, psi.conj())
        initial_states.append(rho.flatten(order='F'))

    # Stack into a 9x6 matrix
    states = np.column_stack(initial_states)

    # Precompute propagators for each step
    propagators = []
    for i in range(N_STEPS):
        I_val = pulse[i, 0]
        Q_val = pulse[i, 1]
        L_super = _liouvillian_superoperator(I_val, Q_val, detuning, scale, a, a_dag, n)
        U_super = expm(L_super * DT)
        propagators.append(U_super)

    # Apply propagators
    for U_super in propagators:
        states = U_super @ states

    # Compute fidelities
    X_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    inputs = [psi_0, psi_1, psi_2, psi_3, psi_4, psi_5]

    fidelities = []
    for j in range(6):
        psi_in = inputs[j]
        psi_target = X_mat @ psi_in

        # Extract rho_out
        rho_vec = states[:, j]
        rho_out = rho_vec.reshape((3, 3), order='F')

        fidelity = np.real(np.conj(psi_target) @ rho_out @ psi_target)
        fidelity = max(0.0, min(1.0, fidelity))
        fidelities.append(fidelity)

    return np.mean(fidelities)


def _check_validity(pulse):
    """Check amplitude and slew constraints."""
    amps = np.sqrt(pulse[:, 0] ** 2 + pulse[:, 1] ** 2)
    if np.any(amps > AMPLITUDE_BOUND + 1e-12):
        return False

    slew0 = np.linalg.norm(pulse[0]) / DT
    if slew0 > SLEW_BOUND + 1e-12:
        return False

    for i in range(N_STEPS - 1):
        diff = np.linalg.norm(pulse[i + 1] - pulse[i]) / DT
        if diff > SLEW_BOUND + 1e-12:
            return False

    slew_last = np.linalg.norm(pulse[-1]) / DT
    if slew_last > SLEW_BOUND + 1e-12:
        return False

    return True


def _compute_score(pulse, a, a_dag, n):
    """Compute the score: mean + max infidelity over 9 cases."""
    infidelities = []
    for detuning in [DE_TUNING_BOUNDS[0], 0.0, DE_TUNING_BOUNDS[1]]:
        for scale in [DRIVE_SCALE_BOUNDS[0], 1.0, DRIVE_SCALE_BOUNDS[1]]:
            fid = _simulate_pulse_fast(pulse, detuning, scale, a, a_dag, n)
            infid = 1.0 - fid
            infidelities.append(infid)

    return np.mean(infidelities) + np.max(infidelities)


def _legendre_basis(N, order):
    """
    Generate Legendre polynomial values on N evenly spaced points in [0, 1].
    Returns array of shape (N, order).
    """
    t = np.linspace(0, 1, N)
    # Map to [-1, 1] for standard Legendre
    x = 2 * t - 1
    basis = np.zeros((N, order))
    for k in range(order):
        # Use numpy's legendre to get coefficients
        from numpy.polynomial.legendre import legval
        P_k = np.polynomial.legendre.Legendre.basis(k)
        basis[:, k] = P_k(x)
    return basis


def _generate_pulse_from_coeffs(coeffs, basis_I, basis_Q):
    """Generate pulse from coefficient vectors for I and Q channels."""
    I = basis_I @ coeffs[:8]
    Q = basis_Q @ coeffs[8:16]
    pulse = np.column_stack([I, Q])

    # Enforce amplitude bound
    amps = np.sqrt(pulse[:, 0] ** 2 + pulse[:, 1] ** 2)
    max_amp = np.max(amps)
    if max_amp > AMPLITUDE_BOUND:
        scale_factor = AMPLITUDE_BOUND / max_amp
        pulse *= scale_factor

    return pulse


def _slew_penalty(pulse):
    """Compute slew violation penalty."""
    penalty = 0.0
    # From zero to first step
    slew0 = np.linalg.norm(pulse[0]) / DT
    if slew0 > SLEW_BOUND:
        penalty += (slew0 - SLEW_BOUND) ** 2

    # Between steps
    for i in range(N_STEPS - 1):
        diff = np.linalg.norm(pulse[i + 1] - pulse[i]) / DT
        if diff > SLEW_BOUND:
            penalty += (diff - SLEW_BOUND) ** 2

    # From last step to zero
    slew_last = np.linalg.norm(pulse[-1]) / DT
    if slew_last > SLEW_BOUND:
        penalty += (slew_last - SLEW_BOUND) ** 2

    return penalty


def _generate_pulse(params):
    """Generate pulse from parameters [A, sigma, beta] for backward compatibility."""
    A, sigma, beta = params

    t = np.arange(N_STEPS) * DT
    t_center = 20.0

    # Gaussian
    gauss = np.exp(-0.5 * ((t - t_center) / sigma) ** 2)

    # Offset to 0 at edges
    edge_val = np.exp(-0.5 * ((0 - t_center) / sigma) ** 2)
    gauss = gauss - edge_val

    # Normalize area to pi
    area = np.sum(gauss) * DT
    if area > 0:
        gauss_norm = gauss / area  # Area is 1
    else:
        gauss_norm = gauss

    # Derivative
    dgauss = np.zeros(N_STEPS)
    dgauss[1:] = (gauss_norm[1:] - gauss_norm[:-1]) / DT
    dgauss[0] = dgauss[1]
    dgauss[-1] = dgauss[-2]

    I = A * gauss_norm
    Q = beta * A * dgauss

    pulse = np.column_stack([I, Q])

    # Enforce amplitude bound
    amps = np.sqrt(pulse[:, 0] ** 2 + pulse[:, 1] ** 2)
    max_amp = np.max(amps)
    if max_amp > AMPLITUDE_BOUND:
        scale_factor = AMPLITUDE_BOUND / max_amp
        pulse *= scale_factor

    return pulse


def pulse(target_gate, hardware_model):
    """Main entry point."""
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin

    a, a_dag, n = _build_matrices()

    # Precompute Legendre basis for 80 points, order 8
    basis_full = _legendre_basis(N_STEPS, 8)
    basis_I = basis_full
    basis_Q = basis_full

    # Initial guess: DRAG pulse
    x0_gauss = np.array([0.3, 5.0, 0.2])
    best_pulse = _generate_pulse(x0_gauss)
    best_score = _compute_score(best_pulse, a, a_dag, n)

    # Try Legendre coefficient optimization
    # 16 coefficients: 8 for I, 8 for Q
    # Initialize with coefficients that approximate a Gaussian-like shape
    # For a centered Gaussian, the first few Legendre coefficients matter
    # P0=1, P1=2t-1, etc.
    # A simple pulse centered at t=0.5 would have significant P0 and small higher terms

    # Let's start with a simple pulse: I ~ constant in middle, Q ~ derivative
    # Coefficients for a box-like shape in [0.2, 0.8]
    # This is complex, so let's use random initializations and the optimizer

    def objective_legendre(coeffs):
        p = _generate_pulse_from_coeffs(coeffs, basis_I, basis_Q)
        slew_pen = _slew_penalty(p)
        if slew_pen > 1e-6:
            return _compute_score(p, a, a_dag, n) + slew_pen * 10.0
        return _compute_score(p, a, a_dag, n)

    # Multiple random starts for Legendre optimization
    np.random.seed(42)
    legendre_starts = []

    # Start 1: All zeros except P0 for I (constant drive)
    c1 = np.zeros(16)
    c1[0] = 0.1  # Small constant I
    legendre_starts.append(c1)

    # Start 2: Approximate Gaussian with P0, P2, P4
    c2 = np.zeros(16)
    c2[0] = 0.15
    c2[2] = -0.05
    c2[4] = 0.02
    c2[8] = 0.0  # Q P0
    c2[9] = 0.05  # Q P1 (derivative-like)
    legendre_starts.append(c2)

    # Start 3: Random small coefficients
    c3 = np.random.randn(16) * 0.05
    legendre_starts.append(c3)

    # Start 4: Another random
    c4 = np.random.randn(16) * 0.03
    legendre_starts.append(c4)

    # Start 5: DRAG-like: I is Gaussian, Q is derivative
    # Gaussian ~ P0 + a*P2 + b*P4, derivative ~ P1 + c*P3
    c5 = np.zeros(16)
    c5[0] = 0.12
    c5[2] = -0.04
    c5[4] = 0.015
    c5[8] = 0.0
    c5[9] = 0.03  # Q ~ derivative
    c5[11] = -0.01
    legendre_starts.append(c5)

    for x_start in legendre_starts:
        if time.time() > deadline - 30.0:
            break

        try:
            # Use L-BFGS-B for constrained optimization
            # Bounds: keep coefficients small to avoid huge amplitudes
            bounds = [(-0.5, 0.5)] * 16

            res = minimize(objective_legendre, x_start, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})

            if res.fun < best_score:
                best_score = res.fun
                best_pulse = _generate_pulse_from_coeffs(res.x, basis_I, basis_Q)

                # Check validity
                if not _check_validity(best_pulse):
                    # If invalid, don't accept
                    best_pulse = _generate_pulse_from_coeffs(res.x, basis_I, basis_Q)
                    # Re-check after amplitude scaling
                    if not _check_validity(best_pulse):
                        # Roll back
                        pass

        except Exception:
            pass

    # Also try the original Gaussian optimization with more starts
    def objective_gauss(params):
        p = _generate_pulse(params)
        if not _check_validity(p):
            return 1.0  # Penalty
        return _compute_score(p, a, a_dag, n)

    gauss_starts = [
        np.array([0.3, 5.0, 0.2]),
        np.array([0.35, 4.0, 0.1]),
        np.array([0.25, 6.0, 0.3]),
        np.array([0.4, 5.0, 0.0]),
        np.array([0.3, 5.0, -0.1]),
        np.array([0.32, 4.5, 0.15]),
        np.array([0.28, 5.5, 0.25]),
    ]

    for x_start in gauss_starts:
        if time.time() > deadline - 30.0:
            break

        try:
            res = minimize(objective_gauss, x_start, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_pulse = _generate_pulse(res.x)
        except Exception:
            pass

    # Final validity check
    if not _check_validity(best_pulse):
        # Fallback to a safe pulse
        safe_pulse = np.zeros((N_STEPS, 2))
        t = np.arange(N_STEPS) * DT
        t_center = 20.0
        sigma = 5.0
        gauss = np.exp(-0.5 * ((t - t_center) / sigma) ** 2)
        edge_val = np.exp(-0.5 * ((0 - t_center) / sigma) ** 2)
        gauss = gauss - edge_val
        area = np.sum(gauss) * DT
        if area > 0:
            gauss_norm = gauss / area
        else:
            gauss_norm = gauss
        safe_pulse[:, 0] = 0.3 * gauss_norm
        # Scale to satisfy slew
        slew0 = np.linalg.norm(safe_pulse[0]) / DT
        if slew0 > SLEW_BOUND:
            safe_pulse *= SLEW_BOUND / slew0
        best_pulse = safe_pulse

    return best_pulse
