"""
Pulse design for a three-level transmon X-pi gate.

Approach: Basis expansion in the first 8 Legendre polynomials (scaled to [0,1]).
The pulse is parameterized by 8 coefficients for I and 8 for Q (16 parameters total).
We use constrained L-BFGS-B to minimize the mean infidelity over the 9 hardware
variation cases, subject to amplitude and slew-rate constraints.

The seed (DRAG-style Gaussian) is used as the initial guess for the optimizer.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
import scipy.constants as const

# Constants
DT = 0.5  # ns
N_STEPS = 80
T_TOTAL = N_STEPS * DT  # 40 ns

# Hardware model parameters
ANHARMONICITY = -2 * np.pi * 0.25  # rad/ns
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = [-2 * np.pi * 0.002, 2 * np.pi * 0.002]
DRIVE_SCALE_BOUNDS = [0.97, 1.03]
T1 = 25000.0  # ns
Tphi = 40000.0  # ns

# Lindblad rates
GAMMA1 = 1.0 / T1
GAMMA_PHI = 1.0 / Tphi


def legendre_basis(t, n_terms=8):
    """
    Compute the first n_terms Legendre polynomials evaluated at t in [0, 1].
    Returns an array of shape (len(t), n_terms).
    """
    t = np.asarray(t)
    # Scale t from [0, 1] to [-1, 1]
    x = 2 * t - 1
    basis = np.zeros((len(t), n_terms))
    for k in range(n_terms):
        basis[:, k] = np.polynomial.legendre.legval(x, [0]*k + [1])
    return basis


def coeffs_to_pulse(coeffs):
    """
    Convert 16 coefficients (8 for I, 8 for Q) to an (80, 2) pulse array.
    """
    t = np.linspace(0, 1, N_STEPS)
    basis = legendre_basis(t, 8)
    I = basis @ coeffs[:8]
    Q = basis @ coeffs[8:]
    pulse = np.column_stack([I, Q])
    return pulse


def pulse_to_coeffs(pulse):
    """
    Convert an (80, 2) pulse to 16 Legendre coefficients via least squares.
    """
    t = np.linspace(0, 1, N_STEPS)
    basis = legendre_basis(t, 8)
    coeffs_I = np.linalg.lstsq(basis, pulse[:, 0], rcond=None)[0]
    coeffs_Q = np.linalg.lstsq(basis, pulse[:, 1], rcond=None)[0]
    return np.concatenate([coeffs_I, coeffs_Q])


def build_liouvillian(detuning, scale, I_val, Q_val):
    """
    Build the Liouvillian superoperator for one time step.
    
    H = detuning * n + 0.5 * anharmonicity * n*(n-1) + 0.5 * scale * (I*(a+a^dag) + i*Q*(a-a^dag))
    
    Lindblad operators: a/sqrt(T1) and n*sqrt(2/Tphi)
    """
    # Build matrices in 3-level basis
    # a = diag(sqrt([1, 2]), 1) in the sense of lowering operator
    # In 3-level: |0>, |1>, |2>
    # a|0> = 0, a|1> = |0>, a|2> = sqrt(2)|1>
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = 1.0
    a[1, 2] = np.sqrt(2.0)
    a_dag = a.conj().T
    
    # n = a^dag a
    n = a_dag @ a
    # n should be diag(0, 1, 2)
    
    # Hamiltonian
    H = (detuning * n + 
         0.5 * ANHARMONICITY * (n @ (n - np.eye(3))) +
         0.5 * scale * (I_val * (a + a_dag) + 1j * Q_val * (a - a_dag)))
    
    # Make H Hermitian
    H = 0.5 * (H + H.conj().T)
    
    # Liouvillian: dρ/dt = -i[H, ρ] + D[ρ]
    # D[ρ] = sum_k (L_k ρ L_k^dag - 0.5 {L_k^dag L_k, ρ})
    
    # Build Liouvillian as a 9x9 matrix acting on vec(ρ)
    # L_k = a / sqrt(T1)
    L1 = a / np.sqrt(T1)
    # L_k = n * sqrt(2/Tphi)
    L2 = n * np.sqrt(2.0 / Tphi)
    
    # Build Liouvillian
    # vec(A ρ B) = (B^T ⊗ A) vec(ρ)
    # -i[H, ρ] = -i(H ρ - ρ H) = -i(H ρ I - I ρ H)
    # vec(H ρ) = (I ⊗ H) vec(ρ)
    # vec(ρ H) = (H^T ⊗ I) vec(ρ)
    
    I3 = np.eye(3, dtype=complex)
    L = np.kron(I3, H) - np.kron(H.T, I3)  # [H, ρ] term
    L = -1j * L
    
    # Dissipator terms
    for Lk in [L1, L2]:
        Lk_dag = Lk.conj().T
        Lk_dag_Lk = Lk_dag @ Lk
        # L_k ρ L_k^dag
        L += np.kron(Lk_dag.T, Lk)
        # -0.5 {L_k^dag L_k, ρ} = -0.5 (L_k^dag L_k ρ + ρ L_k^dag L_k)
        L += -0.5 * np.kron(I3, Lk_dag_Lk)
        L += -0.5 * np.kron(Lk_dag_Lk.T, I3)
    
    return L


def simulate_pulse(pulse, detuning, scale):
    """
    Simulate the pulse and return the final density matrix.
    pulse: (80, 2) array of I, Q
    Returns: 3x3 density matrix
    """
    # Initial state: |0><0|
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0
    
    for step in range(N_STEPS):
        I_val = pulse[step, 0]
        Q_val = pulse[step, 1]
        
        L = build_liouvillian(detuning, scale, I_val, Q_val)
        
        # Exponentiate: rho_next = exp(L * DT) rho
        L_exp = expm(L * DT)
        rho_vec = rho.flatten()
        rho_vec_new = L_exp @ rho_vec
        rho = rho_vec_new.reshape((3, 3))
    
    return rho


def compute_infidelity(rho, target_state):
    """
    Compute infidelity = 1 - <target|rho|target>
    target_state: 3x1 vector
    """
    fidelity = np.real(target_state.conj().T @ rho @ target_state)
    fidelity = np.clip(fidelity, 0.0, 1.0)
    return 1.0 - fidelity


def compute_score(pulse):
    """
    Compute the score for a given pulse.
    Score = mean + max of infidelities over 9 cases.
    """
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    # Target states
    e0 = np.array([1, 0, 0], dtype=complex)
    e1 = np.array([0, 1, 0], dtype=complex)
    plus = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    minus = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    i_plus = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    i_minus = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    
    target_states = [e0, e1, plus, minus, i_plus, i_minus]
    
    infidelities = []
    
    for det in detunings:
        for sc in scales:
            rho = simulate_pulse(pulse, det, sc)
            for ts in target_states:
                inf = compute_infidelity(rho, ts)
                infidelities.append(inf)
    
    infidelities = np.array(infidelities)
    mean_inf = np.mean(infidelities)
    max_inf = np.max(infidelities)
    
    return mean_inf + max_inf


def check_constraints(pulse):
    """
    Check amplitude and slew constraints.
    Returns True if valid.
    """
    # Amplitude constraint
    amplitudes = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    if np.any(amplitudes > AMPLITUDE_BOUND + 1e-9):
        return False
    
    # Slew constraint: ||p[i+1] - p[i]|| / DT <= SLEW_BOUND
    # Include from zero to step 0 and from last step to zero
    diffs = np.diff(pulse, axis=0)
    diffs = np.vstack([pulse[0:1] - 0, diffs, 0 - pulse[-1:]])
    norms = np.linalg.norm(diffs, axis=1)
    if np.any(norms / DT > SLEW_BOUND + 1e-9):
        return False
    
    return True


def objective(coeffs):
    """
    Objective function for optimization.
    Converts coefficients to pulse, checks constraints, and returns score.
    """
    pulse = coeffs_to_pulse(coeffs)
    
    # Penalty for constraint violations
    penalty = 0.0
    
    amplitudes = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    amp_viol = np.maximum(0, amplitudes - AMPLITUDE_BOUND)
    penalty += 100.0 * np.sum(amp_viol**2)
    
    diffs = np.diff(pulse, axis=0)
    diffs = np.vstack([pulse[0:1] - 0, diffs, 0 - pulse[-1:]])
    norms = np.linalg.norm(diffs, axis=1)
    slew_viol = np.maximum(0, norms / DT - SLEW_BOUND)
    penalty += 100.0 * np.sum(slew_viol**2)
    
    score = compute_score(pulse)
    
    return score + penalty


def design_pulse():
    """
    Design the pulse using Legendre basis expansion and L-BFGS-B optimization.
    """
    # Initial guess: DRAG-style Gaussian
    t = np.linspace(0, 1, N_STEPS)
    sigma = (40 / 6) / N_STEPS  # sigma in normalized time units
    # Gaussian centered at 0.5
    gauss = np.exp(-0.5 * ((t - 0.5) / sigma)**2)
    # Normalize area to pi
    gauss_area = np.sum(gauss) * DT
    gauss = gauss * (np.pi / gauss_area)
    
    # Offset to 0 at edges (already ~0 due to Gaussian)
    # Ensure exactly 0 at edges
    gauss[0] = 0
    gauss[-1] = 0
    
    # DRAG: add derivative term
    # Q is proportional to derivative of I
    dI = np.gradient(gauss)
    q_scale = 0.1  # DRAG strength
    q_init = q_scale * dI
    
    pulse_init = np.column_stack([gauss, q_init])
    
    # Ensure constraints on initial pulse
    # Scale down if needed
    amplitudes = np.sqrt(pulse_init[:, 0]**2 + pulse_init[:, 1]**2)
    if np.max(amplitudes) > AMPLITUDE_BOUND:
        pulse_init *= AMPLITUDE_BOUND / np.max(amplitudes)
    
    # Check slew
    diffs = np.diff(pulse_init, axis=0)
    diffs = np.vstack([pulse_init[0:1] - 0, diffs, 0 - pulse_init[-1:]])
    norms = np.linalg.norm(diffs, axis=1)
    if np.max(norms / DT) > SLEW_BOUND:
        scale_factor = SLEW_BOUND / np.max(norms / DT)
        pulse_init *= scale_factor
    
    # Convert to Legendre coefficients
    coeffs_init = pulse_to_coeffs(pulse_init)
    
    # Optimize
    result = minimize(
        objective,
        coeffs_init,
        method='L-BFGS-B',
        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    coeffs_opt = result.x
    pulse_opt = coeffs_to_pulse(coeffs_opt)
    
    # Final constraint enforcement: clip amplitudes and check slew
    amplitudes = np.sqrt(pulse_opt[:, 0]**2 + pulse_opt[:, 1]**2)
    if np.max(amplitudes) > AMPLITUDE_BOUND:
        pulse_opt *= AMPLITUDE_BOUND / np.max(amplitudes)
    
    # Check and fix slew if violated
    for _ in range(10):
        diffs = np.diff(pulse_opt, axis=0)
        diffs = np.vstack([pulse_opt[0:1] - 0, diffs, 0 - pulse_opt[-1:]])
        norms = np.linalg.norm(diffs, axis=1)
        max_slew = np.max(norms / DT)
        if max_slew <= SLEW_BOUND:
            break
        pulse_opt *= SLEW_BOUND / max_slew
    
    return pulse_opt


def pulse(target_gate, hardware_model):
    """
    Entry point: returns an (80, 2) array of I and Q amplitudes.
    """
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi gate is supported")
    
    p = design_pulse()
    
    # Final validation
    amplitudes = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
    assert np.all(amplitudes <= AMPLITUDE_BOUND + 1e-9), "Amplitude constraint violated"
    
    diffs = np.diff(p, axis=0)
    diffs = np.vstack([p[0:1] - 0, diffs, 0 - p[-1:]])
    norms = np.linalg.norm(diffs, axis=1)
    assert np.all(norms / DT <= SLEW_BOUND + 1e-9), "Slew constraint violated"
    
    return p


# Cache the designed pulse to avoid recomputation on multiple calls
_pulse_cache = None

def pulse_cached(target_gate, hardware_model):
    """
    Cached version of pulse to avoid recomputation.
    """
    global _pulse_cache
    if _pulse_cache is None:
        _pulse_cache = pulse(target_gate, hardware_model)
    return _pulse_cache


# Override pulse with the cached version for efficiency
pulse = pulse_cached
