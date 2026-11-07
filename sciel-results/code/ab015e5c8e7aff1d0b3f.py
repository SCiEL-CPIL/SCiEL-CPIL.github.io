"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm
import scipy.optimize

# --- Constants ---
N_STEPS = 80
DT = 0.5
T_TOTAL = N_STEPS * DT  # 40 ns

# Hardware parameters
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUND = 2 * np.pi * 0.002
SCALE_MIN = 0.97
SCALE_MAX = 1.03
T1 = 25000.0
Tphi = 40000.0

# Precomputed static matrices
def _precompute_matrices():
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = 1.0
    a[1, 2] = np.sqrt(2.0)
    adag = a.conj().T
    n = adag @ a
    I3 = np.eye(3, dtype=complex)

    # Coherent parts
    def coh(H):
        return -1j * (np.kron(I3, H) - np.kron(H.T, I3))
    
    # H = detuning * n + 0.5 * anh * n(n-1) + 0.5 * scale * (I*(a+adag) + i*Q*(a-adag))
    # We separate terms to allow fast linear combination during simulation
    H_anh = 0.5 * ANHARMONICITY * (n @ (n - I3))
    base_coh_anh = coh(H_anh)
    
    det_coh = coh(n)
    
    # Drive terms: 0.5 * scale * I * (a+adag) and 0.5 * scale * i * Q * (a-adag)
    # Note: The task says 0.5 * scale * (I (a + a^dagger) + i Q (a - a^dagger))
    # So the coefficient for I in H is 0.5 * scale * (a+adag)
    # And for Q is 0.5 * scale * 1j * (a-adag)
    drive_I_H = 0.5 * (a + adag)
    drive_Q_H = 0.5 * 1j * (a - adag)
    
    coh_drive_I = coh(drive_I_H)
    coh_drive_Q = coh(drive_Q_H)
    
    # Dissipative parts (constant)
    def incoh(Lk):
        Lkd = Lk.conj().T
        LkLkd = Lk @ Lkd
        return (np.kron(Lkd.T, Lk) - 0.5 * np.kron(I3, LkLkd) - 0.5 * np.kron(LkLkd.T, I3))
    
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / Tphi)
    base_inc = incoh(L1) + incoh(L2)
    
    return base_coh_anh, det_coh, coh_drive_I, coh_drive_Q, base_inc

(base_coh_anh, det_coh, coh_drive_I, coh_drive_Q, base_inc) = _precompute_matrices()

# Legendre Basis for pulse representation
# We map t in [0, T] to x in [-1, 1]
_t_grid = np.linspace(0, 1, N_STEPS)
_x_grid = 2 * _t_grid - 1
N_TERMS = 8
# legvander returns a Vandermonde matrix where row i is [P_0(x_i), ..., P_{k-1}(x_i)]
# We want columns to be the basis functions.
_legendre_basis = np.polynomial.legendre.legvander(_x_grid, N_TERMS - 1).T 
# Shape: (N_STEPS, N_TERMS)

def _coeffs_to_pulse(coeffs):
    """
    coeffs: (16,) array. First 8 for I, next 8 for Q.
    Returns I, Q arrays of shape (80,).
    """
    I_coeffs = coeffs[:N_TERMS]
    Q_coeffs = coeffs[N_TERMS:]
    I = _legendre_basis @ I_coeffs
    Q = _legendre_basis @ Q_coeffs
    return I, Q

def _pulse_to_coeffs(I, Q):
    """
    Least squares fit of I and Q to Legendre basis.
    """
    I_coeffs, _, _, _ = np.linalg.lstsq(_legendre_basis, I, rcond=None)
    Q_coeffs, _, _, _ = np.linalg.lstsq(_legendre_basis, Q, rcond=None)
    return np.concatenate([I_coeffs, Q_coeffs])

def _enforce_constraints(I, Q):
    """
    Project I, Q to satisfy amplitude and slew rate constraints.
    """
    # Amplitude constraint
    amp = np.sqrt(I**2 + Q**2)
    if np.max(amp) > AMPLITUDE_BOUND:
        scale = AMPLITUDE_BOUND / np.max(amp)
        I = I * scale
        Q = Q * scale
    
    # Slew rate constraint
    # Iterative projection
    for _ in range(10):
        # From 0 to I[0], Q[0]
        dist0 = np.hypot(I[0], Q[0])
        if dist0 / DT > SLEW_BOUND:
            scale = SLEW_BOUND * DT / dist0
            I[0] *= scale
            Q[0] *= scale
        
        # Between steps
        for i in range(N_STEPS - 1):
            dI = I[i+1] - I[i]
            dQ = Q[i+1] - Q[i]
            dist = np.hypot(dI, dQ)
            if dist / DT > SLEW_BOUND:
                scale = SLEW_BOUND * DT / dist
                # Adjust next point to be closer
                I[i+1] = I[i] + dI * scale
                Q[i+1] = Q[i] + dQ * scale
        
        # From last to 0
        distN = np.hypot(I[-1], Q[-1])
        if distN / DT > SLEW_BOUND:
            scale = SLEW_BOUND * DT / distN
            I[-1] *= scale
            Q[-1] *= scale
            
    # Re-check amplitude after slew adjustments
    amp = np.sqrt(I**2 + Q**2)
    if np.max(amp) > AMPLITUDE_BOUND:
        scale = AMPLITUDE_BOUND / np.max(amp)
        I = I * scale
        Q = Q * scale
        
    return I, Q

def _valid(I, Q):
    if np.max(np.sqrt(I * I + Q * Q)) > AMPLITUDE_BOUND + 1e-9:
        return False
    if np.hypot(I[0], Q[0]) / DT > SLEW_BOUND + 1e-9:
        return False
    if np.hypot(I[-1], Q[-1]) / DT > SLEW_BOUND + 1e-9:
        return False
    for i in range(N_STEPS - 1):
        if np.hypot(I[i + 1] - I[i], Q[i + 1] - Q[i]) / DT > SLEW_BOUND + 1e-9:
            return False
    return True

# Precompute test states and targets
_psi0 = np.array([1, 0, 0], dtype=complex)
_psi1 = np.array([0, 1, 0], dtype=complex)
_test_states = [
    np.outer(_psi0, _psi0.conj()).flatten(),
    np.outer(_psi1, _psi1.conj()).flatten(),
    np.outer((_psi0 + _psi1) / np.sqrt(2), ((_psi0 + _psi1) / np.sqrt(2)).conj()).flatten(),
    np.outer((_psi0 - _psi1) / np.sqrt(2), ((_psi0 - _psi1) / np.sqrt(2)).conj()).flatten(),
    np.outer((_psi0 + 1j * _psi1) / np.sqrt(2), ((_psi0 + 1j * _psi1) / np.sqrt(2)).conj()).flatten(),
    np.outer((_psi0 - 1j * _psi1) / np.sqrt(2), ((_psi0 - 1j * _psi1) / np.sqrt(2)).conj()).flatten()
]

# X gate swaps |0> and |1>
def _apply_x(psi_vec):
    p = psi_vec.copy()
    p[0], p[1] = p[1], p[0]
    return p

_target_states = [
    np.outer(_apply_x(np.array([1, 0, 0])), np.array([1, 0, 0])).flatten(),
    np.outer(_apply_x(np.array([0, 1, 0])), np.array([0, 1, 0])).flatten(),
    np.outer(_apply_x((np.array([1, 0, 0]) + np.array([0, 1, 0])) / np.sqrt(2)), ((np.array([1, 0, 0]) + np.array([0, 1, 0])) / np.sqrt(2))).flatten(),
    np.outer(_apply_x((np.array([1, 0, 0]) - np.array([0, 1, 0])) / np.sqrt(2)), ((np.array([1, 0, 0]) - np.array([0, 1, 0])) / np.sqrt(2))).flatten(),
    np.outer(_apply_x((np.array([1, 0, 0]) + 1j * np.array([0, 1, 0])) / np.sqrt(2)), ((np.array([1, 0, 0]) + 1j * np.array([0, 1, 0])) / np.sqrt(2))).flatten(),
    np.outer(_apply_x((np.array([1, 0, 0]) - 1j * np.array([0, 1, 0])) / np.sqrt(2)), ((np.array([1, 0, 0]) - 1j * np.array([0, 1, 0])) / np.sqrt(2))).flatten()
]

def _score(I, Q):
    """
    Compute the score: mean + max infidelity over 9 cases.
    """
    infidelities = np.empty(9)
    k = 0
    det_vals = [-DETUNING_BOUND, 0.0, DETUNING_BOUND]
    sc_vals = [SCALE_MIN, 1.0, SCALE_MAX]
    
    # Pre-build Liouvillian base parts for each detuning/scale? 
    # It's cheaper to just add the variable parts in the loop since expm is the bottleneck.
    
    for di_val in det_vals:
        for sc_val in sc_vals:
            S = np.eye(9, dtype=complex)
            for i in range(N_STEPS):
                # L = base_anh + base_inc + di * det_coh + sc * (I[i] * coh_I + Q[i] * coh_Q)
                L = base_coh_anh + base_inc
                L += di_val * det_coh
                L += sc_val * (I[i] * coh_drive_I + Q[i] * coh_drive_Q)
                
                U = expm(L * DT)
                S = U @ S
            
            f = np.empty(6)
            for j in range(6):
                rho_final_vec = S @ _test_states[j]
                # Fidelity = Tr(rho_final * target)
                # Since target is pure state |t><t|, Fidelity = <t|rho_final|t> = Tr(|t><t| rho_final)
                # Which is vdot(target_vec, rho_final_vec)
                fid = np.real(np.vdot(_target_states[j], rho_final_vec))
                # Ensure numerical stability
                if fid < 0: fid = 0.0
                if fid > 1: fid = 1.0
                f[j] = fid
            infidelities[k] = 1.0 - np.mean(f)
            k += 1
    
    return np.mean(infidelities) + np.max(infidelities)

def pulse(target_gate, hardware_model):
    """
    Main entry point.
    Returns an (80, 2) array of I and Q amplitudes.
    """
    start_time = time.time()
    deadline = start_time + 300.0  # 300s safety margin
    
    if target_gate != 'X_pi':
        return np.zeros((N_STEPS, 2))
    
    # Generate initial seed pulse (Gaussian)
    t = (np.arange(N_STEPS) + 0.5) * DT
    center = T_TOTAL / 2.0
    sigma = T_TOTAL / 6.0
    I_seed = np.exp(-0.5 * ((t - center) / sigma)**2)
    area = np.sum(I_seed) * DT
    if area > 0:
        I_seed *= (np.pi / area)
    Q_seed = np.zeros(N_STEPS)
    
    I_seed, Q_seed = _enforce_constraints(I_seed, Q_seed)
    coeffs_seed = _pulse_to_coeffs(I_seed, Q_seed)
    
    best_score = _score(I_seed, Q_seed)
    best_coeffs = coeffs_seed.copy()
    best_I, best_Q = I_seed.copy(), Q_seed.copy()

    # Optimization using L-BFGS-B
    def objective(coeffs):
        I, Q = _coeffs_to_pulse(coeffs)
        I, Q = _enforce_constraints(I, Q)
        return _score(I, Q)

    # Initial bounds for coefficients
    # Legendre coefficients can be large if the pulse is high, but we scale down in _enforce_constraints
    # So we can use generous bounds
    bounds = [(-2.0, 2.0)] * (2 * N_TERMS)
    
    if time.time() < deadline - 60.0:
        try:
            res = scipy.optimize.minimize(
                objective, 
                coeffs_seed, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
                best_I, best_Q = _coeffs_to_pulse(best_coeffs)
                best_I, best_Q = _enforce_constraints(best_I, best_Q)
        except Exception:
            pass

    # If we have time, try a second round with tighter bounds or different start
    if time.time() < deadline - 30.0:
        try:
            # Try a slightly different start or tighter bounds
            # Just re-run with current best as start, maybe tighter bounds
            bounds_tight = [(-1.0, 1.0)] * (2 * N_TERMS)
            res2 = scipy.optimize.minimize(
                objective, 
                best_coeffs, 
                method='L-BFGS-B',
                bounds=bounds_tight,
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res2.fun < best_score:
                best_score = res2.fun
                best_coeffs = res2.x.copy()
                best_I, best_Q = _coeffs_to_pulse(best_coeffs)
                best_I, best_Q = _enforce_constraints(best_I, best_Q)
        except Exception:
            pass

    return np.column_stack([best_I, best_Q])

# EVOLVE-BLOCK-END
