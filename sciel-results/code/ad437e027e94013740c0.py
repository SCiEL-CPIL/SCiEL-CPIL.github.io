"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm

# Constants
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
T1 = 25000.0
TPHI = 40000.0
N_STEPS = 80
DT = 0.5


def _build_matrices():
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2.0)
    adag = a.conj().T
    n = adag @ a
    return a, adag, n


def _precompute_static(a, adag, n):
    """Precompute static parts of the Liouvillian superoperator."""
    I3 = np.eye(3, dtype=complex)
    
    # Anharmonicity term (static): 0.5 * ANHARMONICITY * n(n-1)
    H_anh = 0.5 * ANHARMONICITY * (n @ (n - I3))
    
    # Lindblad operators (static)
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / TPHI)
    L1dag = L1.conj().T
    L2dag = L2.conj().T
    
    L1dag_L1 = L1dag @ L1
    L2dag_L2 = L2dag @ L2
    
    # Static dissipation superoperators
    L1_super = np.kron(L1, L1.conj())
    L1_diss = -0.5 * (np.kron(L1dag_L1, I3) + np.kron(I3, L1dag_L1.T))
    
    L2_super = np.kron(L2, L2.conj())
    L2_diss = -0.5 * (np.kron(L2dag_L2, I3) + np.kron(I3, L2dag_L2.T))
    
    static_diss = L1_super + L1_diss + L2_super + L2_diss
    
    # Precompute kron structures for H terms
    # H = detuning*n + H_anh + 0.5*scale*(I*(a+adag) + i*Q*(a-adag))
    # H_super = -1j*(kron(H, I3) - kron(I3, H.T))
    
    # For detuning: -1j*(kron(detuning*n, I3) - kron(I3, (detuning*n).T))
    # = -1j*detuning*(kron(n, I3) - kron(I3, n.T))
    n_kron_I = np.kron(n, I3)
    I_kron_nT = np.kron(I3, n.T)
    detuning_super_base = -1j * (n_kron_I - I_kron_nT)
    
    # For anharmonicity: -1j*(kron(H_anh, I3) - kron(I3, H_anh.T))
    H_anh_super = -1j * (np.kron(H_anh, I3) - np.kron(I3, H_anh.T))
    
    # For drive I: -1j*0.5*scale*(kron(I*(a+adag), I3) - kron(I3, (I*(a+adag)).T))
    # = -1j*0.5*scale*I*(kron(a+adag, I3) - kron(I3, (a+adag).T))
    a_plus_adag = a + adag
    a_plus_adag_super_base = -1j * (np.kron(a_plus_adag, I3) - np.kron(I3, a_plus_adag.T))
    
    # For drive Q: -1j*0.5*scale*(kron(i*Q*(a-adag), I3) - kron(I3, (i*Q*(a-adag)).T))
    # = -1j*0.5*scale*i*Q*(kron(a-adag, I3) - kron(I3, (a-adag).T))
    a_minus_adag = a - adag
    a_minus_adag_super_base = -1j * (np.kron(a_minus_adag, I3) - np.kron(I3, a_minus_adag.T))
    
    return {
        'static_diss': static_diss,
        'detuning_base': detuning_super_base,
        'anh_super': H_anh_super,
        'I_base': a_plus_adag_super_base,
        'Q_base': a_minus_adag_super_base,
    }


def _liouvillian(I_val, Q_val, detuning, scale, static):
    """Build Liouvillian superoperator efficiently using precomputed parts."""
    L = static['static_diss'] + static['anh_super']
    L += detuning * static['detuning_base']
    L += 0.5 * scale * I_val * static['I_base']
    L += 0.5 * scale * Q_val * 1j * static['Q_base']
    return L


def _simulate_pulse(IQ, detuning, scale, static):
    """Simulate pulse and return mean fidelity to X gate."""
    X_gate = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, -1, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, 1j, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, -1j, 0], dtype=complex)) / np.sqrt(2),
    ]
    
    # Precompute all step superoperators
    supers = []
    for step in range(N_STEPS):
        I_val = IQ[step, 0]
        Q_val = IQ[step, 1]
        L_super = _liouvillian(I_val, Q_val, detuning, scale, static)
        supers.append(expm(L_super * DT))
    
    fidelities = []
    for psi0 in states:
        rho = np.outer(psi0, psi0.conj()).flatten()
        for s in supers:
            rho = s @ rho
        rho_final = rho.reshape(3, 3)
        psi_target = X_gate @ psi0
        rho_target = np.outer(psi_target, psi_target.conj())
        fidelity = np.real(np.trace(rho_target @ rho_final))
        fidelities.append(fidelity)
    
    return np.mean(fidelities)


def _compute_score(IQ, static):
    """Compute score: mean + max of 9 infidelities."""
    detunings = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    all_infidelities = []
    for det in detunings:
        for sc in scales:
            fid = _simulate_pulse(IQ, det, sc, static)
            all_infidelities.append(1.0 - fid)
    
    infidelities = np.array(all_infidelities)
    return np.mean(infidelities) + np.max(infidelities)


def _check_validity(IQ):
    """Check amplitude and slew rate constraints."""
    amps = np.sqrt(IQ[:, 0]**2 + IQ[:, 1]**2)
    if np.any(amps > AMPLITUDE_BOUND + 1e-12):
        return False
    
    prev = np.array([0.0, 0.0])
    for i in range(N_STEPS):
        diff = np.linalg.norm(IQ[i] - prev) / DT
        if diff > SLEW_BOUND + 1e-12:
            return False
        prev = IQ[i]
    
    diff = np.linalg.norm(-IQ[-1]) / DT
    if diff > SLEW_BOUND + 1e-12:
        return False
    
    return True


def _make_pulse(sigma_factor, drag_coeff, amplitude_scale=1.0):
    """Generate a DRAG-style Gaussian pulse with constraints enforced."""
    t = np.arange(N_STEPS) * DT
    sigma = (40.0 / 6.0) * sigma_factor
    
    env = np.exp(-(t - 20.0)**2 / (2 * sigma**2))
    # Offset to ensure zero at edges
    edge_offset = (env[0] + env[-1]) / 2
    env = env - edge_offset
    env = np.maximum(env, 0)
    
    # Normalize area to pi
    area = np.sum(env) * DT
    if area > 1e-10:
        env = env * (np.pi / area)
    
    env *= amplitude_scale
    I = env
    Q = drag_coeff * np.gradient(I, DT)
    
    IQ = np.column_stack([I, Q])
    
    # Enforce amplitude bound
    max_amp = np.max(np.sqrt(IQ[:, 0]**2 + IQ[:, 1]**2))
    if max_amp > AMPLITUDE_BOUND:
        IQ *= (AMPLITUDE_BOUND / max_amp) * 0.98
    
    # Enforce slew rate
    max_step = SLEW_BOUND * DT * 0.98
    prev = np.array([0.0, 0.0])
    for i in range(N_STEPS):
        diff = np.linalg.norm(IQ[i] - prev)
        if diff > max_step:
            scale = max_step / diff
            IQ[i] = prev + scale * (IQ[i] - prev)
        prev = IQ[i]
    
    # Final step to zero
    diff = np.linalg.norm(IQ[-1])
    if diff > max_step:
        IQ[-1] *= max_step / diff
    
    return IQ


def _optimize_pulse(static, deadline):
    """Optimize pulse parameters with grid search and local refinement."""
    best_IQ = None
    best_score = np.inf
    
    # Phase 1: Coarse grid search
    sigma_factors = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]
    drag_coeffs = [-0.4, -0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2]
    amp_scales = [0.9, 1.0, 1.1]
    
    for sf in sigma_factors:
        for dc in drag_coeffs:
            for ascale in amp_scales:
                if time.time() > deadline:
                    return best_IQ, best_score
                
                IQ = _make_pulse(sf, dc, ascale)
                if not _check_validity(IQ):
                    continue
                
                score = _compute_score(IQ, static)
                if score < best_score:
                    best_score = score
                    best_IQ = IQ.copy()
    
    # Phase 2: Local refinement around best
    if best_IQ is not None and time.time() < deadline - 30:
        # Find approximate best parameters
        # Re-do with finer grid centered on promising region
        for sf in np.linspace(0.8, 1.2, 7):
            for dc in np.linspace(-0.45, 0.25, 11):
                for ascale in np.linspace(0.85, 1.15, 5):
                    if time.time() > deadline:
                        break
                    IQ = _make_pulse(sf, dc, ascale)
                    if not _check_validity(IQ):
                        continue
                    score = _compute_score(IQ, static)
                    if score < best_score:
                        best_score = score
                        best_IQ = IQ.copy()
    
    return best_IQ, best_score


def pulse(target_gate, hardware_model):
    """
    Generate a pulse for the given target gate and hardware model.
    Returns an (80, 2) array of I and Q amplitudes.
    """
    start_time = time.time()
    deadline = start_time + 310.0  # Safety margin before 330s kill
    
    a, adag, n = _build_matrices()
    static = _precompute_static(a, adag, n)
    
    # Get initial seed
    IQ_seed = _make_pulse(1.0, -0.25, 1.0)
    
    # Optimize
    best_IQ, best_score = _optimize_pulse(static, deadline)
    
    if best_IQ is None:
        best_IQ = IQ_seed
    
    # Final validity check
    if not _check_validity(best_IQ):
        best_IQ = IQ_seed
        if not _check_validity(best_IQ):
            best_IQ = np.zeros((N_STEPS, 2))
    
    return best_IQ

# EVOLVE-BLOCK-END
