"""Robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
import time

# Hardware parameters
ANHARMONICITY = -2 * np.pi * 0.25  # rad/ns
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = [-2 * np.pi * 0.002, 2 * np.pi * 0.002]
DRIVE_SCALE_BOUNDS = [0.97, 1.03]
T1 = 25000.0
T_PHI = 40000.0

N_STEPS = 80
DT = 0.5
TOTAL_TIME = N_STEPS * DT  # 40 ns


def build_operators():
    """Build a, a_dagger, n for 3-level system."""
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2.0)
    a_dag = a.conj().T
    n = a_dag @ a
    return a, a_dag, n


a_op, a_dag_op, n_op = build_operators()


def hamiltonian(detuning, scale, I, Q):
    """Build Hamiltonian for given parameters."""
    H = detuning * n_op + 0.5 * ANHARMONICITY * n_op @ (n_op - np.eye(3))
    H += 0.5 * scale * (I * (a_op + a_dag_op) + 1j * Q * (a_op - a_dag_op))
    return H


def lindblad_operators():
    """Build Lindblad operators."""
    L1 = a_op / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / T_PHI)
    return [L1, L2]


L_ops = lindblad_operators()


def liouvillian_step(detuning, scale, I, Q):
    """
    Compute the superoperator for one step of evolution.
    Returns the matrix that maps density matrix (vectorized) to new density matrix.
    """
    H = hamiltonian(detuning, scale, I, Q)
    U = expm(-1j * H * DT)
    
    dim = 3
    dim2 = dim * dim
    
    # Unitary part
    S_u = np.kron(U.T, U)
    
    # Lindblad part
    S_l = np.zeros((dim2, dim2), dtype=complex)
    for L in L_ops:
        Ldag = L.conj().T
        S_l += np.kron(L.T, L)
        LdagL = Ldag @ L
        S_l -= 0.5 * (np.kron(np.eye(dim), LdagL) + np.kron(LdagL.T, np.eye(dim)))
    
    S_total = S_u + DT * S_l
    
    return S_total


def evolve_pulse(pulse, detuning, scale):
    """
    Evolve the state under the given pulse.
    pulse: (80, 2) array of (I, Q)
    Returns final density matrix starting from |0>.
    """
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0
    
    for i in range(N_STEPS):
        I = pulse[i, 0]
        Q = pulse[i, 1]
        S = liouvillian_step(detuning, scale, I, Q)
        rho_vec = rho.flatten()
        rho_vec = S @ rho_vec
        rho = rho_vec.reshape(3, 3)
    
    return rho


def compute_fidelity_matrix(pulse, detuning, scale):
    """
    Compute the process matrix (Choi-like) by evolving basis states.
    Returns a 9x9 matrix where element (i,j) corresponds to
    <i| rho_final |j> for initial state |j> evolved to final.
    Actually, we compute the process by evolving |0>, |1>, |2> and their
    coherences. For fidelity with X, we need to track how each input
    basis state maps.
    
    We compute the superoperator S_total applied over all steps.
    Then fidelity for state |psi> is <X|psi><psi|X| S(|psi><psi|) |X|psi><psi|X>
    For X gate: X|0>=|1>, X|1>=|0>, X|2>=|2> (approximately, but |2> stays |2>)
    Actually for the 5 states in the task:
    |0> -> |1>
    |1> -> |0>
    (|0>+|1>)/sqrt(2) -> (|1>+|0>)/sqrt(2) = same state
    (|0>-|1>)/sqrt(2) -> (|1>-|0>)/sqrt(2) = -(|0>-|1>)/sqrt(2), fidelity with target
    (|0>+i|1>)/sqrt(2) -> (|1>+i|0>)/sqrt(2) = i(|0>-i|1>)/sqrt(2)
    
    Let's just evolve each of the 5 states and compute fidelity.
    """
    states = [
        np.array([1.0, 0.0, 0.0], dtype=complex),
        np.array([0.0, 1.0, 0.0], dtype=complex),
        (np.array([1.0, 0.0, 0.0]) + np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0),
        (np.array([1.0, 0.0, 0.0]) - np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0),
        (np.array([1.0, 0.0, 0.0]) + 1j * np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0),
        (np.array([1.0, 0.0, 0.0]) - 1j * np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0),
    ]
    
    # X gate targets
    X_gate = np.array([[0.0, 1.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0]], dtype=complex)
    target_states = [X_gate @ st for st in states]
    
    fidelities = []
    for i, (psi, psi_target) in enumerate(zip(states, target_states)):
        rho = np.outer(psi, psi.conj())
        for k in range(N_STEPS):
            I = pulse[k, 0]
            Q = pulse[k, 1]
            S = liouvillian_step(detuning, scale, I, Q)
            rho_vec = rho.flatten()
            rho_vec = S @ rho_vec
            rho = rho_vec.reshape(3, 3)
        
        # Fidelity = <psi_target| rho |psi_target>
        fid = np.real(np.vdot(psi_target, rho @ psi_target))
        fidelities.append(fid)
    
    return fidelities


def compute_score(pulse):
    """
    Compute the score for a given pulse.
    Returns (mean_infidelity, max_infidelity).
    """
    fidelities = []
    
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    for det in detunings:
        for sc in scales:
            fids = compute_fidelity_matrix(pulse, det, sc)
            mean_fid = np.mean(fids)
            infidelity = 1.0 - mean_fid
            fidelities.append(infidelity)
    
    fidelities = np.array(fidelities)
    return np.mean(fidelities), np.max(fidelities)


def check_validity(pulse):
    """Check if pulse satisfies amplitude and slew constraints."""
    for i in range(N_STEPS):
        I = pulse[i, 0]
        Q = pulse[i, 1]
        if np.sqrt(I**2 + Q**2) > AMPLITUDE_BOUND + 1e-9:
            return False
    
    I0 = 0.0
    Q0 = 0.0
    I1 = pulse[0, 0]
    Q1 = pulse[0, 1]
    if np.sqrt((I1 - I0)**2 + (Q1 - Q0)**2) / DT > SLEW_BOUND + 1e-9:
        return False
    
    for i in range(N_STEPS - 1):
        dI = pulse[i + 1, 0] - pulse[i, 0]
        dQ = pulse[i + 1, 1] - pulse[i, 1]
        if np.sqrt(dI**2 + dQ**2) / DT > SLEW_BOUND + 1e-9:
            return False
    
    I_last = pulse[-1, 0]
    Q_last = pulse[-1, 1]
    if np.sqrt(I_last**2 + Q_last**2) / DT > SLEW_BOUND + 1e-9:
        return False
    
    return True


def drag_gaussian_pulse(beta=0.1, width_scale=1.0, center_shift=0.0):
    """
    Generate a DRAG-style Gaussian pulse.
    """
    t = np.linspace(0, TOTAL_TIME, N_STEPS)
    sigma = TOTAL_TIME / 6.0 * width_scale
    center = TOTAL_TIME / 2.0 + center_shift
    
    gauss = np.exp(-0.5 * ((t - center) / sigma) ** 2)
    
    area = np.sum(gauss) * DT
    if area > 0:
        gauss = gauss * (np.pi / area)
    
    gauss_deriv = np.gradient(gauss, DT)
    
    I = gauss
    Q = beta * gauss_deriv
    
    pulse = np.column_stack([I, Q])
    
    # Ensure validity
    if not check_validity(pulse):
        max_amp = np.max(np.sqrt(I**2 + Q**2))
        if max_amp > AMPLITUDE_BOUND:
            scale = AMPLITUDE_BOUND / max_amp
            pulse *= scale
        
        for i in range(N_STEPS - 1):
            dI = pulse[i + 1, 0] - pulse[i, 0]
            dQ = pulse[i + 1, 1] - pulse[i, 1]
            slew = np.sqrt(dI**2 + dQ**2) / DT
            if slew > SLEW_BOUND:
                scale = SLEW_BOUND / slew
                pulse *= scale
    
    return pulse


def cosine_pulse(beta=0.0):
    """Generate a cosine-window pulse with optional DRAG."""
    t = np.linspace(0, TOTAL_TIME, N_STEPS)
    g = 0.5 * (1 - np.cos(2 * np.pi * t / TOTAL_TIME))
    area = np.sum(g) * DT
    if area > 0:
        g = g * (np.pi / area)
    
    if beta != 0:
        g_deriv = np.gradient(g, DT)
        Q = beta * g_deriv
    else:
        Q = np.zeros_like(g)
    
    I = g
    pulse = np.column_stack([I, Q])
    
    if not check_validity(pulse):
        max_amp = np.max(np.sqrt(I**2 + Q**2))
        if max_amp > AMPLITUDE_BOUND:
            scale = AMPLITUDE_BOUND / max_amp
            pulse *= scale
        for i in range(N_STEPS - 1):
            dI = pulse[i + 1, 0] - pulse[i, 0]
            dQ = pulse[i + 1, 1] - pulse[i, 1]
            slew = np.sqrt(dI**2 + dQ**2) / DT
            if slew > SLEW_BOUND:
                scale = SLEW_BOUND / slew
                pulse *= scale
    
    return pulse


def gradient_descent(pulse, max_iter=50, lr=0.01, deadline=None):
    """
    Simple gradient descent on the pulse amplitudes using finite differences.
    This is a fallback if we have time.
    """
    current_pulse = pulse.copy()
    best_pulse = current_pulse.copy()
    best_score = np.inf
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        mean_inf, max_inf = compute_score(current_pulse)
        score = mean_inf + max_inf
        
        if score < best_score:
            best_score = score
            best_pulse = current_pulse.copy()
        
        # Compute gradient using central differences
        grad = np.zeros_like(current_pulse)
        eps = 0.001
        
        for i in range(N_STEPS):
            for j in range(2):
                # Forward
                p_plus = current_pulse.copy()
                p_plus[i, j] += eps
                if not check_validity(p_plus):
                    p_plus[i, j] = current_pulse[i, j]
                    continue
                s_plus = compute_score(p_plus)[0] + compute_score(p_plus)[1]
                
                # Backward
                p_minus = current_pulse.copy()
                p_minus[i, j] -= eps
                if not check_validity(p_minus):
                    p_minus[i, j] = current_pulse[i, j]
                    continue
                s_minus = compute_score(p_minus)[0] + compute_score(p_minus)[1]
                
                grad[i, j] = (s_plus - s_minus) / (2 * eps)
        
        # Update
        current_pulse -= lr * grad
        
        # Project back to feasible set
        if not check_validity(current_pulse):
            # Simple projection: scale down
            max_amp = np.max(np.sqrt(current_pulse[:, 0]**2 + current_pulse[:, 1]**2))
            if max_amp > AMPLITUDE_BOUND:
                current_pulse *= AMPLITUDE_BOUND / max_amp
            for i in range(N_STEPS - 1):
                dI = current_pulse[i + 1, 0] - current_pulse[i, 0]
                dQ = current_pulse[i + 1, 1] - current_pulse[i, 1]
                slew = np.sqrt(dI**2 + dQ**2) / DT
                if slew > SLEW_BOUND:
                    current_pulse *= SLEW_BOUND / slew
            
            if not check_validity(current_pulse):
                current_pulse = best_pulse.copy()
    
    return best_pulse


def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes for the X-pi gate.
    """
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi gate is supported")
    
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    # Generate diverse candidate pulses
    candidates = []
    
    # DRAG Gaussian pulses with various parameters
    for width_scale in [0.7, 0.8, 0.9, 1.0, 1.1]:
        for center_shift in [-1.0, 0.0, 1.0]:
            for beta in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                if time.time() > deadline:
                    break
                p = drag_gaussian_pulse(beta=beta, width_scale=width_scale, center_shift=center_shift)
                if check_validity(p):
                    candidates.append(p)
    
    # Cosine pulses
    for beta in [0.0, 0.1, 0.2, 0.3]:
        if time.time() > deadline:
            break
        p = cosine_pulse(beta=beta)
        if check_validity(p):
            candidates.append(p)
    
    # Evaluate all candidates
    best_pulse = None
    best_score = np.inf
    
    for p in candidates:
        if time.time() > deadline:
            break
        try:
            mean_inf, max_inf = compute_score(p)
            score = mean_inf + max_inf
            if score < best_score:
                best_score = score
                best_pulse = p.copy()
        except:
            continue
    
    # If we have time, try gradient descent from the best candidate
    if best_pulse is not None and time.time() < deadline - 30:
        try:
            refined = gradient_descent(best_pulse, max_iter=20, lr=0.005, deadline=deadline - 5)
            if check_validity(refined):
                mean_inf, max_inf = compute_score(refined)
                score = mean_inf + max_inf
                if score < best_score:
                    best_pulse = refined
                    best_score = score
        except:
            pass
    
    if best_pulse is None:
        # Fallback
        best_pulse = drag_gaussian_pulse(beta=0.2, width_scale=1.0, center_shift=0.0)
    
    return best_pulse
# EVOLVE-BLOCK-END
