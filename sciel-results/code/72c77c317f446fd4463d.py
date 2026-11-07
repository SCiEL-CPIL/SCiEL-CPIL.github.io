"""Robust control pulses for a three-level transmon."""
import numpy as np
import time
from scipy.linalg import expm

# Hardware parameters
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
T1 = 25000.0
T2 = 40000.0

N_STEPS = 80
DT = 0.5
DIM = 3

# Build ladder operators for 3-level system
def build_operators():
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2.0)
    a_dag = a.conj().T
    n = a_dag @ a
    return a, a_dag, n

a_op, a_dag_op, n_op = build_operators()

# Precompute the dissipation superoperator (independent of detuning, scale, I, Q)
def build_dissipation_superop():
    dim = DIM
    L_diss = np.zeros((dim * dim, dim * dim), dtype=complex)
    L1 = a_op / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / T2)
    eye = np.eye(dim)
    for L_op in [L1, L2]:
        L_dag = L_op.conj().T
        L_dag_L = L_dag @ L_op
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    for m in range(dim):
                        L_diss[i * dim + j, k * dim + m] += L_op[i, k] * L_dag[m, j]
                        L_diss[i * dim + j, k * dim + m] -= 0.5 * L_dag_L[i, k] * (eye[j, m] + eye[m, j])
    return L_diss

L_DISSIPATION = build_dissipation_superop()

def hamiltonian_superop(H):
    """Build the -i[H, .] superoperator from a 3x3 Hamiltonian H."""
    dim = DIM
    S = np.zeros((dim * dim, dim * dim), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                S[i * dim + j, k * dim + j] -= 1j * H[i, k]
                S[i * dim + j, k * dim + i] += 1j * H[k, j]
    return S

A_mat = a_op + a_dag_op
B_mat = 1j * (a_op - a_dag_op)

L_A = hamiltonian_superop(A_mat)
L_B = hamiltonian_superop(B_mat)
L_N = hamiltonian_superop(n_op)

n_minus_1 = n_op - np.eye(DIM)
anh_term = n_op @ n_minus_1
H_anh = 0.5 * ANHARMONICITY * anh_term
L_ANH = hamiltonian_superop(H_anh)

def build_liouvillian_fast(detuning, scale, I, Q):
    """Build full Liouvillian using precomputed components."""
    L = L_DISSIPATION + L_ANH + detuning * L_N
    L += 0.5 * scale * (I * L_A + Q * L_B)
    return L

def check_validity(pulse):
    """Check if the pulse satisfies all constraints."""
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    if np.any(amps > AMPLITUDE_BOUND + 1e-10):
        return False
    diffs = np.diff(pulse, axis=0, prepend=np.zeros((1, 2)))
    slew = np.linalg.norm(diffs, axis=1) / DT
    if np.any(slew > SLEW_BOUND + 1e-10):
        return False
    if np.linalg.norm(pulse[-1]) / DT > SLEW_BOUND + 1e-10:
        return False
    return True

def seed_pulse():
    """DRAG-style Gaussian seed pulse."""
    t = np.linspace(0, N_STEPS * DT, N_STEPS)
    sigma = 40.0 / 6.0
    center = N_STEPS * DT / 2.0
    env = np.exp(-0.5 * ((t - center) / sigma)**2)
    edge_offset = env[0]
    env = env - edge_offset
    env = np.maximum(env, 0)
    area = np.sum(env) * DT
    if area > 0:
        env = env * (np.pi / area)
    denv = np.gradient(env, DT)
    drag_coeff = 0.1
    I = env
    Q = drag_coeff * denv
    pulse = np.column_stack([I, Q])
    if not check_validity(pulse):
        max_amp = np.max(np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2))
        if max_amp > AMPLITUDE_BOUND:
            pulse *= AMPLITUDE_BOUND / max_amp
        for i in range(1, N_STEPS):
            delta = np.linalg.norm(pulse[i] - pulse[i-1]) / DT
            if delta > SLEW_BOUND:
                scale = SLEW_BOUND / delta
                pulse[i:] *= scale
    return pulse

# Precompute all 54 initial density matrices (6 states)
STATES = [
    np.array([1, 0, 0], dtype=complex),
    np.array([0, 1, 0], dtype=complex),
    (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2),
    (np.array([1, -1, 0], dtype=complex)) / np.sqrt(2),
    (np.array([1, 1j, 0], dtype=complex)) / np.sqrt(2),
    (np.array([1, -1j, 0], dtype=complex)) / np.sqrt(2),
]
TARGET_RHOS = []
for st in STATES:
    ts = np.zeros(3, dtype=complex)
    ts[0] = st[1]
    ts[1] = st[0]
    ts[2] = st[2]
    TARGET_RHOS.append(np.outer(ts, ts.conj()))

INITIAL_RHOS = [np.outer(st, st.conj()) for st in STATES]
INITIAL_RHOS_VEC = np.array([r.flatten() for r in INITIAL_RHOS])

# Precompute fidelity coefficients
FID_COEFFS = np.zeros((6, 9), dtype=complex)
for s in range(6):
    T = TARGET_RHOS[s]
    for i in range(3):
        for j in range(3):
            FID_COEFFS[s, i*3+j] = T[j, i]

def compute_score_fast(pulse):
    """Fast vectorized score computation."""
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    case_infidelities = []
    
    for det in detunings:
        for scale in scales:
            rhos_vec = INITIAL_RHOS_VEC.copy()
            
            for i in range(N_STEPS):
                I_val = pulse[i, 0]
                Q_val = pulse[i, 1]
                L = build_liouvillian_fast(det, scale, I_val, Q_val)
                U = expm(L * DT)
                rhos_vec = rhos_vec @ U.T
            
            fids = np.real(np.sum(rhos_vec * FID_COEFFS, axis=1))
            mean_fid = np.mean(fids)
            infid = 1.0 - mean_fid
            case_infidelities.append(infid)
    
    score = np.mean(case_infidelities) + np.max(case_infidelities)
    return score

def pulse(target_gate, hardware_model):
    """Main entry point."""
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Start with seed
    best_pulse = seed_pulse()
    best_score = compute_score_fast(best_pulse)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 120:
        return best_pulse
    
    # Phase 1: Try diverse DRAG variants
    np.random.seed(42)
    
    candidates = [best_pulse]
    for sigma_mult in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]:
        for drag_coeff in [-0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            t = np.linspace(0, N_STEPS * DT, N_STEPS)
            sigma = 40.0 / 6.0 * sigma_mult
            center = N_STEPS * DT / 2.0
            env = np.exp(-0.5 * ((t - center) / sigma)**2)
            edge_offset = env[0]
            env = env - edge_offset
            env = np.maximum(env, 0)
            area = np.sum(env) * DT
            if area > 0:
                env = env * (np.pi / area)
            denv = np.gradient(env, DT)
            I = env
            Q = drag_coeff * denv
            cand = np.column_stack([I, Q])
            if check_validity(cand):
                candidates.append(cand)
    
    # Evaluate all candidates
    for cand in candidates:
        sc = compute_score_fast(cand)
        if sc < best_score:
            best_score = sc
            best_pulse = cand
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 90:
        return best_pulse
    
    # Phase 2: Gradient-based optimization using finite differences
    # We'll compute gradients numerically and use them for a custom descent
    
    def objective(params):
        pulse_arr = params.reshape(N_STEPS, 2)
        if not check_validity(pulse_arr):
            return 1e10
        return compute_score_fast(pulse_arr)
    
    x0 = best_pulse.flatten()
    
    # Try a few optimization approaches
    # Approach 1: Nelder-Mead with careful settings
    try:
        from scipy.optimize import minimize
        result = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 800, 'xatol': 1e-7, 'fatol': 1e-9,
                                  'adaptive': True})
        if result.fun < best_score:
            candidate = result.x.reshape(N_STEPS, 2)
            if check_validity(candidate):
                best_pulse = candidate
                best_score = result.fun
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 60:
        return best_pulse
    
    # Phase 3: Adaptive random search with momentum
    step_size = 0.02
    no_improve_count = 0
    max_no_improve = 80
    momentum = np.zeros_like(best_pulse)
    lr = 0.3
    
    for iteration in range(1000):
        if time.time() > deadline - 15:
            break
        if no_improve_count > max_no_improve:
            break
        
        if iteration % 40 == 0 and iteration > 0:
            step_size *= 0.88
        
        perturbation = np.random.randn(N_STEPS, 2) * step_size
        candidate = best_pulse + momentum + perturbation
        
        # Clip to amplitude bound
        amps = np.sqrt(candidate[:, 0]**2 + candidate[:, 1]**2)
        mask = amps > AMPLITUDE_BOUND
        if np.any(mask):
            candidate[mask] *= AMPLITUDE_BOUND / amps[mask]
        
        if check_validity(candidate):
            score = compute_score_fast(candidate)
            if score < best_score:
                momentum = lr * (candidate - best_pulse)
                best_pulse = candidate
                best_score = score
                no_improve_count = 0
            else:
                momentum = lr * momentum
                no_improve_count += 1
    
    return best_pulse
