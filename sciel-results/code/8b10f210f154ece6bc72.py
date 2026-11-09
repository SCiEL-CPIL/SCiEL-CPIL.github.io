"""Direct 160-D coordinate-descent pulse optimization for 3-level transmon X-pi gate."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg as sla
import time
from scipy.optimize import minimize

# --- Constants from task ---
DT = 0.5
N_STEPS = 80
GATE_TIME = 40.0
T1 = 25000.0
Tphi = 40000.0
ANHARMONICITY = -2 * np.pi * 0.25
DETUNING_VALUES = np.array([-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002])
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

# Lindblad operators
G1 = A_MAT / np.sqrt(T1)
G2 = N_MAT * np.sqrt(2.0 / Tphi)
G1DAG = G1.conj().T
G2DAG = G2.conj().T
G1DAG_G1 = G1DAG @ G1
G2DAG_G2 = G2DAG @ G2

DISS_L = (np.kron(G1, G1.conj()) - 0.5 * (np.kron(G1DAG_G1, I3) + np.kron(I3, G1DAG_G1.T)) +
          np.kron(G2, G2.conj()) - 0.5 * (np.kron(G2DAG_G2, I3) + np.kron(I3, G2DAG_G2.T)))

# Test states and X matrix
STATES = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0] / np.sqrt(2),
    [1, -1, 0] / np.sqrt(2),
    [1, 1j, 0] / np.sqrt(2),
    [1, -1j, 0] / np.sqrt(2),
], dtype=complex)

X3 = np.array([[0, 1, 0],
               [1, 0, 0],
               [0, 0, 1]], dtype=complex)

# Precompute rho0 vectors and target vectors for each state
RHO0_VECTORS = []
TARGET_VECTORS = []
for psi in STATES:
    rho0 = np.outer(psi, psi.conj())
    RHO0_VECTORS.append(rho0.reshape(-1).copy())
    psi_t = X3 @ psi
    TARGET_VECTORS.append(psi_t)

# Precompute the 9 (detuning, scale) pairs
CASES = [(d, s) for d in DETUNING_VALUES for s in SCALE_VALUES]


def build_hamiltonian(detuning, scale, I, Q):
    """Build the 3-level Hamiltonian matrix."""
    drive = 0.5 * scale * (I * (A_MAT + AD_MAT) + 1j * Q * (A_MAT - AD_MAT))
    return detuning * N_MAT + 0.5 * ANHARMONICITY * NN1 + drive


def evolve_case(pulse, detuning, scale):
    """Evolve all 6 states for one (detuning, scale) case, return mean fidelity."""
    # Build expm matrices for this case
    exp_mats = []
    for i in range(N_STEPS):
        I_val, Q_val = pulse[i]
        H = build_hamiltonian(detuning, scale, I_val, Q_val)
        L = -1j * (np.kron(H, I3) - np.kron(I3, H.T)) + DISS_L
        exp_mats.append(sla.expm(L * DT))
    
    # Evolve each state
    fidelities = []
    for k in range(6):
        rho_vec = RHO0_VECTORS[k].copy()
        for i in range(N_STEPS):
            rho_vec = exp_mats[i] @ rho_vec
        rho = rho_vec.reshape(3, 3)
        psi_t = TARGET_VECTORS[k]
        f = np.vdot(psi_t, rho @ psi_t).real
        f = np.clip(f, 0, 1)
        fidelities.append(f)
    return np.mean(fidelities)


def score_only(pulse):
    """Compute score (mean + max infidelity) over 9 cases."""
    infidelities = np.zeros(9)
    for idx, (d, s) in enumerate(CASES):
        f = evolve_case(pulse, d, s)
        infidelities[idx] = 1.0 - f
    return np.mean(infidelities) + np.max(infidelities)


def check_validity(pulse):
    """Check amplitude and slew constraints."""
    for i in range(len(pulse)):
        amp = np.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
        if amp > AMP_BOUND + 1e-9:
            return False
    prev = np.array([0.0, 0.0])
    for i in range(len(pulse)):
        curr = pulse[i]
        diff = np.linalg.norm(curr - prev)
        if diff > MAX_DIFF + 1e-9:
            return False
        prev = curr
    diff = np.linalg.norm(prev)
    if diff > MAX_DIFF + 1e-9:
        return False
    return True


def enforce_constraints(pulse):
    """Project pulse onto feasible set: amplitude + slew (forward/backward)."""
    pulse = pulse.copy()
    
    # Enforce amplitude bound
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    mask = amps > AMP_BOUND
    if np.any(mask):
        pulse[mask] *= (AMP_BOUND / amps[mask])[:, None]
    
    # Enforce slew bound forward
    prev = np.array([0.0, 0.0])
    for i in range(N_STEPS):
        curr = pulse[i].copy()
        diff = curr - prev
        diff_norm = np.linalg.norm(diff)
        if diff_norm > MAX_DIFF:
            curr = prev + diff * (MAX_DIFF / diff_norm)
        pulse[i] = curr
        prev = curr
    
    # Enforce slew bound backward
    prev = np.array([0.0, 0.0])
    for i in range(N_STEPS - 1, -1, -1):
        curr = pulse[i].copy()
        diff = curr - prev
        diff_norm = np.linalg.norm(diff)
        if diff_norm > MAX_DIFF:
            curr = prev + diff * (MAX_DIFF / diff_norm)
        pulse[i] = curr
        prev = curr
    
    # Re-enforce amplitude after slew projection
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    mask = amps > AMP_BOUND
    if np.any(mask):
        pulse[mask] *= (AMP_BOUND / amps[mask])[:, None]
    
    return pulse


def seed_pulse():
    """Generate a DRAG-style Gaussian seed pulse."""
    t = np.arange(N_STEPS) * DT
    # Gaussian with sigma = 40/6 ns, centered at 20 ns
    sigma = 40.0 / 6.0
    center = 20.0
    # Amplitude chosen so that integral ~ pi (area = amp * sigma * sqrt(2*pi))
    amp = np.pi / (sigma * np.sqrt(2 * np.pi))
    
    g = amp * np.exp(-0.5 * ((t - center) / sigma)**2)
    # Offset to make edges ~0
    g -= g[0]
    
    # DRAG Q component: proportional to derivative of I
    dg = g * (-(t - center) / sigma**2)
    beta = 0.5 * np.abs(ANHARMONICITY) / (2 * np.pi)  # rough DRAG beta
    q = beta * dg
    
    pulse = np.column_stack((g, q))
    pulse = enforce_constraints(pulse)
    return pulse


def pulse(target_gate, hardware_model):
    """Main entry point: optimize the 160-D pulse directly."""
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    start_time = time.time()
    deadline = start_time + 290  # 40s safety margin
    
    # Start from seed
    p0 = seed_pulse()
    best_pulse = p0.copy()
    best_score = score_only(best_pulse)
    
    if time.time() > deadline:
        return best_pulse
    
    # Flatten pulse to 160-D vector
    x0 = best_pulse.reshape(-1).copy()
    
    def objective(x):
        p = x.reshape(N_STEPS, 2)
        p = enforce_constraints(p)
        return score_only(p)
    
    # Nelder-Mead on 160-D: use small initial simplex and adaptive scaling
    # This is expensive, so we limit iterations
    try:
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 300,
                'xatol': 1e-5,
                'fatol': 1e-7,
                'adaptive': True,
                'disp': False,
                'initial_simplex': _make_simplex(x0),
            }
        )
        
        if result.fun < best_score:
            opt_pulse = enforce_constraints(result.x.reshape(N_STEPS, 2))
            opt_score = score_only(opt_pulse)
            if opt_score < best_score:
                best_pulse = opt_pulse
                best_score = opt_score
                
    except Exception:
        pass
    
    # If time remains, do a second Nelder-Mead from a perturbed start
    if time.time() < deadline - 60:
        try:
            x1 = best_pulse.reshape(-1).copy()
            x1 += 0.01 * np.random.randn(160)
            x1 = enforce_constraints(x1.reshape(N_STEPS, 2)).reshape(-1)
            
            result2 = minimize(
                objective,
                x1,
                method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'xatol': 1e-5,
                    'fatol': 1e-7,
                    'adaptive': True,
                    'disp': False,
                    'initial_simplex': _make_simplex(x1),
                }
            )
            if result2.fun < best_score:
                opt_pulse = enforce_constraints(result2.x.reshape(N_STEPS, 2))
                opt_score = score_only(opt_pulse)
                if opt_score < best_score:
                    best_pulse = opt_pulse
                    best_score = opt_score
        except Exception:
            pass
    
    return best_pulse


def _make_simplex(x0):
    """Create a small initial simplex for Nelder-Mead in 160-D."""
    n = len(x0)
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        # Perturb one coordinate by a small amount
        delta = 0.005 if abs(x0[i]) < 0.1 else 0.01
        xi[i] += delta
        simplex.append(xi)
    return np.array(simplex)
# EVOLVE-BLOCK-END
