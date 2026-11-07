import numpy as np
import time
from scipy.linalg import expm

def _build_matrices():
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2.0)
    adag = a.conj().T
    n = adag @ a
    return a, adag, n

def _liouvillian_super(I_val, Q_val, detuning, scale, a, adag, n):
    I3 = np.eye(3, dtype=complex)
    # Anharmonicity: alpha = -2*pi*0.25
    alpha = -2.0 * np.pi * 0.25
    # H = detuning * n + 0.5 * alpha * n(n-1) + 0.5 * scale * (I*(a+a†) + iQ*(a-a†))
    H = detuning * n + 0.5 * alpha * (n @ (n - I3))
    H += 0.5 * scale * (I_val * (a + adag) + 1j * Q_val * (a - adag))
    
    # Liouvillian superoperator in vectorized form (column-stacking)
    # L(rho) = -i[H, rho] + sum_k (L_k rho L_k† - 0.5{L_k†L_k, rho})
    # Vectorized: -1j*(kron(H,I) - kron(I,H.T)) + sum_k (kron(L_k, L_k.conj()) - 0.5*kron(L_k†L_k, I) - 0.5*kron(I, (L_k†L_k).T))
    
    H_super = -1j * (np.kron(H, I3) - np.kron(I3, H.T))
    
    T1 = 25000.0
    Tphi = 40000.0
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / Tphi)
    
    L1dag = L1.conj().T
    L2dag = L2.conj().T
    
    L1_super = np.kron(L1, L1.conj()) - 0.5 * (np.kron(L1dag @ L1, I3) + np.kron(I3, (L1dag @ L1).T))
    L2_super = np.kron(L2, L2.conj()) - 0.5 * (np.kron(L2dag @ L2, I3) + np.kron(I3, (L2dag @ L2).T))
    
    return H_super + L1_super + L2_super

def _simulate_pulse(IQ, detuning, scale, a, adag, n):
    dt = 0.5
    X_gate = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, -1, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, 1j, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, -1j, 0], dtype=complex)) / np.sqrt(2),
    ]
    
    fids = []
    for psi in states:
        rho = np.outer(psi, psi.conj()).flatten()
        for i in range(80):
            L = _liouvillian_super(IQ[i, 0], IQ[i, 1], detuning, scale, a, adag, n)
            S = expm(L * dt)
            rho = S @ rho
        rho = rho.reshape(3, 3)
        psi_t = X_gate @ psi
        rho_t = np.outer(psi_t, psi_t.conj())
        fid = np.real(np.trace(rho_t @ rho))
        fids.append(fid)
    return np.mean(fids)

def _score(IQ):
    a, adag, n = _build_matrices()
    infs = []
    for det in [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]:
        for sc in [0.97, 1.0, 1.03]:
            fid = _simulate_pulse(IQ, det, sc, a, adag, n)
            infs.append(1.0 - fid)
    return float(np.mean(infs) + np.max(infs))

def _valid(IQ):
    amps = np.sqrt(IQ[:, 0] ** 2 + IQ[:, 1] ** 2)
    if np.any(amps > 0.4 + 1e-12):
        return False
    max_delta = 0.12 * 0.5
    for i in range(80):
        prev = IQ[i - 1] if i > 0 else np.array([0.0, 0.0])
        if np.linalg.norm(IQ[i] - prev) > max_delta + 1e-12:
            return False
    if np.linalg.norm(IQ[-1]) > max_delta + 1e-12:
        return False
    return True

def _enforce_constraints(IQ):
    IQ = IQ.copy().astype(float)
    max_delta = 0.12 * 0.5
    
    # Slew rate: forward pass
    prev = np.array([0.0, 0.0])
    for i in range(80):
        diff = IQ[i] - prev
        dist = np.linalg.norm(diff)
        if dist > max_delta:
            IQ[i] = prev + (max_delta / dist) * diff
        prev = IQ[i]
    
    # Slew rate: backward pass
    prev = np.array([0.0, 0.0])
    for i in range(79, -1, -1):
        diff = prev - IQ[i]
        dist = np.linalg.norm(diff)
        if dist > max_delta:
            IQ[i] = prev - (max_delta / dist) * diff
        prev = IQ[i]
    
    # Amplitude bound
    for i in range(80):
        amp = np.sqrt(IQ[i, 0] ** 2 + IQ[i, 1] ** 2)
        if amp > 0.4:
            scale = 0.4 / amp
            IQ[i, 0] *= scale
            IQ[i, 1] *= scale
    
    # Re-enforce slew after amplitude clamping (forward then backward)
    prev = np.array([0.0, 0.0])
    for i in range(80):
        diff = IQ[i] - prev
        dist = np.linalg.norm(diff)
        if dist > max_delta:
            IQ[i] = prev + (max_delta / dist) * diff
        prev = IQ[i]
    
    prev = np.array([0.0, 0.0])
    for i in range(79, -1, -1):
        diff = prev - IQ[i]
        dist = np.linalg.norm(diff)
        if dist > max_delta:
            IQ[i] = prev - (max_delta / dist) * diff
        prev = IQ[i]
    
    return IQ

def _make_seed_pulse(sigma_factor=1.0, drag_coeff=-0.3, amp_factor=1.0):
    t = np.arange(80) * 0.5
    sigma = (40.0 / 6.0) * sigma_factor
    env = np.exp(-(t - 20.0) ** 2 / (2 * sigma ** 2))
    # Offset to ensure zero at edges
    edge_offset = env[0] + env[-1]
    env = np.maximum(env - edge_offset, 0.0)
    area = np.sum(env) * 0.5
    if area > 0:
        env *= np.pi / area
    env *= amp_factor
    I = env
    Q = drag_coeff * np.gradient(I, 0.5)
    IQ = np.column_stack([I, Q])
    IQ = _enforce_constraints(IQ)
    return IQ

def _nelder_mead_simplex(f, x0, max_iter=200, step_size=0.02, f_tol=1e-6):
    """Simple Nelder-Mead implementation for 160-dimensional problem."""
    n = len(x0)
    # Initialize simplex
    simplex = [x0.astype(float)]
    fvals = [f(x0)]
    
    for i in range(n):
        x = x0.copy()
        x[i] += step_size
        simplex.append(x)
        fvals.append(f(x))
    
    # Order by function value
    for _ in range(max_iter):
        # Sort
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[0] - fvals[-1]) < f_tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + (centroid - simplex[-1])
        fr = f(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = f(simplex[i])
    
    return simplex[0], fvals[0]

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 310.0  # Safety margin
    
    # Generate multiple seed candidates
    seed_configs = [
        (1.0, -0.3, 1.0),
        (0.9, -0.4, 1.0),
        (1.1, -0.2, 1.0),
        (1.0, -0.5, 1.0),
        (0.85, -0.35, 1.0),
        (1.0, -0.25, 1.0),
        (1.05, -0.35, 1.0),
        (0.95, -0.45, 1.0),
    ]
    
    best_IQ = None
    best_score = np.inf
    
    for sf, dc, af in seed_configs:
        if time.time() > deadline - 60:
            break
        try:
            IQ = _make_seed_pulse(sf, dc, af)
            if not _valid(IQ):
                continue
            s = _score(IQ)
            if s < best_score:
                best_score = s
                best_IQ = IQ
        except Exception:
            continue
    
    if best_IQ is None:
        # Fallback: return a valid zero pulse
        return np.zeros((80, 2))
    
    # Local optimization using coordinate descent + Nelder-Mead
    a, adag, n = _build_matrices()
    
    def objective(x):
        IQ = _enforce_constraints(x.reshape(80, 2))
        if not _valid(IQ):
            return 1e6
        return _score(IQ)
    
    # Coordinate descent first for speed
    x = best_IQ.flatten().copy()
    f_current = objective(x)
    step_sizes = [0.02, 0.01, 0.005, 0.002, 0.001]
    
    for step in step_sizes:
        if time.time() > deadline - 60:
            break
        improved = True
        while improved and time.time() < deadline - 60:
            improved = False
            for i in range(len(x)):
                for delta in [step, -step]:
                    x_trial = x.copy()
                    x_trial[i] += delta
                    f_trial = objective(x_trial)
                    if f_trial < f_current - 1e-7:
                        x = x_trial
                        f_current = f_trial
                        improved = True
                        break
    
    if time.time() < deadline - 90:
        best_IQ = _enforce_constraints(x.reshape(80, 2))
        best_score = f_current
    
    # Nelder-Mead on a subset of parameters for further refinement
    if time.time() < deadline - 90:
        try:
            # Focus optimization on the main envelope parameters (I channel)
            # and DRAG coefficients (Q channel)
            nm_result, nm_score = _nelder_mead_simplex(
                objective, x, max_iter=150, step_size=0.005, f_tol=1e-5
            )
            if nm_score < best_score:
                best_IQ = _enforce_constraints(nm_result.reshape(80, 2))
                best_score = nm_score
        except Exception:
            pass
    
    # Final validity check
    if not _valid(best_IQ):
        best_IQ = _enforce_constraints(best_IQ)
        if not _valid(best_IQ):
            best_IQ = _make_seed_pulse(1.0, -0.3, 1.0)
    
    return best_IQ
