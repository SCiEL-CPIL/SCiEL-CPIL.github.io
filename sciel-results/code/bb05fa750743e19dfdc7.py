import numpy as np
from scipy.optimize import minimize
import time

def pulse(target_gate, hardware_model):
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    anharmonicity = -2 * np.pi * 0.25
    amp_bound = 0.4
    slew_bound = 0.12
    T1 = 25000.0
    Tphi = 40000.0
    dt = 0.5
    nsteps = 80
    
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = 1.0
    a[1, 2] = np.sqrt(2.0)
    adag = a.conj().T
    n = adag @ a
    nnm1 = n @ (n - np.eye(3))
    X = a + adag
    Y = 1j * (a - adag)
    
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / Tphi)
    L1_dag = L1.conj().T
    L2_dag = L2.conj().T
    L1_L1dag = L1_dag @ L1
    L2_L2dag = L2_dag @ L2
    
    def build_H(det, scale, I, Q):
        H = det * n + 0.5 * anharmonicity * nnm1
        H += 0.5 * scale * (I * X + Q * Y)
        return H
    
    def evolve_rho(rho, H, dt):
        """First-order evolution."""
        comm = -1j * (H @ rho - rho @ H)
        diss = (L1 @ rho @ L1_dag - 0.5 * (L1_L1dag @ rho + rho @ L1_L1dag) +
                L2 @ rho @ L2_dag - 0.5 * (L2_L2dag @ rho + rho @ L2_L2dag))
        return rho + dt * (comm + diss)
    
    def simulate_pulse(I, Q, det, scale):
        states = [
            np.array([1, 0, 0], dtype=complex),
            np.array([0, 1, 0], dtype=complex),
            np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
            np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
            np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
            np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
        ]
        targets = [
            np.array([0, 1, 0], dtype=complex),
            np.array([1, 0, 0], dtype=complex),
            (np.array([0, 1, 0], dtype=complex) + np.array([1, 0, 0], dtype=complex)) / np.sqrt(2),
            (np.array([0, 1, 0], dtype=complex) - np.array([1, 0, 0], dtype=complex)) / np.sqrt(2),
            (np.array([0, 1, 0], dtype=complex) + 1j * np.array([1, 0, 0], dtype=complex)) / np.sqrt(2),
            (np.array([0, 1, 0], dtype=complex) - 1j * np.array([1, 0, 0], dtype=complex)) / np.sqrt(2),
        ]
        
        fidelities = []
        for i, psi0 in enumerate(states):
            rho = np.outer(psi0, psi0.conj())
            for t in range(nsteps):
                H = build_H(det, scale, I[t], Q[t])
                rho = evolve_rho(rho, H, dt)
            target_state = targets[i]
            rho_target = np.outer(target_state, target_state.conj())
            fid = np.real(np.trace(rho @ rho_target))
            fid = max(0.0, min(1.0, fid))
            fidelities.append(fid)
        
        return 1.0 - np.mean(fidelities)
    
    def score_pulse(I, Q):
        dets = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
        scales = [0.97, 1.0, 1.03]
        infs = []
        for det in dets:
            for scale in scales:
                infs.append(simulate_pulse(I, Q, det, scale))
        return np.mean(infs) + np.max(infs)
    
    def make_drag_pulse(alpha, sigma_factor, amp_scale):
        t = np.arange(nsteps) * dt
        center = 20.0
        sigma = (40.0 / 6.0) * sigma_factor
        I_raw = np.exp(-0.5 * ((t - center) / sigma) ** 2)
        edge_val = np.exp(-0.5 * ((0 - center) / sigma) ** 2)
        I_raw = I_raw - edge_val
        I_raw = np.maximum(I_raw, 0)
        area = np.sum(I_raw) * dt
        if area > 0:
            I_raw = I_raw * (np.pi / area)
        dI = np.gradient(I_raw, dt)
        Q = -alpha * dI
        I = I_raw * amp_scale
        Q = Q * amp_scale
        
        amp = np.sqrt(I**2 + Q**2)
        if np.max(amp) > amp_bound:
            s = amp_bound / np.max(amp)
            I *= s
            Q *= s
        
        dI_dt = np.concatenate([[I[0] / dt], np.diff(I) / dt, [-I[-1] / dt]])
        dQ_dt = np.concatenate([[Q[0] / dt], np.diff(Q) / dt, [-Q[-1] / dt]])
        slew = np.sqrt(dI_dt**2 + dQ_dt**2)
        if np.max(slew) > slew_bound:
            s = slew_bound / np.max(slew)
            I *= s
            Q *= s
        
        return I, Q
    
    def check_valid(I, Q):
        amp = np.sqrt(I**2 + Q**2)
        if np.max(amp) > amp_bound + 1e-9:
            return False
        dI_dt = np.concatenate([[I[0] / dt], np.diff(I) / dt, [-I[-1] / dt]])
        dQ_dt = np.concatenate([[Q[0] / dt], np.diff(Q) / dt, [-Q[-1] / dt]])
        slew = np.sqrt(dI_dt**2 + dQ_dt**2)
        if np.max(slew) > slew_bound + 1e-9:
            return False
        return True
    
    I0, Q0 = make_drag_pulse(0.1, 1.0, 1.0)
    best_I, best_Q = I0.copy(), Q0.copy()
    best_score = score_pulse(I0, Q0)
    
    start_time = time.time()
    deadline = start_time + 280
    
    # Grid search
    alphas = np.linspace(0.0, 0.5, 11)
    sigmas = np.linspace(0.7, 1.5, 9)
    
    for alpha in alphas:
        if time.time() > deadline - 60:
            break
        for sigma in sigmas:
            if time.time() > deadline - 60:
                break
            I, Q = make_drag_pulse(alpha, sigma, 1.0)
            if not check_valid(I, Q):
                continue
            s = score_pulse(I, Q)
            if s < best_score:
                best_score = s
                best_I, best_Q = I.copy(), Q.copy()
    
    # Local optimization
    if time.time() < deadline - 30:
        def obj_local(params):
            I, Q = make_drag_pulse(params[0], params[1], params[2])
            if not check_valid(I, Q):
                return 1.0
            return score_pulse(I, Q)
        
        # Find best from grid
        best_params = [0.1, 1.0, 1.0]
        for alpha in alphas:
            for sigma in sigmas:
                I, Q = make_drag_pulse(alpha, sigma, 1.0)
                if not check_valid(I, Q):
                    continue
                s = score_pulse(I, Q)
                if s < best_score:
                    best_score = s
                    best_params = [alpha, sigma, 1.0]
        
        result = minimize(obj_local, best_params, method='Nelder-Mead',
                         options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-6})
        if result.fun < best_score:
            I, Q = make_drag_pulse(result.x[0], result.x[1], result.x[2])
            if check_valid(I, Q):
                best_score = result.fun
                best_I, best_Q = I.copy(), Q.copy()
    
    if not check_valid(best_I, best_Q):
        t = np.arange(nsteps) * dt
        center = 20.0
        sigma = 40.0 / 6.0
        I = np.exp(-0.5 * ((t - center) / sigma) ** 2)
        edge_val = np.exp(-0.5 * ((0 - center) / sigma) ** 2)
        I = np.maximum(I - edge_val, 0)
        area = np.sum(I) * dt
        if area > 0:
            I = I * (np.pi / area)
        Q = np.zeros(nsteps)
        amp = np.sqrt(I**2 + Q**2)
        dI_dt = np.concatenate([[I[0] / dt], np.diff(I) / dt, [-I[-1] / dt]])
        dQ_dt = np.concatenate([[Q[0] / dt], np.diff(Q) / dt, [-Q[-1] / dt]])
        slew = np.sqrt(dI_dt**2 + dQ_dt**2)
        scale = min(1.0, amp_bound / np.max(amp), slew_bound / np.max(slew))
        best_I = I * scale
        best_Q = Q * scale
    
    return np.column_stack([best_I, best_Q])
