"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _apply_ry(state, n, qubit, theta):
    """Apply Ry(theta) to qubit on state of n qubits. Vectorized."""
    dim = 1 << n
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    mask = (1 << qubit)
    idx0 = np.array([i for i in range(dim) if (i & mask) == 0])
    idx1 = idx0 | mask
    new_state = state.copy()
    new_state[idx0] = c * state[idx0] - s * state[idx1]
    new_state[idx1] = s * state[idx0] + c * state[idx1]
    return new_state

def _apply_cz(state, n, q1, q2):
    """Apply CZ between qubits q1 and q2 (q1 < q2) on state of n qubits."""
    dim = 1 << n
    new_state = state.copy()
    for i in range(dim):
        if ((i >> q1) & 1) and ((i >> q2) & 1):
            new_state[i] = -state[i]
    return new_state

def _circuit_state(n, layers, params):
    """Compute the state produced by the circuit.
    params: array of shape (layers, n)
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0  # |0>^n
    
    for layer in range(layers):
        # Apply Ry to every qubit
        for q in range(n):
            state = _apply_ry(state, n, q, params[layer, q])
        # Apply CZ on adjacent pairs (except after last layer)
        if layer < layers - 1:
            for q in range(n - 1):
                state = _apply_cz(state, n, q, q + 1)
    
    return state

def _hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix H = -coupling * sum Z_i Z_{i+1} - field * sum X_i"""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    # ZZ terms: diagonal
    for i in range(n - 1):
        for b in range(dim):
            bi = (b >> i) & 1
            bj = (b >> (i + 1)) & 1
            zval = (1 - 2 * bi) * (1 - 2 * bj)
            H[b, b] += -coupling * zval
    
    # X terms: off-diagonal (each pair added once)
    for i in range(n):
        for b in range(dim):
            b_flip = b ^ (1 << i)
            if b < b_flip:  # only add each pair once
                H[b, b_flip] += -field
                H[b_flip, b] += -field
    
    return H

def _energy(state, H):
    """Compute <state|H|state>"""
    return np.real(np.vdot(state, H @ state))

def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy by diagonalizing H"""
    H = _hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0]

def _objective(params, n, coupling, field, layers, H, e_ground):
    """Compute the objective: (circuit_energy - ground_energy) / n"""
    state = _circuit_state(n, layers, params)
    e_circuit = _energy(state, H)
    return (e_circuit - e_ground) / n

def _nelder_mead(objective, x0, maxiter=200, tol=1e-8):
    """Simple Nelder-Mead implementation"""
    n = len(x0)
    # Initial simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        step = 0.1 if x0[i] == 0 else 0.05 * abs(x0[i])
        xi[i] += step
        simplex.append(xi)
    
    fvals = [objective(x) for x in simplex]
    
    for it in range(maxiter):
        # Sort by function value
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[0] - fvals[-1]) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + (centroid - simplex[-1])
        fr = objective(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + 2 * (centroid - simplex[-1])
            fe = objective(xe)
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
            fc = objective(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])
    
    order = np.argsort(fvals)
    return simplex[order[0]], fvals[order[0]]

def angles(n, coupling, field, layers):
    """Optimize circuit parameters to minimize mean energy gap per spin."""
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # Precompute ground energy and Hamiltonian
    e_ground = _exact_ground_energy(n, coupling, field)
    H = _hamiltonian(n, coupling, field)
    
    def obj(params):
        state = _circuit_state(n, layers, params)
        e_circuit = _energy(state, H)
        return (e_circuit - e_ground) / n
    
    # Initial guess: all zeros
    x0 = np.zeros((layers, n))
    
    best_params = x0.copy()
    best_val = obj(x0)
    
    np.random.seed(42)
    
    # Multi-restart: use scipy L-BFGS-B followed by Nelder-Mead polish
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline - 10:  # leave 10s for final polish
            break
        
        if restart == 0:
            x_init = x0.copy()
        elif restart == 1:
            # Symmetric guess: all angles same
            x_init = np.full((layers, n), np.pi / 6)
        else:
            x_init = np.random.uniform(-np.pi / 2, np.pi / 2, (layers, n))
        
        x_flat = x_init.flatten()
        
        def obj_flat(xf):
            return obj(xf.reshape((layers, n)))
        
        # Phase 1: L-BFGS-B (uses gradient approximation)
        try:
            res = minimize(obj_flat, x_flat, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.success or res.fun < best_val:
                x_flat = res.x
        except Exception:
            pass
        
        # Phase 2: Nelder-Mead polish for high precision
        if time.time() < deadline - 5:
            try:
                x_nm, f_nm = _nelder_mead(obj_flat, x_flat, maxiter=150, tol=1e-9)
                if f_nm < best_val:
                    best_val = f_nm
                    best_params = x_nm.reshape((layers, n))
            except Exception:
                pass
        
        # Update best from L-BFGS-B result if better
        if res.fun < best_val:
            best_val = res.fun
            best_params = x_flat.reshape((layers, n))
    
    # Final polish if time allows
    if time.time() < deadline - 2:
        try:
            x_flat = best_params.flatten()
            res = minimize(obj_flat, x_flat, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except Exception:
            pass
    
    return np.clip(best_params, -np.pi, np.pi)
# EVOLVE-BLOCK-END
