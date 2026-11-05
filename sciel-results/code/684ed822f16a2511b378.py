"""Quantum information: shallow-circuit Ising ground states.
Optimize 3-layer Ry/CZ circuit angles to approximate ground states of
open transverse-field Ising chains.
"""
import numpy as np
import time
from scipy.optimize import minimize

# Global deadline guard
_DEADLINE = 150.0
_START_TIME = None


def _get_elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def _time_remaining():
    return _DEADLINE - _get_elapsed()


def _build_hamiltonian(n, coupling, field):
    """Build the full Hamiltonian matrix for open transverse-field Ising chain.
    H = -coupling * sum_i Z_i Z_{i+1} - field * sum_i X_i
    Qubit i is bit i (little-endian).
    """
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.float64)
    
    # Z Z interaction terms
    for i in range(n - 1):
        # Z_i Z_{i+1}: diagonal in computational basis
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            zz = (1 - 2 * bit_i) * (1 - 2 * bit_ip1)
            H[state, state] += -coupling * zz
    
    # X terms: flip bit i
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H[state, flipped] += -field
    
    return H


def _ry_matrix(theta):
    """Ry rotation matrix for angle theta."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _apply_ry_to_qubit(state, qubit, theta, n):
    """Apply Ry(theta) to qubit in n-qubit state vector."""
    dim = 2 ** n
    new_state = np.zeros_like(state)
    # Iterate over basis states where qubit is 0
    for state_idx in range(dim):
        if (state_idx >> qubit) & 1 == 0:
            bit0 = state_idx
            bit1 = state_idx | (1 << qubit)
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            new_state[bit0] += c * state[bit0] - s * state[bit1]
            new_state[bit1] += s * state[bit0] + c * state[bit1]
    return new_state


def _apply_cz(state, i, j, n):
    """Apply CZ gate between qubits i and j."""
    dim = 2 ** n
    new_state = state.copy()
    for state_idx in range(dim):
        bit_i = (state_idx >> i) & 1
        bit_j = (state_idx >> j) & 1
        if bit_i == 1 and bit_j == 1:
            new_state[state_idx] = -state[state_idx]
    return new_state


def _apply_circuit(theta, n, layers):
    """Apply the 3-layer Ry/CZ circuit to |0>^n.
    theta: shape (layers, n)
    Each layer: apply Ry(theta[l]) to every qubit.
    Between layers: apply CZ on adjacent pairs (i, i+1) in ascending order.
    No entangler after final layer.
    """
    dim = 2 ** n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0  # |0>^n
    
    for l in range(layers):
        # Apply Ry to every qubit
        for q in range(n):
            state = _apply_ry_to_qubit(state, q, theta[l, q], n)
        
        # Apply CZ between layers (not after final layer)
        if l < layers - 1:
            for i in range(n - 1):
                state = _apply_cz(state, i, i + 1, n)
    
    return state


def _circuit_energy(state, H, n):
    """Compute <state|H|state>."""
    return np.real(np.vdot(state, H @ state))


def _objective(theta_flat, n, coupling, field, layers, H, E0):
    """Objective: mean (circuit energy - exact ground energy) / n."""
    theta = theta_flat.reshape(layers, n)
    state = _apply_circuit(theta, n, layers)
    E_circuit = _circuit_energy(state, H, n)
    return (E_circuit - E0) / n


def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy via diagonalization."""
    H = _build_hamiltonian(n, coupling, field)
    evals = np.linalg.eigvalsh(H)
    return evals[0], H


def angles(n, coupling, field, layers):
    """Optimize circuit angles to approximate ground state.
    Returns: finite real array of shape (layers, n) in radians.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    # Compute exact ground state energy and Hamiltonian
    E0, H = _exact_ground_energy(n, coupling, field)
    
    # Multi-start optimization
    best_theta = None
    best_val = np.inf
    
    # Initial guesses
    inits = []
    
    # 1. All zeros
    inits.append(np.zeros((layers, n)))
    
    # 2. All pi/2 (maximally mixed-like)
    inits.append(np.full((layers, n), np.pi / 2))
    
    # 3. Based on field strength: small field -> small angles
    for f_scale in [0.1, 0.3, 0.5, 0.7, 1.0]:
        init = np.zeros((layers, n))
        for l in range(layers):
            for q in range(n):
                init[l, q] = f_scale * np.pi / 2 * (1.0 + 0.1 * np.sin(q * np.pi / n))
        inits.append(init)
    
    # 4. Random inits
    rng = np.random.RandomState(42)
    for _ in range(8):
        inits.append(rng.uniform(-np.pi, np.pi, size=(layers, n)))
    
    # 5. Symmetric inits (all qubits same angle per layer)
    for a1 in [0.1, 0.3, 0.5, 0.8]:
        for a2 in [0.1, 0.3, 0.5, 0.8]:
            for a3 in [0.1, 0.3, 0.5, 0.8]:
                init = np.zeros((layers, n))
                init[0, :] = a1 * np.pi
                init[1, :] = a2 * np.pi
                init[2, :] = a3 * np.pi
                inits.append(init)
    
    # Sort inits by objective value
    init_vals = []
    for init in inits:
        if _time_remaining() < 5.0:
            break
        val = _objective(init.flatten(), n, coupling, field, layers, H, E0)
        init_vals.append((val, init))
    
    init_vals.sort(key=lambda x: x[0])
    
    # Optimize from best inits
    max_starts = 12
    for val, init in init_vals[:max_starts]:
        if _time_remaining() < 10.0:
            break
        
        theta_flat = init.flatten()
        
        # Try multiple optimizers
        results = []
        
        # L-BFGS-B
        try:
            res = minimize(
                _objective, theta_flat,
                args=(n, coupling, field, layers, H, E0),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.success or res.fun < val:
                results.append(res)
        except Exception:
            pass
        
        # Nelder-Mead for local refinement
        try:
            res_nm = minimize(
                _objective, theta_flat,
                args=(n, coupling, field, layers, H, E0),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res_nm.success or res_nm.fun < val:
                results.append(res_nm)
        except Exception:
            pass
        
        for res in results:
            if res.fun < best_val:
                best_val = res.fun
                best_theta = res.x.reshape(layers, n)
        
        # Polish the best found so far
        if best_theta is not None and _time_remaining() > 15.0:
            try:
                res_polish = minimize(
                    _objective, best_theta.flatten(),
                    args=(n, coupling, field, layers, H, E0),
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if res_polish.fun < best_val:
                    best_val = res_polish.fun
                    best_theta = res_polish.x.reshape(layers, n)
            except Exception:
                pass
    
    # Final ultra-precision polish
    if best_theta is not None and _time_remaining() > 5.0:
        try:
            res_final = minimize(
                _objective, best_theta.flatten(),
                args=(n, coupling, field, layers, H, E0),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if res_final.fun < best_val:
                best_val = res_final.fun
                best_theta = res_final.x.reshape(layers, n)
        except Exception:
            pass
    
    # Safety: if optimization failed, return best initial
    if best_theta is None:
        best_theta = init_vals[0][1]
    
    # Ensure finite
    best_theta = np.nan_to_num(best_theta, nan=0.0, posinf=np.pi, neginf=-np.pi)
    
    return best_theta
