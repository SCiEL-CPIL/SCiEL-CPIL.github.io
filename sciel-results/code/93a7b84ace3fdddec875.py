"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix for the open transverse-field Ising chain."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            zz = 1.0 if bit_i == bit_ip1 else -1.0
            H[state, state] += -coupling * zz
    
    # X_i terms
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H[flipped, state] += -field
    
    return H

def _apply_circuit(params, n, layers):
    """Apply the fixed circuit to |0>^n and return the statevector."""
    dim = 1 << n
    # Start with |0>^n
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    
    for layer in range(layers):
        # Apply Ry(theta) to every qubit
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            # Ry = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
            new_psi = np.zeros(dim, dtype=np.complex128)
            for state in range(dim):
                bit = (state >> q) & 1
                if bit == 0:
                    # |0> -> c|0> + s|1>
                    state1 = state | (1 << q)
                    new_psi[state] += c * psi[state]
                    new_psi[state1] += s * psi[state]
                else:
                    # |1> -> -s|0> + c|1>
                    state0 = state & ~(1 << q)
                    new_psi[state] += c * psi[state]
                    new_psi[state0] += -s * psi[state]
            psi = new_psi
        
        # Apply CZ on every adjacent pair (except after final layer)
        if layer < layers - 1:
            for i in range(n - 1):
                for state in range(dim):
                    bit_i = (state >> i) & 1
                    bit_ip1 = (state >> (i + 1)) & 1
                    if bit_i == 1 and bit_ip1 == 1:
                        psi[state] *= -1.0
    
    return psi

def _circuit_energy(params, n, coupling, field, H, layers=3):
    """Compute <psi|H|psi> for the circuit state."""
    psi = _apply_circuit(params, n, layers)
    energy = np.vdot(psi, H @ psi).real
    return energy

def _exact_ground_energy(n, coupling, field, H):
    """Compute exact ground energy via diagonalization."""
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0]

def _objective_flat(x, n, coupling, field, H, layers=3):
    """Flattened objective for optimizer."""
    params = x.reshape((layers, n))
    energy = _circuit_energy(params, n, coupling, field, H, layers)
    return energy

def angles(n, coupling, field, layers):
    """Optimize circuit parameters to approximate ground state."""
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Build Hamiltonian
    H = _build_hamiltonian(n, coupling, field)
    e0 = _exact_ground_energy(n, coupling, field, H)
    
    # Objective: minimize circuit energy (since e0 is constant, minimizing E_circuit minimizes gap)
    def obj(x):
        params = x.reshape((layers, n))
        energy = _circuit_energy(params, n, coupling, field, H, layers)
        return energy
    
    best_params = np.zeros((layers, n))
    best_energy = np.inf
    
    # Multi-restart optimization
    n_restarts = 8
    np.random.seed(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Initial guess
        if restart == 0:
            # Start near |+> state
            x0 = np.zeros(layers * n)
            x0[(layers-1)*n : layers*n] = np.pi / 2
        else:
            # Random initialization
            x0 = np.random.uniform(-np.pi, np.pi, size=layers * n)
        
        try:
            result = minimize(
                obj, x0, method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if result.fun < best_energy:
                best_energy = result.fun
                best_params = result.x.reshape((layers, n))
        except Exception:
            continue
    
    # If we have time, do a polish pass with L-BFGS-B
    if time.time() < deadline - 5.0:
        try:
            result2 = minimize(
                obj, best_params.flatten(), method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            if result2.fun < best_energy:
                best_energy = result2.fun
                best_params = result2.x.reshape((layers, n))
        except Exception:
            pass
    
    return best_params
# EVOLVE-BLOCK-END
