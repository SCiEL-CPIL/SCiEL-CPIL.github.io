"""Optimize 3-layer Ry/CZ circuit parameters for transverse-field Ising ground states."""
import numpy as np
from scipy.optimize import minimize
import time

def _exact_ground_energy(n, J, h):
    """Compute exact ground state energy of open TFIM chain via exact diagonalization."""
    # Build Hamiltonian in computational basis
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms: -J * Z_i Z_{i+1}
    for i in range(n - 1):
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            z_i = 1 - 2 * bit_i
            z_ip1 = 1 - 2 * bit_ip1
            H[state, state] += -J * z_i * z_ip1
    
    # X terms: -h * X_i
    for i in range(n):
        for state in range(dim):
            partner = state ^ (1 << i)
            H[state, partner] += -h
    
    # Diagonalize
    eigvals = np.linalg.eigvalsh(H)
    return float(eigvals[0])


def _simulate_circuit(n, thetas, J, h):
    """
    Simulate the 3-layer Ry/CZ circuit and compute expectation of H.
    thetas: array of shape (3, n)
    Returns: (energy, statevector)
    """
    # Start with |0>^n
    state = np.zeros(1 << n, dtype=complex)
    state[0] = 1.0
    
    for layer in range(3):
        # Apply Ry(theta) to every qubit
        for q in range(n):
            theta = thetas[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            # Ry = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]]
            # For qubit q, we need to apply this to the q-th qubit
            # Iterate over all basis states
            new_state = np.zeros(1 << n, dtype=complex)
            for idx in range(1 << n):
                bit = (idx >> q) & 1
                # The other bits remain the same
                base = idx & ~(1 << q)
                if bit == 0:
                    # |0> -> c|0> + s|1>
                    new_state[base] += c * state[idx]
                    new_state[base | (1 << q)] += s * state[idx]
                else:
                    # |1> -> -s|0> + c|1>
                    new_state[base] += -s * state[idx]
                    new_state[base | (1 << q)] += c * state[idx]
            state = new_state
        
        # Apply CZ on adjacent pairs (ascending order), except after last layer
        if layer < 2:
            for i in range(n - 1):
                # CZ: |11> -> -|11>, others unchanged
                for idx in range(1 << n):
                    bit_i = (idx >> i) & 1
                    bit_ip1 = (idx >> (i + 1)) & 1
                    if bit_i == 1 and bit_ip1 == 1:
                        state[idx] *= -1
    
    # Compute energy expectation
    # E = -J * sum Z_i Z_{i+1} - h * sum X_i
    energy = 0.0
    for i in range(n - 1):
        for idx in range(1 << n):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            z_i = 1 - 2 * bit_i
            z_ip1 = 1 - 2 * bit_ip1
            energy += -J * z_i * z_ip1 * (np.abs(state[idx])**2).real
    for i in range(n):
        for idx in range(1 << n):
            partner = idx ^ (1 << i)
            energy += -h * (state[idx].conjug() * state[partner]).real
    
    return energy, state


def _energy_only(n, thetas_flat, J, h, e0):
    """Compute (energy - e0) / n for optimization."""
    thetas = thetas_flat.reshape(3, n)
    energy, _ = _simulate_circuit(n, thetas, J, h)
    return (energy - e0) / n


def _objective_wrapper(n, J, h, e0):
    def obj(thetas_flat):
        return _energy_only(n, thetas_flat, J, h, e0)
    return obj


def _physics_init(n, J, h):
    """Generate physics-informed initial guess for thetas."""
    thetas = np.zeros((3, n))
    # For strong field (h >> J), ground state is close to |+>^n, so theta ~ pi/2
    # For strong coupling (J >> h), ground state is close to |0>^n or |1>^n
    ratio = h / J if J != 0 else 1.0
    # Interpolate: at ratio=0, theta~0; at ratio->inf, theta~pi/2
    base_theta = np.pi / 2 * min(1.0, ratio)
    for layer in range(3):
        for q in range(n):
            thetas[layer, q] = base_theta
    return thetas


def angles(n, coupling, field, layers):
    """
    Optimize circuit parameters for the given TFIM parameters.
    Returns: array of shape (layers, n) with optimal angles in radians.
    """
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    # Compute exact ground energy
    e0 = _exact_ground_energy(n, coupling, field)
    
    # Objective function
    obj = _objective_wrapper(n, coupling, field, e0)
    
    best_theta = None
    best_val = np.inf
    
    # Generate multiple starting points
    starts = []
    
    # Physics-informed start
    starts.append(_physics_init(n, coupling, field))
    
    # Uniform pi/2 (product |+>)
    t = np.zeros((3, n))
    t[-1] = np.pi / 2
    starts.append(t.copy())
    
    # All zero
    starts.append(np.zeros((3, n)))
    
    # Random starts
    np.random.seed(42)
    for _ in range(5):
        t = np.random.uniform(-np.pi/2, np.pi/2, (3, n))
        starts.append(t)
    
    # Grid-based coarse search for n=4 (small enough)
    if n == 4:
        # Try a coarse grid on the last layer angles
        grid_vals = np.linspace(-np.pi/2, np.pi/2, 5)
        for g1 in grid_vals:
            for g2 in grid_vals:
                t = np.zeros((3, n))
                t[0, :] = g1
                t[1, :] = g1
                t[2, :] = g2
                starts.append(t.copy())
    
    # Optimize from each starting point
    for s_idx, t0 in enumerate(starts):
        if time.time() > deadline - 10:
            break
        
        t0_flat = t0.flatten()
        try:
            result = minimize(
                obj,
                t0_flat,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            val = result.fun
            if val < best_val:
                best_val = val
                best_theta = result.x.reshape(3, n)
        except Exception:
            pass
    
    # If we have time, do a finer polish
    if best_theta is not None and time.time() < deadline - 15:
        best_flat = best_theta.flatten()
        try:
            result = minimize(
                obj,
                best_flat,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_theta = result.x.reshape(3, n)
        except Exception:
            pass
    
    if best_theta is None:
        # Fallback: return physics init
        best_theta = _physics_init(n, coupling, field)
    
    return best_theta
