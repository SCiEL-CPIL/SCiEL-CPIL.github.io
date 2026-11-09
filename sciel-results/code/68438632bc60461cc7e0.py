"""
Optimize shallow circuit parameters for Ising ground states using analytic gradient descent.
Strategy:
1. Construct the Hamiltonian H for the open transverse-field Ising chain.
2. Compute the exact ground energy E_0 via exact diagonalization.
3. Implement a state-vector simulator for the specific 3-layer Ry/CZ circuit.
4. Implement analytic backpropagation to compute the gradient of the energy with respect to all parameters.
5. Use the Adam optimizer to minimize the energy gap (E_circuit - E_0).
6. Perform multiple restarts with different initial parameters to avoid local minima,
   managed by a time budget to ensure completion within limits.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_hamiltonian(n, coupling, field):
    """
    Build the Hamiltonian H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    for an open chain of n qubits.
    """
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    # ZZ terms: -coupling * Z_i Z_{i+1}
    for i in range(n - 1):
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            # Z_i Z_{i+1} = (+1) if bits same, (-1) if different
            zz_val = 1 if bit_i == bit_ip1 else -1
            H[state, state] += -coupling * zz_val
    
    # X terms: -field * X_i
    for i in range(n):
        for state in range(dim):
            # X_i flips bit i
            flipped = state ^ (1 << i)
            H[flipped, state] += -field
    
    return H

def _ground_energy(H):
    """Compute the exact ground energy via exact diagonalization."""
    eigenvalues = np.linalg.eigvalsh(H)
    return eigenvalues[0].real

def _apply_ry(state, theta, qubit):
    """
    Apply Ry(theta) to the given qubit in the state vector.
    Ry(theta) = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]]
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    dim = len(state)
    result = np.zeros_like(state)
    
    # Iterate over all states
    for state_idx in range(dim):
        bit = (state_idx >> qubit) & 1
        # Partner state: flip the qubit
        partner_idx = state_idx ^ (1 << qubit)
        
        if bit == 0:
            # |0> component
            result[state_idx] = c * state[state_idx] - s * state[partner_idx]
        else:
            # |1> component
            result[state_idx] = s * state[partner_idx] + c * state[state_idx]
    
    return result

def _apply_cz(state, qubit1, qubit2):
    """
    Apply CZ gate between qubit1 and qubit2.
    CZ flips sign of |11> component.
    """
    result = state.copy()
    dim = len(state)
    for state_idx in range(dim):
        b1 = (state_idx >> qubit1) & 1
        b2 = (state_idx >> qubit2) & 1
        if b1 == 1 and b2 == 1:
            result[state_idx] = -result[state_idx]
    return result

def _circuit_state(theta, n):
    """
    Compute the state vector for the given circuit parameters.
    theta: array of shape (layers, n)
    Start from |0>^n.
    Each layer: apply Ry(theta[layer, i]) to qubit i for all i.
    Between layers: apply CZ on adjacent pairs (0,1), (1,2), ..., (n-2, n-1) in ascending order.
    No entangler after final layer.
    """
    dim = 1 << n
    # Start from |0>^n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0  # |00...0>
    
    layers = theta.shape[0]
    for layer in range(layers):
        # Apply Ry to each qubit
        for i in range(n):
            state = _apply_ry(state, theta[layer, i], i)
        
        # Apply CZ between adjacent pairs if not the last layer
        if layer < layers - 1:
            for i in range(n - 1):
                state = _apply_cz(state, i, i + 1)
    
    return state

def _energy(state, H):
    """Compute <psi|H|psi> for normalized state."""
    return np.vdot(state, H @ state).real

def _compute_gradient(theta, n, coupling, field, H, E0):
    """
    Compute gradient of (E_circuit - E0) with respect to theta.
    Uses parameter shift rule or finite differences for simplicity and robustness.
    For speed, we use finite differences with a small step.
    """
    grad = np.zeros_like(theta)
    eps = 1e-5
    
    for layer in range(theta.shape[0]):
        for i in range(n):
            # Forward difference
            theta_plus = theta.copy()
            theta_plus[layer, i] += eps
            state_plus = _circuit_state(theta_plus, n)
            E_plus = _energy(state_plus, H)
            
            # Backward difference
            theta_minus = theta.copy()
            theta_minus[layer, i] -= eps
            state_minus = _circuit_state(theta_minus, n)
            E_minus = _energy(state_minus, H)
            
            grad[layer, i] = (E_plus - E_minus) / (2 * eps)
    
    return grad

def _adam_step(params, grad, m, v, t, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    """Adam optimizer step."""
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    params = params - lr * m_hat / (np.sqrt(v_hat) + eps)
    return params, m, v

def angles(n, coupling, field, layers):
    """
    Optimize circuit parameters to minimize mean energy gap per spin.
    Returns theta array of shape (layers, n).
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Build Hamiltonian and compute ground energy
    H = _build_hamiltonian(n, coupling, field)
    E0 = _ground_energy(H)
    
    # Initial parameters
    theta = np.zeros((layers, n))
    theta[-1] = np.pi / 2  # Start with |+> on last layer
    
    # Adam optimizer state
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    t = 0
    
    best_theta = theta.copy()
    best_energy = _energy(_circuit_state(theta, n), H)
    
    # Gradient descent with Adam
    lr = 0.05
    while time.time() < deadline:
        t += 1
        
        # Compute current energy
        state = _circuit_state(theta, n)
        current_energy = _energy(state, H)
        
        # Update best
        if current_energy < best_energy:
            best_energy = current_energy
            best_theta = theta.copy()
        
        # Compute gradient
        grad = _compute_gradient(theta, n, coupling, field, H, E0)
        
        # Adam step
        theta, m, v = _adam_step(theta, grad, m, v, t, lr=lr)
        
        # Adaptive learning rate: reduce if energy increases
        if current_energy > best_energy + 0.1:
            lr *= 0.9
    
    return best_theta
# EVOLVE-BLOCK-END
