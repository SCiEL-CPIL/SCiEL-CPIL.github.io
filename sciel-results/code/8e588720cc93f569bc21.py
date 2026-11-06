"""Tensor-network-inspired initialization for shallow Ising circuit."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm


def _exact_ground_state(n, coupling, field):
    """Compute the exact ground state of the open transverse-field Ising chain.
    
    H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    Qubit i is bit i (little-endian).
    """
    # Build Pauli matrices
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms
    for i in range(n - 1):
        term = I.copy()
        for j in range(n):
            if j == i or j == i + 1:
                term = np.kron(term, Z)
            else:
                term = np.kron(term, I)
        H += -coupling * term
    
    # X terms
    for i in range(n):
        term = I.copy()
        for j in range(n):
            if j == i:
                term = np.kron(term, X)
            else:
                term = np.kron(term, I)
        H += -field * term
    
    # Diagonalize
    eigvals, eigvecs = np.linalg.eigh(H)
    ground_idx = np.argmin(eigvals)
    ground_state = eigvecs[:, ground_idx]
    ground_energy = eigvals[ground_idx]
    
    return ground_state, ground_energy


def _simulate_circuit(angles, n, layers):
    """Simulate the Ry/CZ circuit starting from |0>^n.
    
    angles: (layers, n) array of Ry rotation angles.
    Each layer: apply Ry(theta_i) to qubit i for all i.
    Between layers (after layer k, before layer k+1): apply CZ on adjacent pairs in ascending order.
    No entangler after final layer.
    """
    I = np.eye(2, dtype=complex)
    Ry = lambda theta: expm(-0.5j * theta * np.array([[0, 1], [1, 0]], dtype=complex))
    CZ = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, -1]], dtype=complex)
    
    dim = 2 ** n
    # Start from |0>^n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    
    for layer in range(layers):
        # Apply Ry to each qubit
        for i in range(n):
            U = I.copy()
            for j in range(n):
                if j == i:
                    U = np.kron(U, Ry(angles[layer, i]))
                else:
                    U = np.kron(U, I)
            state = U @ state
        
        # Apply CZ between layers (not after last layer)
        if layer < layers - 1:
            for i in range(n - 1):
                U = I.copy()
                for j in range(n):
                    if j == i or j == i + 1:
                        U = np.kron(U, CZ)
                    else:
                        U = np.kron(U, I)
                state = U @ state
    
    return state


def _energy(state, n, coupling, field):
    """Compute expectation value of H for a given state."""
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    dim = 2 ** n
    energy = 0.0
    
    # ZZ terms
    for i in range(n - 1):
        term = I.copy()
        for j in range(n):
            if j == i or j == i + 1:
                term = np.kron(term, Z)
            else:
                term = np.kron(term, I)
        energy += -coupling * np.vdot(state, term @ state).real
    
    # X terms
    for i in range(n):
        term = I.copy()
        for j in range(n):
            if j == i:
                term = np.kron(term, X)
            else:
                term = np.kron(term, I)
        energy += -field * np.vdot(state, term @ state).real
    
    return energy


def _fit_angles_via_mps(n, coupling, field, layers):
    """Fit circuit angles by comparing circuit state to exact ground state.
    
    Uses a tensor-network-inspired approach: initialize angles based on
    the structure of the ground state, then refine with local optimization.
    """
    # Get exact ground state
    ground_state, ground_energy = _exact_ground_state(n, coupling, field)
    
    # Strategy: The circuit applies Ry rotations and CZ gates.
    # We can think of this as a tensor network. For small n, we can directly
    # optimize the angles to maximize overlap with the ground state.
    
    # Initialize angles using a heuristic based on the field strength
    # For transverse field, the ground state has components in both Z and X bases.
    # A reasonable initial guess is angles that rotate towards the ground state direction.
    
    # Use gradient-free optimization (Nelder-Mead) to minimize energy
    # or maximize overlap with ground state
    
    def objective(angles_flat):
        angles_reshaped = angles_flat.reshape(layers, n)
        # Clamp to valid range
        angles_reshaped = np.clip(angles_reshaped, -np.pi, np.pi)
        state = _simulate_circuit(angles_reshaped, n, layers)
        # Minimize energy (equivalent to maximizing overlap for ground state)
        return _energy(state, n, coupling, field)
    
    # Initial guess: based on field strength
    # For strong field, angles closer to pi/2 (X basis)
    # For weak field, angles closer to 0 (Z basis)
    # Use a smooth interpolation
    theta0 = np.zeros((layers, n))
    # The optimal single-qubit rotation angle depends on the local field
    # Approximate: arctan(field/coupling) gives the tilt angle
    if coupling != 0:
        base_angle = np.arctan2(field, coupling)
    else:
        base_angle = np.pi / 2
    
    # Distribute the rotation across layers
    for l in range(layers):
        theta0[l, :] = base_angle / layers
    
    # Use Nelder-Mead optimization
    from scipy.optimize import minimize
    
    x0 = theta0.flatten()
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
    
    best_angles = result.x.reshape(layers, n)
    best_angles = np.clip(best_angles, -np.pi, np.pi)
    
    # Try multiple initializations to avoid local minima
    best_energy = objective(best_angles.flatten())
    
    # Additional initializations
    inits = [
        np.zeros((layers, n)),
        np.full((layers, n), np.pi / 2),
        np.full((layers, n), base_angle / layers),
        np.random.uniform(-np.pi/4, np.pi/4, (layers, n)),
    ]
    
    for init in inits:
        x0 = init.flatten()
        res = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        e = objective(res.x.flatten())
        if e < best_energy:
            best_energy = e
            best_angles = res.x.reshape(layers, n)
            best_angles = np.clip(best_angles, -np.pi, np.pi)
    
    return best_angles


def angles(n, coupling, field, layers):
    """Return circuit parameters (layers, n) in radians."""
    # Use the tensor-network-inspired fitting approach
    try:
        best_angles = _fit_angles_via_mps(n, coupling, field, layers)
    except Exception:
        # Fallback: simple heuristic
        if coupling != 0:
            base_angle = np.arctan2(field, coupling)
        else:
            base_angle = np.pi / 2
        best_angles = np.full((layers, n), base_angle / layers)
    
    return best_angles
# EVOLVE-BLOCK-END
