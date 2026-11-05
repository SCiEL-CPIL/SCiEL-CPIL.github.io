import numpy as np
from scipy.optimize import minimize
import time

# Precomputed constants and caches
_hamiltonian_cache = {}
_ground_energy_cache = {}

def _build_hamiltonian(n, coupling, field):
    key = (n, coupling, field)
    if key in _hamiltonian_cache:
        return _hamiltonian_cache[key]
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    
    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        op = np.array([[1]], dtype=complex)
        for q in range(n - 1, -1, -1):
            if q == i or q == i + 1:
                op = np.kron(op, Z)
            else:
                op = np.kron(op, I)
        H -= coupling * op
    
    # X_i terms
    for i in range(n):
        op = np.array([[1]], dtype=complex)
        for q in range(n - 1, -1, -1):
            if q == i:
                op = np.kron(op, X)
            else:
                op = np.kron(op, I)
        H -= field * op
    
    _hamiltonian_cache[key] = H
    return H


def _get_ground_energy(n, coupling, field):
    key = (n, coupling, field)
    if key in _ground_energy_cache:
        return _ground_energy_cache[key]
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    e0 = float(eigvals[0])
    _ground_energy_cache[key] = e0
    return e0


def _apply_ry_vectorized(psi, n, theta_layer):
    """Apply Ry(theta_layer[i]) to qubit i for all i simultaneously.
    theta_layer shape (n,)."""
    dim = 1 << n
    # Reshape into tensor of shape (2,)*n where axis k corresponds to qubit (n-1-k)
    # Actually, let's use a direct approach.
    # For qubit i (bit i), we process blocks.
    for i in range(n):
        t = theta_layer[i]
        c = np.cos(t / 2)
        s = np.sin(t / 2)
        block = 1 << i
        # Vectorized update
        # We can reshape psi into shape (-1, 2, block) where the middle axis is bit i
        # But careful with memory layout.
        # Let's use the standard trick:
        # psi.reshape(-1, 2, block) -> axis 1 is bit i (0 or 1), axis 2 is lower bits
        # Wait, if block = 2^i, then the dimension for bit i is 2, and lower bits is block.
        # The higher bits form the -1 dimension.
        # Shape: (dim // (2*block), 2, block)
        reshape_shape = (dim // (2 * block), 2, block)
        psi_reshaped = psi.reshape(reshape_shape)
        psi0 = psi_reshaped[:, 0, :]
        psi1 = psi_reshaped[:, 1, :]
        
        new_psi0 = c * psi0 - s * psi1
        new_psi1 = s * psi0 + c * psi1
        
        psi_reshaped[:, 0, :] = new_psi0
        psi_reshaped[:, 1, :] = new_psi1
        
    return psi


def _apply_cz_vectorized(psi, n, i):
    """Apply CZ to qubits i and i+1."""
    dim = 1 << n
    # Find indices where bit i and bit i+1 are both 1
    # We can use a mask or just iterate over the relevant blocks.
    # Since n is small (4 or 6), we can precompute indices or use vectorized masking.
    # Let's precompute indices for each (n, i) pair for speed.
    key = (n, i)
    if not hasattr(_apply_cz_vectorized, 'index_cache'):
        _apply_cz_vectorized.index_cache = {}
    
    if key not in _apply_cz_vectorized.index_cache:
        indices = []
        for idx in range(dim):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            if bit_i == 1 and bit_ip1 == 1:
                indices.append(idx)
        _apply_cz_vectorized.index_cache[key] = np.array(indices, dtype=int)
    
    idxs = _apply_cz_vectorized.index_cache[key]
    psi[idxs] *= -1
    return psi


def _circuit_state(theta, n):
    """Simulate the 3-layer Ry/CZ circuit. theta shape (layers, n)."""
    dim = 1 << n
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    
    layers = theta.shape[0]
    
    for layer in range(layers):
        # Apply Ry to all qubits
        psi = _apply_ry_vectorized(psi, n, theta[layer])
        
        # Apply CZ gates if not the last layer
        if layer < layers - 1:
            for i in range(n - 1):
                psi = _apply_cz_vectorized(psi, n, i)
    
    return psi


def _energy_from_state(psi, H):
    """Compute <psi|H|psi>."""
    return float(np.real(np.vdot(psi, H @ psi)))


def _objective(params, n, coupling, field, H, e0):
    """Objective: mean energy gap per spin."""
    theta = params.reshape(3, n)
    psi = _circuit_state(theta, n)
    e = _energy_from_state(psi, H)
    return (e - e0) / n


def _mean_field_init(n, coupling, field):
    """
    Initialize parameters using a mean-field approximation.
    For the transverse-field Ising model, the ground state in the mean-field approximation
    can be approximated by a product state with angles derived from the local fields.
    """
    # Simple heuristic: initialize with angles that align with the effective field
    # For small field, angles near 0; for large field, angles near pi/2
    theta = np.zeros((3, n), dtype=float)
    # Scale angles based on field strength
    scale = np.arctan(field / coupling) if coupling > 0 else np.pi/4
    theta[:, :] = scale * 0.5  # Start with half the scale for all layers
    return theta.flatten()


def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    H = _build_hamiltonian(n, coupling, field)
    e0 = _get_ground_energy(n, coupling, field)
    
    # Initial guess: mean-field inspired
    best_params = _mean_field_init(n, coupling, field)
    best_obj = _objective(best_params, n, coupling, field, H, e0)
    
    # Also try zero initialization
    zero_params = np.zeros(3 * n)
    zero_obj = _objective(zero_params, n, coupling, field, H, e0)
    if zero_obj < best_obj:
        best_obj = zero_obj
        best_params = zero_params.copy()
    
    # Multi-start optimization with L-BFGS-B
    rng = np.random.default_rng(42)
    
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            x0 = best_params.copy()
        else:
            x0 = rng.uniform(-np.pi, np.pi, size=3 * n)
        
        try:
            result = minimize(
                _objective,
                x0,
                args=(n, coupling, field, H, e0),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            obj_val = result.fun
            if np.isfinite(obj_val) and obj_val < best_obj:
                best_obj = obj_val
                best_params = result.x.copy()
        except Exception:
            continue
    
    # Final polish with tighter tolerance if time permits
    if time.time() < deadline - 5:
        try:
            result = minimize(
                _objective,
                best_params,
                args=(n, coupling, field, H, e0),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    theta = best_params.reshape(3, n)
    # Ensure finite
    theta = np.where(np.isfinite(theta), theta, 0.0)
    return theta
