"""Optimized quantum circuit for Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import expm

def _ry_matrix(theta):
    """Return the 2x2 Ry rotation matrix."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)

def _cz_matrix():
    """Return the 4x4 CZ matrix."""
    return np.array([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,1,0],
                     [0,0,0,-1]], dtype=np.complex128)

def _apply_ry_to_qubit(state, qubit, theta, n):
    """Apply Ry(theta) to qubit in an n-qubit state vector."""
    if theta == 0:
        return state
    ry = _ry_matrix(theta)
    # Build the full operator: I ⊗ ... ⊗ Ry ⊗ ... ⊗ I
    # Qubit qubit is bit qubit (little-endian: qubit 0 is least significant)
    # In numpy kron, we need to be careful about ordering
    # State is in basis |q_{n-1} ... q_1 q_0> where q_0 is least significant
    # So qubit 0 is the rightmost (last) in the kron chain
    # We'll build the operator from most significant to least significant
    ops = []
    for i in range(n - 1, -1, -1):
        if i == qubit:
            ops.append(ry)
        else:
            ops.append(np.eye(2, dtype=np.complex128))
    full_op = ops[0]
    for op in ops[1:]:
        full_op = np.kron(full_op, op)
    return full_op @ state

def _apply_cz_to_pair(state, q1, q2, n):
    """Apply CZ between qubits q1 and q2 in an n-qubit state."""
    # Build CZ operator for the pair
    # CZ on qubits q1, q2: flips sign when both are |1>
    dim = 2 ** n
    op = np.eye(dim, dtype=np.complex128)
    for i in range(dim):
        bits = np.array([int((i >> j) & 1) for j in range(n)])
        if bits[q1] == 1 and bits[q2] == 1:
            op[i, i] = -1
    return op @ state

def _build_circuit_state(theta, n, layers):
    """Build the state vector from the circuit parameters.
    
    theta: array of shape (layers, n)
    n: number of qubits
    layers: number of layers (3)
    """
    state = np.zeros(2 ** n, dtype=np.complex128)
    state[0] = 1.0  # |0>^n
    
    for layer in range(layers):
        # Apply Ry to all qubits
        for q in range(n):
            state = _apply_ry_to_qubit(state, q, theta[layer, q], n)
        
        # Apply CZ between adjacent pairs (except after final layer)
        if layer < layers - 1:
            for q in range(n - 1):
                state = _apply_cz_to_pair(state, q, q + 1, n)
    
    return state

def _compute_energy(state, n, coupling, field):
    """Compute <H> for the given state.
    H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    """
    dim = 2 ** n
    energy = 0.0
    
    # ZZ terms
    for i in range(n - 1):
        for j in range(dim):
            bits = np.array([int((j >> k) & 1) for k in range(n)])
            # Z eigenvalue: +1 if bit is 0, -1 if bit is 1
            z_i = 1.0 if bits[i] == 0 else -1.0
            z_j = 1.0 if bits[i + 1] == 0 else -1.0
            zz_val = z_i * z_j
            energy += -coupling * zz_val * np.abs(state[j]) ** 2
    
    # X terms
    for i in range(n):
        for j in range(dim):
            bits = np.array([int((j >> k) & 1) for k in range(n)])
            # X flips bit i
            flipped_bits = bits.copy()
            flipped_bits[i] = 1 - flipped_bits[i]
            flipped_idx = 0
            for k in range(n):
                flipped_idx |= (flipped_bits[k] << k)
            # <state|X_i|state> = sum_j conj(state_j) * X_i * state_j
            # X_i |j> = |j with bit i flipped>
            # So <state|X_i|state> = sum_j conj(state_j) * state_{j^flip_i}
            # This equals 2 * Re(sum over j where bit i = 0 of conj(state_j) * state_{j with bit i = 1})
            # More directly: for each j, X_i|j> = |j^flip_i>, so
            # <state|X_i|state> = sum_j conj(state_j) * <j|X_i|j> = sum_j conj(state_j) * delta_{j, j^flip_i}
            # Wait, that's not right. Let me think again.
            # X_i |j> = |j with bit i flipped>. Let j' = j with bit i flipped.
            # Then <state|X_i|state> = sum_j <state|j> <j|X_i|state> = sum_j conj(state_j) * <j|X_i|state>
            # X_i|state> = sum_k state_k X_i|k> = sum_k state_k |k^flip_i>
            # So <j|X_i|state> = sum_k state_k <j|k^flip_i> = sum_k state_k delta_{j, k^flip_i} = state_{j^flip_i}
            # Therefore <state|X_i|state> = sum_j conj(state_j) * state_{j^flip_i}
            # Since j^flip_i is a permutation, this equals sum_j conj(state_j) * state_{j^flip_i}
            # Note that if we pair j with j^flip_i, we get conj(state_j)*state_{j^flip_i} + conj(state_{j^flip_i})*state_j = 2*Re(conj(state_j)*state_{j^flip_i})
            # But we can just compute it directly for each j
            pass
    
    # Let me redo the X term computation more carefully
    energy_x = 0.0
    for i in range(n):
        for j in range(dim):
            bits = np.array([int((j >> k) & 1) for k in range(n)])
            flipped_bits = bits.copy()
            flipped_bits[i] = 1 - flipped_bits[i]
            flipped_idx = 0
            for k in range(n):
                flipped_idx |= (flipped_bits[k] << k)
            energy_x += np.conj(state[j]) * state[flipped_idx]
        energy += -field * np.real(energy_x)
    
    return energy

def _objective(params, n, coupling, field, layers, ground_energy):
    """Objective function: mean energy gap per spin."""
    theta = params.reshape(layers, n)
    state = _build_circuit_state(theta, n, layers)
    energy = _compute_energy(state, n, coupling, field)
    gap = (energy - ground_energy) / n
    return gap

def _compute_ground_energy(n, coupling, field):
    """Compute exact ground energy by diagonalization."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    # ZZ terms
    for i in range(n - 1):
        for j in range(dim):
            bits = np.array([int((j >> k) & 1) for k in range(n)])
            z_i = 1.0 if bits[i] == 0 else -1.0
            z_j = 1.0 if bits[i + 1] == 0 else -1.0
            H[j, j] += -coupling * z_i * z_j
    
    # X terms
    for i in range(n):
        for j in range(dim):
            bits = np.array([int((j >> k) & 1) for k in range(n)])
            flipped_bits = bits.copy()
            flipped_bits[i] = 1 - flipped_bits[i]
            flipped_idx = 0
            for k in range(n):
                flipped_idx |= (flipped_bits[k] << k)
            H[flipped_idx, j] += -field
    
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0]

def angles(n, coupling, field, layers):
    """Optimize circuit parameters to approximate ground state."""
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # Compute exact ground energy
    ground_energy = _compute_ground_energy(n, coupling, field)
    
    # Initial guess: small angles
    best_theta = np.zeros((layers, n))
    best_score = np.inf
    
    # Try multiple initializations
    np.random.seed(42)
    
    n_params = layers * n
    max_iters = 500
    
    for init_try in range(5):
        if time.time() > deadline:
            break
        
        if init_try == 0:
            x0 = np.zeros(n_params)
        elif init_try == 1:
            x0 = np.random.uniform(-0.5, 0.5, n_params)
        elif init_try == 2:
            x0 = np.random.uniform(-1.0, 1.0, n_params)
        elif init_try == 3:
            # Try to start near |+> state
            x0 = np.zeros(n_params)
            x0[-n:] = np.pi / 2  # Last layer Ry(pi/2) gives |+>
        else:
            x0 = np.random.uniform(-1.5, 1.5, n_params)
        
        try:
            result = minimize(
                _objective,
                x0,
                args=(n, coupling, field, layers, ground_energy),
                method='L-BFGS-B',
                options={'maxiter': max_iters, 'maxfun': 2000}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_theta = result.x.reshape(layers, n)
                
        except Exception:
            continue
    
    # If optimization didn't improve, fall back to a reasonable heuristic
    if best_score >= np.inf or best_score > 0.5:
        # Try a simple heuristic: all angles = pi/4
        theta_heuristic = np.full((layers, n), np.pi / 4)
        state = _build_circuit_state(theta_heuristic, n, layers)
        energy = _compute_energy(state, n, coupling, field)
        score = (energy - ground_energy) / n
        if score < best_score:
            best_score = score
            best_theta = theta_heuristic
    
    # Ensure finite values
    best_theta = np.where(np.isfinite(best_theta), best_theta, 0.0)
    
    return best_theta
# EVOLVE-BLOCK-END
