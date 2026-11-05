"""Optimized program: Unitary composition strategy for fast energy evaluation."""
import numpy as np
import time
from scipy.optimize import minimize

# Precompute exact ground energies for the known cases
# H = -coupling * sum Z_i Z_{i+1} - field * sum X_i, open chain

def _build_hamiltonian(n, coupling, field):
    """Build the sparse Hamiltonian matrix."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    
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
            H[state, flipped] += -field
    
    return H


def _get_ground_energy(n, coupling, field):
    """Compute the exact ground state energy."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return float(eigvals[0])


# Precompute ground energies for known cases
_GROUND_ENERGY_CACHE = {}

def _ground_energy_cached(n, coupling, field):
    key = (n, coupling, field)
    if key not in _GROUND_ENERGY_CACHE:
        _GROUND_ENERGY_CACHE[key] = _get_ground_energy(n, coupling, field)
    return _GROUND_ENERGY_CACHE[key]

_U_CZ_CACHE = {}

def _get_u_cz(n):
    """Get the CZ layer unitary for chain of length n."""
    if n in _U_CZ_CACHE:
        return _U_CZ_CACHE[n]
    
    dim = 2 ** n
    # Identity of size dim
    U_cz = np.eye(dim, dtype=complex)
    
    # Build CZ for each adjacent pair and multiply
    # CZ_{i, i+1}
    for i in range(n - 1):
        # Create CZ gate on qubits i and i+1
        # We can build this by iterating over basis states or using kron products
        # Efficient way: 
        # CZ = |0><0|_i \otimes I_{i+1} + |1><1|_i \otimes Z_{i+1}
        # Let's build the full matrix for this pair interaction
        cz_pair = np.zeros((dim, dim), dtype=complex)
        
        # Iterate over all basis states
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            
            if bit_i == 0:
                # Identity on i+1
                cz_pair[state, state] += 1.0
            else:
                # Z on i+1: flip sign if bit_ip1 is 1
                if bit_ip1 == 0:
                    cz_pair[state, state] += 1.0
                else:
                    cz_pair[state, state] += -1.0
        
        U_cz = cz_pair @ U_cz
        
    _U_CZ_CACHE[n] = U_cz
    return U_cz

def _build_ry_unitary(thetas, n):
    """Build the unitary for a layer of Ry gates."""
    # thetas is array of shape (n,)
    # Ry(theta) = [[c, -s], [s, c]]
    # U_ry = Ry(t0) \otimes Ry(t1) \otimes ... \otimes Ry(t_{n-1})
    
    # Note: Qubit i is bit i. In standard kron, leftmost is most significant.
    # If we use np.kron(A, B), A acts on the "left" (higher index in tensor product if we think of indices).
    # Standard convention: |q_{n-1} ... q_0>.
    # np.kron(Ry(t_{n-1}), ..., Ry(t_0))
    # Let's verify with n=2.
    # State |00> index 0.
    # Ry(t0) on qubit 0, Ry(t1) on qubit 1.
    # U = Ry(t1) \otimes Ry(t0).
    # Vector |00> -> Ry(t1)|0> \otimes Ry(t0)|0>.
    
    matrices = []
    # We need to order them correctly.
    # If we do np.kron(M1, M2), the resulting matrix acts on V1 \otimes V2.
    # If qubit i corresponds to the i-th tensor factor from the right (least significant),
    # then qubit 0 is the rightmost factor.
    # So U = Ry(t_{n-1}) \otimes ... \otimes Ry(t_0).
    
    for i in range(n - 1, -1, -1):
        theta = thetas[i]
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        ry = np.array([[c, -s], [s, c]], dtype=complex)
        matrices.append(ry)
        
    if n == 1:
        return matrices[0]
        
    U = matrices[0]
    for m in matrices[1:]:
        U = np.kron(U, m)
        
    return U

def _circuit_unitary(params, n, layers, u_cz):
    """Compute the total circuit unitary."""
    # params shape (layers, n)
    # U = U_ry(L-1) @ U_cz @ U_ry(L-2) @ ... @ U_cz @ U_ry(0)
    
    U_total = np.eye(2**n, dtype=complex)
    
    for layer in range(layers - 1, -1, -1):
        U_ry = _build_ry_unitary(params[layer], n)
        U_total = U_ry @ U_total
        
        if layer > 0:
            U_total = u_cz @ U_total
            
    return U_total

def _energy_from_unitary(U, H):
    """Compute energy expectation value given the unitary."""
    # |psi> = U |0...0> = first column of U
    psi = U[:, 0]
    # E = <psi| H |psi>
    H_psi = H @ psi
    energy = np.vdot(psi, H_psi).real
    return energy

def angles(n, coupling, field, layers):
    """Optimize circuit parameters to minimize mean energy gap per spin."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Get exact ground energy
    e0 = _ground_energy_cached(n, coupling, field)
    
    # Build Hamiltonian
    H = _build_hamiltonian(n, coupling, field)
    
    # Get CZ unitary
    u_cz = _get_u_cz(n)
    
    # Objective: mean energy gap per spin = (E_circuit - E_0) / n
    def objective(params_flat):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return np.inf
        params = params_flat.reshape(layers, n)
        U = _circuit_unitary(params, n, layers, u_cz)
        e = _energy_from_unitary(U, H)
        return (e - e0) / n
    
    # Bounds: theta in [-pi, pi]
    bounds = [(-np.pi, np.pi)] * (layers * n)
    
    # Initial guess: try several starting points
    best_params = None
    best_energy = np.inf
    
    # Try a few initial points
    init_candidates = [
        np.zeros(layers * n),
        np.pi / 2 * np.ones(layers * n),
        np.random.randn(layers * n) * 0.5,
        np.random.randn(layers * n) * 0.5,
        np.random.randn(layers * n) * 0.5,
        np.random.randn(layers * n) * 0.5,
    ]
    
    for init in init_candidates:
        elapsed = time.time() - start_time
        if elapsed > deadline - 10.0:
            break
        
        try:
            result = minimize(
                objective,
                init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < best_energy and np.isfinite(result.fun):
                best_energy = result.fun
                best_params = result.x
        except Exception:
            continue
    
    # If no optimization succeeded, use the |+> state
    if best_params is None:
        params = np.zeros((layers, n))
        params[-1] = np.pi / 2
        return params
    
    return best_params.reshape(layers, n)
