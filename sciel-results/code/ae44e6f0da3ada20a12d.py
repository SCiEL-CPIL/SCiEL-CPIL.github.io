"""Optimized 3-layer Ry/CZ circuit for open transverse-field Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize

# Precompute Pauli matrices
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
I2 = np.eye(2, dtype=np.complex128)

# CZ gate (2x2x2x2 tensor or 4x4 matrix)
CZ_4x4 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, -1]], dtype=np.complex128)

# Cache for precomputed matrices
_cache = {}

def _get_cache_key(n, coupling, field):
    return (n, coupling, field)

def _build_cz_layer_unitary(n):
    """Build the full unitary for the CZ layer: CZ_{n-2,n-1} ... CZ_{0,1}"""
    # Identity on n qubits
    U = np.eye(2**n, dtype=np.complex128)
    
    # Apply CZ gates in ascending order of pairs
    # The unitary for a sequence of gates G1, G2, ... is G_k ... G_2 G_1
    # So we start with I, then multiply by CZ_0,1, then CZ_1,2, etc.
    # U_total = CZ_last ... CZ_first
    # Wait, state evolution: |psi> -> CZ_1 |psi> -> CZ_2 CZ_1 |psi>
    # So U = CZ_last ... CZ_first
    
    for i in range(n - 1):
        # Build CZ_{i, i+1} as 2^n x 2^n matrix
        # It acts on qubits i and i+1
        # We can build it by Kronecker products
        ops = [I2] * n
        # CZ on i, i+1
        # In basis |... b_i b_{i+1} ...>, CZ flips sign if b_i=1 and b_{i+1}=1
        # Matrix for CZ on 2 qubits is CZ_4x4
        # We need to embed CZ_4x4 into the n-qubit space
        
        # Efficient way:
        # Permute qubits i, i+1 to the end? Or just build the full matrix.
        # Since n is small, building full matrix is fine.
        
        # Create a 2x2x...x2 tensor for CZ_{i, i+1}
        # Actually, simpler:
        # U_cz = I ... I \otimes CZ_4x4 \otimes I ... I (with appropriate ordering)
        
        # Qubit ordering: qubit 0 is least significant bit (rightmost in tensor product if using standard kron)
        # Standard np.kron(A, B) corresponds to A acting on higher index?
        # Let's stick to the convention:
        # |q_{n-1} ... q_0>
        # np.kron(A, B) -> A acts on left (higher index), B on right (lower index).
        
        # So if we want CZ on i, i+1:
        # We need to place CZ_4x4 at the position corresponding to bits i and i+1.
        
        # Let's build the list of 2x2 or 4x4 matrices from right to left (qubit 0 to n-1)
        mats = []
        for q in range(n):
            if q == i:
                # This is the first qubit of the pair (lower index)
                # In kron order, rightmost is qubit 0.
                # So we need to handle the pair together.
                pass
            
        # Alternative: Build the matrix by iterating over all basis states?
        # 2^n is small.
        U_cz = np.eye(2**n, dtype=np.complex128)
        for state in range(2**n):
            b_i = (state >> i) & 1
            b_ip1 = (state >> (i+1)) & 1
            if b_i == 1 and b_ip1 == 1:
                U_cz[state, state] = -1.0
        
        U = U_cz @ U
        
    return U

def _build_hamiltonian(n, coupling, field):
    """Build the full Hamiltonian matrix."""
    H = np.zeros((2**n, 2**n), dtype=np.complex128)
    
    # ZZ terms
    for i in range(n - 1):
        for state in range(2**n):
            b_i = (state >> i) & 1
            b_ip1 = (state >> (i+1)) & 1
            z_i = 1 if b_i == 0 else -1
            z_ip1 = 1 if b_ip1 == 0 else -1
            H[state, state] -= coupling * z_i * z_ip1
            
    # X terms
    for i in range(n):
        for state in range(2**n):
            # X flips bit i
            flipped = state ^ (1 << i)
            H[flipped, state] -= field
            
    return H

def _get_precomputed(n, coupling, field):
    key = _get_cache_key(n, coupling, field)
    if key not in _cache:
        H = _build_hamiltonian(n, coupling, field)
        ge = float(np.real(np.linalg.eigvalsh(H)[0]))
        U_cz = _build_cz_layer_unitary(n)
        phi_0 = np.zeros(2**n, dtype=np.complex128)
        phi_0[0] = 1.0
        _cache[key] = {
            'H': H,
            'ge': ge,
            'U_cz': U_cz,
            'phi_0': phi_0
        }
    return _cache[key]

def _ry_matrix(theta):
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)

def _build_ry_unitary(theta_vec, n):
    """Build unitary for a layer of Ry gates given theta_vec of length n."""
    U = np.array([[1.0]], dtype=np.complex128)
    # Qubit 0 is least significant.
    # np.kron(A, B) -> A is left (higher), B is right (lower).
    # So we iterate from n-1 down to 0.
    for q in range(n - 1, -1, -1):
        Ry = _ry_matrix(theta_vec[q])
        U = np.kron(U, Ry)
    return U

def _compute_energy(theta_flat, n, coupling, field, precomputed):
    H = precomputed['H']
    U_cz = precomputed['U_cz']
    phi_0 = precomputed['phi_0']
    
    theta = theta_flat.reshape(3, n)
    
    U0 = _build_ry_unitary(theta[0], n)
    U1 = _build_ry_unitary(theta[1], n)
    U2 = _build_ry_unitary(theta[2], n)
    
    # U_total = U2 U_cz U1 U_cz U0
    # To save time, we can apply to phi_0 step by step
    psi = U0 @ phi_0
    psi = U_cz @ psi
    psi = U1 @ psi
    psi = U_cz @ psi
    psi = U2 @ psi
    
    # Energy
    Hp = H @ psi
    energy = np.vdot(psi, Hp)
    return float(np.real(energy))

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    precomputed = _get_precomputed(n, coupling, field)
    ge = precomputed['ge']
    
    def objective(theta_flat):
        e = _compute_energy(theta_flat, n, coupling, field, precomputed)
        return e - ge
        
    # Initial guess: seed
    theta_init = np.zeros(layers * n)
    theta_init[-n:] = np.pi / 2
    
    best_theta = theta_init.copy()
    best_val = objective(theta_init)
    
    bounds = [(-np.pi, np.pi)] * (layers * n)
    
    # Optimizer settings
    # Use L-BFGS-B
    options = {
        'maxiter': 50,
        'maxfun': 100,
        'gtol': 1e-5
    }
    
    # Try main optimization
    try:
        res = minimize(
            objective,
            theta_init,
            method='L-BFGS-B',
            bounds=bounds,
            options=options
        )
        if res.fun < best_val:
            best_theta = res.x
            best_val = res.fun
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        return best_theta.reshape(layers, n)
        
    # Random restarts
    rng = np.random.RandomState(42)
    max_restarts = 20
    for i in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        # Random initialization
        theta_rand = rng.uniform(-np.pi, np.pi, size=(layers, n)).flatten()
        
        try:
            res = minimize(
                objective,
                theta_rand,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 30, 'maxfun': 50, 'gtol': 1e-5}
            )
            if res.fun < best_val:
                best_theta = res.x
                best_val = res.fun
        except Exception:
            pass
            
    # Final check
    theta = np.where(np.isfinite(best_theta), best_theta, 0.0)
    return theta.reshape(layers, n)

# EVOLVE-BLOCK-END
