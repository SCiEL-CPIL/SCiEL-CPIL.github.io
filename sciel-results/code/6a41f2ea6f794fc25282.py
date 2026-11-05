import numpy as np
import scipy.optimize as opt
import time
from functools import lru_cache

# Global cache for angles
_angle_cache = {}

def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix for the open transverse-field Ising chain."""
    dim = 2**n
    H = np.zeros((dim, dim), dtype=complex)
    
    # Z_i Z_{i+1} terms
    # Z = [[1, 0], [0, -1]]
    Z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    
    # Kronecker products for operators
    def kron_ops(ops):
        res = np.array([[1]])
        for op in ops:
            res = np.kron(res, op)
        return res

    # Sum Z_i Z_{i+1}
    ZZ_sum = np.zeros((dim, dim), dtype=complex)
    for i in range(n - 1):
        ops = [I] * n
        ops[i] = Z
        ops[i+1] = Z
        ZZ_sum += kron_ops(ops)
        
    # Sum X_i
    X_sum = np.zeros((dim, dim), dtype=complex)
    for i in range(n):
        ops = [I] * n
        ops[i] = X
        X_sum += kron_ops(ops)
        
    H = -coupling * ZZ_sum - field * X_sum
    return H

def _get_ground_energy(H):
    """Compute the exact ground state energy."""
    eigvals, eigvecs = np.linalg.eigh(H)
    return eigvals[0]

def _apply_ry(state, qubit, theta):
    """Apply Ry(theta) to a specific qubit in the state vector."""
    dim = len(state)
    n = int(np.log2(dim))
    # Create Ry matrix
    ry = np.array([
        [np.cos(theta/2), -np.sin(theta/2)],
        [np.sin(theta/2), np.cos(theta/2)]
    ], dtype=complex)
    
    # Apply via kron
    ops = [np.eye(2)] * n
    ops[qubit] = ry
    gate = np.kron(np.kron(ops[0], ops[1]), np.kron(ops[2], ops[3])) if n==4 else None
    # Generic kron application
    gate = np.array([[1]])
    for op in ops:
        gate = np.kron(gate, op)
        
    return gate @ state

def _apply_cz(state, q1, q2):
    """Apply CZ gate between q1 and q2."""
    dim = len(state)
    n = int(np.log2(dim))
    cz = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex)
    
    ops = [np.eye(2)] * n
    # This simple kron replacement doesn't work directly for non-adjacent or specific qubits in a flat list without reordering logic.
    # Easier: iterate over basis states.
    # Or use the fact that CZ is diagonal.
    # CZ |a, b> = (-1)^(a*b) |a, b>
    # We can construct the diagonal vector.
    
    # Let's just build the full matrix for small n. It's efficient enough for n<=6.
    gate = np.array([[1]])
    for i in range(n):
        if i == q1 or i == q2:
            # We need to place the 2x2 part of CZ correctly
            # This is tricky with simple kron if q1, q2 are not 0,1.
            pass
            
    # Alternative efficient method for CZ:
    # The CZ gate is diagonal. 
    # For each basis state |b>, the phase is (-1) if bit q1 is 1 AND bit q2 is 1.
    diag = np.ones(dim, dtype=complex)
    mask = 0
    # Set bits for q1 and q2
    # Bit i corresponds to 2^i in the binary representation if we assume little-endian or big-endian?
    # Standard quantum info often uses |q_{n-1} ... q_0>. 
    # Let's stick to the matrix multiplication for safety and simplicity, n is small.
    
    # Construct full gate matrix
    gate = np.eye(dim, dtype=complex)
    # Iterate through basis states to set diagonal entries
    for basis in range(dim):
        b_q1 = (basis >> q1) & 1
        b_q2 = (basis >> q2) & 1
        if b_q1 and b_q2:
            gate[basis, basis] = -1
            
    return gate @ state

def _circuit_state(theta, n):
    """
    theta: (layers, n) array
    n: number of qubits
    Returns state vector.
    """
    dim = 2**n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0 # |0...0>
    
    for l in range(len(theta)):
        # Apply Ry to all qubits
        for i in range(n):
            state = _apply_ry(state, i, theta[l, i])
        
        # Apply CZ if not the last layer
        if l < len(theta) - 1:
            # CZ on every adjacent pair in ascending order: (0,1), (1,2), ...
            for i in range(n - 1):
                state = _apply_cz(state, i, i+1)
                
    return state

def _energy(theta, n, coupling, field, H, E0):
    """
    Calculate mean energy gap per spin.
    """
    state = _circuit_state(theta, n)
    # Energy = <state| H |state>
    e = np.vdot(state, H @ state).real
    gap = e - E0
    return gap / n

def angles(n, coupling, field, layers):
    # Check cache
    key = (n, coupling, field, layers)
    if key in _angle_cache:
        return _angle_cache[key]
        
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    H = _build_hamiltonian(n, coupling, field)
    E0 = _get_ground_energy(H)
    
    # Initial guess: Seed program logic (Product |+>)
    # theta[-1] = pi/2, others 0
    theta_init = np.zeros((layers, n))
    theta_init[-1] = np.pi / 2
    
    # Objective function
    def objective(theta_flat):
        theta_reshaped = theta_flat.reshape(layers, n)
        return _energy(theta_reshaped, n, coupling, field, H, E0)
        
    # Optimization
    best_theta = theta_init
    best_val = objective(best_theta.flatten())
    
    # Try a few random restarts + local optimization
    # n=4,6 -> 12, 18 parameters. 
    # L-BFGS-B is good.
    
    np.random.seed(42) # Reproducibility
    
    # Restart 0: Seed
    res = opt.minimize(objective, best_theta.flatten(), method='L-BFGS-B', 
                        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
    if res.success and res.fun < best_val:
        best_val = res.fun
        best_theta = res.x.reshape(layers, n)
        
    # Check time
    if time.time() > deadline:
        _angle_cache[key] = best_theta
        return best_theta
        
    # Random restarts
    n_restarts = 5
    for i in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        theta_rand = np.random.uniform(-np.pi, np.pi, (layers, n))
        # Sometimes near 0 or pi/2 is better, mix them
        if i % 2 == 0:
             theta_rand = np.random.uniform(-np.pi/4, np.pi/4, (layers, n))
             
        res = opt.minimize(objective, theta_rand.flatten(), method='L-BFGS-B', 
                            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_theta = res.x.reshape(layers, n)
            
    # If we have time, try Nelder-Mead on the best found so far for polishing?
    # Or Differential Evolution for global search?
    # Given small n, DE might be overkill but robust.
    # Let's try one DE run if time permits.
    
    if time.time() < deadline - 20.0:
        # Differential Evolution
        bounds = [(-np.pi, np.pi)] * (layers * n)
        de_res = opt.differential_evolution(objective, bounds, maxiter=500, tol=1e-12, 
                                            popsize=10, seed=42, polish=True)
        if de_res.fun < best_val:
            best_val = de_res.fun
            best_theta = de_res.x.reshape(layers, n)

    _angle_cache[key] = best_theta
    return best_theta
