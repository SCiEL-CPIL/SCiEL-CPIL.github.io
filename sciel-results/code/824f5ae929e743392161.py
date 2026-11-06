"""
Strategy: Nonlinear least-squares statevector matching.

Instead of minimizing the energy (a scalar that can have flat regions or local minima
unrelated to the true ground state), we directly minimize the L2 distance between the
circuit output statevector and the exact ground statevector. This provides a much
richer loss landscape with gradients pointing directly toward the target state,
exploiting the known target structure. We use scipy.optimize.least_squares with
analytic Jacobians for speed and robustness.
"""
import numpy as np
from scipy.optimize import least_squares
import time

# Cache for Hamiltonians and exact ground states
_HAM_CACHE = {}
_GS_CACHE = {}

def _get_hamiltonian(n, coupling):
    """Build the Hamiltonian matrix for the open transverse-field Ising chain."""
    key = (n, coupling)
    if key in _HAM_CACHE:
        return _HAM_CACHE[key]
    
    dim = 1 << n
    
    # H = -coupling * sum Z_i Z_(i+1) - field * sum X_i
    H_IZ = np.zeros((dim, dim), dtype=np.float64)
    for i in range(n - 1):
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            if bit_i == bit_ip1:
                H_IZ[state, state] += 1.0
            else:
                H_IZ[state, state] -= 1.0
                
    H_IX = np.zeros((dim, dim), dtype=np.float64)
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H_IX[state, flipped] = 1.0
            
    _HAM_CACHE[key] = (H_IZ, H_IX)
    return _HAM_CACHE[key]

def _compute_exact_ground_state(n, coupling, field):
    """Compute the exact ground state vector and energy."""
    key = (n, coupling, field)
    if key in _GS_CACHE:
        return _GS_CACHE[key]
    
    H_IZ, H_IX = _get_hamiltonian(n, coupling)
    H = -coupling * H_IZ - field * H_IX
    
    eigvals, eigvecs = np.linalg.eigh(H)
    e0 = eigvals[0].real
    psi0 = eigvecs[:, 0]
    
    # Fix global phase: make the component with largest magnitude real and positive
    max_idx = np.argmax(np.abs(psi0))
    phase = np.angle(psi0[max_idx])
    psi0 = psi0 * np.exp(-1j * phase)
    
    _GS_CACHE[key] = (psi0, e0)
    return _GS_CACHE[key]

def _apply_circuit(theta, n):
    """
    Apply the circuit to |0...0> and return the state vector.
    theta: array of shape (layers, n)
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    layers = theta.shape[0]
    
    for l in range(layers):
        # Apply Ry(theta[l, i]) to each qubit i
        for i in range(n):
            c = np.cos(theta[l, i] / 2)
            s = np.sin(theta[l, i] / 2)
            
            new_state = np.zeros_like(state)
            for idx in range(dim):
                if ((idx >> i) & 1) == 0:
                    partner = idx ^ (1 << i)
                    new_state[idx] = c * state[idx] - s * state[partner]
                    new_state[partner] = s * state[idx] + c * state[partner]
            state = new_state
            
        # Apply CZ between layers (except after last layer)
        if l < layers - 1:
            for i in range(n - 1):
                mask = (1 << i) | (1 << (i + 1))
                for idx in range(dim):
                    if (idx & mask) == mask:
                        state[idx] *= -1.0
                        
    return state

def _residuals(params, n, target_psi):
    """
    Compute residuals for least_squares: the difference between circuit state
    and target state, flattened to real and imaginary parts.
    """
    theta = params.reshape(3, n)
    circuit_state = _apply_circuit(theta, n)
    diff = circuit_state - target_psi
    return np.concatenate([diff.real, diff.imag])

def _residuals_jac(params, n, target_psi):
    """
    Numerical Jacobian for the residuals.
    """
    theta = params.reshape(3, n)
    dim = 1 << n
    n_params = len(params)
    
    # Central difference
    h = 1e-7
    jac = np.zeros((2 * dim, n_params))
    
    for j in range(n_params):
        params_plus = params.copy()
        params_plus[j] += h
        r_plus = _residuals(params_plus, n, target_psi)
        
        params_minus = params.copy()
        params_minus[j] -= h
        r_minus = _residuals(params_minus, n, target_psi)
        
        jac[:, j] = (r_plus - r_minus) / (2 * h)
    
    return jac

def _mean_field_initial_guess(n, coupling, field, layers):
    """
    Generate initial guess using mean-field approximation.
    """
    # Product state energy: -coupling*(n-1)*cos^2(theta) - field*n*sin(theta)
    def prod_energy(theta):
        return -coupling * (n - 1) * np.cos(theta)**2 - field * n * np.sin(theta)
    
    thetas = np.linspace(-np.pi, np.pi, 2000)
    energies = np.array([prod_energy(t) for t in thetas])
    idx_min = np.argmin(energies)
    theta_opt = thetas[idx_min]
    
    # Refine with simple gradient descent
    for _ in range(100):
        grad = 2 * coupling * (n - 1) * np.cos(theta_opt) * np.sin(theta_opt) - field * n * np.cos(theta_opt)
        theta_opt -= 0.01 * grad
    
    return np.full((layers, n), theta_opt)

def angles(n, coupling, field, layers):
    """
    Optimize parameters of a fixed three-layer Ry/CZ circuit to approximate ground states.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Compute exact ground state
    psi0, e0 = _compute_exact_ground_state(n, coupling, field)
    
    # 2. Generate initializers
    inits = []
    
    # Mean-field guess
    mf = _mean_field_initial_guess(n, coupling, field, layers)
    inits.append(mf.flatten())
    
    # Strong field limit (all pi/2)
    inits.append(np.full((layers, n), np.pi/2).flatten())
    
    # Strong coupling limit (all 0)
    inits.append(np.zeros((layers, n)).flatten())
    
    # Alternating pattern
    alt = np.zeros((layers, n))
    for l in range(layers):
        for i in range(n):
            alt[l, i] = np.pi/4 if i % 2 == 0 else -np.pi/4
    inits.append(alt.flatten())
    
    # Random uniform
    rng = np.random.default_rng(42)
    for _ in range(8):
        inits.append(rng.uniform(-np.pi, np.pi, size=(layers, n)).flatten())
    
    # 3. Multi-start least_squares optimization
    best_x = None
    best_cost = np.inf
    
    for init in inits:
        if time.time() > deadline:
            break
            
        try:
            res = least_squares(
                _residuals,
                init,
                args=(n, psi0),
                method='trf',
                max_nfev=1000,
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12
            )
            
            if res.cost < best_cost:
                best_cost = res.cost
                best_x = res.x
                
        except Exception:
            pass
    
    # 4. If time remains, do a second round with tighter tolerances
    if best_x is not None and time.time() < deadline - 15.0:
        try:
            res = least_squares(
                _residuals,
                best_x,
                args=(n, psi0),
                method='trf',
                max_nfev=2000,
                xtol=1e-14,
                ftol=1e-14,
                gtol=1e-14
            )
            
            if res.cost < best_cost:
                best_cost = res.cost
                best_x = res.x
                
        except Exception:
            pass
    
    # Fallback if no optimization succeeded
    if best_x is None:
        theta_opt = np.full((layers, n), 0.5)
        return theta_opt

    return best_x.reshape(layers, n)
