# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares


def _exact_ground_state(n, coupling, field):
    """Compute the exact ground state of the open transverse-field Ising chain."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    for basis in range(dim):
        zz_energy = 0.0
        for i in range(n - 1):
            bit_i = (basis >> i) & 1
            bit_ip1 = (basis >> (i + 1)) & 1
            z_i = 1.0 if bit_i == 0 else -1.0
            z_ip1 = 1.0 if bit_ip1 == 0 else -1.0
            zz_energy += z_i * z_ip1
        H[basis, basis] += -coupling * zz_energy
        
        for i in range(n):
            flipped = basis ^ (1 << i)
            H[flipped, basis] += -field

    eigvals, eigvecs = np.linalg.eigh(H)
    ground_idx = np.argmin(eigvals)
    return eigvecs[:, ground_idx], eigvals[ground_idx]


def _apply_circuit(state, angles, n):
    """Apply the 3-layer Ry/CZ circuit to the initial |0>^n state."""
    layers = angles.shape[0]
    state = state.copy()
    dim = 2 ** n
    
    # Precompute bit masks for speed
    # Actually, the current loop is fine. Let's just keep it.
    
    for layer in range(layers):
        for i in range(n):
            theta = angles[layer, i]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            new_state = np.zeros_like(state)
            for basis in range(dim):
                bit_i = (basis >> i) & 1
                if bit_i == 0:
                    new_state[basis] += c * state[basis]
                    new_state[basis | (1 << i)] += s * state[basis]
                else:
                    new_state[basis & ~(1 << i)] += -s * state[basis]
                    new_state[basis] += c * state[basis]
            state = new_state
        
        if layer < layers - 1:
            for i in range(n - 1):
                mask = (1 << i) | (1 << (i + 1))
                for basis in range(dim):
                    if (basis & mask) == mask:
                        state[basis] *= -1
                        
    return state


def _circuit_state(angles, n):
    dim = 2 ** n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    return _apply_circuit(state, angles, n)


def _residuals(params, n, coupling, field, target_state):
    angles = params.reshape((3, n))
    circuit_state = _circuit_state(angles, n)
    diff = circuit_state - target_state
    return np.concatenate([diff.real, diff.imag])


def _generate_inits(n, target_state, num_random=10):
    inits = []
    
    # 1. Fourier-based init
    real_state = target_state.real.copy()
    fft_coeffs = np.fft.fft(real_state)
    magnitudes = np.abs(fft_coeffs)
    top_indices = np.argsort(magnitudes)[::-1][:5]
    
    for idx in top_indices:
        freq = idx / (2 ** n)
        base_angle = 2 * np.pi * freq
        init = np.zeros((3, n))
        for layer in range(3):
            for i in range(n):
                init[layer, i] = base_angle * (i + 1) / (n + 1) + layer * 0.1
        inits.append(init.flatten())
        
    # 2. Uniform inits
    inits.append(np.zeros(3 * n))
    init_pi2 = np.zeros((3, n))
    init_pi2[-1] = np.pi / 2
    inits.append(init_pi2.flatten())
    
    # 3. Random inits
    np.random.seed(42)
    for _ in range(num_random):
        init = np.random.uniform(-np.pi, np.pi, (3, n))
        inits.append(init.flatten())
        
    return inits


def angles(n, coupling, field, layers):
    target_state, ground_energy = _exact_ground_state(n, coupling, field)
    inits = _generate_inits(n, target_state, num_random=10)
    
    best_params = None
    best_cost = np.inf
    
    for init in inits:
        try:
            result = least_squares(
                _residuals,
                init,
                args=(n, coupling, field, target_state),
                method='trf',
                max_nfev=2000,
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12
            )
            cost = result.cost
            if cost < best_cost:
                best_cost = cost
                best_params = result.x
        except Exception:
            continue
    
    if best_params is None:
        return np.zeros((layers, n))
    
    return best_params.reshape((layers, n))
# EVOLVE-BLOCK-END
