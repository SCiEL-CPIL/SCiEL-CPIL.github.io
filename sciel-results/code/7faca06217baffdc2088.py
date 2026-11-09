"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools

def _mps_init(n):
    """Initialize MPS for |0>^n state."""
    # Each site is a tensor of shape (2, 2) where first index is physical, second is bond
    # For |0>^n, all sites are |0> with trivial bonds
    mps = []
    for i in range(n):
        # Tensor of shape (2, 2): physical index, bond index
        # |0> = [1, 0], so tensor[0, 0] = 1, tensor[1, 0] = 0
        t = np.zeros((2, 2), dtype=complex)
        t[0, 0] = 1.0
        mps.append(t)
    return mps

def _apply_ry_to_mps(mps, qubit, theta):
    """Apply Ry(theta) to qubit in MPS."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    ry = np.array([[c, -s], [s, c]], dtype=complex)
    
    new_mps = mps.copy()
    # Ry acts on physical index: new_tensor[a, bond] = sum_b ry[a, b] * tensor[b, bond]
    t = mps[qubit]
    new_t = ry @ t
    new_mps[qubit] = new_t
    return new_mps

def _apply_cz_to_mps(mps, q1, q2):
    """Apply CZ gate between qubits q1 and q2 in MPS."""
    # CZ = diag(1, 1, 1, -1)
    # This couples the two qubits, so we need to merge them, apply CZ, then split
    
    # Assume q1 < q2
    if q1 > q2:
        q1, q2 = q2, q1
    
    # Merge q1 and q2 into a single site
    t1 = mps[q1]
    t2 = mps[q2]
    
    # t1 has shape (2, d1), t2 has shape (2, d2)
    # After merge, we have a tensor of shape (2, 2, d1, d2) -> (4, d1*d2)
    merged = np.einsum('pa,qb->pqab', t1, t2)
    d1, d2 = t1.shape[1], t2.shape[1]
    merged = merged.reshape(4, d1 * d2)
    
    # Apply CZ
    cz = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, -1]], dtype=complex)
    merged = cz @ merged
    
    # Split back: need to find a good split
    # Reshape to (2, 2, d1, d2)
    merged = merged.reshape(2, 2, d1, d2)
    
    # SVD to split into two tensors
    # merged has shape (2, 2, d1, d2), we want to split into (2, d_new) and (2, d_new)
    # Reshape to (2, 2*d1*d2) and SVD
    merged_2d = merged.reshape(2, -1)
    U, S, Vt = np.linalg.svd(merged_2d, full_matrices=False)
    
    # Truncate to keep bond dimension manageable
    max_bond = min(d1 * d2, 16)
    U = U[:, :max_bond]
    S = S[:max_bond]
    Vt = Vt[:max_bond, :]
    
    # Reconstruct
    t1_new = (U * S).reshape(2, max_bond)
    t2_new = (Vt.T * S).reshape(2, max_bond)
    
    new_mps = mps.copy()
    new_mps[q1] = t1_new
    new_mps[q2] = t2_new
    return new_mps

def _mps_expectation(mps, n, coupling, field):
    """Compute <psi|H|psi> for MPS state."""
    dim = 2**n
    energy = 0.0
    
    # Build full state vector from MPS for simplicity (n is small)
    state = np.zeros(dim, dtype=complex)
    for bits in itertools.product([0, 1], repeat=n):
        amp = 1.0
        for i in range(n):
            amp *= mps[i][bits[i], 0]  # This is wrong, MPS has bonds
        state[sum(b << i for i, b in enumerate(bits))] = amp
    
    # Actually, let me just use the full state vector approach since n is small
    # The MPS approach is overkill for n=4,6
    pass

def _circuit_state(theta, n, layers):
    """Build the circuit state vector."""
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1.0
    
    for l in range(layers):
        for i in range(n):
            c = np.cos(theta[l, i] / 2)
            s = np.sin(theta[l, i] / 2)
            gate = np.array([[c, -s], [s, c]], dtype=complex)
            
            shape = [2] * n
            state_tensor = state.reshape(shape)
            perm = [i] + [j for j in range(n) if j != i]
            state_permuted = np.transpose(state_tensor, perm)
            orig_shape = state_permuted.shape
            state_reshaped = state_permuted.reshape(2, -1)
            new_state_reshaped = gate @ state_reshaped
            new_state_permuted = new_state_reshaped.reshape(orig_shape)
            inv_perm = [0] * n
            for k, p in enumerate(perm):
                inv_perm[p] = k
            state = np.transpose(new_state_permuted, inv_perm).reshape(-1)
        
        if l < layers - 1:
            for i in range(n - 1):
                shape = [2] * n
                state_tensor = state.reshape(shape)
                others = [j for j in range(n) if j != i and j != i + 1]
                perm = [i, i + 1] + others
                state_permuted = np.transpose(state_tensor, perm)
                orig_shape = state_permuted.shape
                state_reshaped = state_permuted.reshape(4, -1)
                cz = np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, -1]], dtype=complex)
                new_state_reshaped = np.einsum('ab,b...->a...', cz, state_reshaped)
                new_state_permuted = new_state_reshaped.reshape(orig_shape)
                inv_perm = [0] * n
                for k, p in enumerate(perm):
                    inv_perm[p] = k
                state = np.transpose(new_state_permuted, inv_perm).reshape(-1)
    
    return state

def _expectation_value_fast(state, n, coupling, field):
    """Compute <psi|H|psi> efficiently."""
    dim = 2**n
    energy = 0.0
    
    bits_list = [[(idx >> k) & 1 for k in range(n)] for idx in range(dim)]
    
    for i in range(n - 1):
        zz_exp = 0.0
        for idx in range(dim):
            z_val = 1 if bits_list[idx][i] == bits_list[idx][i+1] else -1
            zz_exp += np.abs(state[idx])**2 * z_val
        energy += -coupling * zz_exp.real
    
    for i in range(n):
        x_exp = 0.0
        for idx in range(dim):
            new_idx = idx ^ (1 << i)
            x_exp += state[idx].conjugate() * state[new_idx]
        energy += -field * x_exp.real
    
    return energy

def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy."""
    dim = 2**n
    H = np.zeros((dim, dim), dtype=complex)
    
    for i in range(n - 1):
        for bits in itertools.product([0, 1], repeat=n):
            z_val = 1 if bits[i] == bits[i + 1] else -1
            idx = sum(b << i for i, b in enumerate(bits))
            H[idx, idx] += -coupling * z_val
    
    for i in range(n):
        for bits in itertools.product([0, 1], repeat=n):
            idx = sum(b << i for i, b in enumerate(bits))
            new_idx = idx ^ (1 << i)
            H[idx, new_idx] += -field
    
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0].real

def angles(n, coupling, field, layers):
    """
    Optimize circuit parameters to minimize mean energy gap per spin.
    Returns theta of shape (layers, n).
    """
    e_ground = _exact_ground_energy(n, coupling, field)
    
    def objective(theta_flat):
        theta = theta_flat.reshape(layers, n)
        state = _circuit_state(theta, n, layers)
        energy = _expectation_value_fast(state, n, coupling, field)
        return (energy - e_ground) / n
    
    def grad(theta_flat):
        theta = theta_flat.reshape(layers, n)
        eps = 1e-4
        g = np.zeros_like(theta)
        for l in range(layers):
            for i in range(n):
                theta_p = theta.copy()
                theta_p[l, i] += eps
                sp = _circuit_state(theta_p, n, layers)
                ep = _expectation_value_fast(sp, n, coupling, field)
                
                theta_m = theta.copy()
                theta_m[l, i] -= eps
                sm = _circuit_state(theta_m, n, layers)
                em = _expectation_value_fast(sm, n, coupling, field)
                
                g[l, i] = (ep - em) / (2 * eps)
        return (g / n).flatten()
    
    best_theta = None
    best_val = np.inf
    
    np.random.seed(42)
    
    initializations = []
    initializations.append(np.zeros((layers, n)))
    
    theta_pi2 = np.zeros((layers, n))
    theta_pi2[-1] = np.pi / 2
    initializations.append(theta_pi2)
    
    for scale in [0.3, 0.5, 0.7, 0.9]:
        theta_f = np.zeros((layers, n))
        for l in range(layers):
            for i in range(n):
                theta_f[l, i] = scale * np.arctan(field / max(coupling, 0.01))
        initializations.append(theta_f)
    
    for _ in range(40):
        initializations.append(np.random.uniform(-np.pi, np.pi, (layers, n)))
    
    for init in initializations:
        try:
            result = minimize(
                objective,
                init.flatten(),
                jac=grad,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            val = result.fun
            if val < best_val:
                best_val = val
                best_theta = result.x.reshape(layers, n)
        except Exception:
            continue
    
    if best_theta is None:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    
    return best_theta
# EVOLVE-BLOCK-END
