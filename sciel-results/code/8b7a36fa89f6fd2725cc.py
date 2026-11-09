"""Optimize 3-layer Ry/CZ circuit angles for transverse-field Ising ground states."""
import numpy as np
import time

_START = time.time()
_DEADLINE = 115.0  # safety margin before 120s kill

# Precompute single-qubit rotation matrices
def _ry(theta):
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]])

# Precompute Pauli matrices
_I2 = np.eye(2)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_CZ = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, -1]], dtype=complex)


def _build_hamiltonian(n, coupling, field):
    """Build H = -coupling * sum Z_i Z_{i+1} - field * sum X_i for open chain."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    # ZZ terms
    for i in range(n - 1):
        for basis in range(dim):
            bi = (basis >> i) & 1
            bj = (basis >> (i + 1)) & 1
            z = (1 if bi == 0 else -1) * (1 if bj == 0 else -1)
            H[basis, basis] += -coupling * z
    # X terms
    for i in range(n):
        for basis in range(dim):
            target = basis ^ (1 << i)
            H[target, basis] += -field
    return H


def _circuit_state(theta, n, layers):
    """Simulate the circuit. theta shape (layers, n). Returns state vector (2^n,)."""
    dim = 1 << n
    # Start |0>^n
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    # We represent state as tensor of shape (2,)*n for efficient single-qubit ops
    psi_t = psi.reshape([2] * n)
    for layer in range(layers):
        # Apply Ry(theta[layer, i]) to every qubit i
        for i in range(n):
            th = theta[layer, i]
            c = np.cos(th / 2.0)
            s = np.sin(th / 2.0)
            # R_y = [[c, -s],[s, c]]
            # For each configuration of other qubits, transform the 2-dim subspace at axis i
            # Use einsum-style manual loop
            shape = [2] * n
            new_t = np.zeros_like(psi_t)
            # Move axis i to front for easier indexing
            for idx in range(dim):
                bits = [(idx >> b) & 1 for b in range(n)]
                # value of psi_t at this index
                v0 = psi_t[tuple(bits)]
                # apply Ry on qubit i: new[bits] where bit i flipped uses the other bit
                # Ry acts on basis |b_i>: 
                #   |0> -> c|0> + s|1>
                #   |1> -> -s|0> + c|1>
                # So new[b_i=0] = c*old[b_i=0] + s*old[b_i=1]
                #    new[b_i=1] = -s*old[b_i=0] + c*old[b_i=1]
                pass
        # This loop approach is O(dim) per qubit per layer = O(n*dim) per layer, fine for dim<=64
        # Let me redo with a cleaner tensor approach
        psi_t = _apply_all_ry(psi_t, theta[layer], n)
        if layer < layers - 1:
            psi_t = _apply_cz_layer(psi_t, n)
    return psi_t.reshape(dim)


def _apply_all_ry(psi_t, thetas, n):
    """Apply Ry(thetas[i]) to each qubit i independently (they commute since on different qubits)."""
    dim = 1 << n
    # For each qubit i, do a 2x2 rotation on that axis
    # We process qubit by qubit
    for i in range(n):
        th = thetas[i]
        c = np.cos(th / 2.0)
        s = np.sin(th / 2.0)
        # Build index arrays
        # We'll create new tensor
        new_t = np.zeros_like(psi_t)
        # Iterate over all indices
        for idx in range(dim):
            bits = [(idx >> b) & 1 for b in range(n)]
            b_i = bits[i]
            # old value at this index
            old0 = psi_t[tuple(bits)]
            # partner index with bit i flipped
            bits_flip = bits[:]
            bits_flip[i] = 1 - b_i
            old1 = psi_t[tuple(bits_flip)]
            # Ry: |0> -> c|0> + s|1>, |1> -> -s|0> + c|1>
            # new[b_i=0] = c*old[b_i=0] + s*old[b_i=1]
            # new[b_i=1] = -s*old[b_i=0] + c*old[b_i=1]
            if b_i == 0:
                new_t[tuple(bits)] = c * old0 + s * old1
                new_t[tuple(bits_flip)] = -s * old0 + c * old1
            # else handled when we hit the flipped index
        psi_t = new_t
    return psi_t


def _apply_cz_layer(psi_t, n):
    """Apply CZ on every adjacent pair (i, i+1) in ascending order."""
    dim = 1 << n
    for i in range(n - 1):
        # CZ is diagonal: phase -1 when both bits are 1
        for idx in range(dim):
            bi = (idx >> i) & 1
            bj = (idx >> (i + 1)) & 1
            if bi == 1 and bj == 1:
                psi_t[tuple([(idx >> b) & 1 for b in range(n)])] *= -1.0
    return psi_t


def _energy_and_grad(theta, n, coupling, field, H, E0):
    """Compute energy E = <psi|H|psi> and gradient w.r.t. theta. Returns (E, grad)."""
    psi = _circuit_state(theta, n, layers=theta.shape[0])
    E = np.real(np.vdot(psi, H @ psi))
    # Gradient via parameter-shift: for Ry, dE/dtheta = (E(theta+pi/2) - E(theta-pi/2))/2
    # But that's expensive (2*params evaluations). Instead use analytic gradient.
    # For a single Ry gate on qubit i at layer L:
    #   d/dtheta |psi> = (1/2) R_y'(theta) ... 
    # R_y(theta) = [[c,-s],[s,c]], R_y'(theta) = (1/2)[[-s,-c],[c,-s]]
    # The gradient of E = <psi|H|psi> w.r.t. theta_i^L is:
    #   2 Re[ <psi| (dRy/dtheta) |psi> ] projected through the rest of circuit
    # This is complex. Let's use finite-difference-free analytic:
    # Actually, simplest robust: parameter-shift rule for Ry:
    #   dE/dtheta = (E(theta + pi/2) - E(theta - pi/2)) / 2
    # This is exact for rotation gates. Cost: 2 * (layers*n) state sims.
    # For n=6, layers=3: 36 sims. Each sim is O(n * dim) with dim=64 -> ~2304 ops. Total ~83k ops. Fast.
    grad = np.zeros_like(theta)
    for L in range(theta.shape[0]):
        for i in range(n):
            theta_plus = theta.copy()
            theta_plus[L, i] += np.pi / 2.0
            theta_minus = theta.copy()
            theta_minus[L, i] -= np.pi / 2.0
            psi_plus = _circuit_state(theta_plus, n, layers=theta.shape[0])
            psi_minus = _circuit_state(theta_minus, n, layers=theta.shape[0])
            E_plus = np.real(np.vdot(psi_plus, H @ psi_plus))
            E_minus = np.real(np.vdot(psi_minus, H @ psi_minus))
            grad[L, i] = (E_plus - E_minus) / 2.0
    return E, grad


def _objective(theta, n, coupling, field, H, E0):
    """Return mean energy gap per spin: (E - E0)/n."""
    E, _ = _energy_and_grad(theta, n, coupling, field, H, E0)
    return (E - E0) / n


def angles(n, coupling, field, layers):
    global _START, _DEADLINE
    if time.time() - _START > _DEADLINE:
        # Return a safe default
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta

    dim = 1 << n
    H = _build_hamiltonian(n, coupling, field)
    # Exact ground energy
    eigvals = np.linalg.eigvalsh(H)
    E0 = np.min(eigvals).real

    best_theta = None
    best_val = np.inf

    # Initial guesses
    inits = []
    # 1. All zero
    inits.append(np.zeros((layers, n)))
    # 2. Last layer pi/2 (product |+>)
    t2 = np.zeros((layers, n))
    t2[-1] = np.pi / 2
    inits.append(t2)
    # 3. Heuristic: earlier layers small, later layers larger
    for L in range(layers):
        for i in range(n):
            pass
    t3 = np.zeros((layers, n))
    t3[0] = 0.2
    t3[1] = 0.4
    t3[2] = 0.8
    inits.append(t3)
    # 4. Random
    np.random.seed(42)
    for _ in range(5):
        inits.append(np.random.uniform(-np.pi/2, np.pi/2, size=(layers, n)))

    from scipy.optimize import minimize

    def obj_flat(x):
        theta = x.reshape(layers, n)
        E, _ = _energy_and_grad(theta, n, coupling, field, H, E0)
        return (E - E0) / n

    def grad_flat(x):
        theta = x.reshape(layers, n)
        _, g = _energy_and_grad(theta, n, coupling, field, H, E0)
        return g.reshape(-1)

    bounds = [(-np.pi, np.pi)] * (layers * n)

    for init in inits:
        if time.time() - _START > _DEADLINE - 5:
            break
        x0 = init.reshape(-1)
        try:
            res = minimize(obj_flat, x0, jac=grad_flat, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-12})
            val = res.fun
            if val < best_val:
                best_val = val
                best_theta = res.x.reshape(layers, n)
        except Exception:
            pass

    # Final polish with tighter tolerance on the best
    if best_theta is not None and time.time() - _START < _DEADLINE - 10:
        x0 = best_theta.reshape(-1)
        try:
            res = minimize(obj_flat, x0, jac=grad_flat, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_theta = res.x.reshape(layers, n)
        except Exception:
            pass

    if best_theta is None:
        best_theta = np.zeros((layers, n))
        best_theta[-1] = np.pi / 2

    return best_theta
