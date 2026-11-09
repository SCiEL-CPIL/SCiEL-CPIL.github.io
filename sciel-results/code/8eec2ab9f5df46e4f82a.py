import numpy as np
import time
from scipy.optimize import minimize

def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix for the Ising chain."""
    # Basis size 2^n
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=float)
    
    # Interaction term: -coupling * sum Z_i Z_{i+1}
    for i in range(n - 1):
        # Z_i Z_{i+1}
        # For each basis state |b>, Z_i gives +1 if b_i=0, -1 if b_i=1.
        # We can iterate over all basis states
        pass # Using bitwise is easier for small n, but matrix construction via kron is standard.
        # Let's use bitwise for speed and simplicity in this context.
        
    # Actually, let's just use the bitwise energy calculation for the circuit state, 
    # and exact diagonalization for the ground energy.
    
    # For exact diagonalization, we need the matrix.
    # Z_i Z_{i+1} is diagonal.
    # X_i is off-diagonal.
    
    for b in range(dim):
        # Diagonal part from ZZ terms
        e_diag = 0.0
        for i in range(n - 1):
            bit_i = (b >> i) & 1
            bit_j = (b >> (i + 1)) & 1
            # Z eigenvalue: 1 if 0, -1 if 1
            z_i = 1 if bit_i == 0 else -1
            z_j = 1 if bit_j == 0 else -1
            e_diag -= coupling * z_i * z_j
            
        H[b, b] += e_diag
        
    # Off-diagonal part from X terms
    # X_i flips bit i.
    for i in range(n):
        for b in range(dim):
            # If we flip bit i, we go to state b'
            b_prime = b ^ (1 << i)
            # H_{b, b'} += -field
            # Since H is Hermitian, H_{b', b} is also -field
            # We add to H[b, b_prime]
            H[b, b_prime] -= field
            
    return H

def _exact_ground_energy(n, coupling, field):
    H = _build_hamiltonian(n, coupling, field)
    # Use np.linalg.eigvalsh for Hermitian matrices
    evals = np.linalg.eigvalsh(H)
    return evals[0]

def _circuit_energy(theta, n, coupling, field):
    """
    theta: array of shape (layers, n)
    Returns energy of the state produced by the circuit.
    """
    layers = theta.shape[0]
    dim = 1 << n
    # Start with |0>^n
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    
    # Precompute rotation matrices for unique angles? 
    # Ry(theta) = cos(theta/2) I - i sin(theta/2) X
    # We apply Ry to all qubits in a layer.
    
    # To be efficient, we can apply Ry to each qubit sequentially.
    # Or construct the full layer operator.
    # For n=6, dim=64. 3 layers.
    # Applying Ry to qubit i involves tensor product with I.
    
    # Let's iterate through layers
    for l in range(layers):
        # Apply Ry(theta_l, i) for all i
        for i in range(n):
            th = theta[l, i]
            c = np.cos(th / 2)
            s = -1j * np.sin(th / 2)
            
            # Apply to qubit i
            # We need to reshape psi to (2, 2, ..., 2) with n axes
            # Or use bitwise logic.
            # Bitwise logic:
            # For each basis state b, if bit i is 0, new amplitude gets c*old + s*old_{b^1<<i}
            # If bit i is 1, new amplitude gets c*old + s*old_{b^1<<i} ... wait.
            # Ry = [[c, s], [s*, c]]? No.
            # Ry(theta) = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]] in real?
            # Standard Ry(theta) = exp(-i theta Y / 2) = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]]
            # Wait, Y = [[0, -i], [i, 0]].
            # exp(-i theta/2 Y) = cos(theta/2) I - i sin(theta/2) Y
            # = [[cos, -sin], [sin, cos]]?
            # -i * Y = [[0, -1], [1, 0]].
            # So Ry = [[c, -s], [s, c]] where s = sin(theta/2).
            # Let's check: theta=pi/2. c=1/sqrt2, s=1/sqrt2.
            # Ry(pi/2) |0> = 1/sqrt2 |0> + 1/sqrt2 |1>. Correct.
            
            c = np.cos(th / 2)
            s = np.sin(th / 2)
            
            new_psi = np.zeros_like(psi)
            for b in range(dim):
                bit_i = (b >> i) & 1
                if bit_i == 0:
                    # State |...0...>
                    # Amplitude at b comes from c * psi[b] + s * psi[b ^ (1<<i)]
                    # Wait, Ry acts on the subspace.
                    # U |0> = c|0> + s|1>
                    # U |1> = -s|0> + c|1>
                    # So if we have state |b> where bit i is 0:
                    # It receives contribution from psi[b] * c and psi[b^1<<i] * (-s)?
                    # Let's trace:
                    # psi = a|0> + b|1>
                    # U psi = a(c|0> + s|1>) + b(-s|0> + c|1>)
                    # = (ac - bs)|0> + (as + bc)|1>
                    # So new_amp[0] = c*a - s*b
                    # new_amp[1] = s*a + c*b
                    # Here a = psi[b] (bit 0), b = psi[b^1<<i] (bit 1)
                    # So new_psi[b] = c * psi[b] - s * psi[b ^ (1<<i)]
                    # new_psi[b ^ (1<<i)] = s * psi[b] + c * psi[b ^ (1<<i)]
                    
                    partner = b ^ (1 << i)
                    new_psi[b] = c * psi[b] - s * psi[partner]
                    new_psi[partner] = s * psi[b] + c * psi[partner]
                # If bit_i == 1, we already handled it as partner of the 0 case
            psi = new_psi
            
        # Apply CZ to adjacent pairs
        # CZ = |0><0| \otimes I + |1><1| \otimes Z
        # CZ |a, b> = (-1)^{a*b} |a, b>
        # So it just flips the sign of the amplitude if both bits are 1.
        for i in range(n - 1):
            mask_i = 1 << i
            mask_j = 1 << (i + 1)
            for b in range(dim):
                if (b & mask_i) and (b & mask_j):
                    psi[b] *= -1

    # Compute Energy
    # E = <psi| H |psi>
    # H = -J sum ZZ - h sum X
    # <psi| ZZ |psi> can be computed from probabilities?
    # No, ZZ is diagonal. <psi| Z_i Z_j |psi> = sum_b |psi[b]|^2 * (z_i(b) * z_j(b))
    # <psi| X_i |psi> = sum_b psi[b]^* psi[b ^ (1<<i)]
    
    energy = 0.0
    probs = np.abs(psi)**2
    
    # ZZ terms
    for i in range(n - 1):
        for b in range(dim):
            bit_i = (b >> i) & 1
            bit_j = (b >> (i + 1)) & 1
            z_i = 1 if bit_i == 0 else -1
            z_j = 1 if bit_j == 0 else -1
            energy -= coupling * probs[b] * z_i * z_j
            
    # X terms
    for i in range(n):
        for b in range(dim):
            partner = b ^ (1 << i)
            # <psi| X_i |psi> = sum_b psi[b]^* psi[partner]
            # Since X is Hermitian and real in this basis?
            # X |b> = |partner>
            # <psi| X |psi> = sum_b <psi|b> <b|X|psi> = sum_b psi[b]^* <b|psi_partner>
            # Wait, X|psi> = sum_c psi[c] X|c> = sum_c psi[c] |c^1<<i>
            # <psi| X|psi> = sum_b psi[b]^* (X|psi>)_b = sum_b psi[b]^* psi[b^1<<i]
            energy -= field * (np.conj(psi[b]) * psi[partner]).real
            
    return energy.real

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Determine symmetry parameters
    if n == 4:
        num_unique = 2
        # Map unique params u[0], u[1] to 4 angles
        # [u0, u1, u1, u0]
        def expand_params(u):
            # u is flat array of length layers * num_unique
            u = u.reshape(layers, num_unique)
            theta = np.zeros((layers, n))
            for l in range(layers):
                theta[l, 0] = u[l, 0]
                theta[l, 1] = u[l, 1]
                theta[l, 2] = u[l, 1]
                theta[l, 3] = u[l, 0]
            return theta
    elif n == 6:
        num_unique = 3
        # Map unique params u[0], u[1], u[2] to 6 angles
        # [u0, u1, u2, u2, u1, u0]
        def expand_params(u):
            u = u.reshape(layers, num_unique)
            theta = np.zeros((layers, n))
            for l in range(layers):
                theta[l, 0] = u[l, 0]
                theta[l, 1] = u[l, 1]
                theta[l, 2] = u[l, 2]
                theta[l, 3] = u[l, 2]
                theta[l, 4] = u[l, 1]
                theta[l, 5] = u[l, 0]
            return theta
    else:
        # Fallback: no symmetry assumption, optimize all
        num_unique = n
        def expand_params(u):
            return u.reshape(layers, n)

    e_ground = _exact_ground_energy(n, coupling, field)
    
    def loss(u):
        if time.time() > deadline:
            return 1e6
        theta = expand_params(u)
        e_circ = _circuit_energy(theta, n, coupling, field)
        return e_circ - e_ground

    best_u = None
    best_loss = float('inf')
    
    # Multi-start optimization
    # Try a few random starts and one zero start
    np.random.seed(42)
    starts = [np.zeros(layers * num_unique)]
    for _ in range(10):
        starts.append(np.random.uniform(-np.pi, np.pi, size=layers * num_unique))
        
    for u0 in starts:
        if time.time() > deadline:
            break
        try:
            res = minimize(loss, u0, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 2000, 'maxfev': 2000})
            if res.fun < best_loss:
                best_loss = res.fun
                best_u = res.x
        except Exception:
            pass
            
    if best_u is None:
        # Fallback to zero
        best_u = np.zeros(layers * num_unique)
        
    theta = expand_params(best_u)
    return theta
