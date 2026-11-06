"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations


def _expected_cut(n, edges, gammas, betas):
    """Compute expected cut size for given QAOA angles using exact state simulation."""
    dim = 1 << n
    # Initial state: |+>^n = (1/sqrt(2^n)) * ones
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    # Build cost operator C = sum_{(i,j) in edges} (1 - Z_i Z_j) / 2
    # For MaxCut, C = sum_{(i,j) in edges} (I - Z_i Z_j) / 2
    # exp(-i gamma C) = exp(-i gamma/2 * (|E|*I - sum Z_i Z_j))
    # = exp(-i gamma |E|/2) * exp(i gamma/2 * sum Z_i Z_j)
    # The global phase doesn't matter, so we apply exp(i gamma/2 * sum Z_i Z_j)
    
    # Build adjacency matrix for efficient application
    adj = np.zeros((n, n), dtype=int)
    for (u, v) in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    
    for gamma, beta in zip(gammas, betas):
        # Apply cost term: exp(i * gamma/2 * sum_{(i,j) in E} Z_i Z_j)
        # Z_i Z_j is diagonal with entry +1 if bits i,j same, -1 if different
        # For each basis state, the eigenvalue is product of Z_i Z_j over edges
        # Z_i Z_j = +1 if bit_i == bit_j, -1 otherwise
        # So sum Z_i Z_j = (# same edges) - (# different edges) = 2*(# same) - |E|
        
        # Compute for all states at once
        phases = np.zeros(dim, dtype=float)
        for state in range(dim):
            bits = [(state >> k) & 1 for k in range(n)]
            same_count = 0
            diff_count = 0
            for (u, v) in edges:
                if bits[u] == bits[v]:
                    same_count += 1
                else:
                    diff_count += 1
            # Z_i Z_j eigenvalue: +1 if same, -1 if diff
            # sum = same_count - diff_count
            phase = gamma / 2.0 * (same_count - diff_count)
            phases[state] = phase
        
        psi = psi * np.exp(1j * phases)
        
        # Apply mixer: exp(-i beta * sum X_k) = product_k exp(-i beta X_k)
        # exp(-i beta X) = cos(beta) I - i sin(beta) X
        # X flips the bit
        # We can apply this efficiently by iterating over qubits
        for k in range(n):
            # For each pair of states differing only in bit k
            # new_psi[a] = cos(beta)*psi[a] - i*sin(beta)*psi[b] where b = a with bit k flipped
            # new_psi[b] = cos(beta)*psi[b] - i*sin(beta)*psi[a]
            c = np.cos(beta)
            s = np.sin(beta)
            new_psi = np.zeros_like(psi)
            mask = 1 << k
            for state in range(dim):
                if state & mask == 0:
                    partner = state | mask
                    new_psi[state] = c * psi[state] - 1j * s * psi[partner]
                    new_psi[partner] = c * psi[partner] - 1j * s * psi[state]
            psi = new_psi
    
    # Compute expected cut: <psi| C |psi> where C = sum_{(i,j)} (1 - Z_i Z_j)/2
    # = |E|/2 - 1/2 * sum_{(i,j)} <Z_i Z_j>
    # <Z_i Z_j> = sum_state |psi[state]|^2 * (1 if bits same, -1 if diff)
    probs = np.abs(psi) ** 2
    expected_cut = 0.0
    for (u, v) in edges:
        corr = 0.0
        for state in range(dim):
            bits_u = (state >> u) & 1
            bits_v = (state >> v) & 1
            if bits_u == bits_v:
                corr += probs[state]
            else:
                corr -= probs[state]
        expected_cut += (1.0 - corr) / 2.0
    
    return expected_cut


def _classical_optimum(n, edges):
    """Find exact MaxCut optimum by brute force."""
    best = 0
    for state in range(1 << n):
        bits = [(state >> k) & 1 for k in range(n)]
        cut = 0
        for (u, v) in edges:
            if bits[u] != bits[v]:
                cut += 1
        if cut > best:
            best = cut
    return best


def _spectral_init(n, edges):
    """Initialize angles using spectral/degree-based heuristics."""
    # Compute degrees
    degrees = np.zeros(n)
    for (u, v) in edges:
        degrees[u] += 1
        degrees[v] += 1
    
    # Simple heuristic: gamma ~ 1/sqrt(max_degree), beta ~ 1/n
    max_deg = max(degrees) if len(degrees) > 0 else 1
    gamma_init = 1.0 / np.sqrt(max(max_deg, 1))
    beta_init = 1.0 / n
    
    return np.array([[gamma_init, beta_init], [gamma_init * 0.8, beta_init * 0.7]])


def angles(n, edges, p):
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 100.0  # Leave 20s margin
    
    # Compute classical optimum
    opt = _classical_optimum(n, edges)
    if opt == 0:
        return np.zeros((p, 2))
    
    # Initialize with spectral heuristic
    init = _spectral_init(n, edges)
    
    def objective(params):
        if time.time() > deadline:
            return 0.0
        gammas = params[:p]
        betas = params[p:]
        ec = _expected_cut(n, edges, gammas, betas)
        return -ec / opt  # Minimize negative ratio
    
    # Try multiple starting points
    best_angles = init
    best_val = -1.0
    
    # Starting points: spectral + a few random
    starts = [init.flatten()]
    rng = np.random.RandomState(42)
    for _ in range(3):
        starts.append(rng.uniform(0, np.pi, size=2 * p))
    
    for start in starts:
        if time.time() > deadline:
            break
        try:
            result = minimize(objective, start, method='BFGS', 
                            options={'maxiter': 50, 'gtol': 1e-4})
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
                best_angles = np.column_stack([best_params[:p], best_params[p:]])
        except Exception:
            pass
    
    if best_val < 0:
        # Fallback to initial
        return init
    
    return best_angles
# EVOLVE-BLOCK-END
